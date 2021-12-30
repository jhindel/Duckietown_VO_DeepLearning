
import numpy as np
import torch
import pytorch_lightning as pl
from .loss import DeepVO_loss, CTCNet_loss
from .model import ConvNet, ConvNet2, ConvLstmNet
from .dataset import DuckietownDataset, DuckietownDatasetCTC
from .ctc_block_utils import get_all_subsequences, get_all_compositions

class DeepVONet(pl.LightningModule):

    # TODO patience

    def __init__(self, args):
        super().__init__()
        if args["model"] == "ConvNet":
            self.architecture = ConvNet(args["resize"], args["dropout_p"])
        elif args["model"] == "ConvNet2":
            self.architecture = ConvNet2(args["resize"], args["dropout_p"])
        elif args["model"] == "ConvLstmNet":
            self.architecture = ConvLstmNet(args["resize"], args["dropout_p"])
        self.args = args
        self.test_data = DuckietownDataset(self.args["test_split"], self.args)
        self.trajectories = [None] * len(self.test_data)

    def forward(self, x):
        return self.architecture(x)

    def compute_loss(self, batch):
        images_stacked = batch[0]
        relative_pose = batch[1]

        # Initialize with zeros the Variable containing estimated relative_pose
        shape = (relative_pose.shape[1], relative_pose.shape[0],
                 relative_pose.shape[2])  # (trajectory_length,batch_size,3)
        relative_pose_pred = torch.zeros(shape)
        relative_pose_pred = relative_pose_pred.type_as(relative_pose)
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)  # (trajectory_length, batch_size, 3,64,64)

        # TODO check if can vectorize it
        for t in range(len(images_stacked)):
            # input (batch_size, 3, 64, 64), output (batch_size, 3)
            # relative_pose_pred:(trajectory_length, batch_size, 3)
            relative_pose_pred[t] = self(images_stacked[t])

        relative_pose_pred = relative_pose_pred.permute(1, 0, 2)  # (batch_size, trajectory_length, 3)

        loss = DeepVO_loss(relative_pose, relative_pose_pred, self.args["K"])

        return loss, relative_pose_pred

    def training_step(self, batch, batch_nb):
        loss, _ = self.compute_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, _ = self.compute_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        loss, relative_pose_predicted = self.compute_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.trajectories[batch_nb] = np.array(relative_pose_predicted.cpu().data)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.architecture.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])

    def train_dataloader(self):
        train_data = DuckietownDataset(self.args["train_split"], self.args)
        return torch.utils.data.DataLoader(train_data, batch_size=self.args["bsize"], num_workers=2, shuffle=False, drop_last=True)

    def val_dataloader(self):
        val_data = DuckietownDataset(self.args["val_split"], self.args)
        return torch.utils.data.DataLoader(val_data, batch_size=self.args["bsize"], num_workers=2, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=1, num_workers=2, shuffle=False, drop_last=True)

class CTCNet(DeepVONet):
    def __init__(self, args):
        super().__init__(args)
        if args["model"] == "ConvNet":
            self.noisy_estimator = ConvNet(args["resize"], args["dropout_p"])
        elif args["model"] == "ConvNet2":
            self.noisy_estimator = ConvNet2(args["resize"], args["dropout_p"])
        elif args["model"] == "ConvLstmNet":
            self.noisy_estimator = ConvLstmNet(args["resize"], args["dropout_p"])
        self.architecture.load_state_dict(torch.load(args["pretrained_DeepVO_model_path"]))
        self.noisy_estimator.load_state_dict(torch.load(args["noisy_estimator_path"]))
        self.noisy_estimator.eval()
        for params in self.noisy_estimator.parameters():
            params.requires_grad = False
        # self.noisy_estimator = self.architecture

    def compute_training_loss(self, batch):
        batch = batch.permute(1, 0, 2, 3, 4) # (trajectory_length, batch_size, 3, 64, 64)

        trajectories = get_all_subsequences(batch, self.args["max_step_size"])
        poses = []
        for i in range(batch.shape[0]-1):
            stacked_images = torch.cat((batch[i], batch[i+1]), 1)
            # do not compute gradients at this point: ?!
            pose = self.architecture(stacked_images)
            poses.append(pose)
        poses_composition_lists = get_all_compositions(poses, self.args["max_step_size"])

        loss = 0
        for trajectory, poses_composition_list in zip(trajectories, poses_composition_lists):
            shape = (trajectory.shape[0]-1, batch.shape[1], 3)
            poses_direct = torch.zeros(shape)
            poses_DeepVO = torch.zeros(shape)
            poses_composition = torch.zeros(shape)
            for t, elem in enumerate(poses_composition_list):
                # input (batch_size, 3, 64, 64), output (batch_size, 3)
                stacked_images = torch.cat((trajectory[t], trajectory[t+1]), 1)
                poses_direct[t] = self.forward(stacked_images)
                poses_DeepVO[t] = self.noisy_estimator(stacked_images)
                poses_composition[t] = elem
            poses_composition = poses_composition.permute(1, 0, 2)  # (batch_size, trajectory_length, 3)
            poses_direct = poses_direct.permute(1, 0, 2)
            poses_DeepVO = poses_DeepVO.permute(1, 0, 2)
            loss += CTCNet_loss(poses_DeepVO, poses_composition, poses_direct, K=self.args["K"], alpha=self.args["alpha"], beta=self.args["beta"])
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.compute_training_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.compute_training_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        loss, relative_pose_predicted = self.compute_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.trajectories[batch_nb] = np.array(relative_pose_predicted.cpu().data)
        return loss

    def train_dataloader(self):
        train_data = DuckietownDatasetCTC(self.args["train_split"], self.args)
        return torch.utils.data.DataLoader(train_data, batch_size=self.args["bsize"], num_workers=2, shuffle=False, drop_last=True)

    def val_dataloader(self):
        val_data = DuckietownDatasetCTC(self.args["val_split"], self.args)
        return torch.utils.data.DataLoader(val_data, batch_size=self.args["bsize"], num_workers=2, shuffle=False, drop_last=True)
