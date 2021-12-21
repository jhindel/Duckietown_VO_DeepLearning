
import numpy as np
import torch
import pytorch_lightning as pl
from .loss import DeepVO_loss
from .model import ConvNet, ConvLstmNet
from .dataset import DuckietownDataset

class DeepVONet(pl.LightningModule):

    # TODO patience

    def __init__(self, args):
        super().__init__()
        if args["model"] == "ConvNet":
            self.architecture = ConvNet(args["resize"], args["dropout_p"])
        elif args["model"] == "ConvLstmNet":
            self.architecture = ConvLstmNet(args["resize"], args["dropout_p"])
        self.args = args
        self.test_data = DuckietownDataset(self.args["test_split"], self.args)
        self.trajectories = np.zeros(len(self.test_data))

    def forward(self, x):
        return self.architecture(x)

    def compute_loss(self, batch):
        images_stacked = batch[0]
        relative_pose = batch[1]

        # Initialize with zeros the Variable containing estimated relative_pose
        shape = (relative_pose.shape[1], relative_pose.shape[0],
                 relative_pose.shape[2])  # (trajectory_length,batch_size,3)
        relative_pose_pred = torch.zeros(shape)
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
        self.trajectories[batch_nb] = relative_pose_predicted
        return loss

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.architecture.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])

    def train_dataloader(self):
        train_data = DuckietownDataset(self.args["train_split"], self.args)
        return torch.utils.data.DataLoader(train_data, batch_size=self.args["bsize"], num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        val_data = DuckietownDataset(self.args["val_split"], self.args)
        return torch.utils.data.DataLoader(val_data, batch_size=self.args["bsize"], num_workers=4, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=1, num_workers=4, shuffle=False, drop_last=True)




