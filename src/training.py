import copy
import datetime
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
from torch.autograd import Variable
import os
from tqdm.autonotebook import tqdm
from .utils import relative2absolute
from .loss import DeepVO_loss


def train_model(model, train_loader, val_loader, args):
    """

    :param model: network architecture to be trained
    :param train_loader: training loader
    :param val_loader: validation loader
    :param args: hyperparameters
    :return:
    """

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    # wandb.watch(model, criterion, log="all", log_freq=10)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Creating logs : dictionary of lists for train_loss, val_loss
    logs = dict(train_loss=[], val_loss=[])

    # Counting patience
    count_patience = 0

    # Training and validating on each epoch
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    # TODO check if we can use pre-trained weights from KITTI

    for epoch in range(1, args["epochs"] + 1):

        if count_patience == args["patience"]:
            break

        print('Epoch {}/{}; '.format(epoch, args["epochs"]), end='')
        # print('-' * 10)

        for phase in ["train", "val"]:

            if phase == 'train':
                data_loader = train_loader
                model.train()
            else:
                data_loader = val_loader
                model.eval()

            running_loss = 0.0
            nb_batch = 0

            for batch_idx, (images_stacked, relative_pose) in enumerate(tqdm(data_loader)):

                if torch.cuda.is_available():
                    images_stacked, relative_pose = images_stacked.cuda(), relative_pose.cuda()

                images_stacked = images_stacked.permute(1, 0, 2, 3, 4)  # (trajectory_length, batch_size, 3,64,64)
                images_stacked, relative_pose = Variable(images_stacked), Variable(relative_pose)

                # Initialize with zeros the Variable containing estimated relative poses
                relative_pose_pred = Variable(
                    torch.zeros(relative_pose.shape))  # (batch_size, trajectory_length,3)
                relative_pose_pred = relative_pose_pred.permute(1, 0, 2)  # (trajectory_length, batch_size, 3)

                if torch.cuda.is_available():
                    relative_pose_pred = relative_pose_pred.cuda()

                # TODO check but should be done automatically
                model.reset_hidden_states(bsize=args["bsize"], zero=True)  # reset to 0 the hidden states of RNN
                # TODO check if can't vectorize it
                for t in range(len(images_stacked)):
                    relative_pose_pred[t] = model(
                        images_stacked[t])  # input (batch_size, 3, 64, 64), output (batch_size, 3)
                    # relative_pose_pred:(trajectory_length, batch_size, 3)

                relative_pose_pred = relative_pose_pred.permute(1, 0, 2)  # (batch_size, trajectory_length, 3)

                loss = DeepVO_loss(relative_pose, relative_pose_pred, args["K"])

                # if phase is 'train', compute gradient and do optimizer step
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                nb_batch += 1
                # print(nb_batch)
            epoch_loss = running_loss / nb_batch
            wandb.log({f"{phase}_loss": epoch_loss}, step=epoch)
            logs[phase + '_loss'].append(epoch_loss)

            if phase == 'train':
                print('{} loss, {:.4f}; '.format(phase, epoch_loss), end='')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                count_patience = 0
                print('{} loss, {:.4f}.'.format(phase, epoch_loss))
            elif phase == 'val' and args["patience"] is not None:
                count_patience += 1
                print('{} loss, {:.4f}; '.format(phase, epoch_loss), end='')
                print("patience, {}.".format(count_patience))
            elif phase == 'val' and args["patience"] is None:
                print('{} loss, {:.4f}.'.format(phase, epoch_loss))

        # save parameters after training for one specific epoch
        if epoch % 10 == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict()}
            torch.save(state, os.path.join(args["checkpoint_path"], "checkpoint_{}.pth".format(epoch)))
            wandb.save(state)

    time_elapsed = time.time() - since

    # determine number of trained epochs and best epoch
    trained_epochs = len(logs['train_loss'])
    if args["patience"] is not None and trained_epochs != args["epochs"]:
        best_epoch = trained_epochs - args["patience"]
    else:
        best_epoch = np.argmin(logs['val_loss']) + 1
    logs['trained_epochs'] = trained_epochs
    logs['best_epoch'] = best_epoch

    # print training time and best validation loss
    print('\nTraining complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

    # Pickle logs
    filename = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')) + "_logs.pkl"
    filepath = os.path.join(args["checkpoint_path"], filename)

    with open(filepath, "wb") as f:
        pickle.dump(logs, f)

        # load best model weights
    model.load_state_dict(best_model)

    state = {'best_epoch': best_epoch, 'state_dict': best_model}
    filename = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')) + \
               "_best_model_state_dict.pth"
    filepath = os.path.join(args["checkpoint_path"], filename)
    torch.save(state, filepath)
    wandb.save(filepath)

    return model, logs, args


def test_model(model, test_loader, args):
    """

    :param model: best model to be tested
    :param test_loader: test dataloader
    :param args: hyperparameters
    :return:
    """

    print("TESTING")

    # to save complete predicted trajectories
    trajectories_nb = len(test_loader)
    trajectories_pred = Variable(torch.zeros((trajectories_nb, args["trajectory_length"], 3)))
    trajectories = Variable(torch.zeros((trajectories_nb, args["trajectory_length"], 3)))

    # For BN and dropout layers
    model.eval()

    for batch_id, (images_stacked, relative_pose) in enumerate(tqdm(test_loader)):
        # images_stacked.shape = (1,trajectory_length,3,64,64), relative_pose.shape = (1,trajectory_length,3)

        trajectories[batch_id] = relative_pose

        # Computation of estimated relative_pose
        if torch.cuda.is_available():
            images_stacked, relative_pose = images_stacked.cuda(), relative_pose.cuda()

        images_stacked, relative_pose = Variable(images_stacked), Variable(relative_pose)

        # Initialize with zeros the Variable containing estimated relative_pose
        relative_pose_pred = Variable(torch.zeros(relative_pose.shape))
        relative_pose_pred = relative_pose_pred.permute(1, 0, 2)  # (trajectory_length,batch_size,3)

        if torch.cuda.is_available():
            relative_pose_pred = relative_pose_pred.cuda()

        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)  # (trajectory_length,batch_size,3,64,64)

        # Initialize hidden states of RNN to zero before predicting TODO check this (reset hidden state after each
        #  prediction)
        model.reset_hidden_states(bsize=1, zero=True)

        for t in range(len(images_stacked)):
            relative_pose_pred[t] = model(images_stacked[t])  # input (batch_size, 3, 64, 64), output (batch_size, 3)
            # relative_pose_pred: (trajectory_length, batch_size, 3)

        relative_pose_pred = relative_pose_pred.permute(1, 0, 2)  # (batch_size, trajectory_length, 3)

        # add to list of all predictions
        trajectories_pred[batch_id] = relative_pose_pred

    # calculate loss of all predictions
    loss = DeepVO_loss(trajectories, trajectories_pred, args["K"])

    print('{} loss: {:.4f}'.format('test', loss))

    # Pickle test_data: trajectories_pred
    # reshape all predicitions into array of size (batch_size * 3)
    trajectories_pred = np.asarray(trajectories_pred.data)
    # print(trajectories_pred.shape)
    trajectories_pred = trajectories_pred.reshape(-1, trajectories_pred.shape[-1])
    filename = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')) + \
               "_trajectories_estimated_relative_transformation.pkl"
    filepath = os.path.join(args["checkpoint_path"], filename)

    with open(filepath, "wb") as f:
        pickle.dump(trajectories_pred, f)

    wandb.run.summary["test_loss"] = loss

    return loss, trajectories_pred


def plot_train_valid(logs, args):
    # number of trained epochs and best epoch
    trained_epochs = logs['trained_epochs']
    best_epoch = logs['best_epoch']

    # Results summary
    print('\nBest result at epoch %d; training loss, %.4f; validation loss, %.4f.' %
          (best_epoch, logs['train_loss'][best_epoch - 1], logs['val_loss'][best_epoch - 1]))

    # Graphic
    print('\nGraphic:')
    fig, ax = plt.subplots()
    line_up, = ax.plot(list(range(1, trained_epochs + 1)), logs['train_loss'])
    line_down, = ax.plot(list(range(1, trained_epochs + 1)), logs['val_loss'])
    ax.legend([line_up, line_down], ['train', 'validation'])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.axvline(x=best_epoch, color='k', linestyle=':')
    wandb.log({"full training graph": wandb.Image(fig)})
    plt.show()


def plot_test(test_data, relative_poses_pred):
    fig, ax = plt.subplots()
    absolute_poses = test_data.get_absolute_poses().to_numpy()
    absolute_poses_pred = relative2absolute(relative_poses_pred, absolute_poses[0])
    ax.plot(absolute_poses_pred[:, 0], absolute_poses_pred[:, 1], label='predicted trajectory')
    ax.plot(absolute_poses[:, 0], absolute_poses[:, 1], label='ground truth trajectory')
    fig.legend()
    wandb.log({"trajectory": wandb.Image(fig)})
    plt.show()
