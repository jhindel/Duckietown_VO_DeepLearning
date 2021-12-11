import torch


def DeepVO_loss(relative_pose_change, relative_pose_change_pred, K):
    criterion = torch.nn.MSELoss()
    return (criterion(relative_pose_change_pred[:, :, :2], relative_pose_change[:, :, :2]) +
                        K * criterion(relative_pose_change_pred[:, :, 2], relative_pose_change[:, :, 2]))


# TODO add CTCNet loss in this file