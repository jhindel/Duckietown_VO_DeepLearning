import torch


def DeepVO_loss(relative_pose_change, relative_pose_change_pred, K):
    # automatically calculates mean over batches
    criterion = torch.nn.MSELoss()
    loss = (criterion(relative_pose_change_pred[:, :, :2], relative_pose_change[:, :, :2]) +
                        K * criterion(relative_pose_change_pred[:, :, 2], relative_pose_change[:, :, 2]))
    return loss


# TODO add CTCNet loss in this file