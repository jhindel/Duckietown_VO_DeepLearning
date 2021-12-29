import torch

criterion = torch.nn.MSELoss()

def DeepVO_loss(relative_pose_change, relative_pose_change_pred, K):
    # automatically calculates mean over batches
    # print(criterion(relative_pose_change[:, :, :2], relative_pose_change_pred[:, :, :2]))
    # print(criterion(relative_pose_change[:, :, 2], relative_pose_change_pred[:, :, 2]))
    loss_vector = (criterion(relative_pose_change[:, :, :2], relative_pose_change_pred[:, :, :2]) +
                        K * criterion(relative_pose_change[:, :, 2], relative_pose_change_pred[:, :, 2]))
    # print("angle", criterion(relative_pose_change_pred[:, :, 2], relative_pose_change[:, :, 2]))
    loss = torch.mean(loss_vector)
    return loss


# TODO add CTCNet loss in this file