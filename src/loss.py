import torch

criterion = torch.nn.MSELoss()

def DeepVO_loss(relative_pose_change, relative_pose_change_pred, K):
    # automatically calculates mean over batches
    loss = (criterion(relative_pose_change_pred[:, :, :2], relative_pose_change[:, :, :2]) +
                        K * criterion(relative_pose_change_pred[:, :, 2], relative_pose_change[:, :, 2]))
    return loss

def CTCNet_loss(pose_DeepVO, pose_composition, pose_direct, K=1, alpha=1, beta=1):
    CTC_loss = criterion(pose_direct[:, :, :2], pose_composition[:, :, :2]) + K * criterion(pose_direct[:, :, 2], pose_composition[:, :, 2])
    regularization_loss = criterion(pose_direct[:, :, :2], pose_DeepVO[:, :, :2]) + K * criterion(pose_direct[:, :, 2], pose_DeepVO[:, :, 2])
    loss = alpha * CTC_loss + beta * regularization_loss
    return loss
