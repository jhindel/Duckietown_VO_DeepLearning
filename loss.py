import torch
import numpy as np


def CTC_loss(estimated_relative_poses, relative_poses, alpha, beta):
    # assume relative pos are: t->t+1, t+1->t+2,  ..., t->t+n

    criterion = torch.nn.MSELoss()

    normal_loss = (criterion(estimated_relative_poses[:, :, :2], relative_poses[:, :, :2]) +
                   100 * criterion(estimated_relative_poses[:, :, 2], relative_poses[:, :, 2]))

    estimated_relative_poses_startend = estimated_relative_poses[-1]

    matrix_representation_next = np.apply_along_axis(get_matrix_repr, 0, estimated_relative_poses[:-1])

    matrix_cum = np.prod(matrix_representation_next) # doesn't give correct result (check github CTCNet once back up)

    relative_pos_cum = [matrix_cum.item((0, 2)), matrix_cum.item((1, 2)), np.sum[estimated_relative_poses[:, :, 2]]]

    CTC_loss = criterion(relative_pos_cum, estimated_relative_poses_startend)

    return alpha*normal_loss + beta*CTC_loss

def get_matrix_repr(relative_pose):
    return np.asarray([[np.cos(relative_pose[2]), -np.sin(relative_pose[2]), relative_pose[0]],
                       [np.sin(relative_pose[2]), np.cos(relative_pose[2]), relative_pose[1]],
                        [0, 0, 1]])





