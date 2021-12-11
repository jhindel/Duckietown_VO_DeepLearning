import numpy as np


# Make large numbers readable
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# Count number of parameters of a neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rotation(pose):
    x_n, y_n, theta_n = pose
    return np.array([[np.cos(theta_n), -np.sin(theta_n), x_n],
                     [np.sin(theta_n), np.cos(theta_n), y_n],
                     [0, 0, 1]])


def inverse_rotation(pose):
    x_n, y_n, theta_n = pose
    return np.array([[np.cos(theta_n), np.sin(theta_n), -np.sin(theta_n) * y_n - np.cos(theta_n) * x_n],
                     [-np.sin(theta_n), np.cos(theta_n), -np.cos(theta_n) * y_n + np.sin(theta_n) * x_n],
                     [0, 0, 1]])


def absolute2relative(absolute_poses):
    n_relative_poses = absolute_poses.shape[0] - 1
    relative_poses = np.zeros((n_relative_poses, 3), dtype='float32')
    relative_thetas = np.zeros(n_relative_poses, dtype='float32')

    absolute_thetas = absolute_poses[:, -1]

    copy_absolute_poses = np.copy(absolute_poses)
    copy_absolute_poses[:, -1] = 1

    for i in range(n_relative_poses):
        relative_poses[i] = inverse_rotation(absolute_poses[i]).dot(copy_absolute_poses[i + 1])
        relative_thetas[i] = absolute_thetas[i + 1] - absolute_thetas[i]

    relative_poses[:, -1] = relative_thetas

    return relative_poses


def relative2absolute(relative_poses, absolute_pose_0):
    n_absolute_poses = relative_poses.shape[0] + 1
    absolute_poses = np.zeros((n_absolute_poses, 3), dtype='float32')
    absolute_thetas = np.zeros(n_absolute_poses, dtype='float32')

    relative_thetas = relative_poses[:, -1]

    copy_relative_poses = np.copy(relative_poses)
    copy_relative_poses[:, -1] = 1

    absolute_poses[0] = absolute_pose_0
    absolute_thetas[0] = absolute_pose_0[-1]

    for i in range(n_absolute_poses - 1):
        absolute_poses[i + 1] = rotation(absolute_poses[i]).dot(copy_relative_poses[i])
        absolute_thetas[i + 1] = relative_thetas[i] + absolute_thetas[i]
        absolute_poses[i + 1][-1] = absolute_thetas[i + 1]

    absolute_poses[:, -1] = absolute_thetas

    return absolute_poses
