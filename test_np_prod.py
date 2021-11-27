import numpy as np


def get_matrix_repr(relative_pose):
    print(relative_pose)
    return np.asarray([[np.cos(relative_pose[2]), -np.sin(relative_pose[2]), relative_pose[0]],
                       [np.sin(relative_pose[2]), np.cos(relative_pose[2]), relative_pose[1]],
                       [0, 0, 1]])


estimated_relative_poses_next = [[1, 1, 0], [1, 2, 0]]

matrix_representation_next = np.apply_along_axis(get_matrix_repr, 1, estimated_relative_poses_next)

print(matrix_representation_next)

print(np.dot(matrix_representation_next[0], matrix_representation_next[1]))

matrix_cum = np.prod(matrix_representation_next, axis=0)

print(matrix_cum)


