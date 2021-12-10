import numpy as np


def get_matrix_repr(relative_pose):
    print("received pose", relative_pose)
    n = np.asarray([[np.cos(relative_pose[2]), -np.sin(relative_pose[2]), relative_pose[0]],
                    [np.sin(relative_pose[2]), np.cos(relative_pose[2]), relative_pose[1]],
                    [0, 0, 1]])
    print(n)
    return n


# 110 120
estimated_relative_poses_next = np.asarray([[1, 1, 1], [1, 2, 3], [0, 0, 0]])
# print(get_matrix_repr(estimated_relative_poses_next[:,0]))

matrix_representation_next = np.apply_along_axis(get_matrix_repr, axis=0, arr=estimated_relative_poses_next)

print("matrices", matrix_representation_next)

print(np.dot(matrix_representation_next[0], matrix_representation_next[1]))

matrix_cum = np.prod(matrix_representation_next, axis=0)

print(matrix_cum)

print([matrix_cum.item((0, 2)), matrix_cum.item((1, 2))])

print(estimated_relative_poses_next[:, :, 2])
