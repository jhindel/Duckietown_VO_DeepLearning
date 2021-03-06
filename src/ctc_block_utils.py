import numpy as np
import torch

def exp_map(pose):
	x = pose[:, 0]
	y = pose[:, 1]
	theta = pose[:, 2]
	transformation = torch.zeros((3*pose.shape[0], 3))
	for i in range(pose.shape[0]):
		transformation[3*i:3*(i+1), :] = torch.tensor([[torch.cos(theta[i]), -torch.sin(theta[i]), x[i]], 
			                                     	   [torch.sin(theta[i]), torch.cos(theta[i]), y[i]],
			                                      	   [0, 0, 1]])
	return transformation

def log_map(transformation):
	batch_size = transformation.shape[0] // 3
	pose = torch.zeros((batch_size, 3))
	for i in range(batch_size):
		pose[i, 0] = transformation[3*i, 2]
		pose[i, 1] = transformation[3*i + 1, 2]
		pose[i, 2] = torch.atan2(transformation[3*i + 1, 0], transformation[3*i, 0])
	return pose

def composition(poses):
	composition_transformation = torch.eye(3).repeat(poses[0].shape[0], 1)
	for pose in poses:
		exp_pose = exp_map(pose)
		for i in range(pose.shape[0]):
		  composition_transformation[3*i:3*(i+1), :] = composition_transformation[3*i:3*(i+1), :] @ exp_pose[3*i:3*(i+1), :]
	composition_pose = log_map(composition_transformation)
	return composition_pose

def get_subsequences(trajectory, step_size):
	trajectories = []
	for i in range(step_size):
		trajectories.append(trajectory[i::step_size])
	return trajectories

def get_all_subsequences(trajectory, max_step_size=3):
	trajectories = []
	for i in range(2, max_step_size+1):
		trajectories += get_subsequences(trajectory, i)
	return trajectories

def get_compositions(poses, step_size):
	pose_lists = []
	for i in range(step_size):
		pose_list = [composition(poses[j:j+step_size]) for j in range(i, len(poses)-step_size+1, step_size)]
		pose_lists.append(pose_list)
	return pose_lists

def get_all_compositions(poses, max_step_size=3):
	pose_lists = []
	for i in range(2, max_step_size+1):
		pose_lists += get_compositions(poses, i)
	return pose_lists
