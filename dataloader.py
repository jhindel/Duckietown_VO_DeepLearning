import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from os import walk
import numpy as np
import torch


class DuckietownDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_fwf(annotations_file)
        self.img_dir = img_dir
        filenames_dir = sorted(next(walk(self.img_dir), (None, None, []))[2])
        self.img_labels['img'] = filenames_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["img"])
        image = read_image(img_path)
        image = preprocess(image)
        label = self.img_labels.iloc[idx][["x", "y", "theta_correct"]]
        return image, label


def preprocess(img):
    # adapted from: https: // github.com / krrish94 / DeepVO / blob / master / KITTIDataset.py
    # TODO adapt size of image according to DeepVO
    height, width = 224, 224
    img = np.resize(img, (height, width, 3))

    channelwise_mean = [0.4, 0.45, 0.2]  # [1.39663424e-05, 1.09073426e-05, 1.15871203e-05]
    channelwise_stddev = [0.22, 0.21, 0.15]  # [0.00236842, 0.00190088, 0.00201849]
    # Subtract the mean R,G,B pixels
    img[:, :, 0] = ((img[:, :, 0] / 255.0) - channelwise_mean[0]) / (channelwise_stddev[0])
    img[:, :, 1] = ((img[:, :, 1] / 255.0) - channelwise_mean[1]) / (channelwise_stddev[1])
    img[:, :, 2] = ((img[:, :, 2] / 255.0) - channelwise_mean[2]) / (channelwise_stddev[2])

    # Torch expects NCWH
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)

    return img
