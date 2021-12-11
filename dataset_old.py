import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from utils import absolute2relative


preprocess = transforms.Compose([
    transforms.Resize((args["resize"]//2, args["resize"])),
    transforms.ToTensor(),
    # The following means and stds have been pre-computed
    transforms.Normalize(mean=[0.424, 0.459, 0.218], std=[0.224, 0.215, 0.158])
])


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class DeepVOdata(torch.utils.data.Dataset):

    def __init__(self, datapath, trajectory_length, kind='train',
                 loader=default_image_loader):

        self.datapath = datapath
        self.trajectory_length = trajectory_length
        self.kind = kind
        self.size = 0
        self.loader = loader
        self.poses = self.load_poses()

    def load_poses(self):

        with open(os.path.join(self.datapath, 'poses', self.kind + '.txt')) as f:
            data = f.readlines()

        poses_str = []
        for line in data:
            poses_str.append(line.split())

        poses_float = []
        for line in poses_str[1:]:
            poses_float.append(list(float(x.replace(',', '')) for x in line))

        poses = np.array(poses_float, dtype=np.float32)
        poses = poses[:, (0, 1, 3)]

        # Size of the DeepVOdata intance = number of distinct trajectories
        self.size = len(poses_float) - self.trajectory_length

        return poses

    def get_cropped_image(self, kind, index):

        image_path = os.path.join(self.datapath, kind, 'frame' + '%06d' % index + '.png')
        image = self.loader(image_path)
        area = (0, 160, 640, 480)
        cropped_image = image.crop(area)

        return cropped_image

    def __getitem__(self, index):

        images_stacked = []
        rel_poses = []

        for i in range(index, index + self.trajectory_length):
            img1 = self.get_cropped_image(self.kind, i)
            img2 = self.get_cropped_image(self.kind, i + 1)
            pose1 = self.poses[i]
            pose2 = self.poses[i + 1]
            absolute_poses = np.vstack((pose1, pose2))
            rel_pose = absolute2relative(absolute_poses).squeeze()

            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                images_stacked.append(np.concatenate([img1, img2], axis=1))  # trajectory_length * (3, 640, 640)
            else:
                images_stacked.append(np.concatenate([img1, img2], axis=0))  # trajectory_length * (640, 640, 3)

            rel_poses.append(rel_pose)

        return np.asarray(images_stacked), np.asarray(rel_poses)

    def __len__(self):
        return self.size

def number_of_data(filepath):
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            pass
    return i

def reindex(filename, offset=1):

    index = int(filename[5:11])
    index -= offset
    index_str = str(index).zfill(6)
    filename = filename[0:5] + index_str + filename [11:]
    return filename

def reindex_files(datapath, offset=1):

    for f in sorted(os.listdir(datapath)):
        g = reindex(f, offset=offset)
        os.rename(os.path.join(datapath, f), os.path.join(datapath ,g))

def sample_training_and_validation_data(input_images_directory_path,
                                        input_poses_file_path,
                                        output_directory_path,
                                        size_train_dataset=None):

    size_input_dataset = number_of_data(input_poses_file_path)
    if size_train_dataset is None:
        size_train_dataset = size_input_datase t -size_input_datase t/ /3

    # Creating training and validation folders
    dirname_train = os.path.join(output_directory_path, "train")
    dirname_val = os.path.join(output_directory_path, "val")
    dirname_poses = os.path.join(output_directory_path, "poses")
    if os.path.exists(output_directory_path):
        print('Sampling of training and validation data already done!')
        return
    else:
        os.mkdir(output_directory_path)
        os.mkdir(dirname_train)
        os.mkdir(dirname_val)
        os.mkdir(dirname_poses)
        train_filename = os.path.join(dirname_poses, "train.txt")
        val_filename = os.path.join(dirname_poses, "val.txt")

    # Creating the list of filenames for train and val folders in output_directory
    filenames = sorted([os.path.join(input_images_directory_path, f)
                        for f in os.listdir(input_images_directory_path)])

    # Creating the list of filenames for training and validation dataset
    train_filenames = filenames[:size_train_dataset]
    val_filenames = filenames[size_train_dataset:]

    # Copying image files to train and val directories
    for f in train_filenames:
        shutil.copy(f, dirname_train)

    for f in val_filenames:
        shutil.copy(f, dirname_val)

    reindex_files(os.path.join(output_directory_path, "val"), offset=size_train_dataset)

    # Read poses and write them on train and val files
    with open(input_poses_file_path, 'r') as poses, \
            open(train_filename, 'w') as train_poses, \
            open(val_filename, 'w') as val_poses:

        for i, line in enumerate(poses):
            if i <= size_train_dataset:
                train_poses.write(line)
            elif i == (size_train_dataset + 1):
                val_poses.write("         x          y      theta\n")
                val_poses.write(line)
            else:
                val_poses.write(line)

    print("Size of the input dataset: {}".format(size_input_dataset))
    print("Size of the training dataset: {}".format(size_train_dataset))


sample_training_and_validation_data("drive/My Drive/Data_for_training_testing/Images/alex_3small_loops_images",
                                    "drive/My Drive/Data_for_training_testing/Ground_truth_Modified/best/alex_3small_loops_ground_truth.txt",
                                    "drive/My Drive/VOdata/alex_3small_loops",
                                    size_train_dataset=660)