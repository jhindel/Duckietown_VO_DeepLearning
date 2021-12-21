import torch
import torch.nn as nn
import math
import wandb

from torch.autograd import Variable
from torch.nn.init import calculate_gain
import pytorch_lightning as pl
from .loss import DeepVO_loss
from .dataset import DuckietownDataset


# Count output size, given the input size, the kernel size, the padding and the stride
def oc(inp, k, p, s):
    return int(math.floor((inp + 2 * p - k) / s) + 1)


# Initialize neural network layers according to their type
def weights_init(m):
    # TODO pre-trained weights
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


class ConvLstmNet(nn.Module):

    def __init__(self, size, dropout_p=0.5):

        super().__init__()

        self.i_col = size
        self.i_row = size
        self.dropout_p = dropout_p

        self.o_col = oc(oc(oc(oc(oc(oc(oc(oc(oc(self.i_col, 7, 3, 2), 5, 2, 2), 5, 2, 2), 3, 1, 1),
                                    3, 1, 2), 3, 1, 1), 3, 1, 2), 3, 1, 1), 3, 1, 2)
        self.o_row = oc(oc(oc(oc(oc(oc(oc(oc(oc(self.i_row, 7, 3, 2), 5, 2, 2), 5, 2, 2), 3, 1, 1),
                                    3, 1, 2), 3, 1, 1), 3, 1, 2), 3, 1, 1), 3, 1, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # ReLU layer
        self.relu = nn.ReLU(inplace=True)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True)
        self.bn3 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True)
        self.bn3_1 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True)
        self.bn4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True)
        self.bn4_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True)
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True)
        self.bn5_1 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True)
        self.bn6 = nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True)

        # LSTM layers
        self.lstm1 = nn.LSTMCell(self.o_col * self.o_row * 1024, 100)
        self.lstm2 = nn.LSTMCell(100, 100)

        # Linear layer
        self.fc = nn.Linear(in_features=100, out_features=3)

        # Dropout layers
        self.dropout = nn.Dropout2d(p=0.8)
        self.dropout_hidden = nn.Dropout(p=self.dropout_p)  # default p = 0.5

        # Initialization of all linear, convolutional and BN layers, initialization of hidden states of LSTMs
        self.apply(weights_init)
        self.reset_hidden_states()

    # TODO check but should be done automatically
    # model.reset_hidden_states(bsize=args["bsize"], zero=True, phase=phase)  # reset to 0 the hidden states of RNN
    def reset_hidden_states(self, bsize=1, zero=True, phase='eval', cpu=False):

        if zero == True:
            self.hx1 = torch.zeros(bsize, 100)
            self.cx1 = torch.zeros(bsize, 100)
            self.hx2 = torch.zeros(bsize, 100)
            self.cx2 = torch.zeros(bsize, 100)
        else:
            self.hx1 = self.hx1.detach().clone()
            self.cx1 = self.cx1.detach().clone()
            self.hx2 = self.hx2.detach().clone()
            self.cx2 = self.cx2.detach().clone()

        if next(self.parameters()).is_cuda == True:
            self.hx1 = self.hx1.cuda()
            self.cx1 = self.cx1.cuda()
            self.hx2 = self.hx2.cuda()
            self.cx2 = self.cx2.cuda()

        if phase == 'train':
            self.hx1 = self.hx1.requires_grad_()
            self.cx1 = self.cx1.requires_grad_()
            self.hx2 = self.hx2.requires_grad_()
            self.cx2 = self.cx2.requires_grad_()

        if cpu:
            self.hx1 = self.hx1.cpu()
            self.cx1 = self.cx1.cpu()
            self.hx2 = self.hx2.cpu()
            self.cx2 = self.cx2.cpu()


    def forward(self, x):

        x = self.dropout(x)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.bn3_1(self.conv3_1(x))
        x = self.relu(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = self.bn4_1(self.conv4_1(x))
        x = self.relu(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.bn5_1(self.conv5_1(x))
        x = self.relu(x)
        x = self.bn6(self.conv6(x))

        x = x.view(x.size(0), self.o_col * self.o_row * 1024)
        x = self.dropout_hidden(x)

        self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))

        x = self.hx1
        # x = self.bn1d(x)
        x = self.dropout_hidden(x)

        self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))

        x = self.hx2
        # x = self.bn1d(x)
        x = self.dropout_hidden(x)

        x = self.fc(x)

        return x


class ConvNet(nn.Module):

    def __init__(self, size, dropout):
        super().__init__()
        self.i_col = size
        self.i_row = size
        self.dropout_p = dropout


        self.o_col = oc(oc(oc(oc(oc(self.i_col, 7, 3, 2), 5, 2, 2), 5, 2, 2), 3, 1, 1),
                        3, 1, 2)
        self.o_row = oc(oc(oc(oc(oc(self.i_row, 7, 3, 2), 5, 2, 2), 5, 2, 2), 3, 1, 1),
                        3, 1, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # ReLU layer
        self.relu = nn.ReLU(inplace=True)

        # LSTM layers
        self.dense = nn.Linear(in_features=self.o_col * self.o_row * 256, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=3)

        # Dropout layers
        self.dropout = nn.Dropout2d(p=0.8)
        self.dropout_hidden = nn.Dropout(p=self.dropout_p)  # default p = 0.5


    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = x.view(x.size(0), self.o_col * self.o_row * 256)
        x = self.dropout_hidden(x)
        x = self.relu(self.dense(x))
        x = self.dropout_hidden(x)
        x = self.relu(self.dense2(x))
        x = self.dense3(x)

        return x

