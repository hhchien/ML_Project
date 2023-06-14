"""
Input shape = (bs, 5, 33, 3) 
Output shape = 4 
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class NSBlock(nn.Module):

    def __init__(self, dim, ks_1=3, ks_2=3, dl_1=1, dl_2=1, mp_ks=3, mp_st=1):
        super(NSBlock, self).__init__()
        self.dim = dim
        self.conv_r1 = nn.Conv2d(
            dim, dim, kernel_size=ks_1, dilation=dl_1, padding=(dl_1 * (ks_1 - 1)) // 2)
        self.bn_r1 = nn.BatchNorm2d(dim)
        self.conv_r2 = nn.Conv2d(
            dim, dim, kernel_size=ks_2, dilation=dl_2, padding=(dl_2 * (ks_2 - 1)) // 2)
        self.bn_r2 = nn.BatchNorm2d(dim)
        self.pool_r2 = nn.MaxPool2d((1, mp_ks), padding=(
            0, (mp_ks - 1) // 2), stride=(1, mp_st))

    def forward(self, x):
        y1 = (F.leaky_relu(self.bn_r1(self.conv_r1(x))))
        y2 = (self.bn_r2(self.conv_r2(y1)))
        y3 = x + y2
        z = self.pool_r2(y3)
        return z


class NeuSomaticNet(nn.Module):

    def __init__(self, num_channels, dim, window_size=16):
        super(NeuSomaticNet, self).__init__()
        self.dim = dim
        self.ws = window_size
        self.conv1 = nn.Conv2d(num_channels, self.dim, kernel_size=(
            1, 3), padding=(0, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pool1 = nn.MaxPool2d((1, 3), padding=(0, 1), stride=(1, 1))
        self.nsblocks = [
            [3, 5, 1, 1, 3, 1],
            [3, 5, 1, 1, 3, 2],
            [3, 5, 2, 1, 3, 2],
            [3, 5, 4, 2, 3, 2],
        ]
        res_layers = []
        for ks_1, ks_2, dl_1, dl_2, mp_ks, mp_st in self.nsblocks:
            rb = NSBlock(self.dim, ks_1, ks_2, dl_1, dl_2, mp_ks, mp_st)
            res_layers.append(rb)
        self.res_layers = nn.Sequential(*res_layers)

        if self.ws == 8 or self.ws == 10:
            self.fc_dim = self.dim * 5 * 3 # window size = 10, 8
        elif self.ws == 12:
            self.fc_dim = self.dim * 5 * 4  # window size = 16
        elif self.ws == 16:
            self.fc_dim = self.dim * 5 * 5  # window size = 16
        elif self.ws == 20:
            self.fc_dim = self.dim * 5 * 6 # window size = 20
        elif self.ws == 32:
            self.fc_dim = self.dim * 5 * 9 # window size = 32
        print(self.fc_dim)
        self.fc1 = nn.Linear(self.fc_dim, 240)

        self.fc2 = nn.Linear(240, 3)
        # self.fc3 = nn.Linear(240, 1)
        # self.fc4 = nn.Linear(240, 4)

        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        internal_outs = [x]
        x = self.res_layers(x)
        x = self.drop(x)
        internal_outs.append(x)
        # print(x.shape)
        x2 = x.view(-1, self.fc_dim)
        x3 = F.leaky_relu(self.fc1(x2))
        internal_outs.extend([x2, x3])
        o1 = self.fc2(x3)  

        # o2 = self.fc3(x3)
        # o3 = self.fc4(x3)
        # return [o1, o2, o3], internal_outs
        
        return o1, internal_outs


# # test network
# import torch
# model = NeuSomaticNet(num_channels=3, dim=16, window_size=12)
# # print(model)

# input = torch.rand(16, 3, 5, 25)
# x, _ = model(input)

# print(x.detach().numpy().shape)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out