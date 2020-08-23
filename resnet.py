from torch import nn
import torch
from res_block import ResBlock
from torch.nn import Conv2d, Conv1d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Linear, Softmax
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class Flatten(nn.Module):

    def __init__(self, start_dim):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, input_tensor):
        return torch.flatten(input_tensor, self.start_dim)


class Sigmoid(nn.Module):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input_tensor):
        return torch.sigmoid(input_tensor)

class ResNet(nn.Module):

    def __init__(self):
        print("ResNet Loaded")
        super(ResNet, self).__init__()
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        #self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.layers = nn.ModuleList([
            #Conv1d(1, 64, 1, 0),
            Linear(42, 64),
            #BatchNorm2d(64),
            ReLU(),
            #axPool2d(3, 2),
            #ResBlock(64, 64, 3),
            #ResBlock(64, 128, 2),
            #ResBlock(128, 256, 2),
            #ResBlock(256, 512, 2),
            #AvgPool2d(10),
            #Flatten(1),
            Linear(64, 7),

        ])


    def forward(self, input_tensor):
        for layer in self.layers:
            print(layer)
            input_tensor = layer(input_tensor.float())

        return input_tensor
