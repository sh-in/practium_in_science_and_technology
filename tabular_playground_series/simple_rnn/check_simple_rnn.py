import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import transforms
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import math

# Create RNN Model
class RNNHardCell(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(RNNHardCell, self).__init__()

        # Number of input dimensions
        self.n_input = torch.tensor(n_input, dtype=torch.int)
        self.n_input = self.n_input.to(device)

        # Number of hidden dimensions
        self.n_hidden = torch.tensor(n_hidden, dtype=torch.int)
        self.n_hidden = self.n_hidden.to(device)

        # State is checking if there is a previous output or not 
        self.in_h = nn.Linear(self.n_input, self.n_hidden, bias=True)
        self.h_h = nn.Linear(self.n_hidden, self.n_hidden, bias=True)

    def forward(self, x, y):
        # First state doesn't have a previous state, so just go through hardtanh
        # First state can be also dealt with this equation, because first y is array of 0
        y = F.hardtanh(self.in_h(x) + self.h_h(y))
        return y


# Connect FC to RNNHardCell
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = RNNHardCell(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim, bias=True)
        self.num_layers = num_layers

    # xs is the batch data of train_dataset and test_data_set per data length
    # data length means how long term treat as one block
    def forward(self, xs):
        # create tensor
        y = torch.zeros([self.hidden_dim], dtype=torch.float)
        ys = torch.zeros([xs.size(0), self.hidden_dim], dtype=torch.float)

        # move to device
        y = y.to(device)
        ys = ys.to(device)

        xs = xs.permute(1, 0, 2)
        # print("xs.shape", xs.shape) # ttorch.Size([60, 64, 13])

        # loop for each time series
        for x in xs:
            ys = self.rnn(x, ys)

        # print("ys.shape", ys.shape) # torch.Size([64, 128])
        ys = self.out(ys)

        return ys


# Set seed and device
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter
input_dim = 3
hidden_dim = 4
output_dim = 2

model = RNNModel(input_dim, hidden_dim, output_dim)
print(list(model.named_parameters()))
summary(model)