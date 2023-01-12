import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import math

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Create RNN Model
class RNNHardCell(nn.Module):
    def __init__(self, n_input, n_hidden, state=None):
        super(RNNHardCell, self).__init__()

        # Number of input dimensions
        self.n_input = torch.tensor(n_input, dtype=torch.int)
        self.n_input = self.n_input.to(device)

        # Number of hidden dimensions
        self.n_hidden = torch.tensor(n_hidden, dtype=torch.int)
        self.n_hidden = self.n_hidden.to(device)

        # State is checking if there is a previous output or not 
        self.in_h = nn.Linear(self.n_input, self.n_hidden, bias=False)
        self.h_h = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.state = torch.tensor(0, dtype=torch.int)
    #     self.register_parameter()

    # def register_parameter(self):
    #     stdv = 1.0 / math.sqrt(self.n_hidden)
    #     for weight in self.parameters():
    #         nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, y, state=None):
        self.state = state

        if self.state == 0:
            # print(self.state)
            self.state = torch.tensor(1, dtype=torch.int)
            y = F.hardtanh(self.in_h(x))
        else:
            # print(self.state)
            y = F.hardtanh(self.in_h(x) + self.h_h(y))
        return y, self.state


# Connect FC to RNNHardCell
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = RNNHardCell(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim, bias=False)
        self.num_layers = num_layers

    # xs is the data train and test data per data length
    # data length means how long term treat as one block
    def forward(self, xs, state=None):
        # create tensor
        y = torch.tensor(0, dtype=torch.float)
        ys = torch.zeros([xs.size(0), 128], dtype=torch.float)
        state = torch.tensor(0, dtype=torch.int)

        # move to device
        y = y.to(device)
        ys = ys.to(device)
        state = state.to(device)

        # loop for each batch
        for i, x in enumerate(xs):
            state = torch.tensor(0, dtype=torch.int)
            # loop for each sequence
            for s in x:
                y, state = self.rnn(s, y, state)
            ys[i] = y
        ys = self.out(ys)

        return ys


# output test
output_path = "output path" # eg. ./output
# path for output
if os.path.exists(output_path + "mnist/RNN_MNIST") == False:
    os.makedirs(output_path + "mnist/RNN_MNIST")
output_path = output_path + "mnist/RNN_MNIST/"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare Dataset
# Load data
train = pd.read_csv("train data path", dtype=np.float32) # eg. ./data/train.csv
Test = pd.read_csv("test data path", dtype=np.float32)

# Split data into features and labels
targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != "label"].values/255
# normalize test data
Test = Test.values/255

# train test split
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)
print("features_train.shape", features_train.shape) # (33600, 784)
# print("targets_train.shape", targets_train.shape) # (33600,)
print("features_test.shape", features_test.shape) # (8400, 784)
# print("targets_test.shape", targets_test.shape) # (8400,)

# Create feature and targets tensor for train set.
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
print("featuresTrain.shape", featuresTrain.shape) # torch.Size([33600, 784])
print("targetsTrain.shape", targetsTrain.shape)

# Create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
print("featuresTest.shape", featuresTest.shape) # torch.Size([8400, 784])
print("targetsTest.shape", targetsTest.shape)

# Create test tensor
TTest = torch.from_numpy(Test)

# Pytorch train and test sets
train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest)
print("len(train)", len(train)) # 33600
# test dataset
DTest = TensorDataset(TTest)


# Parameter
epochs = 1
input_dim = 28
hidden_dim = 128
output_dim = 10
num_layers = 1
batch_size = 64
batch_size_test = 28
data_size = len(train)
test_size = len(test)
seq_size = 28
feature_size = 28

# Data loader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False)
print("len(train_loader)", len(train_loader)) # 525 -> 525 * 64 = 33600

# Visualize one of the images in data set
fig = plt.figure()
plt.imshow(features_numpy[10].reshape(28, 28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig(output_path + "graph.png")

# model, optimizer, loss function
model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
summary(model)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training and evaluate the model
def train_step(x, t):
    # print("x.shape", x.shape) # torch.Size([64, 28, 28])
    # print("t.shape", t.shape) # t.shape torch.Size([64])
    model.train()
    optimizer.zero_grad()
    preds = model(x)
    # print("preds.shape", preds.shape) # torch.Size([64, 10])
    loss = criterion(preds, t)
    loss.backward()
    optimizer.step()

    return loss, preds

def test_step(x, t):
    model.eval()
    preds = model(x)
    # print("t.shape", t.shape) # torch.Size([64])
    # print("preds.shape", preds.shape) # torch.Size([64, 10])
    loss = criterion(preds, t)

    return loss, preds

# list for training and validation(test)
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(1, epochs+1):
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0

    # training set
    for i, (x, t) in enumerate(train_loader):
        x, t = x.to(device), t.to(device)
        x = x.reshape(batch_size, seq_size, feature_size)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # validation set
    for i, (x, t) in enumerate(test_loader):
        x, t = x.to(device), t.to(device)
        x = x.reshape(batch_size_test, seq_size, feature_size)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    # append to list
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    
    # display results
    print("Epoch: {}, Training loss: {:.3f}, Training Acc: {:.3f}, Val loss: {:.3f}, Val Acc{:.3f}".format(epoch, train_loss, train_acc, test_loss, test_acc))

# save graph
# plot loss graph
plt.figure(figsize=(5, 4))
plt.plot(train_loss_list, label="training")
plt.plot(test_loss_list, label="test")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss (Cross Entropy)")
plt.savefig(output_path + "loss.png")
plt.clf()
plt.close()

# plot accuracy graph
plt.figure(figsize=(5, 4))
plt.plot(train_acc_list, label="training")
plt.plot(test_acc_list, label="test")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(output_path + "acc.png")
plt.clf()
plt.close()


# Predict results
outputs = []
model.eval()
preds = model(DTest)
preds = np.argmax(preds, axis=1)
preds = pd.Series(outputs, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), preds], axis=1)
submission.to_csv(output_path+"rnn_submission.csv", index=False)
