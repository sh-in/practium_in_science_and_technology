import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import math
import time

# nn.Module is made from two parts. __init__() part and forward() part.
class Block(nn.Module):
    # In this method, create a layer instance and apply it forward method by your desired order.
    def __init__(self, channel_in, channel_out):
        # Don't forget this code supre().__init__()
        super().__init__()
        channel = channel_out // 4

        # 1*1 Conv
        self.conv1s = nn.ModuleList([nn.Conv2d(channel_in, channel, kernel_size=(1, 1)) for i in range(13)])
        # self.conv1_1 = nn.Conv2d(channel_in, channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        # 3*3 Conv
        self.conv2s = nn.ModuleList([nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1) for i in range(13)])
        # self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        # 1*1 Conv
        self.conv3s = nn.ModuleList([nn.Conv2d(channel, channel_out, kernel_size=(1, 1), padding=0) for i in range(13)])
        # self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)

        self.relu3 = nn.ReLU()

        # for skip connection
        self.shortcut = self._shortcut(channel_in, channel_out)

    def forward(self, x):
        y = []
        # print("x.shape(in Block): ", x.shape)
        h = self.conv1s[i](x)
        # h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2s[i](h)
        # h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3s[i](h)
        # h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)
        return y

    # This method check the channel_in and channel_out for skip connection
    def _shortcut(self, channel_in, channel_out):
        # if channne_in and channel_out are not same, ajust channel_in to channel_out
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    # if cannel_in and channel_out are not same, ajust it by using 1*1 Conv
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


# Need to adjust for 13 sensors.
class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        # First few layers doesn't made from block, so just write it
        self.conv1s = nn.ModuleList([nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3) for i in range(13)])
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Block 1
        # First block has different input and output channels
        self.block0 = self._building_block(256, channel_in=64)
        # Second and third block has same input and output channels
        self.block1 = nn.ModuleList([self._building_block(256) for _ in range(2)])
        # Block1 and Block2 has a difference in channels and output sizes.
        # Therefore, set input and output channels and stride as 2 to adjust output size.
        self.conv2s = nn.ModuleList([nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2)) for i in range(13)])
        # self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))

        # Block 2
        # In Block2, input and output channels are same because of the previous operation.
        self.block2 = nn.ModuleList([self._building_block(512) for _ in range(4)])
        # Block2 and 3 has a difference in channels and output sizes.
        # Therefore, set input and output channels and stride as 2 to adjust output size.
        self.conv3s = nn.ModuleList([nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2)) for i in range(13)])
        # self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))

        # Block 3
        # In Block 3, input and output channels are same because of the previous operation.
        self.block3 = nn.ModuleList([self._building_block(1024) for _ in range(6)])
        # Block3 and 4 has a difference in channels and output sizes.
        # Therefore, set input and output channels and stride as 2 to adjust output size.
        self.conv4s = nn.ModuleList([nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2)) for i in range(13)])
        # self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))

        # Block 4
        # In Block 4, input and output channels are same because of the prevoius operation.
        self.block4 = nn.ModuleList([self._building_block(2048) for _ in range(3)])
        # In the last section, we use average pool, 1000-d fc, and softmax
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048*13, 1000)
        self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        hs = []
        for i in range(13):
            h = self.conv1s[i](x[:, :, :, i*60:(i+1)*60])
            # h = self.conv1(x[i])
            h = self.bn1(h)
            h = self.relu1(h)
            h = self.pool1(h)
            h = self.block0(h)
            hs.append(h)
        for block in self.block1:
            for i in range(13):
                hs[i] = block(hs[i])
            # hs = block(hs)
        for i in range(13):
            hs[i] = self.conv2s[i](hs[i])
            # h = self.conv2(h)
        for block in self.block2:
            for i in range(13):
                hs[i] = block(hs[i])
            # hs = block(hs)
        for i in range(13):
            hs[i] = self.conv3s[i](hs[i])
            # h = self.conv3(h)
        for block in self.block3:
            for i in range(13):
                hs[i] = block(hs[i])
            # hs = block(hs)
        for i in range(13):
            hs[i] = self.conv4s[i](hs[i])
            # h = self.conv4(h)
        for block in self.block4:
            for i in range(13):
                hs[i] = block(hs[i])
            # hs = block(hs)
        h = hs[0].cpu()
        for i in range(1, 13):
            h = np.concatenate((h, hs[i].cpu()), axis=1)
        h = torch.from_numpy(h).to(device)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)

        return y

    # Set channel_in and channel_out and build Block.
    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)


# By using torch.nn.functional, import as F, you can write Module's subclass definition consisely.
class GlobalAvgPool2d(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


# Set start time.
start = time.time()

# Set output path.
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/transformer_comp/wavelet") == False:
    os.makedirs(output_path + "result/transformer_comp/wavelet")
output_path = output_path + "result/transformer_comp/wavelet/"

# Set dataset path.
ds_path = "../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/wavelets/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data.
print("Start loading training data...")
Train = []
for i in range(13):
    # Train.append(pd.read_csv(ds_path+f"sensor_{i:02d}_train.csv", dtype=np.float32))
    reader = pd.read_csv(ds_path+f"sensor_{i:02d}_train.csv", dtype=np.float32, chunksize=640)
    t = reader.get_chunk(640)
    Train.append(t)
    Train[i] = Train[i].values
    Train[i] = Train[i].reshape(-1, 1, 60, 60)
# Train_label = pd.read_csv("../../../../datasets/kaggle_tabular_playground_series_apr_2022/train_labels.csv")
Train_label = pd.read_csv("../../../../datasets/kaggle_tabular_playground_series_apr_2022/train_labels.csv", chunksize=640).get_chunk(640)
labels = Train_label["state"]
print("Finish loading training data.")

# Load test data.
print("Start loading testing data...")
Test = []
for i in range(13):
    # Test.append(pd.read_csv(ds_path+f"sensor_{i:02d}_test.csv", dtype=np.float32))
    reader = pd.read_csv(ds_path+f"sensor_{i:02d}_test.csv", dtype=np.float32, chunksize=128)
    t = reader.get_chunk(128)
    Test.append(t)
    Test[i] = Test[i].values
    Test[i] = Test[i].reshape(-1, 1, 60, 60)
print("Finish loading testing data.")

# Merge data.
# train = Train
# test = Test
train = Train[0]
test = Test[0]
print("Merging data...")
for i in range(1, 13):
    train = np.concatenate((train, Train[i]), axis=3)
    test = np.concatenate((test, Test[i]), axis=3)
print("Finish merging data.")

# Check data.
print("Train[0].shape", Train[0].shape)
print("Test[0].shape", Test[0].shape)
# print("train.shape", train.shape)
# print("test.shape", test.shape)
print("len(train)", len(train))
print("len(test)", len(test))
print("train[0].shape", train[0].shape)
print("test[0].shape", test[0].shape)

def get_ticks_label_set(labels, num):
    l = len(labels)
    step = l//(num-1)
    position = np.arange(0, l, step)
    label = labels[::step]
    return position, label


# Sampling interval = 1
dt = 1
t = np.arange(0, 60, 1)
# Sampling frequency
fs = 1/dt
# Nyquist frequency
nq_f = fs/2.0

# Frequencies which I want to analysis.
freqs = np.linspace(1, nq_f, 60)
freqs_rate = freqs/fs

# Wavelet configuration
wavelet_type = "cmor1.5-1.0"

# Scale
scales = 1/freqs_rate
# Reverse
scales = scales[::-1]

x_positions, x_labels = get_ticks_label_set(t, 10)
y_positions, y_labels = get_ticks_label_set(freqs, 10)
y_labels = [math.floor(d * 10 ** 2) / 10 ** 2 for d in y_labels]

plt.figure(figsize=(5, 4))
print(train[0, 0, :, :60].shape)
plt.imshow(train[0, 0, :, :60], aspect="auto")
plt.yticks(y_positions, y_labels)
plt.xticks(x_positions, x_labels)
plt.xlabel("Time[s]")
plt.ylabel("Frequency[Hz]")
plt.savefig(output_path + "check_data" + ".png")
plt.clf()
plt.close()

# Split training and validation set.
print("Shaping training and testing data...")
x_train, x_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, shuffle=False)

# Create x and y tensor for training and validation sets.
xTrain = torch.from_numpy(x_train)
yTrain = torch.tensor(y_train.values)
xTest = torch.from_numpy(x_val)
yTest = torch.tensor(y_val.values)

# Create test tensor.
ttest = torch.from_numpy(test)

# Create TensorDataset.
train_ds = TensorDataset(xTrain, yTrain)
val_ds = TensorDataset(xTest, yTest)
dtest = TensorDataset(ttest)

# Create DataLoader
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=False)
print("Finish shaping data.")

# Parameters.
epochs = 2
input_dim = (batch_size, 1, 60, 60*13)
# input_dim = (batch_size, 1, 60, 60)
output_dim = 2
lr=0.0001

# Build model
model = ResNet50(output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, verbose=True)

# Check model
summary(model, input_size=input_dim)


# For training.
# Apply reshape(-1, 60, 60) to each sequence of np array and make it shape (13, 60, 60).

# Training and evaluate the model
def compute_loss(label, pred):
    return criterion(pred, label)

def train_step(x, t):
    model.train()
    preds = model(x)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, preds

def test_step(x, t):
    model.eval()
    preds = model(x)
    loss = compute_loss(t, preds)

    return loss, preds


for epoch in range(epochs):
    start_train = time.time()
    train_loss = 0.
    train_acc = 0.
    test_loss = 0.
    test_acc = 0.

    for (x, t) in train_loader:
        x, t = x.to(device), t.to(device)
        print("x.shape(in training): ", x.shape)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    for (x, t) in test_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    scheduler.step(test_acc)

    # Display results
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    test_loss    test_acc    duration")
    end_train = time.time()
    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, test_loss, test_acc, end_train-start_train))
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

# Plot results
plt.figure(figsize=(5, 4))
plt.plot(train_loss_list, label="training")
plt.plot(val_loss_list, label="validation")
plt.ylim(0, 0.3)
plt.legend()
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("loss(Cross Entropy)")
# plt.savefig(output_path+"loss"+".png")