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


# Need to adjust for 13 sensors.
class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        # First few layers doesn't made from block, so just write it
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.fc = nn.Linear(128*13, 64)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.pool3(h)

        h = torch.flatten(h, start_dim=1)
        h = self.fc(h)
        h = self.relu4(h)
        h = self.out(h)
        y = self.sigmoid(h)

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


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path="checkpoint_model_one_img.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc = 0
        self.train_acc = 0
        self.val_loss = np.Inf
        self.train_loss = np.Inf
        self.epoch = 0
        self.path = path
    
    def __call__(self, train_acc, train_loss, val_acc, val_loss, epoch, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(train_acc, train_loss, val_acc, val_loss, epoch, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(train_acc, train_loss, val_acc, val_loss, epoch, model)
            self.counter = 0

    def checkpoint(self, train_acc, train_loss, val_acc, val_loss, epoch, model):
        if self.verbose:
            print(f"Validation accuracy increased ({self.val_acc:.6f} --> {val_acc:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.epoch = epoch


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
    Train.append(pd.read_csv(ds_path+f"sensor_{i:02d}_train.csv", dtype=np.float32))
    # reader = pd.read_csv(ds_path+f"sensor_{i:02d}_train.csv", dtype=np.float32, chunksize=640)
    # t = reader.get_chunk(640)
    # Train.append(t)
    Train[i] = Train[i].values
    Train[i] = Train[i].reshape(-1, 1, 60, 60)
Train_label = pd.read_csv("../../../../datasets/kaggle_tabular_playground_series_apr_2022/train_labels.csv", dtype=np.float32)
# Train_label = pd.read_csv("../../../../datasets/kaggle_tabular_playground_series_apr_2022/train_labels.csv", chunksize=640, dtype=np.float32).get_chunk(640)
labels = Train_label["state"]
print("Finish loading training data.")

# Load test data.
print("Start loading testing data...")
Test = []
for i in range(13):
    Test.append(pd.read_csv(ds_path+f"sensor_{i:02d}_test.csv", dtype=np.float32))
    # reader = pd.read_csv(ds_path+f"sensor_{i:02d}_test.csv", dtype=np.float32, chunksize=128)
    # t = reader.get_chunk(128)
    # Test.append(t)
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
epochs = 100
input_dim = (batch_size, 1, 60, 60*13)
# input_dim = (batch_size, 1, 60, 60)
output_dim = 1
lr=0.0001

# Build model
model = CNN(output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()
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
    preds = model(x).squeeze(-1)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, preds

def test_step(x, t):
    model.eval()
    preds = model(x).squeeze(-1)
    loss = compute_loss(t, preds)

    return loss, preds

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

es = EarlyStopping(patience=10, verbose=True)
score_list = []

for epoch in range(1, epochs+1):
    start_train = time.time()
    train_loss = 0.
    train_acc = 0.
    val_loss = 0.
    val_acc = 0.

    for (x, t) in train_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        # print(preds)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), torch.round(preds.detach().cpu()))
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    for (x, t) in val_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        val_loss += loss.item()
        val_acc += accuracy_score(t.tolist(), torch.round(preds.detach().cpu()))
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    scheduler.step(val_acc)

    # Display results
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    test_loss    test_acc    duration")
    end_train = time.time()
    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_train-start_train))
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    # Early Stopping
    es(train_acc, train_loss, val_acc, val_loss, epoch, model)
    if es.early_stop:
        print("Early Stopping!")
        score_list.append([es.epoch, es.train_acc, es.train_loss, es.val_acc, es.val_loss])
        break

# Plot results
# Plot loss graph
plt.figure(figsize=(5, 4))
plt.plot(train_loss_list, label="training")
plt.plot(val_loss_list, label="validation")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("loss(BCE Entropy)")
plt.savefig(output_path+"loss" + str(121) +".png")

# Plot accuracy graph
plt.figure(figsize=(5, 4))
plt.plot(train_acc_list, label = "training")
plt.plot(val_acc_list, label = "validation")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(output_path+"acc" + str(121) + ".png")
plt.clf()
plt.close()

# Load the checkpoint
model.load_state_dict(torch.load("checkpoint_model_one_img.pth"))

# Make predictions
print("Making predictions")
preds = []
with torch.no_grad():
    model.eval()
    for x in test_loader:
        x = x[0].to(device)
        output = model(x)
        preds.append(output.detach().cpu())
print("len(preds): ", len(preds))
print("len(preds[0]): ", len(preds[0]))
print("preds[0]: ", preds[0])
preds = torch.round(torch.from_numpy(np.concatenate(preds, 0).flatten()))
print("len(preds): ", len(preds))
print("preds[0]: ", preds[0])

print("Complete making predictions")
# preds = np.array(preds)
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+len(dtest)), "state": preds})
submissions.to_csv(output_path+"wavelets_submissions" + str(121) + ".csv", index=False, header=True)

# Print total processing time
end = time.time()
print("Total duration {}".format(end-start))