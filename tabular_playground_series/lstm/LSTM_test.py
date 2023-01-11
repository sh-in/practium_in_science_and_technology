# This file is for testing changes for LSTM.py which the file is latest and working correctly.

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
import time

# Create LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        # Number of input dimensions
        self.input_size = torch.tensor(input_size, dtype=torch.int)
        self.input_size = self.input_size.to(device)

        # Number of hidden dimensions
        self.hidden_size = torch.tensor(hidden_size, dtype=torch.int)
        self.hidden_size = self.hidden_size.to(device)

        # These are used for calculating Output, hidden state, cell state
        self.in_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.h_i = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.in_f = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.h_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.in_g = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.h_g = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.in_o = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.h_o = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Initialize weights and bias from Xavier
        self.apply(self._init_weights)

    def forward(self, x_in, h_in, c_in):
        sig = nn.Sigmoid()
        i_t = sig(self.in_i(x_in)+self.h_i(h_in))
        f_t = sig(self.in_f(x_in)+self.h_f(h_in))
        g_t = F.hardtanh(self.in_g(x_in)+self.h_g(h_in))
        o_t = sig(self.in_o(x_in)+self.h_o(h_in))
        c_t = torch.mul(f_t, c_in) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, F.hardtanh(c_t))

        return h_t, h_t, c_t

    # Initialize weights and bias from Xavier(-k^(1/2), k^(1/2)) where k=1/hidden_size
    def _init_weights(self, module):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.uniform_(-stdv, stdv)


# Connect FC to LSTM
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers

    # xs is the batch data of train_dataset and test_data_set per data length
    # data length means how long term treat as one block
    def forward(self, xs):
        # create tensor for Output, hidden state, cell state
        y = torch.zeros([xs.size(0), self.hidden_size], dtype=torch.float)
        h = torch.zeros([xs.size(0), self.hidden_size], dtype=torch.float)
        c = torch.zeros([xs.size(0), self.hidden_size], dtype=torch.float)

        # move to device
        y = y.to(device)
        h = h.to(device)
        c = c.to(device)

        xs = xs.permute(1, 0, 2)
        # print("xs.shape", xs.shape) # ttorch.Size([60, 64, 13])

        # loop for each time series
        for x in xs:
            y, h, c = self.lstm(x, h, c)

        # print("ys.shape", ys.shape) # torch.Size([64, 128])
        y = self.out(y)

        return y


# Set start time
start = time.time()

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "lstm") == False:
    os.makedirs(output_path + "lstm")
output_path = output_path + "lstm/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Set seed and device
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the data
Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
Train_label = pd.read_csv(ds_path + "train_labels.csv")

# Check data
features = Train.columns.tolist()[3:]
# print(features)
print(Train.head())
print(Train_label.head())

sequence = Train["sequence"]
labels = Train_label["state"]
train = Train.drop(["sequence", "subject", "step"], axis=1).values
# print("train.shape", train.shape) # (1558080, 13)
# in train there are 25968 rows which contains 60x13 matrix
# each rows means each sequence or subject's sensor data
train = train.reshape(-1, 60, train.shape[-1])
# print("train.shape", train.shape) # (25968, 60, 13)
# print("labels.shape", labels.shape) # (25968,)

# train validation split
train_x, val_x, train_y, val_y = train_test_split(train, labels, test_size=0.2, shuffle=False)
# print("train_x.shape", train_x.shape) # (20774, 60, 13)
# print("train_y.shape", train_y.shape) # (20774,)
# print("val_x.shape", val_x.shape) # (5194, 60, 13)
# print("val_y.shape", val_y.shape) # (5194,)

# Create x and y tensor for train set
xTrain = torch.from_numpy(train_x)
yTrain = torch.tensor(train_y.values)
# print("xTrain.shape", xTrain.shape) # torch.Size([20774, 60, 13])
# print("yTrain.shape", yTrain.shape) # torch.Size([20774])

# Creat x and y for validation set
xVal = torch.from_numpy(val_x)
yVal = torch.tensor(val_y.values)
# print("xVal.shape", xVal.shape) # torch.Size([5194, 60, 13])
# print("yVal.shape", yVal.shape) # torch.Size([5194])

# Create TensorDataset for train and validation sets
train_ds = TensorDataset(xTrain, yTrain)
val_ds = TensorDataset(xVal, yVal)
# print("len(train_ds)", len(train_ds)) # 20774
# print("len(val_ds)", len(val_ds)) # 5194

# Parameter
epochs = 100
input_size = 13
hidden_size = 128
output_size = 2
num_layers = 1
batch_size = 64
batch_size_val = 28
data_size = len(train_ds)
test_size = len(val_ds)
seq_size = 60
feature_size = 13

# Data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
# print("len(train_loader)", len(train_loader)) # 325

# model, optimizer, loss function
model = LSTMCell(input_size, hidden_size, output_size, num_layers).to(device)
summary(model)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, patience=5)

# Training and evaluate the model
def train_step(x, t):
    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, t)
    loss.backward()
    optimizer.step()

    return loss, preds

def val_step(x, t):
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

# list for training and validation(test)
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# Code to speed up training
torch.backends.cudnn.benchmark = True

for epoch in range(1, epochs+1):
    # Set start time for training
    start_train = time.time()

    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # training set
    model.train()
    for i, (x, t) in enumerate(train_loader):
        x, t = x.to(device), t.to(device)
        if batch_size != x.shape[0]:
            x = x.reshape(x.shape[0], seq_size, feature_size)
        else:
            x = x.reshape(batch_size, seq_size, feature_size)   
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # validation set
    model.eval()
    for i, (x, t) in enumerate(val_loader):
        x, t = x.to(device), t.to(device)
        if batch_size_val != x.shape[0]:
            x = x.reshape(x.shape[0], seq_size, feature_size)
        else:
            x = x.reshape(batch_size_val, seq_size, feature_size)
        loss, preds = val_step(x, t)
        val_loss += loss.item()
        val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
    scheduler.step(val_loss)
    # show learning rate
    # print(optimizer.param_groups[0]['lr'])
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # append to list
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    # display results and training duration
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    val_loss    val_acc    duration")
    end_train = time.time()

    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_train - start_train))

# Save graph
# Plot loss graph
plt.figure(figsize=(5, 4))
plt.plot(train_loss_list, label="training")
plt.plot(val_loss_list, label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss (Cross Entropy)")
plt.savefig(output_path + "loss.png")
plt.clf()
plt.close()

# Plot accuracy graph
plt.figure(figsize=(5, 4))
plt.plot(train_acc_list, label="training")
plt.plot(val_acc_list, label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(output_path + "acc.png")
plt.clf()
plt.close()

# Make predictions
# Load test data
Test = pd.read_csv(ds_path + "test.csv", dtype=np.float32)
# Check test data
print(Test.head())

test_sequence = Test["sequence"]
test = Test.drop(["sequence", "subject", "step"], axis=1).values
test = test.reshape(-1, 60, test.shape[-1])

# Create test tensor
ttest = torch.from_numpy(test)

# Create TensorDataset for test sets
dtest = TensorDataset(ttest)

# Create DataLoader for test sets
test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=False)
# print(len(test_loader))

# Predict
preds = []
with torch.no_grad():
    model.eval()
    for x in test_loader:
        x = x[0].to(device)
        output = model.forward(x)
        pred = output.argmax(dim=-1).tolist()
        preds += pred
# print(len(preds), len(test_sequence)/60, len(test_sequence)/60 + 25968 - 1)
preds = np.array(preds)
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+int(len(test_sequence)/60)), "state": preds})
submissions.to_csv(output_path + "lstm_submissions.csv", index=False, header=True)

# Print processing time
end = time.time()
print("Total duration {}".format(end - start))