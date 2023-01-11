# This is the latest and working correctly file.

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

        # Initialize weights and bias from Xavier
        self.apply(self._init_weights)

    def forward(self, x, y):
        # First state doesn't have a previous state, so just go through hardtanh
        # First state can be also dealt with this equation, because first y is array of 0
        y = F.hardtanh(self.in_h(x) + self.h_h(y))
        return y

    # Initialize weights and bias from Xavier(-k^(1/2), k^(1/2)) where k=1/hidden_size
    def _init_weights(self, module):
        stdv = 1.0 / math.sqrt(self.n_hidden) if self.n_hidden > 0 else 0
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.uniform_(-stdv, stdv)


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


# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "simple_rnn") == False:
    os.makedirs(output_path + "simple_rnn")
output_path = output_path + "simple_rnn/"

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
input_dim = 13
hidden_dim = 128
output_dim = 2
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
model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
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

for epoch in range(1, epochs+1):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # training set
    model.train()
    for i, (x, t) in enumerate(train_loader):
        # if i%10==0:
        #     print("training: {}/{}".format(i, len(train_loader)))
        x, t = x.to(device), t.to(device)
        # print("x.shape", x.shape) # torch.Size([64, 60, 13])
        # print("t.shape", t.shape) # torch.Size([64])
        if batch_size != x.shape[0]:
            x = x.reshape(x.shape[0], seq_size, feature_size)
        else:
            x = x.reshape(batch_size, seq_size, feature_size)        
        # print("x.shape", x.shape) # torch.Size([64, 60, 13])
        # print("t.shape", t.shape) # torch.Size([64])
        # print("(x.permute(1, 0, 2)).shape", (x.permute(1, 0, 2)).shape) # ttorch.Size([60, 64, 13])
        loss, preds = train_step(x, t)
        # print("len(t.tolist())", len(t.tolist())) # 64
        # print(len(preds.argmax(dim=-1).tolist())) # 64
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # validation set
    model.eval()
    for i, (x, t) in enumerate(val_loader):
        # if i%10==0:
        #     print("testing: {}/{}".format(i, len(val_loader)))
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
    
    # display results
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    val_loss    val_acc")

    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc))

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
submissions.to_csv(output_path + "simple_rnn_submissions.csv", index=False, header=True)