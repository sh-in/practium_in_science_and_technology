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

    def forward(self, x, y, state=None):
        self.state = state

        # First state doesn't have a previous state, so just go through hardtanh
        if self.state == 0:
            self.state = torch.tensor(1, dtype=torch.int)
            y = F.hardtanh(self.in_h(x))
        else:
            y = F.hardtanh(self.in_h(x) + self.h_h(y))
        return y, self.state


# Connect FC to RNNHardCell
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = RNNHardCell(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim, bias=False)
        self.num_layers = num_layers

    # xs is the batch data of train_dataset and test_data_set per data length
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


# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "simple_rnn") == False:
    os.makedirs(output_path + "simple_rnn")
output_path = output_path + "simple_rnn/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
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
epochs = 20
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

def val_step(x, t):
    model.eval()
    preds = model(x)
    # print("t.shape", t.shape) # torch.Size([64])
    # print("preds.shape", preds.shape) # torch.Size([64, 10])
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

    # print("epoch: ", epoch)
    # print("start training")
    # training set
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
        # print("(x.reshape(batch_size, seq_size, feature_size)).shape", (x.reshape(batch_size, seq_size, feature_size)).shape) # torch.Size([64, 28, 28])
        loss, preds = train_step(x, t)
        # print("len(t.tolist())", len(t.tolist())) # 64
        # print(len(preds.argmax(dim=-1).tolist())) # 64
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
    # print("finish training")
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # print("start testing")
    # validation set
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
    
    # print("finish testing")
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # append to list
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    # display results
    print("Epoch: {}, Training loss: {:.3f}, Training Acc: {:.3f}, Val loss: {:.3f}, Val Acc{:.3f}".format(epoch, train_loss, train_acc, val_loss, val_acc))

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