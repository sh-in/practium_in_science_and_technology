import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math
import time

class LSTM(nn.Module):
    """
    input_size: 13
    hidden_size: 128
    proj_size: 2
    num_layers: 1
    N = batch_size(for training): 32
    N_val = batch_size(for validation): 28
    L = sequence length: 60
    H_in = input_size: 13
    H_cell = hidden_size: 128
    H_out = proj_size if proj_size > 0 otherwise hidden_size: 2

    For batch_first = True
    Inputs: input, (h_0, c_0)
    input: tensor (N, L, H_in) = (32, 60, 13)
    h_0: tensor (num_layers, N, H_out) = (1, 32, 2). Default to zeros. Hidden state.
    c_0: tensor (num_layers, N, H_cell) = (1, 32, 128). Default to zeros. Cell state.

    Outputs: output, (h_n, c_n)
    output: tensor (N, L, H_out) = (32, 60, 2)
    h_n: tensor (num_layers, N, H_out) = (1, 32, 2). Final hidden state.
    c_n: tensor (num_layers, N, H_cell) = (1, 32, 128). Final cell state.
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.lstm21 = nn.LSTM(hidden_size*2, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.lstm22 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.lstm31 = nn.LSTM(hidden_size*4, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.lstm32 = nn.LSTM(hidden_size*2, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.l4 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y1, _ = self.lstm1(x)
        y21, _ = self.lstm21(y1)
        y22, _ = self.lstm22(x)
        y2 = torch.cat([y21, y22], dim=2)
        y31, _ = self.lstm31(y2)
        y32, _ = self.lstm32(y21)
        y3 = torch.cat([y31, y32], dim=2)
        y3 = torch.max(y3, 1)
        # y31, (hn31, _) = self.lstm31(y2)
        # y32, (hn32, _) = self.lstm32(y21)
        # h31 = torch.cat([hn31[0], hn31[1]], dim=1)
        # h32 = torch.cat([hn32[0], hn32[1]], dim=1)
        # y3 = torch.cat([h31, h32], dim=2)
        selu = nn.SELU()
        y4 = selu(self.l4(y3[0]))
        y5 = self.dropout(y4)
        sig = nn.Sigmoid()
        y = sig(self.fc(y5))
        
        return y


# Set start time
start = time.time()

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "eda_fe/dnn") == False:
    os.makedirs(output_path + "eda_fe/dnn")
output_path = output_path + "eda_fe/dnn/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
# Train = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/newTrain.csv", dtype=np.float32)
# Dataset with lag.
# Train = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/addTrainLag.csv", dtype=np.float)
Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
Train_label = pd.read_csv(ds_path + "train_labels.csv")

# Check data
features = Train.columns.tolist()[3:]
# print(features)
print(Train.head())
# print(Train_label.head())

# sc = StandardScaler()
# Train[features] = sc.fit_transform(Train[features])

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
hidden_dim = 256
output_dim = 2
num_layers = 1
batch_size = 128
batch_size_val = 128
data_size = len(train_ds)
test_size = len(val_ds)
seq_size = 60
feature_size = 13

# Data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
# print("len(train_loader)", len(train_loader)) # 325

# model, optimizer, loss function
model = LSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
summary(model)
# print(model)
# print(list(model.named_parameters()))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# model.to(device)

# Training and Validation
for epoch in range(1, epochs + 1):
    # Set start time for training
    start_train = time.time()
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # train
    model.train()
    for i, (x, t) in enumerate(train_loader):
        x=x.to(device)
        optimizer.zero_grad()
        y = model(x)
        y = y.to("cpu")
        loss = criterion(y, t)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), y.argmax(dim=-1).tolist())
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    train_acc /= len(train_loader)
    train_acc_list.append(train_acc)

    # validation
    model.eval()
    for i, (x, t) in enumerate(val_loader):
        x=x.to(device)
        y=model(x)
        y=y.to("cpu")
        loss = criterion(y, t)
        val_loss += loss.item()
        val_acc += accuracy_score(t.tolist(), y.argmax(dim=-1).tolist())
    scheduler.step(val_acc)
    # show learning rate
    # print(optimizer.param_groups[0]['lr'])
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    val_acc /= len(val_loader)
    val_acc_list.append(val_acc)

    # display results
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    val_loss    val_acc    duration")
    end_train = time.time()

    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_train - start_train))

# Save glaph
# Plot loss graph
plt.figure(figsize=(5, 4))
plt.plot(train_loss_list, label = "training")
plt.plot(val_loss_list, label = "validation")
plt.legend()
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("loss (Cross Entropy))")
plt.savefig(output_path+"loss.png")
plt.clf()
plt.close()

# Plot accuracy graph
plt.figure(figsize=(5, 4))
plt.plot(train_acc_list, label = "training")
plt.plot(val_acc_list, label = "validation")
plt.legend()
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(output_path+"acc.png")
plt.clf()
plt.close()

# Make predictions
# Load train data
Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
sequence = Train["sequence"]
train = Train.drop(["sequence", "subject", "step"], axis=1).values
train = train.reshape(-1, 60, train.shape[-1])
xTrain = torch.from_numpy(train)
yTrain = torch.tensor(labels.values)
train_ds = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
preds = []
with torch.no_grad():
    model.eval()
    for x in train_loader:
        x = x[0].to(device)
        y = model(x)
        y = y.to("cpu")
        pred = [torch.max(i, dim=-1).values if i.argmax(dim=-1)==1 else 1-torch.max(i, dim=-1).values for i in y]
        pred = [i.item() for i in pred]
        preds += pred
preds = np.array(preds)
train_preds = pd.DataFrame({"dnn": preds})
print("train_preds", train_preds)
train_preds.to_csv("../../../../output/Tabular_Playground_Series_Apr_2022/result/transformer_comp/ensemble/dnn_train.csv", index=False, header=True)

# Load test data
# Test = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/newTest.csv", dtype=np.float32)
# Dataset with lag.
# Test = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/addTestLag.csv", dtype=np.float)
Test = pd.read_csv(ds_path + "test.csv", dtype=np.float32)
# Check test data
# features = Train.columns.tolist()[3:]
print(Test.head())

# Test[features] = sc.transform(Test[features])

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
        y = model(x)
        y = y.to("cpu")
        pred = [torch.max(i, dim=-1).values if i.argmax(dim=-1)==1 else 1-torch.max(i, dim=-1).values for i in y]
        pred = [i.item() for i in pred]
        preds += pred
# print(len(preds), len(test_sequence)/60, len(test_sequence)/60 + 25968 - 1)
preds = np.array(preds)
test_preds = pd.DataFrame({"dnn": preds})
print("test_preds", test_preds)
test_preds.to_csv("../../../../output/Tabular_Playground_Series_Apr_2022/result/transformer_comp/ensemble/dnn_test.csv", index=False, header=True)
preds = torch.round(torch.from_numpy(preds)).to(torch.int64)
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+int(len(test_sequence)/60)), "state": preds})
print("submissions", submissions)
submissions.to_csv(output_path + "dnn_submissions.csv", index=False, header=True)

# Print total processing time
end = time.time()
print("Total duration{}".format(end - start))