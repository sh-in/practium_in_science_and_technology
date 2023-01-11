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

from transformers import get_linear_schedule_with_warmup

import math
import time

class Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, n_head):
        super(Transformer_Model, self).__init__()
        self.n_layers = n_layers

        # Input layer
        self.input = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.dropout = nn.Dropout(0.1)
        max_len = 60
        position = torch.arange(max_len).unsqueeze(1)
        # print("position.shape", position.shape) # torch.Size([60, 1]), value: 0~59
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        # torch.arange(0, hidden_size, 2) # torch.Size([256]), value: 0~510 (step 2)
        # print("div_term.shape", div_term.shape) # torch.Size([256]) 
        pe = torch.zeros(1, max_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # print(pe.shape) # torch.Size([1, 60, 512]) 
        self.register_buffer("pe", pe)

        # Multi-Head Attention
        # self.multi_head_attention = nn.MultiheadAttention(hidden_size, n_head, batch_first=True)

        # Feed forward
        # self.dropout = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.enc = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_head, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        # Multi-Head Attention
        x = self.input(x)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        # print("x.shape", x.shape) # torch.Size([32, 60, 512])
        # print("self.pe[:x.size(0)].shape", self.pe[:x.size(0)].shape) # torch.Size([1, 60, 512])
        # l1 = x
        # for i in range(self.n_layers):
        #     l1, _ = self.multi_head_attention(l1, l1, l1)
        #     res = F.normalize(x + l1)
        # # l1, _ = self.multi_head_attention(l1, l1, l1)
        # # res = F.normalize(x + l1)
        # res = self.dropout(res)
        # res = torch.max(res, 1)[0]
        # l2 = res

        # # Feed Forward
        # for i in range(self.n_layers):
        #     l2 = self.fc1(l2)
        #     l2 = self.dropout(l2)
        # # l2 = self.fc1(l2)
        # # l2 = self.dropout(l2)
        # out2 = F.normalize(res+l2)

        out = self.enc(x)

        # y = self.fc2(out2)
        y = torch.max(out, 1)[0]
        y = self.fc2(y)

        return y


# Set start time
start = time.time()

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "eda_fe/transformer") == False:
    os.makedirs(output_path + "eda_fe/transformer")
output_path = output_path + "eda_fe/transformer/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
# Train = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/newTrain.csv", dtype=np.float32)
Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
Train_label = pd.read_csv(ds_path + "train_labels.csv")

# Check data
features = Train.columns.tolist()[3:]
# print(features)
print(Train.head())
print(Train_label.head())

sc = StandardScaler()
Train[features] = sc.fit_transform(Train[features])

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
# input_dim = 17
input_dim = 13
hidden_dim = 512
output_dim = 1
num_layers = 3
batch_size = 32
batch_size_val = 32
n_head = 8

# Data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
# print("len(train_loader)", len(train_loader)) # 325

# model, optimizer, loss function
model = Transformer_Model(input_dim, hidden_dim, output_dim, num_layers, n_head).to(device)
summary(model)
print(model)
# print(list(model.named_parameters()))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4)
num_warmup_steps = int(0.1 * epochs * len(train_loader))
num_training_steps = int(epochs * len(train_loader))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

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
        # y = model(x)
        y = model(x).squeeze(-1)
        # print("train y.shape", y.shape) # torch.Size([32])
        # y = y.to("cpu")
        # loss = criterion(y, t)
        loss = criterion(y, t.to(device).float())
        # train_loss += loss.item()
        train_loss += loss.data.item()
        train_acc += accuracy_score(t.tolist(), torch.round(y.detach().cpu()))
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    train_acc /= len(train_loader)
    train_acc_list.append(train_acc)

    # validation
    model.eval()
    for i, (x, t) in enumerate(val_loader):
        x=x.to(device)
        # y=model(x)
        y=model(x).squeeze(-1)
        # y=y.to("cpu")
        # loss = criterion(y, t)
        loss = criterion(y, t.to(device).float())
        val_loss += loss.data.item()
        val_acc += accuracy_score(t.tolist(), torch.round(y.detach().cpu()))
    # scheduler.step(val_loss)
    # show learning rate
    # print(optimizer.param_groups[0]['lr'])
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    val_acc /= len(val_loader)
    val_acc_list.append(val_acc)

    # display results
    if epoch % 100 == 1:
        print("Ep/MaxEp    train_loss    train_acc    val_loss    val_acc     duration")
        # print("Ep/MaxEp    train_loss    val_loss    duration")
    end_train = time.time()

    print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_train - start_train))
    # print("{:4}/{}{:14.5}{:12.5}{:10.5}".format(epoch, epochs, train_loss, val_loss, end_train - start_train))

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
# Load test data
# Test = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/attention/torch/eda/newTest.csv", dtype=np.float32)
Test = pd.read_csv(ds_path + "test.csv", dtype=np.float32)
# Check test data
# features = Train.columns.tolist()[3:]
print(Test.head())

Test[features] = sc.transform(Test[features])

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
        # output = model.forward(x)
        output = model(x.float())
        preds.append(output.detach().cpu())
        # pred = output.argmax(dim=-1).tolist()
        # preds += pred
preds = torch.round(torch.from_numpy(np.concatenate(preds, 0).flatten()))
# print(len(preds), len(test_sequence)/60, len(test_sequence)/60 + 25968 - 1)
# preds = np.array(preds)
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+int(len(test_sequence)/60)), "state": preds})
submissions.to_csv(output_path + "transformer_submissions.csv", index=False, header=True)

# Print total processing time
end = time.time()
print("Total duration {}".format(end - start))