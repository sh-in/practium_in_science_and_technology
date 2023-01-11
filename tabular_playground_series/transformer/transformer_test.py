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
        self.positional_encoding_layer = PositionalEncoding(hidden_size, 60, 0.1)

        # Encoder
        # Create an encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_head, batch_first=True)
        # Stack the encoder layer num_layers times.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4, norm=None)
        # self.dropout = nn.Dropout()
        
        # Output layer
        self.fc2 = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        # Input layer
        x = self.input(x)
        x = self.positional_encoding_layer(x)

        # Encoder
        # x = self.encoder_layer(x)
        x = self.encoder(x)
        # x = self.dropout(x)

        # Output layer
        # y = torch.max(x, 1)[0]
        y = x.mean(dim=1)
        y = self.fc2(y)

        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        # Positional encoding
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_seq_len
        position = torch.arange(max_seq_len).unsqueeze(1)
        # print("position.shape", position.shape) # torch.Size([60, 1]), value: 0~59
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # torch.arange(0, hidden_size, 2) # torch.Size([256]), value: 0~510 (step 2)
        # print("div_term.shape", div_term.shape) # torch.Size([256]) 
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # print(pe.shape) # torch.Size([1, 60, 512]) 
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path="checkpoint_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.path = path
    
    def __call__(self, val_acc, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_acc, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_acc, model)
            self.counter = 0

    def checkpoint(self, val_acc, model):
        if self.verbose:
            print(f"Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc



# Set start time
start = time.time()

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/transformer_test") == False:
    os.makedirs(output_path + "result/transformer_test")
output_path = output_path + "result/transformer_test/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
# Dataset with lag. input 43
Train = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/newTrainDataset.csv", dtype=np.float32)
# Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
Train_label = pd.read_csv(ds_path + "train_labels.csv")

# Check data
features = Train.columns.tolist()[3:]
# print(features)
print(Train.head())
# print(Train_label.head())

sc = StandardScaler()
Train[features] = sc.fit_transform(Train[features])
print(Train.head())

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
epochs = 0
input_dim = 43
hidden_dim = 512
output_dim = 1
num_layers = 3
batch_size = 256
batch_size_val = 256
n_head = 8

# Data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)
# print("len(train_loader)", len(train_loader)) # 325

# model, optimizer, loss function
model = Transformer_Model(input_dim, hidden_dim, output_dim, num_layers, n_head).to(device)
summary(model)
# print(model)
# print(list(model.named_parameters()))
optimizer = optim.Adam(model.parameters(), lr=0.000001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, verbose=True)
# num_warmup_steps = int(0.1 * epochs * len(train_loader))
# num_training_steps = int(epochs * len(train_loader))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# model.to(device)
early_stopping = EarlyStopping(patience=10, verbose=True)

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
        # scheduler.step()
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
    scheduler.step(val_acc)
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

    # Early Stopping
    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("Early Stopping!")
        break

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

# Load the checkpoint
model.load_state_dict(torch.load("checkpoint_model.pth"))

# Make predictions
# Load test data
Test = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/newTestDataset.csv", dtype=np.float32)
# Test = pd.read_csv(ds_path + "test.csv", dtype=np.float32)
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
        output = model(x)
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
