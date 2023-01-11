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
from sklearn.preprocessing import StandardScaler

import math
import time
from earlystopping import EarlyStopping


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
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers, norm=None)
        self.dropout_enc = nn.Dropout(0.5)
        
        # Output layer
        self.dropout_dec = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        # Input layer
        x = self.input(x)
        x = self.positional_encoding_layer(x)

        # Encoder
        # x = self.encoder_layer(x)
        x = self.encoder(x)
        x = self.dropout_enc(x)

        # Output layer
        # y = torch.max(x, 1)[0]
        y = x.mean(dim=1)
        x = self.dropout_dec(x)
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


# Set start time.
start = time.time()

# Set output path.
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/transformer_comp/ensemble") == False:
    os.makedirs(output_path + "result/transformer_comp/ensemble")
output_path = output_path + "result/transformer_comp/ensemble/"

# Set dataset path.
ds_path = "../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/wavelets/"

# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Wavelets
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
epochs = 1000
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

def training(model, epochs, train_loader, val_loader, es):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    score_list = []
    for epoch in range(1, epochs+1):
        start_train = time.time()
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.

        model.train()
        for (x, t) in train_loader:
            x, t = x.to(device), t.to(device)
            preds = model(x).squeeze(-1)
            loss = compute_loss(t, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(preds)
            train_loss += loss.item()
            train_acc += accuracy_score(t.tolist(), torch.round(preds.detach().cpu()))
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        for (x, t) in val_loader:
            x, t = x.to(device), t.to(device)
            preds = model(x).squeeze(-1)
            loss = compute_loss(t, preds)
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
            return train_loss_list, train_acc_list, val_loss_list, val_acc_list, score_list
        if epoch == epochs:
            return train_loss_list, train_acc_list, val_loss_list, val_acc_list, score_list



def plot_results(train_loss_list, train_acc_list, val_loss_list, val_acc_list, model):
    # Plot results
    # Plot loss graph
    plt.figure(figsize=(5, 4))
    plt.plot(train_loss_list, label="training")
    plt.plot(val_loss_list, label="validation")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(output_path+"loss" + model +".png")

    # Plot accuracy graph
    plt.figure(figsize=(5, 4))
    plt.plot(train_acc_list, label = "training")
    plt.plot(val_acc_list, label = "validation")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(output_path+"acc" + model + ".png")
    plt.clf()
    plt.close()


def make_predictions(test_loader, model):
    preds = []
    with torch.no_grad():
        model.eval()
        for x in test_loader:
            x = x[0].to(device)
            output = model(x)
            preds.append(output.detach().cpu())
    preds = torch.round(torch.from_numpy(np.concatenate(preds, 0).flatten()))
    return preds


# training section can be omitted
# wl_train_loss_list = []
# wl_train_acc_list = []
# wl_val_loss_list = []
# wl_val_acc_list = []

# wl_es = EarlyStopping(name="_wavelets", patience=10, verbose=True)
# wl_score_list = []

# wl_train_loss_list, wl_train_acc_list, wl_val_loss_list, wl_val_acc_list, wl_score_list = training(model, epochs, train_loader, val_loader, wl_es)

# plot_results(wl_train_loss_list, wl_train_acc_list, wl_val_loss_list, wl_val_acc_list, "_wavelets")


# Making prediction for training dataset.
# Load the checkpoint
model.load_state_dict(torch.load("checkpoint_model_wavelets.pth"))

print("Making predictions for training dataset")
wl_train_preds_list = []
# Create data loader for predicting training datasets.
xTrain = torch.from_numpy(train)
yTrain = torch.tensor(labels.values)
train_ds = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for (x, t) in train_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        wl_train_preds_list.append(preds.detach().cpu())
print("len(preds): ", len(wl_train_preds_list))
print("len(preds[0]): ", len(wl_train_preds_list[0]))
print("preds[0]: ", wl_train_preds_list[0])
wl_train_preds_list = torch.from_numpy(np.concatenate(wl_train_preds_list, 0).flatten())
print("len(preds): ", len(wl_train_preds_list))
print("preds[0]: ", wl_train_preds_list[0])
print("accuracy_score(wavelets): ", accuracy_score(labels, torch.round(wl_train_preds_list)))

print("Making predictions for test dataset")
wl_test_preds_list = []
with torch.no_grad():
    for x in test_loader:
        x = x[0].to(device)
        preds = model(x)
        wl_test_preds_list.append(preds.detach().cpu())
wl_test_preds_list = torch.from_numpy(np.concatenate(wl_test_preds_list, 0).flatten())

###############################################################

# Lowpass + noise
# Load data
Train = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/lowpass_gauss_train.csv", dtype=np.float32)
Test = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/lowpass_gauss_test.csv", dtype=np.float32)

# Test in small dataset
# reader = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/lowpass_gauss_train.csv", dtype=np.float32, chunksize=640*60)
# t = reader.get_chunk(640*60)
# Train = t
# print(Train.tail())

# readers = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/lowpass_gauss_test.csv", dtype=np.float32, chunksize=128*60)
# t = reader.get_chunk(128*60)
# Test = t
# print(Test.tail())

features = Train.columns.tolist()[3:]

# Standardize
sc = StandardScaler()
Train[features] = sc.fit_transform(Train[features])
Test[features] = sc.transform(Test[features])

# Shape the training data
sequence = Train["sequence"]
train = Train.drop(["sequence", "subject", "step"], axis=1).values
train = train.reshape(-1, 60, train.shape[-1])

# Shape testing data
test_sequence = Test["sequence"]
test = Test.drop(["sequence", "subject", "step"], axis=1).values
test = test.reshape(-1, 60, test.shape[-1])

# train validation split
train_x, val_x, train_y, val_y = train_test_split(train, labels, test_size=0.2, shuffle=False)

# Create x and y tensor for train set
xTrain = torch.from_numpy(train_x)
yTrain = torch.tensor(train_y.values)
# Create x and y tensor for validation set
xVal = torch.from_numpy(val_x)
yVal = torch.tensor(val_y.values)
# Create test tensor
ttest = torch.from_numpy(test)

# Create TensorDataset for train and validation sets
train_ds = TensorDataset(xTrain, yTrain)
val_ds = TensorDataset(xVal, yVal)
# Create TensorDataset for test sets
dtest = TensorDataset(ttest)

# Parameter
epochs = 1000
input_dim = train_x.shape[-1]
hidden_dim = 256
output_dim = 1
num_layers = 4
batch_size = 128
n_head = 8
lr = 0.0001
print("input_dim: ", input_dim)

# Data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
# Create DataLoader for test sets
test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=False)

# model, optimizer, loss function
model = Transformer_Model(input_dim, hidden_dim, output_dim, num_layers, n_head).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, verbose=True)


# training section can be omitted
# transformer_train_loss_list = []
# transformer_train_acc_list = []
# transformer_val_loss_list = []
# transformer_val_acc_list = []
# transformer_es = EarlyStopping(name="_transformer", patience=10, verbose=True)
# transformer_score_list = []

# transformer_train_loss_list, transformer_train_acc_list, transformer_val_loss_list, transformer_val_acc_list, transformer_score_list = training(model, epochs, train_loader, val_loader, transformer_es,)

# plot_results(transformer_train_loss_list, transformer_train_acc_list, transformer_val_loss_list, transformer_val_acc_list, "_transformer")


# Making prediction for training dataset.
# Load the checkpoint
# model.load_state_dict(torch.load("checkpoint_model_transformer.pth"))
model.load_state_dict(torch.load("checkpoint_model_lowpass_noise.pth"))

print("Making predictions for training dataset")
transformer_train_preds_list = []
# Create data loader for predicting training datasets.
xTrain = torch.from_numpy(train)
yTrain = torch.tensor(labels.values)
train_ds = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for (x, t) in train_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        transformer_train_preds_list.append(preds.detach().cpu())
print("len(preds): ", len(transformer_train_preds_list))
print("len(preds[0]): ", len(transformer_train_preds_list[0]))
print("preds[0]: ", transformer_train_preds_list[0])
transformer_train_preds_list = torch.from_numpy(np.concatenate(transformer_train_preds_list, 0).flatten())
print("len(preds): ", len(transformer_train_preds_list))
print("preds[0]: ", transformer_train_preds_list[0])
print("accuracy_score(transformer lowpass & noise): ", accuracy_score(labels, torch.round(transformer_train_preds_list)))

print("Making predictions for test dataset")
transformer_test_preds_list=[]
with torch.no_grad():
    for x in test_loader:
        x = x[0].to(device)
        preds = model(x)
        transformer_test_preds_list.append(preds.detach().cpu())
transformer_test_preds_list = torch.from_numpy(np.concatenate(transformer_test_preds_list, 0).flatten())

###############################################################

# Create dataframe for ensemble training and test
ensemble_train = pd.DataFrame({"wavelet":wl_train_preds_list, "transformer":transformer_train_preds_list})
ensemble_train.to_csv(output_path+"ensemble_train.csv", index=False, header=True)
ensemble_test = pd.DataFrame({"wavelet":wl_test_preds_list, "transformer":transformer_test_preds_list})
ensemble_test.to_csv(output_path+"ensemble_test.csv", index=False, header=True)
print(ensemble_train.head)
print(ensemble_train.shape)
print(ensemble_test.head)
print(ensemble_test.shape)
ensemble_train = ensemble_train.values
ensemble_test = ensemble_test.values
train_x, val_x, train_y, val_y = train_test_split(ensemble_train, labels, test_size=0.2, shuffle=False)
xTrain = torch.from_numpy(train_x)
yTrain = torch.tensor(train_y.values)
xVal = torch.from_numpy(val_x)
yVal = torch.tensor(val_y.values)
ttest = torch.from_numpy(ensemble_test)
train_ds = TensorDataset(xTrain, yTrain)
val_ds = TensorDataset(xVal, yVal)
dtest = TensorDataset(ttest)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=False)

# Create a model for ensemble training
class Ensemble(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.out = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.out(x)
        y = self.sigmoid(h)

        return y


# model, optimizer, loss function
epochs = 1000
input_dim = 2
output_dim = 1
model = Ensemble(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, verbose=True)

ensemble_train_loss_list = []
ensemble_train_acc_list = []
ensemble_val_loss_list = []
ensemble_val_acc_list = []
ensemble_es = EarlyStopping(name="_ensemble", patience=10, verbose=True)
ensemble_score_list = []

ensemble_train_loss_list, ensemble_train_acc_list, ensemble_val_loss_list, ensemble_val_acc_list, ensemble_score_list = training(model, epochs, train_loader, val_loader, ensemble_es,)

plot_results(ensemble_train_loss_list, ensemble_train_acc_list, ensemble_val_loss_list, ensemble_val_acc_list, "123")

# Load the checkpoint
model.load_state_dict(torch.load("checkpoint_model_ensemble.pth"))

print("Making predictions for test dataset")
preds = make_predictions(test_loader, model)

print("Complete making predictions")
# preds = np.array(preds)
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+len(dtest)), "state": preds})
submissions.to_csv(output_path+"wavelets_submissions" + str(123) + ".csv", index=False, header=True)

# Print total processing time
end = time.time()
print("Total duration {}".format(end-start))