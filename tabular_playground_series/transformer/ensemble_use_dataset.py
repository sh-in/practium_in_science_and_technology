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
from earlystopping import EarlyStopping


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


# Create a model for ensemble training
class Ensemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.hidden(x)
        h = self.out(h)
        y = self.sigmoid(h)

        return y


# Create dataframe for ensemble training and test
ensemble_train = pd.read_csv(output_path+"ensemble_train.csv", dtype=np.float32)
bi_lstm_train = pd.read_csv(output_path+"bi_lstm_train.csv", dtype=np.float32)
lstm_and_gru_train = pd.read_csv(output_path+"lstm_and_gru_train.csv", dtype=np.float32)
dnn_train = pd.read_csv(output_path+"dnn_train.csv", dtype=np.float32)
trains = [bi_lstm_train, lstm_and_gru_train, dnn_train]
for i in trains:
    ensemble_train = pd.merge(ensemble_train, i, left_index=True, right_index=True)

ensemble_test = pd.read_csv(output_path+"ensemble_test.csv", dtype=np.float32)
bi_lstm_test = pd.read_csv(output_path+"bi_lstm_test.csv", dtype=np.float32)
lstm_and_gru_test = pd.read_csv(output_path+"lstm_and_gru_test.csv", dtype=np.float32)
dnn_test = pd.read_csv(output_path+"dnn_test.csv", dtype=np.float32)
tests = [bi_lstm_test, lstm_and_gru_test, dnn_test]
for i in tests:
    ensemble_test = pd.merge(ensemble_test, i, left_index=True, right_index=True)

Train_label = pd.read_csv("../../../../datasets/kaggle_tabular_playground_series_apr_2022/train_labels.csv", dtype=np.float32)
labels = Train_label["state"]
print(ensemble_train.head)
print(ensemble_train.shape)
print(ensemble_test.head)
print(ensemble_test.shape)
print("accuracy_score(wavelets): ", accuracy_score(labels, np.round(ensemble_train["wavelet"])))
print("accuracy_score(transformer): ", accuracy_score(labels, np.round(ensemble_train["transformer"])))
print("accuracy_score(bi_lstm): ", accuracy_score(labels, np.round(ensemble_train["bi_lstm"])))
print("accuracy_score(lstm_and_gru): ", accuracy_score(labels, np.round(ensemble_train["lstm_and_gru"])))
print("accuracy_score(dnn): ", accuracy_score(labels, np.round(ensemble_train["dnn"])))
print("accuracy_score(voting): ", accuracy_score(labels, np.round((ensemble_train["transformer"]+ensemble_train["wavelet"]+ensemble_train["bi_lstm"]+ensemble_train["lstm_and_gru"]+ensemble_train["dnn"])/5.0)))

# Voting
# preds = np.round((ensemble_test["transformer"]+ensemble_test["wavelet"])/2.0)
# submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+ensemble_test.shape[0]), "state": preds})
# submissions.to_csv(output_path+"ensemble_submissions_voting" + str(124) + ".csv", index=False, header=True)

ensemble_train = ensemble_train.values
print("ensemble_train.shape[1]", ensemble_train.shape[1])
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
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dtest, batch_size=batch_size, shuffle=False)

# model, optimizer, loss function
epochs = 1000
input_dim = ensemble_train.shape[1]
hidden_dim = 10
output_dim = 1
lr=0.0001
model = Ensemble(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=4, verbose=True)

ensemble_train_loss_list = []
ensemble_train_acc_list = []
ensemble_val_loss_list = []
ensemble_val_acc_list = []
ensemble_es = EarlyStopping(name="_ensemble", patience=10, verbose=True)
ensemble_score_list = []

ensemble_train_loss_list, ensemble_train_acc_list, ensemble_val_loss_list, ensemble_val_acc_list, ensemble_score_list = training(model, epochs, train_loader, val_loader, ensemble_es)

plot_results(ensemble_train_loss_list, ensemble_train_acc_list, ensemble_val_loss_list, ensemble_val_acc_list, "125")

# Load the checkpoint
model.load_state_dict(torch.load("checkpoint_model_ensemble.pth"))

print("Making predictions for test dataset")
preds = make_predictions(test_loader, model)

print("Complete making predictions")
# preds = np.array(preds)
# Linear
# submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+len(dtest)), "state": preds})
# submissions.to_csv(output_path+"ensemble_submissions_linear" + str(123) + ".csv", index=False, header=True)
# lowpass & noise, wavelet, attention, biLSTM, LSTM
submissions = pd.DataFrame({"sequence": np.arange(25968, 25968+len(dtest)), "state": preds})
submissions.to_csv(output_path+"ensemble_submissions_linear" + str(125) + ".csv", index=False, header=True)

# Print total processing time
end = time.time()
print("Total duration {}".format(end-start))