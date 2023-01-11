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
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers, norm=None)
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



# Set start time
start = time.time()


# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/transformer_comp/data_comp") == False:
    os.makedirs(output_path + "result/transformer_comp/data_comp")
output_path = output_path + "result/transformer_comp/data_comp/"


# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"


# Set seed and device
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the training data
# No added data. input 13
Train0 = pd.read_csv(ds_path + "train.csv", dtype=np.float32)

# Dataset with lag and statistics features. input 43
Train1 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/newTrainDataset.csv", dtype=np.float32)

# Dataset with gaussian noise. input 26
Train2 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/gaussian_noise/add_gaussian_train.csv", dtype=np.float32)

# Dataset with spike. input 26
Train3 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/spike/add_spike_train.csv", dtype=np.float32)

# Dataset with warp(expand). Input 26
Train4 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/warp/add_warp_train.csv", dtype=np.float32)

# Merge data sets
# Train = pd.merge(Train, pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/add_gaussian_train.csv", dtype=np.float32))
Train_label = pd.read_csv(ds_path + "train_labels.csv")


# Load the test data
# No added data
Test0 = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

# Dataset with lag and statistics features. input 43
Test1 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/newTestDataset.csv", dtype=np.float32)

# Dataset with gaussian noise. input 26
Test2 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/add_gaussian_test.csv", dtype=np.float32)

# Dataset with spike noise. input 26
Test3 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/spike/add_spike_test.csv", dtype=np.float32)

# Dataset with warp. input 26
Test4 = pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/warp/add_warp_test.csv", dtype=np.float32)

# Merge data sets
# Test = pd.merge(Test, pd.read_csv("../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/add_gaussian_test.csv", dtype=np.float32))

# Check data
features = Train0.columns.tolist()[3:]
# print(features)
print(Train0.head())
# print(Train_label.head())
print(Test0.head())

# Standardize
sc = StandardScaler()
Train0[features] = sc.fit_transform(Train0[features])
Train1[features] = sc.fit_transform(Train1[features])
Train2[features] = sc.fit_transform(Train2[features])
Train3[features] = sc.fit_transform(Train3[features])
Train4[features] = sc.fit_transform(Train4[features])
# print(Train.head())
Test0[features] = sc.transform(Test0[features])
Test1[features] = sc.transform(Test1[features])
Test2[features] = sc.transform(Test2[features])
Test3[features] = sc.transform(Test3[features])
Test4[features] = sc.transform(Test4[features])


# Shape the training data
sequence = Train0["sequence"]
labels = Train_label["state"]
train0 = Train0.drop(["sequence", "subject", "step"], axis=1).values
train1 = Train1.drop(["sequence", "subject", "step"], axis=1).values
train2 = Train2.drop(["sequence", "subject", "step"], axis=1).values
train3 = Train3.drop(["sequence", "subject", "step"], axis=1).values
train4 = Train4.drop(["sequence", "subject", "step"], axis=1).values
# print("train.shape", train.shape) # (1558080, 13)
# in train there are 25968 rows which contains 60x13 matrix
# each rows means each sequence or subject's sensor data
train0 = train0.reshape(-1, 60, train0.shape[-1])
train1 = train1.reshape(-1, 60, train1.shape[-1])
train2 = train2.reshape(-1, 60, train2.shape[-1])
train3 = train3.reshape(-1, 60, train3.shape[-1])
train4 = train4.reshape(-1, 60, train4.shape[-1])
# print("train.shape", train.shape) # (25968, 60, 13)
# print("labels.shape", labels.shape) # (25968,)

# Shape the testing data
test_sequence = Test0["sequence"]
test0 = Test0.drop(["sequence", "subject", "step"], axis=1).values
test1 = Test1.drop(["sequence", "subject", "step"], axis=1).values
test2 = Test2.drop(["sequence", "subject", "step"], axis=1).values
test3 = Test3.drop(["sequence", "subject", "step"], axis=1).values
test4 = Test4.drop(["sequence", "subject", "step"], axis=1).values
test0 = test0.reshape(-1, 60, test0.shape[-1])
test1 = test1.reshape(-1, 60, test1.shape[-1])
test2 = test2.reshape(-1, 60, test2.shape[-1])
test3 = test3.reshape(-1, 60, test3.shape[-1])
test4 = test4.reshape(-1, 60, test4.shape[-1])


# train validation split
train_x0, val_x0, train_y0, val_y0 = train_test_split(train0, labels, test_size=0.2, shuffle=False)
train_x1, val_x1, train_y1, val_y1 = train_test_split(train1, labels, test_size=0.2, shuffle=False)
train_x2, val_x2, train_y2, val_y2 = train_test_split(train2, labels, test_size=0.2, shuffle=False)
train_x3, val_x3, train_y3, val_y3 = train_test_split(train3, labels, test_size=0.2, shuffle=False)
train_x4, val_x4, train_y4, val_y4 = train_test_split(train4, labels, test_size=0.2, shuffle=False)
# print("train_x.shape", train_x.shape) # (20774, 60, 13)
# print("train_y.shape", train_y.shape) # (20774,)
# print("val_x.shape", val_x.shape) # (5194, 60, 13)
# print("val_y.shape", val_y.shape) # (5194,)


# Create x and y tensor for train set
xTrain0 = torch.from_numpy(train_x0)
xTrain1 = torch.from_numpy(train_x1)
xTrain2 = torch.from_numpy(train_x2)
xTrain3 = torch.from_numpy(train_x3)
xTrain4 = torch.from_numpy(train_x4)
yTrain0 = torch.tensor(train_y0.values)
yTrain1 = torch.tensor(train_y1.values)
yTrain2 = torch.tensor(train_y2.values)
yTrain3 = torch.tensor(train_y3.values)
yTrain4 = torch.tensor(train_y4.values)
# print("xTrain.shape", xTrain.shape) # torch.Size([20774, 60, 13])
# print("yTrain.shape", yTrain.shape) # torch.Size([20774])


# Creat x and y for validation set
xVal0 = torch.from_numpy(val_x0)
xVal1 = torch.from_numpy(val_x1)
xVal2 = torch.from_numpy(val_x2)
xVal3 = torch.from_numpy(val_x3)
xVal4 = torch.from_numpy(val_x4)
yVal0 = torch.tensor(val_y0.values)
yVal1 = torch.tensor(val_y1.values)
yVal2 = torch.tensor(val_y2.values)
yVal3 = torch.tensor(val_y3.values)
yVal4 = torch.tensor(val_y4.values)
# print("xVal.shape", xVal.shape) # torch.Size([5194, 60, 13])
# print("yVal.shape", yVal.shape) # torch.Size([5194])

# Create test tensor
ttest0 = torch.from_numpy(test0)
ttest1 = torch.from_numpy(test1)
ttest2 = torch.from_numpy(test2)
ttest3 = torch.from_numpy(test3)
ttest4 = torch.from_numpy(test4)


# Create TensorDataset for train and validation sets
train_ds0 = TensorDataset(xTrain0, yTrain0)
train_ds1 = TensorDataset(xTrain1, yTrain1)
train_ds2 = TensorDataset(xTrain2, yTrain2)
train_ds3 = TensorDataset(xTrain3, yTrain3)
train_ds4 = TensorDataset(xTrain4, yTrain4)
val_ds0 = TensorDataset(xVal0, yVal0)
val_ds1 = TensorDataset(xVal1, yVal1)
val_ds2 = TensorDataset(xVal2, yVal2)
val_ds3 = TensorDataset(xVal3, yVal3)
val_ds4 = TensorDataset(xVal4, yVal4)
# print("len(train_ds)", len(train_ds)) # 20774
# print("len(val_ds)", len(val_ds)) # 5194

# Create TensorDataset for test sets
dtest0 = TensorDataset(ttest0)
dtest1 = TensorDataset(ttest1)
dtest2 = TensorDataset(ttest2)
dtest3 = TensorDataset(ttest3)
dtest4 = TensorDataset(ttest4)


# Parameter
epochs = 1000
input_dim = 13
input_dims = [13, 43, 26, 26, 26]
hidden_dim = 256
output_dim = 1
num_layers = 4
batch_size = 256
n_head = 8
lr = 0.0001
datasets = ["normal", "statistics", "gaussian", "spike", "warp"]


# array to store acc and loss for each model
score_list = []

# Compare
# for n_id in range(6):
#     hidden_size = hidden_dim[n_id%2]
#     batch_size = batch_sizes[n_id%2]
# Data loader
train_loader0 = DataLoader(train_ds0, batch_size=batch_size, shuffle=False)
train_loader1 = DataLoader(train_ds1, batch_size=batch_size, shuffle=False)
train_loader2 = DataLoader(train_ds2, batch_size=batch_size, shuffle=False)
train_loader3 = DataLoader(train_ds3, batch_size=batch_size, shuffle=False)
train_loader4 = DataLoader(train_ds4, batch_size=batch_size, shuffle=False)
train_loaders = [train_loader0, train_loader1, train_loader2, train_loader3, train_loader4]
val_loader0 = DataLoader(val_ds0, batch_size=batch_size, shuffle=False)
val_loader1 = DataLoader(val_ds1, batch_size=batch_size, shuffle=False)
val_loader2 = DataLoader(val_ds2, batch_size=batch_size, shuffle=False)
val_loader3 = DataLoader(val_ds3, batch_size=batch_size, shuffle=False)
val_loader4 = DataLoader(val_ds4, batch_size=batch_size, shuffle=False)
val_loaders = [val_loader0, val_loader1, val_loader2, val_loader3, val_loader4]
# print("len(train_loader)", len(train_loader)) # 325

# Create DataLoader for test sets
test_loader0 = DataLoader(dtest0, batch_size=batch_size, shuffle=False)
test_loader1 = DataLoader(dtest1, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(dtest2, batch_size=batch_size, shuffle=False)
test_loader3 = DataLoader(dtest3, batch_size=batch_size, shuffle=False)
test_loader4 = DataLoader(dtest4, batch_size=batch_size, shuffle=False)
test_loaders = [test_loader0, test_loader1, test_loader2, test_loader3, test_loader4]
# print(len(test_loader))
for dataset in datasets:
    print("Model: {}, dataset: {}".format(34+datasets.index(dataset), dataset))
    # model, optimizer, loss function
    model = Transformer_Model(input_dims[datasets.index(dataset)], hidden_dim, output_dim, num_layers, n_head).to(device)
    # summary(model)
    # print(model)
    # print(list(model.named_parameters()))
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
    es = EarlyStopping(patience=10, verbose=True)

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
        for i, (x, t) in enumerate(train_loaders[datasets.index(dataset)]):
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
        train_loss /= len(train_loaders[datasets.index(dataset)])
        train_loss_list.append(train_loss)
        train_acc /= len(train_loaders[datasets.index(dataset)])
        train_acc_list.append(train_acc)

        # validation
        model.eval()
        for i, (x, t) in enumerate(val_loaders[datasets.index(dataset)]):
            x=x.to(device)
            # y=model(x)
            y=model(x).squeeze(-1)
            # y=y.to("cpu")
            # loss = criterion(y, t)
            loss = criterion(y, t.to(device).float())
            val_loss += loss.data.item()
            val_acc += accuracy_score(t.tolist(), torch.round(y.detach().cpu()))
        scheduler.step(val_acc)
        val_loss /= len(val_loaders[datasets.index(dataset)])
        val_loss_list.append(val_loss)
        val_acc /= len(val_loaders[datasets.index(dataset)])
        val_acc_list.append(val_acc)

        # display results
        if epoch % 100 == 1:
            print("Ep/MaxEp    train_loss    train_acc    val_loss    val_acc     duration")
        end_train = time.time()

        print("{:4}/{}{:14.5}{:13.5}{:12.5}{:11.5}{:10.5}".format(epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_train - start_train))

        # Early Stopping
        es(train_acc, train_loss, val_acc, val_loss, epoch, model)
        if es.early_stop:
            print("Early Stopping!")
            score_list.append([es.epoch, es.train_acc, es.train_loss, es.val_acc, es.val_loss])
            break

    # Save glaph
    # Plot loss graph
    plt.figure(figsize=(5, 4))
    plt.plot(train_loss_list, label = "training")
    plt.plot(val_loss_list, label = "validation")
    plt.ylim(0, 0.3)
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (Cross Entropy))")
    plt.savefig(output_path+"loss" + str(34+datasets.index(dataset)) + ".png")
    # plt.savefig(output_path+"loss.png")
    plt.clf()
    plt.close()

    # Plot accuracy graph
    plt.figure(figsize=(5, 4))
    plt.plot(train_acc_list, label = "training")
    plt.plot(val_acc_list, label = "validation")
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(output_path+"acc" + str(34+datasets.index(dataset)) + ".png")
    # plt.savefig(output_path+"acc.png")
    plt.clf()
    plt.close()

    # Load the checkpoint
    model.load_state_dict(torch.load("checkpoint_model.pth"))

    # Make predictions
    # Predict
    preds = []
    with torch.no_grad():
        model.eval()
        for x in test_loaders[datasets.index(dataset)]:
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
    submissions.to_csv(output_path + "transformer_submissions" + str(34+datasets.index(dataset)) +  ".csv", index=False, header=True)
    # submissions.to_csv(output_path + "transformer_submissions.csv", index=False, header=True)
        # if n_id%2 and num_layers.index(n_layers)==3:
        #     lr *= 0.1

# Output score_list
print("Model    Epoch    train_acc    train_loss    val_acc    val_loss")
c=0
for i in score_list:
    print("{:5}{:8}{:14.5}{:13.5}{:12.5}{:11.5}".format(c+34, i[0], i[1], i[2], i[3], i[4]))
    c+=1


# Print total processing time
end = time.time()
print("Total duration {}".format(end - start))