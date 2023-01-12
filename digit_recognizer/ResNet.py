import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# nn.Module is made from two parts. __init__() part and forward() part.
class Block(nn.Module):
    # In this method, create a layer instance and apply it forward method by your desired order.
    def __init__(self, channel_in, channel_out):
        # Don't forget this code supre().__init__()
        super().__init__()

        # 3*3 Conv
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu1 = nn.ReLU()

        # 3*3 Conv
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.relu2 = nn.ReLU()

        # for skip connection
        self.shortcut = self._shortcut(channel_in, channel_out)

        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)
        return y

    # This method check the channel_in and channel_out for skip connection
    def _shortcut(self, channel_in, channel_out):
        # if channne_in and channel_out are not same, ajust channel_in to channel_out
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    # if cannel_in and channel_out are not same, ajust it by using 1*1 Conv
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        # First few layers doesn't made from block, so just write it
        self.conv1 = nn.Conv2d(1, 14, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(14)
        self.relu1 = nn.ReLU()

        # Block 1
        # all blocks has same input and output channels
        self.block1 = nn.ModuleList([self._building_block(14) for _ in range(16)])
        # Block1 and Block2 has a difference in channels and output sizes.
        # Therefore, set input and output channels and stride as 2 to adjust output size.
        self.conv2 = nn.Conv2d(14, 28, kernel_size=(1, 1), stride=(2, 2))

        # Block 2
        # In Block2, input and output channels are same because of the previous operation.
        self.block2 = nn.ModuleList([self._building_block(28) for _ in range(16)])
        # Block2 and 3 has a difference in channels and output sizes.
        # Therefore, set input and output channels and stride as 2 to adjust output size.
        self.conv3 = nn.Conv2d(28, 56, kernel_size=(1, 1), stride=(2, 2))

        # Block 3
        # In Block 3, input and output channels are same because of the previous operation.
        self.block3 = nn.ModuleList([self._building_block(56) for _ in range(16)])

        # In the last section, we use average pool, 1000-d fc, and softmax
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(56, output_dim)
        # self.out = nn.Linear(1000, output_dim)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        # h = self.out(h)
        y = torch.log_softmax(h, dim=-1)

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


# main
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# output test
output_path = "output path" # eg. ./output

# Load the data
Train = pd.read_csv("train data path", dtype=np.float32) # eg. ./data/train.csv

# Split data into pixels and labels
x_train = Train.loc[:, Train.columns != "label"].values/255
x_train = x_train.reshape(-1, 1, 28, 28)
y_train = Train.label.values
print(x_train.shape)

# Split training and validation set
x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create feature and targets tensor for train set
XTrain = torch.from_numpy(x_train)
YTrain = torch.from_numpy(y_train).type(torch.LongTensor)

# Create feature and targets tensor for test set
XTest = torch.from_numpy(x_val)
YTest = torch.from_numpy(y_val).type(torch.LongTensor)

# Batch size
batch_size = 100

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(XTrain, YTrain)
test = torch.utils.data.TensorDataset(XTest, YTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle=False)

# check DataLoader
tmp = train_loader.__iter__()
x1, y1 = tmp.next()
# print(x1, y1)
# print(x1[0].shape)

# Show example
plt.imsave("output path for example iamge", x_train[0].reshape(28, 28)) # eg. ./output/egxample.png

# Build model
model = ResNet50(10).to(device)

# Check model
summary(model, input_size=(batch_size, 1, 28, 28))

# Training and evaluate the model
def compute_loss(label, pred):
    return criterion(pred, label)

def train_step(x, t):
    model.train()
    preds = model(x)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, preds

def test_step(x, t):
    model.eval()
    preds = model(x)
    loss = compute_loss(t, preds)

    return loss, preds

criterion = nn.NLLLoss()
optimizer = optimizers.Adam(model.parameters(), weight_decay=0.01)
epochs = 100

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    test_loss = 0.
    test_acc = 0.

    for (x, t) in train_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist()) 

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    for (x, t) in test_loader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print('Epoch: {}, Training loss: {:.3f}, Training Acc: {:.3f}, Val loss: {:.3f}, Val Acc: {:.3f}'.format(epoch+1, train_loss, train_acc, test_loss, test_acc))
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(test_loss)
    val_acc_list.append(test_acc)

# plot results
fig = plt.figure()
plt.plot(train_acc_list, label="Training acc")
plt.plot(val_acc_list, label="Validation acc")
plt.legend()
fig.savefig(output_path+"ResNet_Acc.png")
fig = plt.figure()
plt.plot(train_loss_list, label="Training loss")
plt.plot(val_loss_list, label="Validation loss")
plt.legend()
fig.savefig(output_path+"ResNet_Loss.png")

# Load test data
Test = pd.read_csv("test data path", dtype=np.float32) # eg. ./data/test.csv
Test = Test.values
Test = Test.reshape(-1, 1, 28, 28)
Test = Test/255

batch_size = 1
Test = torch.from_numpy(Test)
test_sets = torch.utils.data.TensorDataset(Test)

new_test_loader = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=False)
print(len(new_test_loader))

# Make predictions
print("Making predictions")
pred_list = []
with torch.no_grad():
    model.eval()
    c = 0
    for images in new_test_loader:
        images = images.to(device)
        output = model.forward(images)
        _, pred = torch.max(output.data, 1)
        pred_list.append(pred.item())

print("Complete making predictions")
pred_list = np.array(pred_list)
ID_lists = np.arange(1, pred_list.shape[0]+1)

submissions = pd.DataFrame({"ImageId": ID_lists, "Label": pred_list})
submissions.to_csv(output_path+"ResNet_submissions.csv", index=False, header=True)
