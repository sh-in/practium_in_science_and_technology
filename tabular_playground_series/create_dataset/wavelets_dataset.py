import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os
import math
import csv

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/wavelets") == False:
    os.makedirs(output_path + "new_dataset/wavelets")
output_path = output_path + "new_dataset/wavelets/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)


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

def wavelet(x, scales, wavelet_type):
    cwtmatr, freqs_rate = pywt.cwt(x, scales=scales, wavelet=wavelet_type)
    return cwtmatr


def get_ticks_label_set(labels, num):
    l = len(labels)
    step = l//(num-1)
    position = np.arange(0, l, step)
    label = labels[::step]
    return position, label


x_positions, x_labels = get_ticks_label_set(t, 10)
y_positions, y_labels = get_ticks_label_set(freqs, 10)
y_labels = [math.floor(d * 10 ** 2) / 10 ** 2 for d in y_labels]

sensor_list = []

# Create dataset
for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    sensor_list.append(sensor_name)
    train = []
    test = []
    for i in range(train_df["sequence"].nunique()):
        train.append(pd.DataFrame(abs(wavelet(train_df[train_df.sequence==i][sensor_name], scales, wavelet_type)).flatten()).transpose())
    for i in range(train_df["sequence"].nunique(), train_df["sequence"].nunique()+test_df["sequence"].nunique()):
        test.append(pd.DataFrame(abs(wavelet(test_df[test_df.sequence==i][sensor_name], scales, wavelet_type)).flatten()).transpose())
    # print("len(train)", len(train))
    # print("len(test)", len(test))
    train_imgs = pd.DataFrame(np.squeeze(train))
    test_imgs = pd.DataFrame(np.squeeze(test))
    # print("train_imgs.shape: ", train_imgs.shape)
    # print("test_imgs.shape: ", test_imgs.shape)

    # print(train_imgs.isna().sum())
    # print(test_imgs.isna().sum())
    train_imgs.to_csv(output_path + sensor_name + "_train.csv", index=False, header=True)
    test_imgs.to_csv(output_path + sensor_name + "_test.csv", index=False, header=True)
    print(sensor_name + " is done.")

# img = pd.read_csv(output_path + "sensor_00_train.csv", dtype=np.float32)
# print("img.shape: ", img.shape)
# img = img.values
# img = img.reshape(-1, 60, 60)
# print("img.shape: ", img.shape)
# print(img[0].shape)

# # Set output path
# output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
# if os.path.exists(output_path + "result/wavelets/test") == False:
#     os.makedirs(output_path + "result/wavelets/test")
# output_path = output_path + "result/wavelets/test/"

# plt.figure(figsize=(5, 4))
# plt.imshow(img[0], aspect="auto")
# plt.yticks(y_positions, y_labels)
# plt.xticks(x_positions, x_labels)
# plt.xlabel("Time[s]")
# plt.ylabel("Frequency[Hz]")
# plt.savefig(output_path + "wavelet_one_0" + ".png")
# plt.clf()
# plt.close()