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
ds_path = "../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/"

# Load the data
df = pd.read_csv(ds_path + "oneDataset.csv", header=None, dtype=np.float32)

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

img_list = []
sensor_list = []

# Create dataset
for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    sensor_list.append(sensor_name)
    img_tmp = pd.DataFrame(abs(wavelet(df[sensor], scales, wavelet_type)).flatten()).transpose()
    img_list.append(img_tmp)
    # print("train.shape: ", len(img_list))



df_img = pd.DataFrame(np.squeeze(img_list))

# I need to create the dataset for image. Each sequence's image should be in one row. Convert it as matrix in training file.
# print(df_img.isna().sum())
# print(df_img.head())

df_img.to_csv(output_path + "oneImageDataset.csv", index=False, header=True)

ds = pd.read_csv(output_path + "oneImageDataset.csv", dtype=np.float32)
print(ds.shape)

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/wavelets/test") == False:
    os.makedirs(output_path + "result/wavelets/test")
output_path = output_path + "result/wavelets/test/"
ds = ds.values
ds = ds.reshape(-1, 60, 60)

for i, img in enumerate(ds):
    plt.figure(figsize=(5, 4))
    plt.imshow(img, aspect="auto")
    plt.yticks(y_positions, y_labels)
    plt.xticks(x_positions, x_labels)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.savefig(output_path + "wavelet" + str(i) + ".png")
    plt.clf()
    plt.close()