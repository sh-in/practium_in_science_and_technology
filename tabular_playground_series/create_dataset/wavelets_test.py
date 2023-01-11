import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os
import math

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/wavelets/img") == False:
    os.makedirs(output_path + "result/wavelets/img")
output_path = output_path + "result/wavelets/img/"

# Set dataset path
# ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"
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

def get_ticks_label_set(labels, num):
    l = len(labels)
    step = l//(num-1)
    position = np.arange(0, l, step)
    label = labels[::step]
    return position, label


x_positions, x_labels = get_ticks_label_set(t, 10)
print(x_positions)
print(x_labels)
# y_positions, y_labels = get_ticks_label_set(freqs[::-1], 10)
y_positions, y_labels = get_ticks_label_set(freqs, 10)
print(y_positions)
print(y_labels)
y_labels = [math.floor(d * 10 ** 2) / 10 ** 2 for d in y_labels]
print(y_labels)

# Analysis
for i in range(13):
    x = df.iloc[:, i]
    cwtmatr, freqs_rate = pywt.cwt(x, scales=scales, wavelet=wavelet_type)
    plt.figure(figsize=(5, 4))
    plt.imshow(np.abs(cwtmatr), aspect="auto")
    plt.yticks(y_positions, y_labels)
    plt.xticks(x_positions, x_labels)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.savefig(output_path + "wavelet" + str(i) + ".png")
    plt.clf()
    plt.close()