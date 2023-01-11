import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq, ifft, ifftn
import matplotlib.pyplot as plt
import math

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/fft/img") == False:
    os.makedirs(output_path + "result/fft/img")
output_path = output_path + "result/fft/img/"

# Set dataset path
# ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"
ds_path = "../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/"

# Load the data
# train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
# test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)
df = pd.read_csv(ds_path + "oneDataset.csv", header=None, dtype=np.float32)
print(df.shape)
for i in range(13):
    x = df.iloc[:, i]
    y = fft(x.values)
    # print(y.shape) # 60
    n = x.size
    t = np.linspace(0, 60, 60)
    # freq = fftfreq(n)[:n//2]
    freq = fftfreq(n)
    # print(freq.shape)
    # iy = ifft(y)


    # FFT
    # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))
    # ax1.set_xlabel("time(s)")
    # ax1.set_ylabel("x")
    # ax1.plot(t, x)

    # ax2.set_xlabel("frequence(Hz)")
    # ax2.set_ylabel("|y|/n")
    # plt.plot(freq, np.abs(y[0:n//2])/n)
    # plt.savefig(output_path + "fft" + str(i) + ".png")
    # plt.clf()
    # plt.close

    # IFFT
    # plt.figure(figsize=(5, 4))
    # plt.plot(t, x, label="original")
    # plt.plot(t, iy, label="IFFT")
    # plt.legend()
    # plt.savefig(output_path + "ifft" + str(i) + ".png")
    # plt.clf()
    # plt.close()

    # Low Pass Filter
    # ignore frequency above tp 
    tp = math.floor(freq.std()*10) / 10.0

    z_lowpass = np.where(abs(freq) > tp, 0, y)
    y_lowpass = ifftn(z_lowpass).real

    plt.figure(figsize=(5, 4))
    plt.plot(t, x, "-", label="original")
    plt.plot(t, y_lowpass, "--", label="lowpass")
    plt.legend()
    plt.savefig(output_path + "lowpass" + str(i) + ".png")
    plt.clf()
    plt.close()

    # Check abnormal values
    diff = abs(y_lowpass - x)
    plt.figure(figsize=(5, 4))
    plt.plot(t, diff)
    plt.savefig(output_path + "diff" + str(i) + ".png")
    plt.clf()
    plt.close()


# print(train_df.isna().sum())
# print(test_df.isna().sum())

# train_df.to_csv(output_path + "newTrainDataset.csv", index=False, header=True)
# test_df.to_csv(output_path + "newTestDataset.csv", index=False, header=True)
