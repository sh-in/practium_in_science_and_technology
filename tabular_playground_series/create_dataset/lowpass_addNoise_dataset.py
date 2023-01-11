import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq, ifft, ifftn
import matplotlib.pyplot as plt
import math

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/fft") == False:
    os.makedirs(output_path + "new_dataset/fft")
output_path = output_path + "new_dataset/fft/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"
# ds_path = "../../../../output/Tabular_Playground_Series_Apr_2022/new_dataset/fft/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

def low_pass(x):
    y = fft(x.values)
    n = x.size
    t = np.linspace(0, 60, 60)
    freq = fftfreq(n)

    tp = math.floor(freq.std()*10) / 10.0
    z_lowpass = np.where(abs(freq) > tp, 0, y)
    y_lowpass = ifftn(z_lowpass).real

    return y_lowpass


def abnormal_vals(x):
    y_lowpass = low_pass(x)
    diff = abs(y_lowpass - x)
    return diff

# Add Gaussian Noise
def gaussian_noise(x, mean, std):
    noise = np.random.normal(mean, std, size=x.shape)
    return x+noise


for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    train_low = []
    test_low = []
    train_abn = []
    test_abn = []
    l = train_df["sequence"].nunique()
    for i in range(train_df["sequence"].nunique()):
        train_low = np.append(train_low, low_pass(train_df[train_df.sequence==i][sensor_name]))
        train_abn = np.append(train_abn, abnormal_vals(train_df[train_df.sequence==i][sensor_name]))
    for i in range(train_df["sequence"].nunique(), train_df["sequence"].nunique() + test_df["sequence"].nunique()):
        test_low = np.append(test_low, low_pass(test_df[test_df.sequence==i][sensor_name]))
        test_abn = np.append(test_abn, abnormal_vals(test_df[test_df.sequence==i][sensor_name]))
    train_df[sensor_name + "_lowpass"] = train_low
    train_df[sensor_name + "_abnormal"] = train_abn
    test_df[sensor_name + "_lowpass"] = test_low
    test_df[sensor_name + "_abnormal"] = test_abn
    train_df[sensor_name + "_gauss_noise"] = gaussian_noise(train_df[sensor_name + "_lowpass"], train_df[sensor_name + "_lowpass"].mean(), train_df[sensor_name + "_lowpass"].std())
    test_df[sensor_name + "_gauss_noise"] = gaussian_noise(test_df[sensor_name + "_lowpass"], test_df[sensor_name + "_lowpass"].mean(), test_df[sensor_name + "_lowpass"].std())



print(train_df.isna().sum())
print(test_df.isna().sum())

train_df.to_csv(output_path + "lowpass_gauss_train.csv", index=False, header=True)
test_df.to_csv(output_path + "lowpass_gauss_test.csv", index=False, header=True)

output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "result/wavelets/lowpass_gauss") == False:
    os.makedirs(output_path + "result/wavelets/lowpass_gauss")
output_path = output_path + "result/wavelets/lowpass_gauss/"

plt.figure(figsize=(5, 4))
plt.plot(train_df[train_df.sequence==0]["sensor_00_lowpass"], "-", label="lowpass")
plt.plot(train_df[train_df.sequence==0]["sensor_00_gauss_noise"], "-", label="noise")
plt.legend()
plt.savefig(output_path + "lowpass_noise.png")
plt.clf()
plt.close()