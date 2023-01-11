import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/gaussian_noise") == False:
    os.makedirs(output_path + "new_dataset/gaussian_noise")
output_path = output_path + "new_dataset/gaussian_noise/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

# Add Gaussian Noise
def gaussian_noise(x, mean, std):
    noise = np.random.normal(mean, std, size=x.shape)
    return x+noise

for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    x = train_df[sensor_name]
    train_df[f"{sensor_name}" + "_gauss_noise"] = gaussian_noise(x, x.mean(), x.std())
    x = test_df[sensor_name]
    test_df[f"{sensor_name}" + "_gauss_noise"] = gaussian_noise(x, x.mean(), x.std())
    plt.figure()
    plt.xlim(train_df[f"{sensor_name}"].mean() - 3*train_df[f"{sensor_name}"].std(), train_df[f"{sensor_name}"].mean() + 3*train_df[f"{sensor_name}"].std())
    sns.kdeplot(train_df[f"{sensor_name}"], label="train")
    sns.kdeplot(train_df[f"{sensor_name}" + "_gauss_noise"], label="train_with_noise")
    sns.kdeplot(test_df[f"{sensor_name}"], label="test")
    sns.kdeplot(test_df[f"{sensor_name}" + "_gauss_noise"], label="test_with_noise")
    plt.legend()
    plt.savefig(output_path + f"{sensor_name}" + "_gauss_noise.png")
    plt.clf()
    plt.close()

print(train_df.isna().sum())
print(test_df.isna().sum())

train_df.to_csv(output_path + "add_gaussian_train.csv", index=False, header=True)
test_df.to_csv(output_path + "add_gaussian_test.csv", index=False, header=True)
