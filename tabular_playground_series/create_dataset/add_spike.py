import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/spike") == False:
    os.makedirs(output_path + "new_dataset/spike")
output_path = output_path + "new_dataset/spike/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

# Add Spike Noise
np.random.seed(1234)

def spike_noise(std):
    r = np.random.randint(60, size=np.random.randint(3)+1)
    noise = np.zeros(60)
    for i in r:
        noise[i] = 2*std*np.random.rand() - std
    return noise

for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    train_x = train_df[sensor_name]
    train_std = train_x.std()
    test_x = test_df[sensor_name]
    test_std = test_x.std()
    train_spike = []
    test_spike = []
    for i in range(train_df["sequence"].nunique()):
        train_spike = np.append(train_spike, spike_noise(train_std))
    for i in range(test_df["sequence"].nunique()):
        test_spike = np.append(test_spike, spike_noise(test_std))
    train_df[sensor_name + "_spike"] = train_df[sensor_name] + train_spike
    test_df[sensor_name + "_spike"] = test_df[sensor_name] + test_spike

    plt.figure()
    plt.plot(train_df.loc[train_df.sequence==0, [sensor_name]], label="train", alpha=1)
    plt.plot(train_df.loc[train_df.sequence==0, [sensor_name + "_spike"]], label="train_spike", alpha=0.5)
    plt.legend()
    plt.savefig(output_path + f"{sensor_name}" + "_spike.png")
    plt.clf()
    plt.close()

print(train_df.isna().sum())
print(test_df.isna().sum())

train_df.to_csv(output_path + "add_spike_train.csv", index=False, header=True)
test_df.to_csv(output_path + "add_spike_test.csv", index=False, header=True)
