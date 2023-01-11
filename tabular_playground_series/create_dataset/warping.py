import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

import time


# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/warp") == False:
    os.makedirs(output_path + "new_dataset/warp")
output_path = output_path + "new_dataset/warp/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

# Add warping noise
np.random.seed(1234)

# shrink noise
# def shrink_noise(x, i):
#     r = np.random.randint(i+3, i+56)
#     # print(x)
#     with_noise = x
#     with_noise[r-1] = x[r-2]
#     if x[r-2] > x[r-3]:
#         with_noise[r-2] = (x[r-2] - x[r-3])/2.0
#     else:
#         with_noise[r-2] = (x[r-3] - x[r-2])/2.0
#     with_noise[r+1] = x[r+2]
#     if x[r+2] > x[r+3]:
#         with_noise[r+2] = (x[r+2] - x[r+3])/2.0
#     else:
#         with_noise[r+2] = (x[r+3] - x[r+2])/2.0
#     return with_noise

def shrink_noise(x, i):
    r = np.random.randint(i+7, i+52)
    noise = np.zeros(60)
    noise += x
    s = 1
    for j in range(-1, -4, -1):
        noise[r+j] = noise[r+j-s]
        s+=1
    d = (x[r-7] + noise[r-3])/4.0
    for j in range(r-4, r-7):
        noise[r+j] = d
        d += d
    
    s = 1
    for j in range(1, 4):
        noise[r+j] = noise[r+j+s]
        s+=1
    d = (x[r+7]+noise[r+3])/4.0
    for j in range(6, 3, -1):
        noise[r+j] = d
        d+=d

    return noise

# expand noise
def expand_noise(x, i):
    r = np.random.randint(i+4, i+55)
    noise = np.zeros(60)
    noise[r-i] = x[r]
    for j in range(i, r-4):
        noise[j-i] = x[j+2]
    for j in range(i+59, r+4, -1):
        noise[j-i] = x[j-2]
    s = 2
    for j in range(r-4, r, 2):
        noise[j-i] = x[j+s]
        s-=1
    s = 2
    for j in range(r+4, r, -2):
        noise[j-i] = x[j-s]
        s-=1
    for j in range(r-3, r+3+1, 2):
        noise[j-i] = (noise[j-i-1] + noise[j-i+1])/2.0
    return noise

for sensor in range(13):
    start_adding_noise = time.time()
    sensor_name = f"sensor_{sensor:02d}"
    train_x = train_df[sensor_name]
    test_x = test_df[sensor_name]
    train_warp = []
    test_warp = []
    # print(train_df.loc[train_df.sequence==0, [sensor_name]])
    for i in range(train_df["sequence"].nunique()):
        train_warp = np.append(train_warp, expand_noise(train_x[i*60:(i+1)*60], i*60))
    n = train_df["sequence"].nunique()
    for i in range(test_df["sequence"].nunique()):
        test_warp = np.append(test_warp, expand_noise(test_x[i*60:(i+1)*60], i*60))
    train_df[sensor_name + "_warp"] = train_warp
    test_df[sensor_name + "_warp"] = test_warp

    plt.figure()
    plt.plot(train_df.loc[train_df.sequence==0, [sensor_name]], label="train", alpha=0.5)
    # print(train_df.loc[train_df.sequence==0, [sensor_name + "_warp"]])
    plt.plot(train_df.loc[train_df.sequence==0, [sensor_name + "_warp"]], label="train_warp", alpha=0.5)
    plt.legend()
    plt.savefig(output_path + f"{sensor_name}" + "_warp.png")
    plt.clf()
    plt.close()
    end_adding_noise = time.time()
    print("Duration: {}".format(end_adding_noise - start_adding_noise))

print(train_df.isna().sum())
print(test_df.isna().sum())

train_df.to_csv(output_path + "add_warp_train.csv", index=False, header=True)
test_df.to_csv(output_path + "add_warp_test.csv", index=False, header=True)
