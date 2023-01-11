import numpy as np
import pandas as pd
import os

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset") == False:
    os.makedirs(output_path + "new_dataset")
output_path = output_path + "new_dataset/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

# Add statistics features to dataset
train_df["sensor_04_mean"] = train_df.groupby("sequence")["sensor_04"].transform("mean")
train_df["sensor_02_std"] = train_df.groupby("sequence")["sensor_02"].transform("std")
kurtosis = []
for i in range(train_df["sequence"].nunique()):
    kurt = train_df[train_df.sequence==i]["sensor_04"].kurt()
    for j in range(60):
        kurtosis.append(kurt)
train_df["sensor_04_kurttosis"] = kurtosis
kurtosis = []
for i in range(train_df["sequence"].nunique()):
    kurt = train_df[train_df.sequence==i]["sensor_10"].kurt()
    for j in range(60):
        kurtosis.append(kurt)
train_df["sensor_10_kurttosis"] = kurtosis

test_df["sensor_04_mean"] = test_df.groupby("sequence")["sensor_04"].transform("mean")
test_df["sensor_02_std"] = test_df.groupby("sequence")["sensor_02"].transform("std")
kurtosis = []
for i in range(train_df["sequence"].nunique(), train_df["sequence"].nunique()  + test_df["sequence"].nunique()):
    kurt = test_df[test_df.sequence==i]["sensor_04"].kurt()
    for j in range(60):
        kurtosis.append(kurt)
test_df["sensor_04_kurttosis"] = kurtosis
kurtosis = []
for i in range(train_df["sequence"].nunique(), train_df["sequence"].nunique()  + test_df["sequence"].nunique()):
    kurt = test_df[test_df.sequence==i]["sensor_10"].kurt()
    for j in range(60):
        kurtosis.append(kurt)
test_df["sensor_10_kurttosis"] = kurtosis

# Add lag to dataset
for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    train_df[f"{sensor_name}" + "_lag1"] = train_df.groupby("sequence")[f"{sensor_name}"].shift(1)
    train_df[f"{sensor_name}" + "_lag1"].fillna(0, inplace=True)
    train_df[f"{sensor_name}" + "_diff1"] = train_df[f"{sensor_name}"] - train_df[f"{sensor_name}" + "_lag1"]

for sensor in range(13):
    sensor_name = f"sensor_{sensor:02d}"
    test_df[f"{sensor_name}" + "_lag1"] = test_df.groupby("sequence")[f"{sensor_name}"].shift(1)
    test_df[f"{sensor_name}" + "_lag1"].fillna(0, inplace=True)
    test_df[f"{sensor_name}" + "_diff1"] = test_df[f"{sensor_name}"] - test_df[f"{sensor_name}" + "_lag1"]

print(train_df.isna().sum())
print(test_df.isna().sum())

train_df.to_csv(output_path + "newTrainDataset.csv", index=False, header=True)
test_df.to_csv(output_path + "newTestDataset.csv", index=False, header=True)
