import numpy as np
import pandas as pd
import os
from scipy import fft
import matplotlib.pyplot as plt
import csv

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "new_dataset/fft") == False:
    os.makedirs(output_path + "new_dataset/fft")
output_path = output_path + "new_dataset/fft/"

# Set dataset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
train_df = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
# test_df = pd.read_csv(ds_path + "test.csv", dtype=np.float32)

train_df = train_df.drop(["sequence", "subject", "step"], axis=1).values
train_df = train_df[:60]

print(train_df.shape)

# print(train_df.isna().sum())
# print(test_df.isna().sum())

f = open(output_path + "oneDataset.csv", "w")
writer = csv.writer(f)
writer.writerows(train_df)
f.close()

# train_df.to_csv(output_path + "oneDataset.csv", index=False, header=True)
# test_df.to_csv(output_path + "newTestDataset.csv", index=False, header=True)