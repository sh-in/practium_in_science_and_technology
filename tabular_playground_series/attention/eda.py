# Exploratory Data Analysis

import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

# Set output path
output_path = "../../../../output/Tabular_Playground_Series_Apr_2022/"
if os.path.exists(output_path + "attention/torch/eda") == False:
    os.makedirs(output_path + "attention/torch/eda")
output_path = output_path + "attention/torch/eda/"

# Set detaset path
ds_path = "../../../../datasets/kaggle_tabular_playground_series_apr_2022/"

# Load the data
# Train = pd.read_csv(ds_path + "train.csv", dtype=np.float32)
# Train_label = pd.read_csv(ds_path + "train_labels.csv")

# Join Data
# df = pd.merge(Train, Train_label, how="left", on="sequence")
# print(df.head())

# Shape of df
# print(df.shape) # (1558080, 17)

# Data type, null values or not
# print(df.info())

# Various summary statistics
# df.describe().to_csv(output_path + "stats.csv")

# Correlation heatmap
# df_corr = df.corr()
# plt.figure(figsize=(8, 10))
# sns.heatmap(df_corr, square=True, vmax=1, vmin=-1, center=0)
# plt.savefig(output_path + "heatmap.png")
# plt.clf()
# plt.close()

# Each sensors's mean of each time steps.
# pandas pivot table
# sensors = df.columns[df.columns.str.contains("sensor")]
# steps_0 = df[df["state"]==0].pivot_table(index=["step"], values=sensors, aggfunc="mean")
# steps_1 = df[df["state"]==1].pivot_table(index=["step"], values=sensors, aggfunc="mean")
# for i in range(13):
#     plt.figure()
#     plt.plot(steps_0.iloc[:, i], label="state=0")
#     plt.plot(steps_1.iloc[:, i], label="state=1")
#     plt.legend()
#     plt.xlabel("steps")
#     plt.ylabel("mean")
#     plt.savefig(output_path + "sensor_mean" + str(i) + ".png")
#     plt.clf()
#     plt.close()

# Distribution
# for i in range(13):
#     if os.path.exists(output_path + "dist") == False:
#         os.makedirs(output_path + "dist")
#     plt.figure()
#     plt.hist(df[sensors[i]], bins=100)
#     plt.title("sensor" + str(i))
#     plt.savefig(output_path + "dist/sensor" + str(i) + ".png")
#     plt.clf()
#     plt.close()

# Skewness and Kurtosis
# skewness = []
# kurtosis = []
# for i in range(13):
#     skewness.append(df[sensors[i]].skew())
#     kurtosis.append(df[sensors[i]].kurt())
# plt.figure()
# plt.bar(sensors, skewness)
# plt.xticks(rotation=90)
# plt.savefig(output_path + "skew.png")
# plt.clf()
# plt.close()

# plt.figure()
# plt.bar(sensors, kurtosis)
# plt.xticks(rotation=90)
# plt.savefig(output_path + "kurt.png")
# plt.clf()
# plt.close()

# Remove outliers
# for i in range(13):
#     if os.path.exists(output_path + "rm_out") == False:
#         os.makedirs(output_path + "rm_out")
#     ave = np.mean(df[sensors[i]])
#     sd = np.std(df[sensors[i]])
#     outlier_min = ave - sd*2
#     outlier_max = ave + sd*2
#     dfi = df[sensors[i]]
#     plt.figure()
#     plt.hist(dfi[(dfi>outlier_min) & (dfi<outlier_max)], bins=100)
#     plt.title("sensor" + str(i))
#     plt.savefig(output_path + "rm_out/sensor" + str(i) + ".png")
#     plt.clf()
#     plt.close()

# Mutual information
# y = df["state"]
# x = df.drop(["sequence", "subject", "state", "step"], axis=1)
# scores = mutual_info_classif(x, y, random_state=0)
# plt.figure()
# plt.bar(sensors, scores)
# plt.xticks(rotation=90)
# plt.savefig(output_path + "MIScore.png")
# plt.clf()
# plt.close()

# Statistics features
# for i in range(13):
#     df[sensors[i] + "_max"] = df.groupby("sequence")[sensors[i]].transform("max")
#     df[sensors[i] + "_min"] = df.groupby("sequence")[sensors[i]].transform("min")
#     df[sensors[i] + "_mean"] = df.groupby("sequence")[sensors[i]].transform("mean")
#     df[sensors[i] + "_std"] = df.groupby("sequence")[sensors[i]].transform("std")
#     df[sensors[i] + "_median"] = df.groupby("sequence")[sensors[i]].transform("median")
#     df[sensors[i] + "_flip"] = df[sensors[i]]*-1

# for i in range(13):
#     kurtosis = []
#     for j in range(df["sequence"].nunique()):
#         kurt = df[df.sequence==j][sensors[i]].kurt()
#         for k in range(60):
#             kurtosis.append(kurt)
#     df[sensors[i] + "_kurtosis"] = kurtosis

# print(df)
# print(df[sensors[12]])
# df.to_csv(output_path+"statistics_features.csv", index=False, header=True)


# Changed df
# dfc = pd.read_csv(output_path + "statistics_features.csv", dtype=np.float32)
# state = []
# length = int(dfc.shape[0]/60)
# print("read csv")

# # Create df for statistics features
# for i in range(length):
#     state.append(dfc.at[60*i, "state"])
# stats_df = pd.DataFrame({"state": state})

# # Get colmun names
# max_cols = [col for col in dfc.columns if "_max" in col]
# print(max_cols)
# min_cols = [col for col in dfc.columns if "_min" in col]
# print(min_cols)
# mean_cols = [col for col in dfc.columns if "_mean" in col]
# print(mean_cols)
# std_cols = [col for col in dfc.columns if "_std" in col]
# print(std_cols)
# median_cols = [col for col in dfc.columns if "_median" in col]
# print(median_cols)
# flip_cols = [col for col in dfc.columns if "_flip" in col]
# print(flip_cols)
# kurtosis_cols = [col for col in dfc.columns if "_kurtosis" in col]
# print(kurtosis_cols)

# # Insert each columns into stats_df
# # Max
# for col in max_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Min
# for col in min_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Mean
# for col in mean_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Std
# for col in std_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Median
# for col in median_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Flip
# for col in flip_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values
# # Kurtosis
# for col in kurtosis_cols:
#     values=[]
#     for i in range(length):
#         values.append(dfc.at[60*i, col])
#     stats_df[col] = values

# # Make df to csv
# stats_df.to_csv(output_path+"stats_df.csv", index=False, header=True)
# print("Finish creating statistics features csv.")

# Read statistics features.
# stats_df = pd.read_csv(output_path + "stats_df.csv", dtype=np.float32)

# Get colmun names
# max_cols = [col for col in stats_df.columns if "_max" in col]
# min_cols = [col for col in stats_df.columns if "_min" in col]
# mean_cols = [col for col in stats_df.columns if "_mean" in col]
# std_cols = [col for col in stats_df.columns if "_std" in col]
# median_cols = [col for col in stats_df.columns if "_median" in col]
# flip_cols = [col for col in stats_df.columns if "_flip" in col]
# kurtosis_cols = [col for col in stats_df.columns if "_kurtosis" in col]

# if os.path.exists(output_path + "stats") == False:
#     os.makedirs(output_path + "stats")

# Max
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in max_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/max.png")
# plt.clf()
# plt.close()

# Min
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in min_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/min.png")
# plt.clf()
# plt.close()

# Mean
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in mean_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/mean.png")
# plt.clf()
# plt.close()

# Std
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in std_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/std.png")
# plt.clf()
# plt.close()

# Median
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in median_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/median.png")
# plt.clf()
# plt.close()

# Flip
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in flip_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/flip.png")
# plt.clf()
# plt.close()

# Kurtosis
# fig, axes = plt.subplots(4, 4, sharex="all", sharey="all", figsize=(20, 25))
# r = 0
# c = 0
# for col in kurtosis_cols:
#     temp = pd.DataFrame({col: stats_df[col].values, "state": stats_df["state"]})
#     temp = temp.sort_values(col)
#     temp.reset_index(inplace=True)
#     axes[r, c].scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
#     axes[r, c].set_title(col)
#     if c==3:
#         c=0
#         r+=1
#     else:
#         c+=1
# plt.savefig(output_path + "stats/kurtosis.png")
# plt.clf()
# plt.close()


# Feature selection
# sensor_04_mean, sensor_02_std, sensor_04_kurtosis, sensor_10_kurtosis seems to be useful.
# Read statistics features.
stats_df = pd.read_csv(output_path + "stats_df.csv", dtype=np.float32)

# Get colmun names
max_cols = [col for col in stats_df.columns if "_max" in col]
min_cols = [col for col in stats_df.columns if "_min" in col]
mean_cols = [col for col in stats_df.columns if "_mean" in col]
mean_cols.remove("sensor_04_mean")
std_cols = [col for col in stats_df.columns if "_std" in col]
std_cols.remove("sensor_02_std")
median_cols = [col for col in stats_df.columns if "_median" in col]
flip_cols = [col for col in stats_df.columns if "_flip" in col]
kurtosis_cols = [col for col in stats_df.columns if "_kurtosis" in col]
kurtosis_cols.remove("sensor_04_kurtosis")
kurtosis_cols.remove("sensor_10_kurtosis")

# Drop useless information
stats_df = stats_df.drop(max_cols, axis=1)
stats_df = stats_df.drop(min_cols, axis=1)
stats_df = stats_df.drop(mean_cols, axis=1)
stats_df = stats_df.drop(std_cols, axis=1)
stats_df = stats_df.drop(median_cols, axis=1)
stats_df = stats_df.drop(flip_cols, axis=1)
stats_df = stats_df.drop(kurtosis_cols, axis=1)

# Make useful data csv
stats_df.to_csv(output_path+"useful_stats_df.csv", index=False, header=True)