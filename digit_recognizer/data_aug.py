import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

# output test
output_path = "../../output/"

try:
    with open(output_path+"test.txt", mode="x") as f:
        f.write("test")
        f.close()
except FileExistsError:
    pass

# Load the data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

y_train = train["label"]

# Drop label column
train = train.drop(labels = ["label"], axis=1)

# Check the data
# print(y_train.value_counts())
# print(train.isnull().any().describe())

# Normalize the data
train = train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)
train = train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Label encoding
y_train = to_categorical(y_train, num_classes=10)

# Split training and validation set
random_seed = 2
train, val, y_train, y_val = train_test_split(train, y_train, test_size=0.1, random_state=random_seed)

# Show example
plt.imsave("../../output/example.png", train[0][:,:,0])

# Data augmentation
def display_aug(datagen, img):
    n=0
    nrows=2
    ncols=3
    row=0
    col=0
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for data in datagen.flow(img, batch_size=1, seed=0):
        show_img=array_to_img(data[0])
        print(show_img)
        ax[row, col].imshow(show_img)
        col+=1
        if col==ncols:
            row+=1
            col=0
        if row==nrows:
            fig.savefig(output_path+"6aug_examples.png")
            break

datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.3,
        shear_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
        )
img = train[0]
img = img[np.newaxis, :, :, :]
display_aug(datagen, img)
