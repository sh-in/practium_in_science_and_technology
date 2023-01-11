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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# output test
output_path = "../../output/"

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
# plt.imsave("../../output/example.png", train[0][:,:,0])

# Define model
model = Sequential([
    # Block One
    Conv2D(filters=32, kernel_size=5, padding="same", activation="relu", input_shape=(28, 28, 1)),
    MaxPool2D(),

    # Block Two
    Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
    MaxPool2D(),

    # Blow Three
    Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
    MaxPool2D(),

    # Head
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Complie the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 20 
batch_size=86

# Test data augmentation
aug = np.arange(0, 0.25, 0.05)
print(aug.shape)

# fig, ax = plt.subplots(5, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# fig.suptitle("Loss(left), Accuracy(right)\ntraining(blue), validation(red)")
# row = 0
# col = 0
n = 1
results = []
for i in aug:
    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=i,
        height_shift_range=i,
        shear_range=0,
        zoom_range=i,
        horizontal_flip=False,
        vertical_flip=False
    )

    datagen.fit(train)

    # Fit the model
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001)
    history = model.fit_generator(
        datagen.flow(train, y_train, batch_size=batch_size),
        validation_data=(val, y_val),
        epochs=epochs,
        steps_per_epoch=train.shape[0]//batch_size,
        verbose=2,
        callbacks=[reduce_lr]
    )

    results.append(model.evaluate(val, y_val))
    # using subplot
    # print(row)
    # print(col)
    # Show the training and validation curves
    # ax[row, col].plot(history.history["loss"], color="b", label="Training loss")
    # ax[row, col].plot(history.history["val_loss"], color="r", label="Validation loss", axes=ax[row, col])
    # col+=1

    # ax[row, col].plot(history.history["accuracy"], color="b", label="Training accuracy")
    # ax[row, col].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")

    # row+=1
    # col=0
    # if row==3:
        # fig.savefig(output_path+"training_validation_curves.png")

    # plot one by one
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax[0].plot(history.history["loss"], color="b", label="Training loss")
    ax[0].plot(history.history["val_loss"], color="r", label="Validation loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")

    ax[1].plot(history.history["accuracy"], color="b", label="Training accuracy")
    ax[1].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    fig.suptitle("Augmentation range"+i.astype("str"))
    fig.savefig(output_path+"aug_test"+str(n)+".png")
    n+=1

i=0
for result in results:
    print("rotation: %d, test loss: %f, test accuracy: %f"% (aug[i], result[0], result[1]))
    i+=1

# Predict results
preds = model.predict(test)
preds = np.argmax(preds, axis=1)
preds = pd.Series(preds, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), preds], axis=1)
submission.to_csv(output_path+"cnn_keras_submission.csv", index=False)
