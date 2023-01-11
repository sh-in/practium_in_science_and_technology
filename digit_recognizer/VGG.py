import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
print(y_train.value_counts())
print(train.isnull().any().describe())

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

# Define model
model = Sequential([
    # Block One
    Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2, padding="same"),

    # Block Two
    Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2, padding="same"),

    # Block Three
    Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2, padding="same"),

    # Block Four
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2, padding="same"),

    # Block Five
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides=2, padding="same"),

    # Head
    Flatten(),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Complie the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
batch_size=64

# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        zoom_range=0.1,
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

# Show the training and validation curves
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history["loss"], color="b", label="Training loss")
ax[0].plot(history.history["val_loss"], color="r", label="Validation loss", axes=ax[0])
legend = ax[0].legend(loc="best", shadow=True)

ax[1].plot(history.history["accuracy"], color="b", label="Training accuracy")
ax[1].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")
legend = ax[1].legend(loc="best", shadow=True)
fig.savefig(output_path+"training_validation_curves.png")

# Predict results
preds = model.predict(test)
preds = np.argmax(preds, axis=1)
preds = pd.Series(preds, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), preds], axis=1)
submission.to_csv(output_path+"cnn_keras_submission.csv", index=False)

# Checking error
# Predict the values from the validation set
y_pred = model.predict(val)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val, axis=1)
# Get errors
errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_val_errors = val[errors]

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n=0
    nrows=2
    ncols=3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            plt.imsave(output_path+"error_example.png", (img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n+=1
    fig.savefig(output_path+"top6_errors.png")


y_pred_errors_prob = np.max(y_pred_errors, axis=1)
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)
