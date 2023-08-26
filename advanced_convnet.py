import keras
import tensorflow as tf
import numpy as np
from keras import layers
from keras.utils import image_dataset_from_directory
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt

train_path = "dataset/dogs-vs-cat/train"
valid_path = "dataset/dogs-vs-cat/valid"
test_path = "dataset/dogs-vs-cat/test"
batch_size = 8

# load datasets
train_ds = image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1,
    label_mode="binary",
)
val_ds = image_dataset_from_directory(
    valid_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1,
    label_mode="binary",
)
test_ds = image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1,
    label_mode="binary",
)

# augmentation and proprocessing layer
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)
proprocessing = keras.Sequential([layers.Rescaling(scale=1.0 / 255.0)])


def block(x, filter_size):
    residual = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)  # activation after normalization
    x = layers.SeparableConv2D(
        filters=filter_size,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(x)
    # no need for bias since batch normalization

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)  # activation after normalization
    x = layers.SeparableConv2D(
        filters=filter_size,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(x)
    # no need for bias since batch normalization

    x = layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same")(x)
    # reshape residual to match the output of maxpooling
    residual = layers.Conv2D(
        filters=filter_size,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="same",
        use_bias=False,
    )(residual)

    # add residual and x
    x = layers.add([x, residual])

    return x


inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = proprocessing(x)

# use traditional conv first
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)
# then use block
for size in [32, 64, 128, 256, 512]:
    x = block(x, filter_size=size)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(units=1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# print(model.summary())

# compile and fit
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_ds, epochs=50, validation_data=val_ds, verbose=1)

# plot result
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# test
test_loss, test_acc = model.evaluate(test_ds)
print("Test:")
print(f"Test accuracy = {test_acc: .4f} - test loss = {test_loss: 4f}")
