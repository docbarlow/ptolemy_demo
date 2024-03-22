# For math
import numpy as np

# For data frames
import pandas as pd

# For generating random numbers (already bundled with Python language)
import random

# Our AI package
import tensorflow as tf

import os

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# see if we have multiple gpus
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# this is convenient - exit the script immediately if it isn't using tensor flow
# built with cuda
if not tf.test.is_built_with_cuda():
    print("Not built with cuda")
    exit(1)

# set path to mnist file
file_path = os.getenv('DATA_FILE_PATH') + "/mnist.npz"

# load the mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data(file_path)

# if this script had internet access in the hpc environment
# one could just load mnist from keras
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape to set up grayscale color channel for each 28x28 image
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

# normalize color between 0 and 1
x_train = x_train.astype ('float32') / 255.0

# create label for it
y_train = to_categorical (y_train)

# setup the model

model = Sequential([
    # Input shape is 28x28x1 like our images
    # There will be 32 convolutions (little parts of the image) that are 3x3
    # The activation function will be RELU; we talked about it in class
    Conv2D(32, (3, 3), activation='relu', strides=1, input_shape=(28, 28,1)),
    # The next layer should pool together all the stuff we learned in the previous
    # layer - (TODO: explain pooling layers and why they are helpful)
    MaxPooling2D((2, 2)),
    # Get a flat tensor - a continuous list of values
    Flatten(),
    # TODO: explain this one
    Dense(100, activation='relu'),
    # This is our last or "output layer". Notice it has 10 values like our labels (0-9).
    # Softmax gives us a probability distribution so we can see which digit is most likely
    # 
    Dense(10, activation='softmax')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)

# compile the model

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[ 'accuracy' ]
)

# fit the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32)

# serialize model to JSON
model_json = model.to_json()

with open("model1.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model1.h5")

print("Saved model to disk")

# Reshape all the test data
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_test = x_test.astype ('float32') / 255.0
y_test = to_categorical(y_test)

# Now let's test the model
score = model.evaluate(x_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



