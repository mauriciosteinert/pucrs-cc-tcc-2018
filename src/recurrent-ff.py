#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np


np.random.seed(1)

dataset = np.load("../datasets/npz_rouge/preprocess2.npz")
X_train = dataset['x'][:200]
X_test = dataset['x'][200:]
Y_train = dataset['y'][:200]
Y_test = dataset['y'][200:]

print("X train shape", X_train.shape)
print("Y train shape", Y_train.shape)

print("X test shape", X_test.shape)
print("Y test shape", Y_test.shape)


model = keras.Sequential([
    keras.layers.Embedding(X_train.shape[1], 1),
    keras.layers.LSTM(1),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(  loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


print(model.summary())
model.fit(X_train, Y_train,
            epochs=5,
            validation_data=(X_test, Y_test))
