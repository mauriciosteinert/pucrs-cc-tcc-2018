#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
from common import *

common = Common()
common.parse_cmd_args()

np.random.seed(1)

dataset = np.load("../datasets/npz_rouge/preprocess2.npz")


# Get longest text
longest_text = 0

for entry in dataset['x']:
    if len(entry) > longest_text:
        longest_text = len(entry)

# Apply padding
padding = np.zeros(100,)

X_res = []
Y_res = []

for entry_x, entry_y in zip(dataset['x'], dataset['y']):
    while len(entry_x) < longest_text:
        entry_x = np.vstack((entry_x, padding))
        entry_y = np.append(entry_y, 0)
    X_res.append(entry_x)
    Y_res.append(entry_y)

X_train = np.array(X_res[:90])
X_test = np.array(X_res[90:])
Y_train = np.array(Y_res[:90])
Y_test = np.array(Y_res[90:])

Y_train = Y_train.reshape(Y_train.shape[0], longest_text, 1)
Y_test = Y_test.reshape(Y_test.shape[0], longest_text, 1)


model = keras.Sequential([
    keras.layers.LSTM(units=10,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            activation=tf.nn.relu,
            return_sequences=True),
    keras.layers.Dense(1, activation='softmax')
])

model.compile(  loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])


print(model.summary())
model.fit(X_train, Y_train,
            epochs=5,
            validation_data=(X_test, Y_test))
