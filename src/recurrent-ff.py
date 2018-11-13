#!/usr/bin/python3

import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
from common import *

common = Common()
common.parse_cmd_args()

np.random.seed(1)

dataset_train = np.load("../datasets/npz2/preprocess-data2-20000-000000.npz")
dataset_test = np.load("../datasets/npz2/preprocess-data2-20000-000001.npz")

# Get longest text
longest_text = 45

# Apply padding
padding = np.zeros(100,)

X_res = []
Y_res = []

# for entry_x, entry_y in zip(dataset_train['x'], dataset_train['y_rouge']):
#     while len(entry_x) < longest_text:
#         entry_x = np.vstack((entry_x, padding))
#         entry_y = np.append(entry_y, 0)
#     X_res.append(entry_x)
#     Y_res.append(entry_y)

for entry_x in dataset_train['x']:
    while len(entry_x) < longest_text:
        entry_x = np.vstack((entry_x, padding))
    X_res.append(entry_x)

for entry_y in dataset_train['y_rouge']:
    sentences_rouge = []
    for sentence in entry_y:
        sentences_rouge.append(sentence[0][0])

    while len(sentences_rouge) < longest_text:
        sentences_rouge.append(0)
    Y_res.append(sentences_rouge)

X_train = np.array(X_res)
Y_train = np.array(Y_res)

print(X_train.shape, Y_train.shape)



X_res = []
Y_res = []

for entry_x in dataset_test['x']:
    while len(entry_x) < longest_text:
        entry_x = np.vstack((entry_x, padding))
    X_res.append(entry_x)

for entry_y in dataset_test['y_rouge']:
    sentences_rouge = []
    for sentence in entry_y:
        sentences_rouge.append(sentence[0][0])

    while len(sentences_rouge) < longest_text:
        sentences_rouge.append(0)
    Y_res.append(sentences_rouge)


X_test = np.array(X_res)
Y_test = np.array(Y_res)


# print(X_test.shape, Y_test.shape)


model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=32,
            activation=tf.nn.relu,
            return_sequences=False),
            input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dense(45)
])

model.compile(  loss='mean_squared_error',
                optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train,
            epochs=1000,
            batch_size=200,
            validation_data=(X_test, Y_test))
