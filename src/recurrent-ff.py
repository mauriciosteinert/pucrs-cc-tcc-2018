#!/usr/bin/python3

import sys
sys.path.append("../lib/python")

import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
from common import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

common = Common()
common.parse_cmd_args()

np.random.seed(1)

# dataset_train = np.load("../datasets/npz2/preprocess-data2-100-000000.npz")
# dataset_test = np.load("../datasets/npz2/preprocess-data2-100-000001.npz")

dataset_train = np.load("../datasets/npz2/preprocess-data2-1000-000000.npz")
dataset_test = np.load("../datasets/npz2/preprocess-data2-1000-000001.npz")


# Get longest text
longest_text = 45

# Apply padding
padding = np.zeros(int(common.config.word_vector_dim),)

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
        sentences_rouge.append(sentence[1][0])
        sentences_rouge.append(sentence[2][0])

    while len(sentences_rouge) < longest_text * 3:
        sentences_rouge.append(0)
        sentences_rouge.append(0)
        sentences_rouge.append(0)
    Y_res.append(sentences_rouge)

X_train = np.array(X_res)
Y_train = np.array(Y_res)

Y_train = Y_train.reshape(Y_train.shape[0], longest_text, 3)

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
        sentences_rouge.append(sentence[1][0])
        sentences_rouge.append(sentence[2][0])

    while len(sentences_rouge) < longest_text * 3:
        sentences_rouge.append(0)
        sentences_rouge.append(0)
        sentences_rouge.append(0)
    Y_res.append(sentences_rouge)


X_test = np.array(X_res)
Y_test = np.array(Y_res)

Y_test = Y_test.reshape(Y_test.shape[0], longest_text, 3)

# print(X_test.shape, Y_test.shape)


model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=8,
            activation=tf.nn.relu,
            return_sequences=True),
            input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dense(3)
    # keras.layers.TimeDistributed(keras.layers.Dense(3))
])

model.compile(  loss='mean_squared_error',
                optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train,
            epochs=int(common.config.num_epochs),
            batch_size=100,
            verbose=2,
            validation_data=(X_test, Y_test))


# Prediction and ROUGE scores for
y_new = model.predict(X_test)


rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

best_rouge_1_list = []
best_rouge_2_list = []
best_rouge_l_list = []


for y_hat, y, file in zip(y_new, dataset_test['y_rouge'], dataset_test['files']):
    y_best_rouge_idx = np.argmax(y, axis=0)[0][0]
    y_hat_best_rouge_idx = np.argmax(y_hat, axis=0)[0]

    try:
        rouge_1_list.append(y[y_hat_best_rouge_idx][0][0])
    except IndexError:
        rouge_1_list.append(0)

    try:
        rouge_2_list.append(y[y_hat_best_rouge_idx][1][0])
    except IndexError:
        rouge_2_list.append(0)

    try:
        rouge_l_list.append(y[y_hat_best_rouge_idx][2][0])
    except IndexError:
        rouge_l_list.append(0)

    best_rouge_1_list.append(y[y_best_rouge_idx][0][0])
    best_rouge_2_list.append(y[y_best_rouge_idx][1][0])
    best_rouge_l_list.append(y[y_best_rouge_idx][2][0])

print("ROUGE-1 MEAN: ", np.mean(np.array(rouge_1_list)))
print("ROUGE-2 MEAN: ", np.mean(np.array(rouge_2_list)))
print("ROUGE-L MEAN: ", np.mean(np.array(rouge_l_list)))

print("ROUGE-1 STDDEV: ", np.std(np.array(rouge_1_list)))
print("ROUGE-2 STDDEV: ", np.std(np.array(rouge_2_list)))
print("ROUGE-L STDDEV: ", np.std(np.array(rouge_l_list)))


print("BEST ROUGE SCORES:")
print("ROUGE-1 MEAN: ", np.mean(np.array(best_rouge_1_list)))
print("ROUGE-2 MEAN: ", np.mean(np.array(best_rouge_2_list)))
print("ROUGE-L MEAN: ", np.mean(np.array(best_rouge_l_list)))

print("ROUGE-1 STDDEV: ", np.std(np.array(best_rouge_1_list)))
print("ROUGE-2 STDDEV: ", np.std(np.array(best_rouge_2_list)))
print("ROUGE-L STDDEV: ", np.std(np.array(best_rouge_l_list)))
