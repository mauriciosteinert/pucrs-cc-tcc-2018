#!/usr/bin/python3

import sys

sys.path.append("../lib/python")
from common import *
import numpy as np
import tensorflow as tf
from tensorflow import keras

common = Common()
common.parse_cmd_args()
# common.prepare_batch()
display_step = 1
device = "/device:GPU:0"
random_seed = 1

np.random.seed(random_seed)
import os
os.environ['PYTHONHASHSEED'] = str(random_seed)
tf.set_random_seed(random_seed)

common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")

# all_batch_run_train, x_train, y_train = common.get_next_batch("training")
# all_batch_run_test, x_test, y_test = common.get_next_batch("test")

# file_train = np.load("../npz-classification-10000/preprocess-data-10000-000000.npz")
# file_test = np.load("../npz-classification-10000/preprocess-data-10000-000001.npz")

file_train = np.load("../datasets/npz2/preprocess-data2-100-000000.npz")
file_test = np.load("../datasets/npz2/preprocess-data2-100-000001.npz")

x_train_1 = file_train['x']
y_train_1 = file_train['y_idx']
x_test_1 = file_test['x']
y_test_1 = file_test['y_idx']

y_test_rouge = file_test['y_rouge']


padding = np.zeros((1, int(common.config.word_vector_dim)))

max_sentences = 45

x_train = []
y_train = []

for entry_x, entry_y in zip(x_train_1, y_train_1):
    while entry_x.shape[0] < max_sentences:
        entry_x = np.vstack((entry_x, padding))

    entry_y_l = np.zeros((max_sentences,))
    entry_y_l[entry_y] = 1

    x_train.append(entry_x.reshape((1, -1))[0])
    y_train.append(entry_y_l.reshape((1, -1))[0])

x_test = []
y_test = []

for entry_x, entry_y in zip(x_test_1, y_test_1):
    while entry_x.shape[0] < max_sentences:
        entry_x = np.vstack((entry_x, padding))

    entry_y_l = np.zeros((max_sentences,))
    entry_y_l[entry_y] = 1

    x_test.append(entry_x.reshape((1, -1))[0])
    y_test.append(entry_y_l.reshape((1, -1))[0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


network = 8

if network == 1:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(8, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 2:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(16, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 3:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(8, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 4:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(16, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 5:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 6:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 7:
    x_train = x_train.reshape(x_train.shape[0], 45, 100)
    x_test = x_test.reshape(y_train.shape[0], 45, 100)

    model = keras.Sequential([
        keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
        keras.layers.MaxPooling1D(pool_size=3),

        keras.layers.Flatten(),
        keras.layers.Dense(8, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(8, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(8, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),
        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])
elif network == 8:
    x_train = x_train.reshape(x_train.shape[0], 45, 100)
    x_test = x_test.reshape(y_train.shape[0], 45, 100)

    model = keras.Sequential([
        keras.layers.Conv1D(filters=2, kernel_size=3,
                            activation='relu',
                            input_shape=(x_train.shape[1], x_train.shape[2]),
                            use_bias=True),
        keras.layers.MaxPooling1D(pool_size=1),
        keras.layers.Conv1D(filters=2, kernel_size=3,
                            activation='relu',
                            use_bias=True),

        keras.layers.MaxPooling1D(pool_size=1),

        keras.layers.Flatten(),
        keras.layers.Dense(4, activation=tf.nn.relu,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'),

        keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
    ])




print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# For convolutional network, uncomment this block
# x_train = x_train.reshape(x_train.shape[0], 45, 100)
# x_test = x_test.reshape(y_train.shape[0], 45, 100)
#
#
# model = keras.Sequential([
#     keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
#     keras.layers.MaxPooling1D(pool_size=3),
#
#     keras.layers.Flatten(),
#     keras.layers.Dense(4, activation=tf.nn.relu,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'),
#
#     keras.layers.Dense(4, activation=tf.nn.relu,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'),
#
#     keras.layers.Dense(4, activation=tf.nn.relu,
#                 kernel_initializer='random_uniform',
#                 bias_initializer='zeros'),
#     keras.layers.Dense(max_sentences, activation=tf.nn.softmax)
# ])



model.compile(  loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.summary()


y_train_p = []

for y_t in y_train:
    y_train_p = np.append(y_train_p, np.argmax(y_t))


y_test_p = []

for y_t in y_test:
    y_test_p = np.append(y_test_p, np.argmax(y_t))


model.fit(x_train, y_train_p,
        batch_size=int(common.config.batch_size),
        epochs=int(common.config.num_epochs),
        validation_data=(x_test, y_test_p),
        verbose=2)

# Prediction and ROUGE scores for
y_new = model.predict_classes(x_test)

rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

log_test = []

for file_name, y_t, y_p, y_rouge in zip(file_test['files'], y_test_1, y_new, y_test_rouge):
    try:
        log_test.append([file_name, y_t, y_rouge[y_t], y_p, y_rouge[y_p]])
    except IndexError:
        log_test.append([file_name, y_t, 0, y_p, 0])

with open(common.config.working_dir + "/" + common.config.session_name + "-results", 'w') as f:
    for entry in log_test:
        f.write("%s\n" % entry)

for y_rouge, y_t, y_p in zip(y_test_rouge, y_test_p, y_new):
    try:
        rouge_1_list.append(y_rouge[0][y_p][0])
        rouge_2_list.append(y_rouge[1][y_p][0])
        rouge_l_list.append(y_rouge[2][y_p][0])
    except IndexError:
        rouge_1_list.append(0)
        rouge_2_list.append(0)
        rouge_l_list.append(0)

print("ROUGE-1 MEAN: ", np.mean(np.array(rouge_1_list)))
print("ROUGE-2 MEAN: ", np.mean(np.array(rouge_2_list)))
print("ROUGE-L MEAN: ", np.mean(np.array(rouge_l_list)))

print("ROUGE-1 STDDEV: ", np.std(np.array(rouge_1_list)))
print("ROUGE-2 STDDEV: ", np.std(np.array(rouge_2_list)))
print("ROUGE-L STDDEV: ", np.std(np.array(rouge_l_list)))
