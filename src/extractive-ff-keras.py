#!/usr/bin/python3

import sys

sys.path.append("../lib/python")
from common import *
import numpy as np
import tensorflow as tf
from tensorflow import keras

common = Common()
common.parse_cmd_args()
common.prepare_batch()
display_step = 1
device = "/device:GPU:0"
random_seed = 1

common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")

all_batch_run_train, x_train, y_train = common.get_next_batch("training")

y_train_p = []

for y_t in y_train:
    y_train_p = np.append(y_train_p, np.argmax(y_t))

print(x_train.shape, y_train.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1],)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(106, activation=tf.nn.softmax)
])

model.compile(  loss='sparse_categorical_crossentropy',
                optimizer=tf.train.AdamOptimizer(),
                metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train_p, epochs=int(common.config.num_epochs))

all_batch_run_test, x_test, y_test = common.get_next_batch("test")

y_test_p = []

for y_t in y_test:
    y_test_p = np.append(y_test_p, np.argmax(y_t))


scores = model.evaluate(x_test, y_test_p, verbose=0)
print("Test Accuracy:", scores)
