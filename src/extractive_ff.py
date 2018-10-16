#!/usr/bin/python3
#
# An extractive text summarization based on feed-forward neural network

import sys

sys.path.append("../lib/python")
from common import *
import numpy as np
import tensorflow as tf

common = Common()
common.parse_cmd_args()
common.prepare_batch()
display_step = 1
device = "/device:GPU:0"
random_seed = 1

common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")

all_batch_run_test, x_input_test, y_label_test = common.get_next_batch("test")

n_hidden_1 = x_input_test.shape[0]
n_hidden_2 = 512

with tf.device(device):
    x_ = tf.placeholder(tf.float32, shape=[None, x_input_test.shape[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, y_label_test.shape[1]])

    weights = {
        'h1': tf.Variable(tf.random_normal([x_input_test.shape[1], n_hidden_1], seed=random_seed)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed=random_seed)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, y_label_test.shape[1]], seed=random_seed))
    }

    biases = {
        'b1': tf.Variable(np.zeros([n_hidden_1], np.float32)),
        'b2': tf.Variable(np.zeros([n_hidden_2], np.float32)),
        'out': tf.Variable(np.zeros(y_label_test.shape[1], np.float32))
    }

    layer_1 = tf.add(tf.matmul(x_, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=out_layer, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=float(common.config.learning_rate))
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./checkpoint_2.ckpt")
        loss = 0
        acc = 0
        loss_test = 0
        acc_test = 0

        for curr_epoch in range(1, int(common.config.num_epochs) + 1):
            #  Implement mini batch passes
            all_batch_run, x_input_batch, y_label_batch = common.get_next_batch("training")
            batch_count = 0
            scores_train = []

            while all_batch_run == 0:
                common.log_message("INFO", "Running batch training " + str(batch_count))
                sess.run(train_op,
                    feed_dict={x_: x_input_batch, y_: y_label_batch})
                all_batch_run, x_input_batch, y_label_batch = common.get_next_batch("training")

                loss, acc = sess.run([loss_op, accuracy],
                    feed_dict={x_: x_input_batch, y_: y_label_batch})
                scores_train.append([loss, acc])

                batch_count += 1

            if curr_epoch % display_step == 0:


                scores_test = []
                all_batch_test = 0
                while all_batch_test == 0:
                    loss_test,acc_test = sess.run([loss_op, accuracy],
                        feed_dict={x_: x_input_test, y_: y_label_test})

                    all_batch_test, x_test, y_test = common.get_next_batch("test")
                    scores_test.append([loss_test, acc_test])

                scores_test_mean = np.mean(np.array(scores_test), axis=0)
                scores_train_mean = np.mean(np.array(scores_train), axis=0)

                common.log_message("INFO", "[" + str(curr_epoch) + "] loss = " \
                        + str(scores_train_mean[0]) + "\tacc = " + str(scores_test_mean[1])
                        + "\ttest loss = " + str(scores_test_mean[0])
                        + "\ttest acc = " + str(scores_test_mean[1]))
                save_path = saver.save(sess, "./checkpoint_" + str(curr_epoch) + ".ckpt")
