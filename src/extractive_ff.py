#!/usr/bin/python3
#
# An extractive text summarization based on feed-forward neural network

import sys

sys.path.append("../")
from common import *
import numpy as np
import tensorflow as tf

common = Common()
common.parse_cmd_args()
common.prepare_batch()
display_step = 1
device = "cpu:0"
random_seed = 1

common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")

common.get_test()

n_hidden_1 = common.test_x.shape[0]
n_hidden_2 = 1000

with tf.device(device):
    x_ = tf.placeholder(tf.float32, shape=[None, common.test_x.shape[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, common.test_y.shape[1]])

    weights = {
        'h1': tf.Variable(tf.random_normal([common.test_x.shape[1], n_hidden_1], seed=random_seed)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed=random_seed)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, common.test_y.shape[1]], seed=random_seed))
    }

    biases = {
        'b1': tf.Variable(np.zeros([n_hidden_1], np.float32)),
        'b2': tf.Variable(np.zeros([n_hidden_2], np.float32)),
        'out': tf.Variable(np.zeros(common.test_y.shape[1], np.float32))
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
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./checkpoint_2.ckpt")
        acc = 0

        for curr_epoch in range(1, int(common.config.num_epochs) + 1):
            #  Implement mini batch passes
            all_batch_run, x_input_batch, y_label_batch = common.get_next_train_batch()

            while all_batch_run == 0:
                sess.run(train_op,
                    feed_dict={x_: x_input_batch,
                    y_: y_label_batch})
                all_batch_run, x_input_batch, y_label_batch = common.get_next_train_batch()

            if curr_epoch % display_step == 0:
                loss, acc = sess.run([loss_op, accuracy],
                    feed_dict={x_: x_input_batch, y_: y_label_batch})
                loss_test,acc_test = sess.run([loss_op, accuracy],
                    feed_dict={x_: common.test_x, y_: common.test_y})

                common.log_message("INFO", "[" + str(curr_epoch) + "] loss = "
                        + str(loss) + "\tacc = " + str(acc)
                        + "\ttest loss = " + str(loss_test)
                        + "\ttest acc = " + str(acc_test))
                # print("[" + str(curr_epoch) + "] loss = "
                #         + str(loss) + "\tacc = " + str(acc)
                #         + "\ttest loss = " + str(loss_test)
                #         + "\ttest acc = " + str(acc_test))
                # save_path = saver.save(sess, "./checkpoint_" + str(curr_step) + ".ckpt")
