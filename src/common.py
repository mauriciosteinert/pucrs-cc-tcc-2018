#!/usr/bin/python3

# This file implements common tasks in summarization process, like:
#
# -- Data set preprocessing, in our case, CNN and Dailymail data set.
# -- Generate word vectors and sentence vector representation of text.
# -- Generate mini batches for neural network learning process

import sys
import argparse
import sent2vec

class Common:


    def hello(self):
        print("Hello folks!")


    def load_word_vec_model(self):
        print("Loading word vector model ...")
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(self.config.word_vector_model)
        print("Word vector loaded!")


    # Argument parse to be used in all tests and models
    def parse_cmd_args(self):
        arg_parser = argparse.ArgumentParser(description='cmdline parameters')
        #  General parameters
        arg_parser.add_argument('--session-name',
                                metavar='session_name',
                                nargs='?',
                                help='Identifier for a specific run.')
        arg_parser.add_argument('--dataset-dir',
                                metavar='dataset_dir',
                                nargs='?',
                                help='Dataset directory to use.')
        arg_parser.add_argument('--working-dir',
                                metavar='working_dir',
                                nargs='?',
                                help='Working directory to save results.')
        arg_parser.add_argument('--word-vector-model',
                                metavar='word_vector_model',
                                nargs='?',
                                help='Word vector model to be used to convert word and sentences.')
        arg_parser.add_argument('--process-n-examples',
                                metavar='process_n_examples',
                                nargs='?',
                                help='Number of examples of the dataset directory to use.')

        # Neural Network parameters
        arg_parser.add_argument('--learning-rate',
                                metavar='nn_learning_rate',
                                nargs='?',
                                help='Learning rate for NN training')
        arg_parser.add_argument('--batch-size',
                                metavar='nn_batch_size',
                                nargs='?',
                                help='Batch size for each learning iteration.')

        self.config = arg_parser.parse_args()
