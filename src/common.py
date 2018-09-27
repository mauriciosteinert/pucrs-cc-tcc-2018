#!/usr/bin/python3

# This file implements common tasks in summarization process, like:
#
# -- Data set preprocessing, in our case, CNN and Dailymail data set.
# -- Generate word vectors and sentence vector representation of text.
# -- Generate mini batches for neural network learning process


import argparse
import sent2vec
import re
import os
import sys


class Common:
    def load_word_vec_model(self):
        if self.config.word_vector_model == None:
            print("No word vector model provided! Aborting ...")
            exit()

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
                                type=int,
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


    # Return a list of files to be processed
    def get_dataset_files(self):
        files = os.listdir(self.config.dataset_dir)
        files.sort()

        if self.config.process_n_examples != None:
            files = files[:self.config.process_n_examples]
        return files


    # Return a list of each sentence in text
    def text_to_sentences(self, text):
        text = text.replace("\xa0", " ")
        sentences = re.split(r'[\.\,\n]', text)

        exclusion = ['', '@highlight']
        # Delete empty entries
        sentences = [sentence for sentence in sentences if sentence not in exclusion]
        return sentences
