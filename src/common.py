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
import time
import nltk
import numpy as np

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
                                help='Identifier for a specific run.')

        arg_parser.add_argument('--dataset-dir',
                                metavar='dataset_dir',
                                help='Dataset directory to use.')

        arg_parser.add_argument('--dataset-file-list',
                                metavar='dataset_file_list',
                                help='Load only files specified in informed file.')

        arg_parser.add_argument('--dataset-exclusion-list',
                                metavar='dataset_exclusion_list',
                                help='Files in dataset that must not be processed.')

        arg_parser.add_argument('--working-dir',
                                metavar='working_dir',
                                help='Working directory to save results.')

        arg_parser.add_argument('--word-vector-model',
                                metavar='word_vector_model',
                                help='Word vector model to be used to convert word and sentences.')

        arg_parser.add_argument('--word-vector-dim',
                                metavar='word_vectordim',
                                help='Word vector model dimensions.')

        arg_parser.add_argument('--process-n-examples',
                                type=int,
                                metavar='process_n_examples',
                                help='Number of examples of the dataset directory to use.')

        # Neural Network parameters
        arg_parser.add_argument('--learning-rate',
                                metavar='nn_learning_rate',
                                help='Learning rate for NN training')

        arg_parser.add_argument('--batch-size',
                                metavar='batch_size',
                                help='Batch size for each learning iteration.')

        arg_parser.add_argument('--batch-test-size',
                                metavar='batch_test_size',
                                help='Batch test size for each learning iteration.')


        arg_parser.add_argument('--percent-train',
                                metavar='percent_train',
                                help='Percentage of examples to be used for training stage.')

        arg_parser.add_argument('--num-epochs',
                                metavar='num_epochs',
                                help='Total of epochs to run during training stage.')

        # Dataset preprocessing
        arg_parser.add_argument('--dataset-chunk-size',
                                metavar='dataset_chunk_size',
                                help='Dataset chunk size per file')

        self.config = arg_parser.parse_args()


    # Log INFO and ERROR messages to respective files
    def log_message(self, type, message):
        if type == 'ERROR':
            log_file = self.config.session_name + "-error.log"
        elif type == 'INFO':
            log_file = self.config.session_name + ".log"

        f = open(self.config.working_dir + "/" + log_file, "a")
        f.write(time.strftime("%Y-%m-%d %H:%m:%S", time.localtime()) + " -- " + message + "\n")
        f.close()


    # Return a list of files to be processed
    def get_dataset_files(self):
        files = os.listdir(self.config.dataset_dir)
        files.sort()

        if self.config.dataset_file_list != None:
            files = []
            f = open(self.config.dataset_file_list, "r")

            for file in f:
                files.append(file[:len(file)-1])

            return files

        if self.config.dataset_exclusion_list != None:
            exclusion_list = []

            f = open(self.config.dataset_exclusion_list, "r")

            for file in f:
                exclusion_list.append(file[:len(file)-1])

            files = [file for file in files if file not in exclusion_list]
            print("Length files", len(files))


        if self.config.process_n_examples != None:
            files = files[:self.config.process_n_examples]

        return files


    # Return a list of each sentence in text
    def text_to_sentences(self, text):
        text = text.replace("\n", " ")
        text = text.replace('@highlight', ".")
        sentences = nltk.sent_tokenize(text)
        sentences = [sentence.replace("  ", " ") for sentence in sentences]

        sentences = [sentence.lower() for sentence in sentences]
        if len(sentences) > 2:
            return sentences[0], sentences[1:]
        else:
            return [], []



    # Return a list of each sentence in text and summaries
    def text_to_sentences2(self, text):
        text = text.replace("  ", " ")
        sentences = text.split("\n")

        sentences = [sentence.strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences if sentence != ""]
        # Find first index of summaries
        idx = 0
        for sentence in sentences:
            if sentence == "@highlight":
                break
            idx += 1

        if len(sentences) < idx:
            return "", []

        summaries = sentences[idx:]
        sentences = sentences[:idx]

        summaries = [summary for summary in summaries if summary != "@highlight"]
        sentences = [sentence for sentence in sentences if len(sentence) > 30]

        summary = ""
        for s in summaries:
            summary += s + ". "

        sentences = [sentence.lower() for sentence in sentences]
        return summary.lower(), sentences



    def rouge_to_list(self, rouge_str):
        rouge_list = [ [rouge_str[0]['rouge-1']['f'], rouge_str[0]['rouge-1']['p'], rouge_str[0]['rouge-1']['r'] ],\
                         [rouge_str[0]['rouge-2']['f'], rouge_str[0]['rouge-2']['p'], rouge_str[0]['rouge-2']['r'] ],\
                         [rouge_str[0]['rouge-l']['f'], rouge_str[0]['rouge-l']['p'], rouge_str[0]['rouge-l']['r'] ]\
                       ]
        return rouge_list


    def euclidian_distance(self, v1, v2):
        return np.sqrt(np.power(np.sum(v1 - v2), 2))


    # Prepare training and test batch
    def prepare_batch(self):
        dataset_files = self.get_dataset_files()
        metadata_file = [m for m in dataset_files if "metadata" in m][0]
        dataset_files = [d for d in dataset_files if "metadata" not in d]

        print("Dataset files = ", str(dataset_files))
        print("Metadata file = " + str(metadata_file))

        metadata = np.load(self.config.dataset_dir + "/" + metadata_file)
        self.max_sentences = metadata['longest_sentence']
        total_dataset_examples = metadata['files_counter']

        # Total of training and test examples
        self.total_training_examples = int(np.floor(total_dataset_examples * float(self.config.percent_train)))
        self.total_test_examples = total_dataset_examples - self.total_training_examples

        self.test_first_chunk_idx = int(self.total_training_examples / int(self.config.dataset_chunk_size))
        self.test_first_example_idx = int(self.total_training_examples % int(self.config.dataset_chunk_size))

        if self.test_first_example_idx > 0:
            self.dataset_files_train = dataset_files[:self.test_first_chunk_idx + 1]
        else:
            self.dataset_files_train = dataset_files[:self.test_first_chunk_idx]

        self.dataset_files_test = dataset_files[self.test_first_chunk_idx:]
        if self.total_test_examples == 0:
            self.dataset_files_test = []


        # Summary
        print("Total of training examples = ", str(self.total_training_examples))
        print("Total of test examples = ", str(self.total_test_examples))
        print("Total of examples = ", str(total_dataset_examples))
        print("Dataset training files = ", str(self.dataset_files_train))
        print("Dataset test files = ", str(self.dataset_files_test))
        print("Test chunk idx = ", str(self.test_first_chunk_idx))
        print("First test example idx = ", str(self.test_first_example_idx))

        # Load initial values
        self.train_curr_chunk_idx = 0
        self.test_curr_chunk_idx = 0

        self.train_curr_example_idx = 0
        self.test_curr_example_idx = self.total_training_examples

        self.train_processed_examples = 0
        self.test_processed_examples = 0

        self.npz_train = np.load(self.config.dataset_dir + "/" \
                        + self.dataset_files_train[self.train_curr_chunk_idx])
        self.train_curr_chunk_idx += 1

        if len(self.dataset_files_test) > 0:
            self.npz_test = np.load(self.config.dataset_dir + "/" \
                            + self.dataset_files_test[self.test_curr_chunk_idx])
            self.test_curr_chunk_idx += 1


    # Return batch values for training and testing sets
    def get_next_batch(self, batch_type):
        x = []
        y = []
        x_res = []
        y_res = []
        all_batch_run = 0

        if batch_type == "training":
            batch_capacity = int(self.config.batch_size)
            chunk_capacity = self.npz_train['x'].shape[0] - self.train_curr_example_idx
        elif batch_type == "test":
            batch_capacity = int(self.config.batch_test_size)
            chunk_capacity = self.npz_test['x'].shape[0] - self.test_curr_example_idx


        while batch_capacity > 0:
            if batch_capacity >= chunk_capacity:
                # Load all chunk into this batch
                # print("Adding full chunk")
                if batch_type == "training":
                    if self.train_processed_examples + chunk_capacity >= self.total_training_examples:
                        x = np.append(x, self.npz_train['x'][self.train_curr_example_idx:\
                            self.train_curr_example_idx + (self.total_training_examples - self.train_processed_examples)])
                        y = np.append(y, self.npz_train['y'][self.train_curr_example_idx:\
                            self.train_curr_example_idx + (self.total_training_examples - self.train_processed_examples)])

                        # Reload first file
                        self.train_curr_chunk_idx = 0
                        self.train_curr_example_idx = 0
                        self.train_processed_examples = 0
                        self.npz_train = np.load(self.config.dataset_dir + "/" \
                                        + self.dataset_files_train[self.train_curr_chunk_idx])
                        self.train_curr_chunk_idx += 1

                        all_batch_run = 1
                        break

                    x = np.append(x, self.npz_train['x'][self.train_curr_example_idx:])
                    y = np.append(y, self.npz_train['y'][self.train_curr_example_idx:])
                    self.train_processed_examples += chunk_capacity
                    batch_capacity -= chunk_capacity

                    # Load next file
                    if self.train_curr_chunk_idx == len(self.dataset_files_train):
                        # Last file, return to first file (shorter mini-batch)
                        self.train_curr_chunk_idx = 0
                        self.train_curr_example_idx = 0
                        break

                    self.npz_train = np.load(self.config.dataset_dir + "/" \
                        + self.dataset_files_train[self.train_curr_chunk_idx])
                    self.train_curr_chunk_idx += 1
                    self.train_curr_example_idx = 0
                    chunk_capacity = self.npz_train['x'].shape[0] - self.train_curr_example_idx

                elif batch_type == "test":
                    if self.test_processed_examples + chunk_capacity >= self.total_test_examples:
                        x = np.append(x, self.npz_test['x'][self.test_curr_example_idx:\
                            self.test_curr_example_idx + (self.total_test_examples - self.test_processed_examples)])
                        y = np.append(y, self.npz_test['y'][self.test_curr_example_idx:\
                            self.test_curr_example_idx + (self.total_test_examples - self.test_processed_examples)])

                        # Reload first file
                        self.test_curr_chunk_idx = 0
                        self.test_curr_example_idx = self.test_first_example_idx
                        self.test_processed_examples = 0
                        self.npz_test = np.load(self.config.dataset_dir + "/" \
                                        + self.dataset_files_test[self.test_curr_chunk_idx])
                        self.test_curr_chunk_idx += 1

                        all_batch_run = 1
                        break

                    x = np.append(x, self.npz_test['x'][self.test_curr_example_idx:])
                    y = np.append(y, self.npz_test['y'][self.test_curr_example_idx:])
                    self.test_processed_examples += chunk_capacity
                    batch_capacity -= chunk_capacity

                    # Load next file
                    if self.test_curr_chunk_idx == len(self.dataset_files_test):
                        # Last file, return to first file (shorter mini-batch)
                        self.test_curr_chunk_idx = 0
                        self.test_curr_example_idx = 0
                        break

                    self.npz_test = np.load(self.config.dataset_dir + "/" \
                        + self.dataset_files_test[self.test_curr_chunk_idx])
                    self.test_curr_chunk_idx += 1
                    self.test_curr_example_idx = 0
                    chunk_capacity = self.npz_test['x'].shape[0] - self.test_curr_example_idx

            else:
                # Load partial chunk
                # print("Adding partial chunk")
                if batch_type == "training":
                    if self.train_processed_examples + chunk_capacity >= self.total_training_examples:
                        if self.total_training_examples - self.train_processed_examples >= batch_capacity:
                            x = np.append(x, self.npz_train['x'][self.train_curr_example_idx:\
                                self.train_curr_example_idx + batch_capacity])
                            y = np.append(y, self.npz_train['y'][self.train_curr_example_idx:\
                                self.train_curr_example_idx + batch_capacity])
                        else:
                            x = np.append(x, self.npz_train['x'][self.train_curr_example_idx:\
                                self.train_curr_example_idx + (self.total_training_examples - self.train_processed_examples)])
                            y = np.append(y, self.npz_train['y'][self.train_curr_example_idx:\
                                self.train_curr_example_idx + (self.total_training_examples - self.train_processed_examples)])

                        # Reload first file
                        self.train_curr_chunk_idx = 0
                        self.train_curr_example_idx = 0
                        self.train_processed_examples = 0
                        self.npz_train = np.load(self.config.dataset_dir + "/" \
                                        + self.dataset_files_train[self.train_curr_chunk_idx])
                        self.train_curr_chunk_idx += 1
                        all_batch_run = 1
                        break

                    if self.train_curr_chunk_idx == len(self.dataset_files_train):
                        # Last file, return to first file (shorter mini-batch)
                        self.train_curr_chunk_idx = 0
                        self.train_curr_example_idx = 0
                        break

                        self.npz_train = np.load(self.config.dataset_dir + "/" \
                            + self.dataset_files_train[self.train_curr_chunk_idx])
                        self.train_curr_chunk_idx += 1
                        self.train_curr_example_idx = 0
                        chunk_capacity = self.npz_train['x'].shape[0] - self.train_curr_example_idx


                    x = np.append(x, self.npz_train['x'][self.train_curr_example_idx:self.train_curr_example_idx + batch_capacity])
                    y = np.append(y, self.npz_train['y'][self.train_curr_example_idx:self.train_curr_example_idx + batch_capacity])
                    self.train_curr_example_idx += batch_capacity
                    self.train_processed_examples += batch_capacity
                    batch_capacity = 0


                elif batch_type == "test":
                    if self.test_processed_examples + chunk_capacity >= self.total_test_examples:
                        if self.total_test_examples - self.test_processed_examples >= batch_capacity:
                            x = np.append(x, self.npz_test['x'][self.test_curr_example_idx:\
                                self.test_curr_example_idx + batch_capacity])
                            y = np.append(y, self.npz_test['y'][self.test_curr_example_idx:\
                                self.test_curr_example_idx + batch_capacity])
                        else:
                            x = np.append(x, self.npz_test['x'][self.test_curr_example_idx:\
                                self.test_curr_example_idx + (self.total_test_examples - self.test_processed_examples)])
                            y = np.append(y, self.npz_test['y'][self.test_curr_example_idx:\
                                self.test_curr_example_idx + (self.total_test_examples - self.test_processed_examples)])

                        # Reload first file
                        self.test_curr_chunk_idx = 0
                        self.test_curr_example_idx = self.test_first_example_idx
                        self.test_processed_examples = 0
                        self.npz_test = np.load(self.config.dataset_dir + "/" \
                                        + self.dataset_files_test[self.test_curr_chunk_idx])
                        self.test_curr_chunk_idx += 1
                        all_batch_run = 1
                        break

                    if self.test_curr_chunk_idx == len(self.dataset_files_test):
                        # Last file, return to first file (shorter mini-batch)
                        self.test_curr_chunk_idx = 0
                        self.test_curr_example_idx = 0
                        break

                        self.npz_test = np.load(self.config.dataset_dir + "/" \
                            + self.dataset_files_test[self.test_curr_chunk_idx])
                        self.test_curr_chunk_idx += 1
                        self.test_curr_example_idx = 0
                        chunk_capacity = self.npz_test['x'].shape[0] - self.test_curr_example_idx


                    x = np.append(x, self.npz_test['x'][self.test_curr_example_idx:self.test_curr_example_idx + batch_capacity])
                    y = np.append(y, self.npz_test['y'][self.test_curr_example_idx:self.test_curr_example_idx + batch_capacity])
                    self.test_curr_example_idx += batch_capacity
                    self.test_processed_examples += batch_capacity
                    batch_capacity = 0

        x_res = []
        y_res = []
        padding = np.zeros((1, int(self.config.word_vector_dim)))

        for entry_x, entry_y in zip(x, y):
            while entry_x.shape[0] < self.max_sentences:
                entry_x = np.vstack((entry_x, padding))
                entry_y = np.append(entry_y, 0)
            x_res.append(entry_x.reshape((1, -1))[0])
            y_res.append(entry_y.reshape((1, -1))[0])

        return all_batch_run, np.array(x_res), np.array(y_res)






    # # Save texts to NPZ files with y one-hot vector representation for summary
    # def dataset_to_npz(self):
    #     # Get files list
    #     files_list = self.get_dataset_files()
    #     files_counter = 0
    #     longest_sentence_len = 0
    #     chunk_counter = 0
    #     x_list = []
    #     y_list = []
    #
    #     for file in files_list:
    #         text = open(self.config.dataset_dir + "/" + file).read()
    #         summary, sentences = self.text_to_sentences2(text)
    #
    #         curr_sentence_len = len(max(sentences, key=len))
    #
    #         if curr_sentence_len > longest_sentence_len:
    #             longest_sentence_len = curr_sentence_len
    #
    #         sentences_vec = model.embed_sentences(sentences)
    #
    #         # Compute best rouge score for this text
    #
    #         x_list.append(sentences)
    #         y_list.append(summary)
    #
    #         files_counter += 1
    #         if files_counter % int(self.config.dataset_chunk_size) == 0:
    #             # Write to file
    #             np.savez(self.config.working_dir + "/" + self.config.session_name + "-" + str("%06d" % chunk_counter), \
    #                         x=x_list, y=y_list)
    #             x_list = []
    #             y_list = []
    #             chunk_counter += 1
    #
    #     # Save metadata
    #     print("Total of files = ", files_counter)
    #     print("Chunk files = ", chunk_counter)
    #     print("Longest sentence in text = ", longest_sentence_len)
    #
    #

    # def prepare_batch(self):
    #     self.dataset_files_train = self.get_dataset_files()
    #     self.dataset_files_test = self.get_dataset_files()
    #
    #     metadata_file = [file for file in self.dataset_files_train if "metadata" in file][0]
    #     self.dataset_files_train = [file for file in self.dataset_files_train if "metadata" not in file]
    #
    #     # Load data from metadata file
    #     m = np.load(self.config.dataset_dir + "/" + metadata_file)
    #
    #     total_dataset_files = m['files_counter']
    #     self.longest_text = m['longest_sentence']
    #
    #     self.total_examples_train = int(np.floor(total_dataset_files * float(self.config.percent_train)))
    #     self.total_examples_test = total_dataset_files - self.total_examples_train
    #     print("Total examples in dataset = ", str(total_dataset_files))
    #     print("Total of training examples = ", str(self.total_examples_train))
    #     print("Total of test examples = ", str(self.total_examples_test))
    #
    #
    #     test_idx = self.total_examples_train
    #     print(test_idx)
    #
    #     test_chunk_idx = int(test_idx / int(self.config.dataset_chunk_size))
    #
    #     i = 0
    #     while self.dataset_files_test[i] != self.dataset_files_test[test_chunk_idx]:
    #         i += 1
    #
    #     self.dataset_files_test = self.dataset_files_test[i:]
    #
    #     print(str(self.dataset_files_train))
    #     print()
    #     print(str(self.dataset_files_test))
    #
    #
    #     # Load first file from dataset - training
    #     self.train_curr_chunk_idx = 0
    #     self.train_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_train[self.train_curr_chunk_idx])
    #     self.train_curr_chunk_idx += 1
    #     self.train_curr_example_idx = 0
    #     self.train_total_processed_examples = 0
    #
    #
    #     # Load first file from dataset - test
    #     self.test_curr_chunk_idx = 0
    #     self.test_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_test[self.test_curr_chunk_idx])
    #     self.test_curr_chunk_idx += 1
    #     self.test_curr_example_idx = 0
    #     self.test_total_processed_examples = 0
    #
    #
    #
    #
    # def get_next_train_batch(self):
    #     x = []
    #     y = []
    #     x_res = []
    #     y_res = []
    #     batch_capacity = int(self.config.batch_size)
    #     all_batch_run = 0
    #
    #     # Adjust dimension for vector dimensionality
    #     padding = np.zeros((1, int(self.config.word_vector_dim)))
    #
    #     while batch_capacity > 0:
    #         if batch_capacity >= (self.train_npz['x'].shape[0] - self.train_curr_example_idx):
    #             if self.train_total_processed_examples + self.train_npz['x'].shape[0] - self.train_curr_example_idx > self.total_examples_train:
    #                 x = np.append(x, self.train_npz['x'][self.train_curr_example_idx:self.train_curr_example_idx + (self.total_examples_train - self.train_total_processed_examples)])
    #                 # y = np.append(x, self.train_npz['y'][self.train_curr_example_idx:self.train_curr_example_idx + (self.total_examples_train - self.total_processed_examples)])
    #
    #                 # Back to first file
    #                 self.train_curr_chunk_idx = 0
    #                 # print("Loading file " + self.dataset_files[self.curr_chunk_idx])
    #                 self.train_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_train[self.train_curr_chunk_idx])
    #                 self.train_curr_chunk_idx += 1
    #                 self.train_curr_example_idx = 0
    #                 self.train_total_processed_examples = 0
    #                 all_batch_run = 1
    #                 break
    #
    #             # Append all file chunk content to this batch
    #             x = np.append(x, self.train_npz['x'][self.train_curr_example_idx:])
    #             # y = np.append(y, self.train_npz['y'][self.train_curr_example_idx:])
    #
    #             batch_capacity -= self.train_npz['x'].shape[0] - self.train_curr_example_idx
    #             self.train_total_processed_examples += self.train_npz['x'].shape[0] - self.train_curr_example_idx
    #
    #             # Load next file chunk
    #             if self.train_curr_chunk_idx == len(self.dataset_files_train):
    #                 # Back to first file
    #                 self.train_curr_chunk_idx = 0
    #                 self.train_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_train[self.train_curr_chunk_idx])
    #                 self.train_curr_chunk_idx += 1
    #                 self.train_curr_example_idx = 0
    #                 self.train_total_processed_examples = 0
    #                 all_batch_run = 1
    #
    #                 # Break and return - shorter mini-batch
    #                 break
    #
    #             # Load next file chunk
    #             # print("Loading file " + self.dataset_files[self.curr_chunk_idx])
    #             self.train_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_train[self.train_curr_chunk_idx])
    #             self.train_curr_chunk_idx += 1
    #             self.train_curr_example_idx = 0
    #         else:
    #             if self.train_total_processed_examples + self.train_npz['x'].shape[0] - self.train_curr_example_idx > self.total_examples_train:
    #                 # print("Batch capacity exceeded 2")
    #                 x = np.append(x, self.train_npz['x'][self.train_curr_example_idx:self.train_curr_example_idx + (self.total_examples_train - self.train_total_processed_examples)])
    #                 # y = np.append(y, self.train_npz['y'][self.train_curr_example_idx:self.train_curr_example_idx + (self.total_examples_train - self.train_total_processed_examples)])
    #
    #                 # Back to first file
    #                 self.train_curr_chunk_idx = 0
    #                 self.train_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_train[self.train_curr_chunk_idx])
    #                 self.train_curr_chunk_idx += 1
    #                 self.train_curr_example_idx = 0
    #                 self.train_total_processed_examples = 0
    #                 all_batch_run = 1
    #                 break
    #
    #             # Load file chunk partially until batch_capacity is filled
    #             x = np.append(x, self.train_npz['x'][self.train_curr_example_idx:self.train_curr_example_idx + batch_capacity])
    #             # y = np.append(y, self.train_npz['y'][self.train_curr_example_idx:self.train_curr_example_idx + batch_capacity])
    #
    #             self.train_curr_example_idx += batch_capacity
    #             self.train_total_processed_examples += batch_capacity
    #             batch_capacity = 0
    #
    #     for entry_x, entry_y in zip(x, y):
    #         while entry_x.shape[0] < self.longest_text:
    #             entry_x = np.vstack((entry_x, padding))
    #             # entry_y = np.append(entry_y, 0)
    #         x_res.append(entry_x.reshape((1,-1))[0])
    #         # y_res.append(entry_y.reshape((1,-1))[0])
    #
    #     return all_batch_run, np.array(x_res), np.array(y_res)
    #
    #
    #
    # # Get next test batch
    # def get_next_test_batch(self):
    #     x = []
    #     y = []
    #     x_res = []
    #     y_res = []
    #     batch_capacity = int(self.config.batch_test_size)
    #     all_batch_run = 0
    #
    #     # Adjust dimension for vector dimensionality
    #     padding = np.zeros((1, int(self.config.word_vector_dim)))
    #
    #     while batch_capacity > 0:
    #         if batch_capacity >= (self.test_npz['x'].shape[0] - self.test_curr_example_idx):
    #             if self.test_total_processed_examples + self.test_npz['x'].shape[0] - self.test_curr_example_idx > self.total_examples_test:
    #                 x = np.append(x, self.test_npz['x'][self.test_curr_example_idx:self.test_curr_example_idx + (self.total_examples_test - self.test_total_processed_examples)])
    #                 # y = np.append(x, self.test_npz['y'][self.test_curr_example_idx:self.test_curr_example_idx + (self.total_examples_test - self.test_total_processed_examples)])
    #
    #                 # Back to first file
    #                 self.test_curr_chunk_idx = 0
    #                 # print("Loading file " + self.dataset_files[self.curr_chunk_idx])
    #                 self.test_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_test[self.test_curr_chunk_idx])
    #                 self.test_curr_chunk_idx += 1
    #                 self.test_curr_example_idx = 0
    #                 self.test_total_processed_examples = 0
    #                 all_batch_run = 1
    #                 break
    #
    #             # Append all file chunk content to this batch
    #             x = np.append(x, self.test_npz['x'][self.test_curr_example_idx:])
    #             # y = np.append(y, self.test_npz['y'][self.test_curr_example_idx:])
    #
    #             batch_capacity -= self.tet_npz['x'].shape[0] - self.test_curr_example_idx
    #             self.test_total_processed_examples += self.test_npz['x'].shape[0] - self.test_curr_example_idx
    #
    #             # Load next file chunk
    #             if self.test_curr_chunk_idx == len(self.dataset_files_test):
    #                 # Back to first file
    #                 self.test_curr_chunk_idx = 0
    #                 self.test_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_test[self.test_curr_chunk_idx])
    #                 self.test_curr_chunk_idx += 1
    #                 self.test_curr_example_idx = 0
    #                 self.test_total_processed_examples = 0
    #                 all_batch_run = 1
    #
    #                 # Break and return - shorter mini-batch
    #                 break
    #
    #             # Load next file chunk
    #             # print("Loading file " + self.dataset_files[self.curr_chunk_idx])
    #             self.test_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_test[self.test_curr_chunk_idx])
    #             self.test_curr_chunk_idx += 1
    #             self.test_curr_example_idx = 0
    #         else:
    #             if self.test_total_processed_examples + self.test_npz['x'].shape[0] - self.test_curr_example_idx > self.total_examples_test:
    #                 # print("Batch capacity exceeded 2")
    #                 x = np.append(x, self.test_npz['x'][self.test_curr_example_idx:self.test_curr_example_idx + (self.total_examples_test - self.test_total_processed_examples)])
    #                 # y = np.append(y, self.test_npz['y'][self.test_curr_example_idx:self.test_curr_example_idx + (self.total_examples_test - self.test_total_processed_examples)])
    #
    #                 # Back to first file
    #                 self.test_curr_chunk_idx = 0
    #                 self.test_npz = np.load(self.config.dataset_dir + "/" + self.dataset_files_test[self.test_curr_chunk_idx])
    #                 self.test_curr_chunk_idx += 1
    #                 self.test_curr_example_idx = 0
    #                 self.test_total_processed_examples = 0
    #                 all_batch_run = 1
    #                 break
    #
    #             # Load file chunk partially until batch_capacity is filled
    #             x = np.append(x, self.test_npz['x'][self.test_curr_example_idx:self.test_curr_example_idx + batch_capacity])
    #             # y = np.append(y, self.test_npz['y'][self.test_curr_example_idx:self.test_curr_example_idx + batch_capacity])
    #
    #             self.test_curr_example_idx += batch_capacity
    #             self.test_total_processed_examples += batch_capacity
    #             batch_capacity = 0
    #
    #     print(x)
    #
    #     for entry_x, entry_y in zip(x, y):
    #         while entry_x.shape[0] < self.longest_text:
    #             entry_x = np.vstack((entry_x, padding))
    #             # entry_y = np.append(entry_y, 0)
    #         x_res.append(entry_x.reshape((1,-1))[0])
    #         # y_res.append(entry_y.reshape((1,-1))[0])
    #
    #     return all_batch_run, np.array(x_res), np.array(y_res)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # def get_test(self):
    #     # Compute index of first test example
    #     test_idx = self.total_examples_train
    #     x = []
    #     y = []
    #     self.test_x = []
    #     self.test_y = []
    #
    #     # Compute first data file chunk with this example
    #     chunk_idx = int(test_idx / int(self.config.dataset_chunk_size))
    #     chunk_example_idx = int(test_idx % int(self.config.dataset_chunk_size))
    #
    #     # print("Test idx = ", test_idx)
    #     # print("Chunk idx = ", chunk_idx)
    #     # print("Chunk example idx = ", chunk_example_idx)
    #     # print("Dataset file", self.dataset_files[chunk_idx])
    #
    #     f = np.load(self.config.dataset_dir + "/" + self.dataset_files[chunk_idx])
    #     x = np.append(x, f['x'][chunk_example_idx:])
    #     y = np.append(y, f['y'][chunk_example_idx:])
    #
    #     for file in self.dataset_files[chunk_idx + 1:]:
    #         f = np.load(self.config.dataset_dir + "/" + file)
    #         x = np.append(x, f['x'])
    #         y = np.append(y, f['y'])
    #
    #     padding = np.zeros((1, int(self.config.word_vector_dim)))
    #
    #     for entry_x, entry_y in zip(x, y):
    #         while entry_x.shape[0] < self.longest_text:
    #             entry_x = np.vstack((entry_x, padding))
    #             entry_y = np.append(entry_y, 0)
    #         self.test_x.append(entry_x.reshape((1,-1))[0])
    #         self.test_y.append(entry_y.reshape((1,-1))[0])
    #     self.test_x = np.array(self.test_x)
    #     self.test_y = np.array(self.test_y)
    #
    #     return self.test_x, self.test_y
