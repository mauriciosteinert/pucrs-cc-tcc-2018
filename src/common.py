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

        arg_parser.add_argument('--percent-train',
                                metavar='percent_train',
                                help='Percentage of examples to be used for training stage.')

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


    # Save texts to NPZ files with y one-hot vector representation for summary
    def dataset_to_npz(self):
        # Get files list
        files_list = self.get_dataset_files()
        files_counter = 0
        longest_sentence_len = 0
        chunk_counter = 0
        x_list = []
        y_list = []

        for file in files_list:
            text = open(self.config.dataset_dir + "/" + file).read()
            summary, sentences = self.text_to_sentences2(text)

            curr_sentence_len = len(max(sentences, key=len))

            if curr_sentence_len > longest_sentence_len:
                longest_sentence_len = curr_sentence_len

            sentences_vec = model.embed_sentences(sentences)

            # Compute best rouge score for this text

            x_list.append(sentences)
            y_list.append(summary)

            files_counter += 1
            if files_counter % int(self.config.dataset_chunk_size) == 0:
                # Write to file
                np.savez(self.config.working_dir + "/" + self.config.session_name + "-" + str("%06d" % chunk_counter), \
                            x=x_list, y=y_list)
                x_list = []
                y_list = []
                chunk_counter += 1

        # Save metadata
        print("Total of files = ", files_counter)
        print("Chunk files = ", chunk_counter)
        print("Longest sentence in text = ", longest_sentence_len)



    def prepare_batch(self):
        self.dataset_files = self.get_dataset_files()

        metadata_file = [file for file in self.dataset_files if "metadata" in file][0]
        self.dataset_files = [file for file in self.dataset_files if "metadata" not in file]

        # Load data from metadata file
        m = np.load(self.config.dataset_dir + "/" + metadata_file)

        total_dataset_files = m['files_counter']
        self.total_dataset_chunks = m['chunks_counter']
        self.longest_text = m['longest_sentence']

        self.total_examples_train = np.floor(total_dataset_files * float(self.config.percent_train))
        self.total_examples_test = total_dataset_files - self.total_examples_train

        # Load first file from dataset
        self.curr_chunk_idx = 0
        self.npz = np.load(self.config.dataset_dir + "/" + self.dataset_files[self.curr_chunk_idx])
        self.curr_chunk_idx += 1
        self.curr_example_idx = 0
        self.total_processed_examples = 0


    def get_next_train_batch(self):
        x = []
        y = []
        batch_capacity = int(self.config.batch_size)

        while batch_capacity > 0:
            if batch_capacity >= (self.npz['x'].shape[0] - self.curr_example_idx):
                # Append all file chunk content to this batch
                x.append(self.npz['x'][self.curr_example_idx:])
                # y.append(self.npz['y'][self.curr_example_idx:])

                batch_capacity -= self.npz['x'].shape[0] - self.curr_example_idx
                self.total_processed_examples += self.npz['x'].shape[0] - self.curr_example_idx

                # Load next file chunk
                if self.curr_chunk_idx == len(self.dataset_files):
                    # Back to first file
                    self.curr_chunk_idx = 0
                    self.npz = np.load(self.config.dataset_dir + "/" + self.dataset_files[self.curr_chunk_idx])
                    self.curr_chunk_idx += 1
                    self.curr_example_idx = 0
                    self.total_processed_examples = 0

                    # return what we got until now - this is a shorter mini-batch
                    return x, y

                # Load next file chunk
                self.npz = np.load(self.config.dataset_dir + "/" + self.dataset_files[self.curr_chunk_idx])
                self.curr_chunk_idx += 1
                self.curr_example_idx = 0
            else:
                # Load file chunk partially until batch_capacity is filled
                x.append(self.npz['x'][self.curr_example_idx:self.curr_example_idx + batch_capacity])
                # y.append(self.npz['y'][self.curr_example_idx:self.curr_example_idx + batch_capacity])
                self.curr_example_idx += batch_capacity
                self.total_processed_examples += batch_capacity
                batch_capacity = 0

        return x, y
