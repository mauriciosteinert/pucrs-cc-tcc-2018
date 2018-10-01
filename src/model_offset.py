#!/usr/bin/python3 -i

# Simple experiment of reading dataset directory and generate embeddings

import sys
sys.path.append("../")

from common import *
import rouge
import numpy as np


common = Common()
common.parse_cmd_args()
common.load_word_vec_model()

rouge = rouge.Rouge()
texts_list_stat = []

common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")
files_counter = 0
files_list = common.get_dataset_files()

for file in files_list:
    print("\n\n[" + str("%.03f" % (files_counter /len(files_list))) + "] Processing file " + file)
    files_counter += 1
    text = open(common.config.dataset_dir + "/" + file).read()

    # Get ground-truth summary and sentences
    summary, sentences = common.text_to_sentences(text)
    # print("Summary = ", summary)
    # print("Sentences = ", sentences)
    # print("------------------------------------------------------------------")

    # Ignore texts with number of sentences less than 8
    if len(sentences) < 10:
        common.log_message("ERROR", "Ignoring file " + file + " due to parse errors or short length.")
        continue

    # Convert summary and text to embeddings
    summary_vec = common.model.embed_sentence(summary)
    sentences_vec = common.model.embed_sentences(sentences)
    text_mean_vec = np.mean(sentences_vec, axis=0)
    text_summary_dif_vec = np.subtract(text_mean_vec, summary_vec)

    sentences_dist = []
    sentence_idx = 0

    # For each sentence, compute offset in relation to ground-truth summary
    for sentence_vec in sentences_vec:
        # print("Processing sentence [" + str(sentence_idx) \
        #         + "] "+ sentences[sentence_idx])
        # print("---------------------------------")
        sentence_vec_dist = np.linalg.norm(np.add(text_summary_dif_vec, sentence_vec))
        # sentence_vec_dist = np.linalg.norm(np.subtract(sentence_vec, summary_vec))

        try:
            rouge_str = rouge.get_scores(summary, sentences[sentence_idx])
        except ValueError:
            common.log_message("ERROR", str(file) + "  -- Error computing ROUGE of sentence " +\
                                sentences[sentence_idx])
            sentence_idx += 1
            continue

        sentences_dist.append([file, sentence_idx, len(sentences[sentence_idx]), sentence_vec_dist, \
                                common.rouge_to_list(rouge_str)])
        sentence_idx += 1

    sentences_dist.sort(key=lambda x: (x[3], x[2]))
    common.log_message("INFO", str(sentences_dist[0]))
    texts_list_stat.append(sentences_dist[0])

    if sentences_dist[0][4][0][0] == 0.0:
        common.log_message("INFO", "File " + file + " with rouge zero score. SENTENCES = " + \
                    str(sentences) + " SUMMARY = " + str(summary))

#
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

for entry in texts_list_stat:
    rouge_1_list.append(entry[4][0])
    rouge_2_list.append(entry[4][1])
    rouge_l_list.append(entry[4][2])

# Consolidated list to compute mean
common.log_message("INFO", "MEAN")
common.log_message("INFO", "\tROUGE-1: " + str(np.mean(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.mean(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: " + str(np.mean(rouge_l_list, axis=0)))

common.log_message("INFO", "STD DEVIATION")
common.log_message("INFO", "\tROUGE-1: " + str(np.std(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.std(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: "+ str(np.std(rouge_l_list, axis=0)))
