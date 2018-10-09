#!/usr/bin/python3 -i

# Simple experiment of reading dataset directory and generate embeddings

import sys
sys.path.append("../lib/python/")
sys.path.append("/home/steinert/dev/pucrs-cc-tcc-2018/lib/rouge/build/lib")

from common import *
import rouge
import numpy as np


common = Common()
common.parse_cmd_args()
common.load_word_vec_model()

rouge = rouge.Rouge()
texts_list_stat = []
texts_list_stat_rouge = []

save_to_npz = True


common.log_message("INFO", "\n\nStarting session " + str(common.config.session_name))
common.log_message("INFO", "Parameters: " + str(common.config))
common.log_message("INFO", "\n")
files_counter = 0
total_match_choice = 0
longest_sentence_len = 0
chunk_counter = 0
x_list = []
y_list = []

files_list = common.get_dataset_files()

for file in files_list:
    if files_counter % 100 == 0:
        print("\n\n[" + str("%.03f" % (files_counter /len(files_list))) + "] Processing file " + file)

    text = open(common.config.dataset_dir + "/" + file).read()

    # Get ground-truth summary and sentences
    summary, sentences = common.text_to_sentences2(text)

    # Ignore texts with number of sentences less than 8
    if len(sentences) < 10:
        common.log_message("ERROR", "Ignoring file " + file + " due to parse errors or short length.")
        continue

    # Convert summary and text to embeddings
    summary_vec = common.model.embed_sentence(summary)
    sentences_vec = common.model.embed_sentences(sentences)

    sentences_dist = []
    sentence_idx = 0

    # For each sentence, compute offset in relation to ground-truth summary
    for sentence_vec in sentences_vec:
        try:
            rouge_str = rouge.get_scores(summary, sentences[sentence_idx])
        except ValueError:
            common.log_message("ERROR", str(file) + "  -- Error computing ROUGE of sentence " +\
                                sentences[sentence_idx])
            sentence_idx += 1
            continue

        sentences_dist.append([file, sentence_idx, common.rouge_to_list(rouge_str)])
        sentence_idx += 1

    sentences_dist.sort(key=lambda x: x[2][0], reverse=True)

    texts_list_stat_rouge.append(sentences_dist[0])
    common.log_message("INFO", "BEST ROUGE SCORE = " + str(sentences_dist[0]))
    common.log_message("INFO", "\n")

    files_counter += 1

    if save_to_npz == True:
        one_hot = np.zeros((len(sentences),))
        one_hot[sentences_dist[0][1]] = 1

        x_list.append(sentences_vec)
        y_list.append(one_hot)

        curr_sentence_len = len(sentences)
        if curr_sentence_len > longest_sentence_len:
            longest_sentence_len = curr_sentence_len

        if files_counter % int(common.config.dataset_chunk_size) == 0:
            # Write to file
            np.savez(common.config.working_dir + "/" + common.config.session_name + "-" + str("%06d" % chunk_counter), \
                x=x_list, y=y_list)
            x_list = []
            y_list = []
            chunk_counter += 1


if save_to_npz == True:
    # Save metadata
    print("Total of files = ", files_counter)
    print("Chunk files = ", chunk_counter)
    print("Longest sentence in text = ", longest_sentence_len)
    metadata = [files_counter, chunk_counter, longest_sentence_len]
    np.savez(common.config.working_dir + "/" + common.config.session_name + "-metadata", \
        files_counter=files_counter, chunks_counter=chunk_counter, longest_sentence=longest_sentence_len)


#
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []


# Consolidated list to compute mean
common.log_message("INFO", "\n")
common.log_message("INFO", "TOTAL OF FILES PROCESSED: " + str(files_counter))

rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

for entry in texts_list_stat_rouge:
    rouge_1_list.append(entry[2][0])
    rouge_2_list.append(entry[2][1])
    rouge_l_list.append(entry[2][2])

common.log_message("INFO", "BEST ROUGE SCORE MEAN")
common.log_message("INFO", "\tROUGE-1: " + str(np.mean(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.mean(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: " + str(np.mean(rouge_l_list, axis=0)))

common.log_message("INFO", "STD DEVIATION")
common.log_message("INFO", "\tROUGE-1: " + str(np.std(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.std(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: "+ str(np.std(rouge_l_list, axis=0)))
