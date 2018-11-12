#!/usr/bin/python3

import sent2vec

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

X_list = []
Y_list = []
file_list_save = []
Y_idx_list = []
files_counter = 0
chunk_counter = 0

# Read files from dataset dir
files_list = common.get_dataset_files()

for file in files_list:
    print("\n\n[" + str("%.03f" % (files_counter /len(files_list))) + "] Processing file " + file)

    text = open(common.config.dataset_dir + "/" + file).read()
    summary, sentences = common.text_to_sentences2(text)

    if len(sentences) < 10 or len(sentences) > 45:
        common.log_message("ERROR", "Ignoring file " + file + " due to parse errors or short length.")
        continue

    try:
        sentences_vec = common.model.embed_sentences(sentences)
    except TypeError:
        common.log_message("ERROR", "Ignoring file " + file + " due to embedding error.")
        continue

    Y_rouge_list = []

    # Compute ROUGE score for each sentence
    for sentence, sentence_vec in zip(sentences, sentences_vec):
        try:
            rouge_str = rouge.get_scores(summary, sentence)
        except ValueError:
            common.log_message("ERROR", str(file) + "  -- Error computing ROUGE of sentence " +\
                                sentence)
            continue

        # Y_rouge_list.append(common.rouge_to_list(rouge_str)[0][2])
        Y_rouge_list.append(common.rouge_to_list(rouge_str))

    Y_list.append(Y_rouge_list)
    Y_idx_list.append(np.argmax(Y_rouge_list, axis=0)[0][0])
    Y_rouge_list = []
    X_list.append(sentences_vec)
    file_list_save.append(file)

    files_counter += 1

    if files_counter % int(common.config.dataset_chunk_size) == 0:
        # Write to file
        np.savez(common.config.working_dir + "/" + common.config.session_name + "-" + str("%06d" % chunk_counter), \
            files=file_list_save, x=np.array(X_list), y_rouge=np.array(Y_list), y_idx=np.array(Y_idx_list))
        X_list = []
        Y_list = []
        Y_idx_list = []
        file_list_save = []
        chunk_counter += 1
