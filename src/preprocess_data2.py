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

# Read files from dataset dir
files_list = common.get_dataset_files()

for file in files_list:
    text = open(common.config.dataset_dir + "/" + file).read()
    summary, sentences = common.text_to_sentences2(text)

    if len(sentences) < 10:
        common.log_message("ERROR", "Ignoring file " + file + " due to parse errors or short length.")
        continue

    sentences_vec = common.model.embed_sentences(sentences)
    Y_rouge_list = []

    # Compute ROUGE score for each sentence
    for sentence, sentence_vec in zip(sentences, sentences_vec):
        try:
            rouge_str = rouge.get_scores(summary, sentence)
        except ValueError:
            common.log_message("ERROR", str(file) + "  -- Error computing ROUGE of sentence " +\
                                sentence)
            continue

        Y_rouge_list.append(common.rouge_to_list(rouge_str)[0][2])
        
    Y_list.append(Y_rouge_list)
    Y_rouge_list = []
    X_list.append(sentences_vec)

    np.savez(common.config.working_dir + "/" + common.config.session_name,
            x=np.array(X_list), y=np.array(Y_list))
