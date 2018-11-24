#!/usr/bin/python3 -i

# Simple experiment of reading dataset directory and generate embeddings

import sys
sys.path.append("../")

from common import *
import rouge
import numpy as np
import matplotlib.pyplot as plt


common = Common()
common.parse_cmd_args()
common.load_word_vec_model()

rouge = rouge.Rouge()
texts_list_stat = []
texts_list_stat_rouge = []

n_top = 10
sentences_top_counter = np.zeros(n_top,)



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
    text_mean_vec = np.mean(sentences_vec, axis=0)
    text_mean_diff_vec = np.subtract(text_mean_vec, summary_vec)

    sentences_dist = []
    sentence_idx = 0

    # For each sentence, compute offset in relation to ground-truth summary
    for sentence_vec in sentences_vec:
        sentence_vec_dist = np.linalg.norm((text_mean_vec, np.add(sentence_vec, text_mean_diff_vec)))

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
    sentence_norm_idx = sentences_dist[0][1]
    common.log_message("INFO", str(sentences_dist[0]))
    texts_list_stat.append(sentences_dist[0])
    sentences_dist.sort(key=lambda x: x[4][0], reverse=True)

    try:
        for i in range(0, n_top):
            if sentence_norm_idx == sentences_dist[i][1]:
                sentences_top_counter[i] += 1
    except IndexError:
        common.log_message("ERROR", str(file) + "  -- Error computing top sentences.")


    texts_list_stat_rouge.append(sentences_dist[0])
    common.log_message("INFO", "BEST ROUGE SCORE = " + str(sentences_dist[0]))
    common.log_message("INFO", "\n")

    files_counter += 1

    if sentences_dist[0][4][0][0] == 0.0:
        common.log_message("INFO", "File " + file + " with rouge zero score.\nSENTENCES = " + \
                    str(sentences) + "\nSUMMARY = " + str(summary) + \
                    " CHOOSEN SUMMARY = " + str(sentences[sentences_dist[0][1]]))

    # Generate T-SNE representation of text
    sentences_vec_tsne = np.vstack((sentences_vec, text_mean_vec))
    U, s, Vh = np.linalg.svd(sentences_vec_tsne, full_matrices=False)

    first_sentence = True

    for i in range(len(sentences_vec)):
        fig = plt.gcf()
        fig.set_size_inches(5, 5)

        if i == len(sentences_vec) - 1:
            plt.plot(U[i,0], U[i,1], 'bs', label='text mean')
        elif i == sentence_norm_idx:
            continue
        else:
            # Plot remaining sentences
            if first_sentence == True:
                plt.plot(U[i,0], U[i,1], 'go', label='sentence')
                first_sentence = False
            else:
                plt.plot(U[i,0], U[i,1], 'go')
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))

        # plt.text(U[i,0], U[i,1], str(i), fontsize=6)
    plt.plot(U[sentence_norm_idx,0], U[sentence_norm_idx, 1], 'r^',
            label='choosen sentence')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05), ncol=3)
    plt.savefig("tsne/" + file + '.png', format='png')
    plt.clf()

#
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

for entry in texts_list_stat:
    rouge_1_list.append(entry[4][0])
    rouge_2_list.append(entry[4][1])
    rouge_l_list.append(entry[4][2])


# Consolidated list to compute mean
common.log_message("INFO", "\n")
common.log_message("INFO", "TOTAL OF FILES PROCESSED: " + str(files_counter))
common.log_message("INFO", "TOTAL MATCH ROUGE / OFFSET: " + str(total_match_choice))
# Consolidated list to compute mean
common.log_message("INFO", "VECTOR OFFSET MEAN")
common.log_message("INFO", "\tROUGE-1: " + str(np.mean(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.mean(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: " + str(np.mean(rouge_l_list, axis=0)))

common.log_message("INFO", "STD DEVIATION")
common.log_message("INFO", "\tROUGE-1: " + str(np.std(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.std(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: "+ str(np.std(rouge_l_list, axis=0)))



#
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

for entry in texts_list_stat_rouge:
    rouge_1_list.append(entry[4][0])
    rouge_2_list.append(entry[4][1])
    rouge_l_list.append(entry[4][2])

common.log_message("INFO", "BEST ROUGE SCORE MEAN")
common.log_message("INFO", "\tROUGE-1: " + str(np.mean(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.mean(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: " + str(np.mean(rouge_l_list, axis=0)))

common.log_message("INFO", "STD DEVIATION")
common.log_message("INFO", "\tROUGE-1: " + str(np.std(rouge_1_list, axis=0)))
common.log_message("INFO", "\tROUGE-2: " + str(np.std(rouge_2_list, axis=0)))
common.log_message("INFO", "\tROUGE-L: "+ str(np.std(rouge_l_list, axis=0)))

common.log_message("INFO", "TOP sentences selection: " + str(sentences_top_counter))
