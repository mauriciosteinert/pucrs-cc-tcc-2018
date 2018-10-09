from common import *

common = Common()
common.parse_cmd_args()

files_list = common.get_dataset_files()
longest_sentence = 0
longest_file = ""
longest_sentences = []

for file in files_list:
    text = open(common.config.dataset_dir + "/" + file, "r").read()
    summary, sentences = common.text_to_sentences2(text)

    if len(sentences) == 0:
        continue

    curr_sentence_len = len(sentences)

    print("Processing " + file + " sentence len = " + str(curr_sentence_len))

    if curr_sentence_len > longest_sentence:
        longest_sentence = curr_sentence_len
        longest_file = file
        longest_sentences = sentences
