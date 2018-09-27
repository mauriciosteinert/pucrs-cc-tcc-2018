#!/usr/bin/python3
#
# Summarization method that computes offset between each sentences and ground-truth
# summary. The sentence with minor offset is selected as summary.


from common import Common
import nltk



print("Initiating model offset ...")
common = Common()
common.parse_cmd_args()
# common.load_word_vec_model()

for file in common.get_dataset_files():
    print(file)
