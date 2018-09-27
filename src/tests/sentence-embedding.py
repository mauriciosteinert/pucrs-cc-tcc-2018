#!/usr/bin/python3 -i

# Simple experiment of reading dataset directory and generate embeddings

import sys
sys.path.append("../")

from common import *

common = Common()
common.parse_cmd_args()


if common.config.word_vector_model != None:
    common.load_word_vec_model()
    print(common.model.embed_sentence("Hello everyone!"))
else:
    print("No word vector model defined! Aborting ...")
    exit()
