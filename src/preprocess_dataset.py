#!/usr/bin/python3


import sys
sys.path.append("../")

import os
from common import *
import numpy as np
import sent2vec


common = Common()

# Preprocess data files
common.parse_cmd_args()
common.dataset_to_npz()
