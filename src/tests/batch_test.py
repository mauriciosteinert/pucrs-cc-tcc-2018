import sys
sys.path.append("../../lib/python")
import numpy as np
from common import *



common = Common()
common.parse_cmd_args()

common.prepare_batch()

# x, y = common.get_next_train_batch()
# print(x)
