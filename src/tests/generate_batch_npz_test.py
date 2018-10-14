# Test
import numpy as np

x=np.array([
np.array([[1,2,3], [4,5,6], [7,8,9]]),
np.array([[10,11,12], [13,14,15]]),
np.array([[16,17,18]])
])
np.savez("file0000", x=x)


x=np.array([
np.array([[19,20,21], [22,23,24]]),
np.array([[25,26,27], [28,29,30], [31,32,33]]),
np.array([[34,35,36]])
])
np.savez("file0001", x=x)



x=np.array([
np.array([[37,38,39]]),
np.array([[40,41,42], [43,44,45], [46,47,48], [49,50,51]]),
np.array([[52,53,54], [55,56,57]]),
])
np.savez("file0002", x=x)


x=np.array([
np.array([[58,59,60], [61,62,63], [64,65,66]]),
np.array([[67,68,69]]),
])
np.savez("file0003", x=x)

np.savez("file-metadata", longest_sentence=4, files_counter=11)
