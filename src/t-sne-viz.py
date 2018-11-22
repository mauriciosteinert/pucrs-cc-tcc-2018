import sent2vec
import numpy as np
import matplotlib.pyplot as plt


model = sent2vec.Sent2vecModel()
model.load_model("../datasets/word_vector/enwiki_sent2vec_100.bin")


words = [
"brazil", "australia", "japan", "italy", "mexico",
"gold", "silver",
"frodo", "sauron", "gandalf", "aragorn", "gimli", "legolas", "bilbo", "saruman", "galadriel", "gollum",
"thor", "freya", "odin", "loki",
"king", "queen", "princess", "prince",
"dog", "cat", "horse", "bull"
]

words_embed = model.embed_sentences(words)
U, s, Vh = np.linalg.svd(words_embed, full_matrices=False)

for i in range(len(words_embed)):
  fig = plt.gcf()
  fig.set_size_inches(20, 20)
  plt.scatter(U[i,0], U[i,1])
  plt.text(U[i,0], U[i,1], words[i], fontsize=32)
  plt.xlim((-0.43, 0.05))
  plt.ylim((-0.14, 0.45))

plt.savefig('viz.pdf')
