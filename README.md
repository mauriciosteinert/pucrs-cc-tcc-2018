About
-----

This repository document all experiments developed for final assignment graduating program in Computer Science at Pontificia Universidade Catolica do Rio Grande do Sul.

Title: Automating news summarization with Deep Learning

Advisor: Felipe Rech Meneguzzi


Requirements:
* Python 3
* Numpy
* Keras with Tensorflow backend
* sent2vec (https://github.com/epfml/sent2vec)
* Natural Language Toolkit (nltk)
* ROUGE (https://github.com/pltrdy/rouge)
* Data set: CNN and DailyMail story files (https://cs.nyu.edu/~kcho/DMQA/)

How to use
----------

At command line:

1. Store pre-processing results (sentence vectors and ROUGE scores) in NUMPY files:

python3 preprocess_data2.py --session-name "preprocess-data2-1000" --dataset-dir ../datasets/all --working-dir ../datasets/npz2 --word-vector-model ../datasets/word_vector/enwiki_sent2vec_100.bin --word-vector-dim 100 --process-n-examples 4000 --dataset-chunk-size 1000

2. For Vector Offset Summarization:

python3 model_offset2.py --session-name "model_offset2_select_tsne" --dataset-dir ../datasets/all --working-dir ../logs --word-vector-model ../datasets/word_vector/enwiki_sent2vec_100.bin --word-vector-dim 100


3. For Feed-forward experiments (training and evaluation over test set):

python3 extractive-ff-keras.py --session-name "extractive-keras-9-test" --dataset-dir ../npz --working-dir ../logs --word-vector-dim 100 --batch-size 200 --num-epochs 100


4. For Recurrent Neural Network experiments (training and evaluation over test set):

python3 -i recurrent-ff.py --session-name "recurrent-ff-20000-d100_out_of_range" --working-dir ../logs --word-vector-dim 100 --num-epochs 100 --batch-size 2000 --dataset-dir ../datasets/all
