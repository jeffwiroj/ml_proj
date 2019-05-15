from keras.applications.resnet50 import ResNet50
import gzip
import pandas as pd
import requests
import io
from os import listdir
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from keras.utils import multi_gpu_model

from keras.models import Sequential,Model
from keras import layers
from keras import backend as K
import tensorflow_hub as hub
# nltk.download("tokenize")
from os import listdir
from keras.layers import Flatten,Activation,GlobalMaxPooling1D,GlobalAveragePooling2D
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,Conv1D,MaxPooling1D
import tensorflow
import keras
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=25, inter_op_parallelism_threads=25)
tensorflow.set_random_seed(1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
import tensorflow as tf


df = pd.read_pickle("frame_no_stem.pkl")

y = []

i = 0
total = len(df.index.values)
images = set(np.load('asin.npy'))
for asin in df.index.values:
    if asin in images:
        item = df.loc[asin]
        y.append(item.categories)
        if i % 1000 == 0:
            print(str(i) + " / " + str(total))
        i += 1
        
y = np.array(y)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

np.random.seed(0)
np.random.shuffle(y)

y_test = y[90000:]

desc_prob = np.loadtxt('desc_prob')
image_prob = np.loadtxt('image_prob')
title_prob = np.loadtxt('title_prob')
desc_prob[desc_prob > 0.5] = 1
desc_prob = desc_prob.astype(int)
image_prob[image_prob > 0.5] = 1
image_prob = image_prob.astype(int)
title_prob[title_prob > 0.5] = 1
title_prob = title_prob.astype(int)


'''
a product is misclassified if the true class has not been predicted
'''
def countMiss(predicted, true_prob):
    cnt = np.zeros(122)
    for i in range(len(predicted)):
        r = true_prob[i] - predicted[i]
        r[r <= 0] = 0
        cnt += r
    return cnt


def getInfos(arr):
    total = np.sum(y_test, axis=0)
    misses = countMiss(arr, y_test)
    missesPercentage = misses / total * 100
    classes = mlb.classes_[:]
    missesPercentage, total, misses, classes = zip(*sorted(zip(missesPercentage, total, misses, classes)))
    return missesPercentage[::-1], total[::-1], misses[::-1], classes[::-1]


missesPercentage, total, misses, classes = getInfos(desc_prob)
for i in range(122):
    print("%-3d" % i, "%-50s" % classes[i], "%-5d/%-5d" % (misses[i], total[i]), "%.2f" % missesPercentage[i])
