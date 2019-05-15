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

import matplotlib.pyplot as plt
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=25, inter_op_parallelism_threads=25)
tensorflow.set_random_seed(1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
import tensorflow as tf

categoryNums = {}

i = 0
total = len(df.index.values)
images = set(np.load('asin.npy'))
for asin in df.index.values:
    if asin in images:
        item = df.loc[asin]
        for cat in item.categories:
            if cat in categoryNums:
                categoryNums[cat] += 1
            else:
                categoryNums[cat] = 0
        
        
        if i % 1000 == 0:
            print(str(i) + " / " + str(total))
        i += 1
        
sortedCat = dict(sorted(categoryNums.items(), key=lambda kv: kv[1])[::-1])  # reverse sorting
data = [i[1] for i in sortedCat.items()]

plt.figure(figsize=(10,4))

plt.xlabel("Number of products in each category")
plt.ylabel("Number of categories")
plt.grid(True)

plt.hist(data, bins=np.linspace(558, 37102, 20) , log=True)
plt.savefig("logplot.pdf", dpi=200, format='pdf')
