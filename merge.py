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

from keras.models import Sequential,Model, Input
from keras import layers
from keras import backend as K
import tensorflow_hub as hub
# nltk.download("tokenize")
from os import listdir
from keras.layers import Flatten,Activation,GlobalMaxPooling1D,GlobalAveragePooling2D, Conv2D
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,Conv1D,MaxPooling1D
import tensorflow
import keras
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tensorflow.set_random_seed(1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

df = pd.read_pickle('frame_no_stem.pkl')

y = []

i = 0
total = len(df.index.values)
images = set(np.load('asin.npy'))
for asin in df.index.values:
    if asin in images:
        item = df.loc[asin]
        y.append(item.categories)
    #     df.set_value(asin, 'description', s)
        if i % 1000 == 0:
            print(str(i) + " / " + str(total))
        i += 1
        
y = np.array(y)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

np.random.seed(0)
np.random.shuffle(y)

y_test = y[90000:]
y_train = y[:90000]

image_outcome = np.loadtxt("image_prob")
desc_outcome = np.loadtxt("desc_prob")
title_outcome = np.loadtxt("title_prob")
image_train = np.loadtxt('image_prob_train')
desc_train = np.loadtxt('desc_prob_train')
title_train = np.loadtxt('title_prob_train')

# Average
merge = np.average([image_outcome, desc_outcome, title_outcome], axis=0)
merge[merge>=0.5] = 1
merge = merge.astype(int)
print("Average:", sklearn.metrics.f1_score(merge,y_test,average = 'micro')*100)


# Regression
total_train = np.c_[image_train, desc_train, title_train]
model = LinearRegression(normalize=True)
model.fit(total_train, y_train)
total_test = np.c_[image_outcome, desc_outcome, title_outcome]
total_outcome = model.predict(total_test)
total_outcome[total_outcome >= 0.5] = 1
total_outcome = total_outcome.astype(int)
print("Linear Regression:", sklearn.metrics.f1_score(total_outcome,y_test,average = 'micro')*100)

# Neural Network
input_im = Input(shape=(122,))
input_de = Input(shape=(122,))
input_ti = Input(shape=(122,))

combined = keras.layers.concatenate([input_im, input_de, input_ti])
z = Dense(330, activation = 'sigmoid')(combined)
z = Dense(270, activation = 'tanh')(z)
z = Dense(122, activation = 'sigmoid')(z)

model = Model(inputs=[input_im, input_de, input_ti], outputs=z)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([image_train, desc_train, title_train], y_train, epochs=20, batch_size=256)

y_out = model.predict([image_outcome, desc_outcome, title_outcome])
y_out[y_out > 0.5] = 1
y_out = y_out.astype(int)
print("NN:", f1_score(y_out, y_test, average='micro'))
