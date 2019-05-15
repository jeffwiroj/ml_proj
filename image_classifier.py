from keras.applications.resnet50 import ResNet50
import gzip
import pandas as pd
import requests
from PIL import Image
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
from os import listdir
from keras.layers import Flatten,Activation,GlobalMaxPooling1D,GlobalAveragePooling2D
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,Conv1D,MaxPooling1D
import tensorflow
import keras
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
tensorflow.set_random_seed(1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def __read_all_images(src):
    files = listdir(src)
    images = {}
    i = 0
    total = len(files)
    for f in files:
        if not (f.endswith(".jpg")):
            continue
        im = Image.open(src + f)
        im = img_to_array(im)
        im = preprocess_input(im)
        images[f[:-4]] = im
        if i % 100 == 0:
            print(str(i) + " / " + str(total))
        i += 1
    return images


def read_all_data(p):
    img_src = "images/"

    df = pd.read_pickle("frame_no_stem.pkl")
    images = __read_all_images(img_src) 
    print("Finished reading images")

    x_images = []
    x_desc = []
    y_category = []
    all_categories = set()

    for asin in df.index.values:
        if asin in images:
            data = images[asin]
            x_images.append(data)

            item = df.loc[asin]
            x_desc.append(item.description)
            cate = item.categories
            y_category.append(cate)
            for c in cate:
                all_categories.add(c)

    print("Finished reading dataframe")
    mlb = MultiLabelBinarizer()
    y_total = mlb.fit_transform(y_category)
    x_images = np.array(x_images)
    x_desc = np.array(x_desc)

    
    return x_images,x_desc, y_total


x_images,x_desc, y_total = read_all_data("resources/")

np.random.seed(0)
state = np.random.get_state()
np.random.shuffle(x_images)
np.random.set_state(state)
np.random.shuffle(y_total)
np.random.set_state(state)
np.random.shuffle(x_desc)

x_imageTest = x_images[90000:]
y_test = y_total[90000:]
x_imageTrain = x_images[:90000]
y_train = y_total[:90000]

model_resnet = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))
x = model_resnet.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(122, activation='sigmoid')(x)
model_image = keras.Model(inputs=model_resnet.input, outputs=predictions)
for x in model_image.layers[0:-24]:
    x.trainable = False
    
model_image.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_image.summary()

model_image.fit(x_imageTrain, y_train, epochs = 25, batch_size = 256)

outcome = model_image.predict(x_imageTrain)

outcome[outcome >=0.5] = 1
outcome[ outcome < 0.5] = 0
outcome = outcome.astype(int)

print(sklearn.metrics.f1_score(outcome,y_train,average='micro')*100)