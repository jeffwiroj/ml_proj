import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.models import Model
from keras import layers
from keras import backend as K
import tensorflow_hub as hub
from os import listdir
from keras.layers import Flatten,Activation,GlobalMaxPooling1D
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

df = pd.read_pickle('frame_no_stem.pkl')

# preprocess the title
def preprocess_sentence(sent):
    words = word_tokenize(sent)
    words = [w.lower() for w in words if w.isalpha() and len(w) < 30]
    return words

X = []
y = []

i = 0
total = len(df.index.values)
images = set([f[:-4] for f in listdir("images/") if f.endswith(".jpg")])
for asin in df.index.values:
    if asin in images:
        item = df.loc[asin]
        X.append(preprocess_sentence(item.title))
        y.append(item.categories)
        if i % 1000 == 0:
            print(str(i) + " / " + str(total))
        i += 1
        
X = np.array(X)
y = np.array(y)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

np.random.seed(0)
state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)

X_train = X[:90000]
y_train = y[:90000]
y_test = y[90000:]
x_test = X[90000:]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
training_dataX = tokenizer.texts_to_sequences(X_train)
test_dataX = tokenizer.texts_to_sequences(x_test)
vocab_size = len(tokenizer.word_index) + 1

training_dataX = np.array(training_dataX)
test_dataX = np.array(test_dataX)
y_train = np.array(y_train)
y_test = np.array(y_test)

maxlen = 54
training_dataX = pad_sequences(training_dataX, padding='post', maxlen=maxlen)
test_dataX = pad_sequences(test_dataX, padding='post', maxlen=maxlen)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            try:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
            except:
                continue

    return embedding_matrix

embedding_dim = 120
embedding_matrix = create_embedding_matrix("glove.840B.300d.txt", tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size) # the coverage

embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True)
sequence_input = keras.layers.Input(shape=(maxlen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)
layer = Conv1D(200, 5, activation='relu')(embedded_sequences)
layer = GlobalMaxPooling1D()(embedded_sequences)
layer = Dense(170, activation='relu')(layer)
layer = Dense(122,name='out_layer',activation = "sigmoid")(layer)
model = Model(sequence_input, layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.summary()

model.fit(np.array(training_dataX), y_train,epochs = 30, batch_size=256,verbose = 1)

outcome = model.predict(test_dataX)

outcome[outcome >= 0.5] = 1
outcome[outcome < 0.5] = 0
outcome= outcome.astype(int)

print(f1_score(y_test,np.array(outcome),average = 'micro')*100)