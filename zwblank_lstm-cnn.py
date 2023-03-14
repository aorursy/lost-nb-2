#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from random import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder


# In[2]:


train = pd.read_csv("../input/finaltrain/newTRAIN.csv")
#train = pd.read_csv("../input/ndsczw/train.csv")
val = pd.read_csv("../input/finaltrain/newVAL.csv")
test = pd.read_csv("../input/ndsczw/test.csv")

def dataset_split(data):
    df_beauty=data[data["Category"]<=16]
    df_fashion=data[data['Category'].between(17, 30, inclusive=True)]
    df_mobile=data[data['Category'].between(31, 57, inclusive=True)]
    return df_beauty

def shuffle_data(df):
    def shuffle_string (string):
        listString = string.split(" ")
        shuffle(listString)
        return " ".join(listString)
    
    trainAppend = df.copy()
    trainAppend.title = df.title.apply(shuffle_string)
    expandedDf = pd.concat([df,trainAppend],ignore_index=True)
    return expandedDf


train = dataset_split(train)
train = shuffle_data(train)
val = dataset_split(val)
test = test[:76545]


# In[3]:


max_features = 90000
tk = Tokenizer(lower = True, filters='', num_words=max_features)
full_text = list(train['title'].values) + list(test["title"].values)
tk.fit_on_texts(full_text)


# In[4]:


train_tokenized = tk.texts_to_sequences(train['title'].fillna('missing'))
test_tokenized = tk.texts_to_sequences(test['title'].fillna('missing'))


# In[5]:


train['title'].apply(lambda x: len(x.split())).plot(kind='hist');
plt.yscale('log');
plt.title('Distribution of question text length in characters')


# In[6]:


max_len = 70
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_val = pad_sequences(tk.texts_to_sequences(val.title.fillna('missing')), maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)


# In[7]:


embedding_path = "../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt"
#embedding_path = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"


# In[8]:


embed_size = 300


# In[9]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
all_embs = np.stack(embedding_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tk.word_index
nb_words = max(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[10]:


ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(train['Category'].values.reshape(-1, 1))


# In[11]:


def build_model(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, weights=[embedding_matrix],trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
    
    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
    
    x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
    max_pool2_gru = GlobalMaxPooling1D()(x_conv2)
    
    
    x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x_conv3)
    max_pool1_lstm = GlobalMaxPooling1D()(x_conv3)
    
    x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool2_lstm = GlobalAveragePooling1D()(x_conv4)
    max_pool2_lstm = GlobalMaxPooling1D()(x_conv4)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool2_lstm, max_pool2_lstm])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    model.summary()
    history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.3, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model


# In[12]:



get_ipython().run_cell_magic('time', '', 'model1 = build_model(lr = 1e-4, lr_d = 0, units = 128, spatial_dr = 0.5, kernel_size1=4, kernel_size2=3, dense_units=58*2, dr=0.1, conv_size=16, epochs=5)')


# In[13]:


pred = model1.predict(X_test, batch_size = 1024, verbose = 1)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
print(np.unique(predictions))


# In[14]:


def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    
    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
    
    x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
    max_pool2_gru = GlobalMaxPooling1D()(x_conv2)

    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    model.summary()
    history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model


# In[15]:


def build_model2(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units * 2, return_sequences = True))(x1)
    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x_gru)
    
    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
    
    x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
    max_pool2_gru = GlobalMaxPooling1D()(x_conv2)
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    model.summary()
    history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model


# In[16]:


get_ipython().run_cell_magic('time', '', 'model2 = build_model2(lr = 1e-4, lr_d = 1e-7, units = 64, spatial_dr = 0.3, kernel_size1=4, kernel_size2=3, dense_units = 64, dr=0.1, conv_size=8, epochs=5)')


# In[17]:


model3 = build_model1(lr = 1e-4, lr_d = 1e-7, units = 256, spatial_dr = 0.1, kernel_size1=4, kernel_size2=3, dense_units = 64, dr=0.1, conv_size=16, epochs=5)


# In[18]:


pred = model3.predict(X_test, batch_size = 1024, verbose = 1)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
print(np.unique(predictions))


# In[19]:


pred = model2.predict(X_test, batch_size = 1024, verbose = 1)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
print(np.unique(predictions))


# In[20]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


# In[21]:


def build_model3(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):
    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units * 2, return_sequences = True))(x1)
    if use_attention:
        x_att = Attention(max_len)(x_gru)
        x = Dropout(dr)(Dense(dense_units, activation='relu') (x_att))
    else:
        x_att = Flatten() (x_gru)
        x = Dropout(dr)(Dense(dense_units, activation='relu') (x_att))

    x = BatchNormalization()(x)
    #x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    #model.summary()
    #history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.1, 
    #                    verbose = 1, callbacks = [check_point, early_stop])
    #model = load_model(file_path)
    return model


# In[22]:


get_ipython().run_cell_magic('time', '', 'file_path = "best_model.hdf5"\ncheck_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,\n                              save_best_only = True, mode = "min")\nearly_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)\nmodel4 = build_model3(lr = 1e-3, lr_d = 1e-7, units = 128, spatial_dr = 0.3, dense_units=58*2, dr=0.1, use_attention=True)\nhistory = model4.fit(X_train, y_ohe, batch_size = 512, epochs = 10, validation_split=0.1, \n                    verbose = 1, callbacks = [check_point, early_stop])')


# In[23]:


def build_model4(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    
    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
       
    x = concatenate([avg_pool1_gru, max_pool1_gru])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    #x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    model.summary()
    history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model


# In[24]:


get_ipython().run_cell_magic('time', '', 'model5 = build_model4(lr = 1e-4, lr_d = 1e-7, units = 128, spatial_dr = 0.3, kernel_size1=3, dense_units=58*2, dr=0.8, conv_size=8, epochs=10)')


# In[25]:


model6 = build_model4(lr = 1e-4, lr_d = 1e-7, units = 256, spatial_dr = 0.3, kernel_size1=4, dense_units=58*2, dr=0.1, conv_size=8, epochs=5)


# In[26]:


def build_model5(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.001)

    inp = Input(shape = (max_len,))
    x = Embedding(max_features + 1, embed_size, trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)
    x_m = Masking()(x1)
    x_gru = LSTM(units)(x_m)

    x = BatchNormalization()(x_gru)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    #x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(17, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    model.summary()
    history = model.fit(X_train, y_ohe, batch_size = 512, epochs = epochs, validation_split=0.3, 
                        verbose = 1, callbacks = [check_point, early_stop, reduce_lr])
    model = load_model(file_path)
    return model


# In[27]:


model7 = build_model5(lr = 1e-4, lr_d = 1e-7, units = 128, spatial_dr = 0.3, kernel_size1=4, dense_units=64, dr=0.8, conv_size=8, epochs=5)


# In[28]:


model8 = build_model5(lr = 1e-4, lr_d = 1e-7, units = 256, spatial_dr = 0.3, kernel_size1=4, dense_units=64, dr=1.2, conv_size=8, epochs=10)


# In[29]:


from nltk import word_tokenize
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{3,}',
            tokenizer=word_tokenize,  ngram_range=(1, 3), use_idf=1,
            smooth_idf=1,sublinear_tf=1, stop_words = 'english')


# In[30]:


tfv.fit(list(train.title.values) + list(val.title.values) + list(test.title.values))
xtrain = tfv.transform(train.title.values)
xval = tfv.transform(val.title.values)
xtest = tfv.transform(test.title.values)


# In[31]:


clf = xgb.XGBClassifier(max_depth=12, n_estimators=2000, colsample_bytree=0.8, random_state = 123, objective='multi:softmax',  num_class = 58, subsample=0.8, n_jobs=-1, learning_rate=0.1, silent = False)


# In[32]:


ytrain, yval = train.Category, val.Category
eval_set_beauty  = [(xtrain,ytrain), (xval,yval)]
clf.fit(xtrain, ytrain, eval_set = eval_set_beauty, eval_metric=['merror'], early_stopping_rounds=40)
print('Beauty Accuracy:', accuracy_score(ytrain, clf.predict(xtrain)))
print('Beauty Cross Validation Accuracy:', accuracy_score(yval, clf.predict(xval)))
#beauty_preds = clf_beauty.predict(test_beauty_tfv, ntree_limit=1000)
xgproba = clf.predict_proba(xtrain, ntree_limit=1000)


# In[33]:


pred1 = model1.predict(X_train, batch_size = 1024, verbose = 1)
pred2 = model2.predict(X_train, batch_size = 1024, verbose = 1)1
pred3 = model3.predict(X_train, batch_size = 1024, verbose = 1)
pred4 = model4.predict(X_train, batch_size = 1024, verbose = 1)
pred5 = model5.predict(X_train, batch_size = 1024, verbose = 1)
pred6 = model6.predict(X_train, batch_size = 1024, verbose = 1)
pred7 = model7.predict(X_train, batch_size = 1024, verbose = 1)
pred8 = model8.predict(X_train, batch_size = 1024, verbose = 1)
xgpred = clf.predict(X_train)
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(xgpred.values.reshape(-1, 1))
#pred = pred / 8


# In[34]:


from sklearn.metrics import accuracy_score
import xgboost as xgb


predictions = np.round(np.argmax(pred, axis=1)).astype(int)
x_ensemble = np.concatenate((pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,y_ohe),axis=1)
gbm = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, colsample_bytree=0.8, objective= 'multi:softmax', nthread= -1, scale_pos_weight=1)
gbm.fit(x_ensemble, train.Category)
ensemble_predictions = gbm.predict(x_ensemble)
print(accuracy_score(ensemble_predictions, train.Category))


# In[35]:


pred1 = model1.predict(X_val, batch_size = 1024, verbose = 1)
pred = pred1
pred2 = model2.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred2
pred3 = model3.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred3
pred4 = model4.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred4
pred5 = model5.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred5
pred6 = model6.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred6
pred7 = model7.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred7
pred8 = model8.predict(X_val, batch_size = 1024, verbose = 1)
pred += pred8
xgpred = clf.predict(X_val)
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(xgpred.values.reshape(-1, 1))
#pred = pred / 8
pred_ensemble = np.concatenate(pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8)
predictions = np.round(np.argmax(pred,axis=1)).astype(int)
print(accuracy_score(predictions, val.Category))
x_ensemble = np.concatenate((pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,y_ohe),axis=1)
ensemble_predictions = gbm.predict(x_ensemble)
print(accuracy_score(ensemble_predictions, val.Category))


# In[36]:


pred1 = model1.predict(X_test, batch_size = 1024, verbose = 1)
pred = pred1
pred2 = model2.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred2
pred3 = model3.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred3
pred4 = model4.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred4
pred5 = model5.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred5
pred6 = model6.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred6
pred7 = model7.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred7
pred8 = model8.predict(X_test, batch_size = 1024, verbose = 1)
pred += pred8
xgpred = clf.predict(X_val)
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(xgpred.values.reshape(-1, 1))


# In[37]:


x_ensemble = np.concatenate((pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,y_ohe),axis=1)
ensemble_predictions = gbm.predict(x_ensemble)


# In[38]:


pd.DataFrame(ensemble_predictions).to_csv("lstm_cnn_submission.csv",index=False)

