#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.concat([
       pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"], usecols=["USER_ID", "FEC_EVENT", "PAGE"]),
       pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"], usecols=["USER_ID", "FEC_EVENT", "PAGE"])
])

data["day"] = data.FEC_EVENT.dt.dayofyear - 1


# In[3]:


y_prev = pd.read_csv("../input/conversiones/conversiones.csv")
y_train = pd.Series(0, index=sorted(data.USER_ID.unique()))
y_train.loc[y_prev[y_prev.mes >= 10].USER_ID.unique()] = 1


# In[4]:


data.shape, data.PAGE.nunique()


# In[5]:


# pages = data[data.FEC_EVENT.dt.month < 10].groupby("PAGE").USER_ID.unique()
# pages = pages.index[pages.apply(lambda x: y_train.loc[x].mean() / y_train.mean() - 1).abs() > 0.05]
# data = data[data.PAGE.isin(pages)]
# data.shape, data.PAGE.nunique()


# In[6]:


# pages = data.PAGE.value_counts()
# pages = pages.index[pages < pages.iloc[int(pages.shape[0] * 0.1)]]
# data = data[data.PAGE.isin(pages)]
# data.shape, data.PAGE.nunique()                     


# In[7]:


# data["PAGE"] = pd.factorize(data.PAGE)[0]
npages = data.PAGE.max() + 1


# In[8]:


history = 60


# In[9]:


def dataPrep(X, nusers, history, npages):
    data = X.copy()
    data["day"] = data.day - data.day.max() + history - 1
    data = data[data.day >= 0]
    data = data.groupby(["USER_ID", "day", "PAGE"]).size().rename("cantidad").reset_index().set_index("USER_ID")
    data = data.astype(np.int32)
    res = np.zeros((nusers, history, npages), dtype=np.float32)
    for user in range(nusers):
        if user in data.index:
            d = data.loc[user]
            res[user, d.day, d.PAGE] = d.cantidad
            res[user] /= (res[user].sum(axis=1) + 1e-16)[: , None]
    return res


# In[10]:


try:
    del X_train, X_test
    gc.collect()
except: pass

X_train = dataPrep(data[data.FEC_EVENT.dt.month < 10], data.USER_ID.max() + 1, history, npages)
X_test = dataPrep(data, data.USER_ID.max() + 1, history, npages)
gc.collect()


# In[11]:


from keras.models import Model, Sequential
from keras.layers import Input, Dense, CuDNNLSTM, add, concatenate, Dropout, Multiply, multiply
from keras.callbacks import EarlyStopping


# In[12]:


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
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
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[13]:


from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import CuDNNLSTM, Bidirectional, Dropout

import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def BidLstm(lstm_size=64):
    inp = Input(shape=(history, npages))
    x = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True))(inp)
    x = Attention(history)(x)
    x = Dense(lstm_size * 2, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auroc])
    return model


# In[14]:


model = BidLstm()
model.summary()


# In[15]:


model.fit(X_train, y_train, batch_size=320, epochs=1000, verbose=1,
             validation_split=0.1,
             callbacks=[EarlyStopping(monitor='val_auroc', patience=10,
                                      verbose=1, mode='max', restore_best_weights=True)])


# In[16]:


test_probs = []
for i in range(10):
    model = model = BidLstm()
    model.fit(X_train, y_train, batch_size=320, epochs=1000, verbose=1,
              validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_auroc', patience=10,
                                       verbose=1, mode='max', restore_best_weights=True)])
    
    test_probs.append(pd.Series(model.predict(X_test)[:, -1], name="fold_" + str(i)))

test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
test_probs.index.name="USER_ID"
test_probs.name="SCORE"

test_probs.to_csv("rnn_benchmark.zip", header=True, compression="zip")


# In[ ]:




