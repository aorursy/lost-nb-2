#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[2]:


INPUT_WIDTH = 19
N_FEATURES = 22

# taken from http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
THRESHOLD = 73 


# In[3]:


train_df = pd.read_csv("../input/train.csv")
train_df[train_df.columns[1:]] = train_df[train_df.columns[1:]].astype(np.float32)
train_df.head(20)


# In[4]:


train_ids = train_df[~np.isnan(train_df.Ref)].Id.unique()
train_new = train_df[np.in1d(train_df.Id, train_ids)]
del train_df, train_ids
train_new.head()


# In[5]:


train_new = train_new.fillna(0.0)
train_new = train_new.reset_index(drop=True)
train_new.head()


# In[6]:


df_temp = pd.DataFrame(train_new.groupby('Id')['Expected'].mean()) # mean, or any value (the same for all)
meaningful_ids = np.array(df_temp[df_temp['Expected'] < THRESHOLD].index)
del df_temp

train_final = train_new[np.in1d(train_new.Id, meaningful_ids)]
del train_new, meaningful_ids
train_final.shape


# In[7]:


train_gp = train_final.groupby("Id")
train_size = len(train_gp)
del train_final

X_train = np.zeros((train_size, INPUT_WIDTH, N_FEATURES), dtype=np.float32)
y_train = np.zeros(train_size, dtype=np.float32)
seq_len_train = np.zeros(train_size, dtype=np.float32)

i = 0
for _, group in train_gp:
    X = group.values
    seq_len = X.shape[0]
    X_train[i,:seq_len,:] = X[:,1:23]
    y_train[i] = X[0,23]
    seq_len_train[i] = seq_len
    i += 1
    del X
    
del train_gp
X_train.shape, y_train.shape


# In[8]:


test_df = pd.read_csv("../input/test.csv")
test_df[test_df.columns[1:]] = test_df[test_df.columns[1:]].astype(np.float32)
test_ids = np.array(test_df.Id.unique())

# Convert all NaNs to zero
test_final = test_df.fillna(0.0)
test_final = test_final.reset_index(drop=True)
del test_df


# In[9]:


test_gp = test_final.groupby("Id")
test_size = len(test_gp)
del test_final

X_test = np.zeros((test_size, INPUT_WIDTH, N_FEATURES), dtype=np.float32)
seq_len_test = np.zeros(test_size, dtype=np.float32)

i = 0
for _, group in test_gp:
    X = group.values
    seq_len = X.shape[0]
    X_test[i,:seq_len,:] = X[:,1:23]
    seq_len_test[i] = seq_len
    i += 1
    del X
    
del test_gp
X_test.shape


# In[10]:


from keras.layers import Input, Dense, CuDNNLSTM, AveragePooling1D, TimeDistributed, Flatten, Bidirectional
from keras.models import Model


# In[11]:


from keras.callbacks import EarlyStopping
es_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


# In[12]:


BATCH_SIZE = 1024
N_EPOCHS = 32


# In[13]:


def get_model_simple(shape=(19,22)):
    inp = Input(shape)
    x = CuDNNLSTM(64, return_sequences=False)(inp)
    x = Dense(1)(x)

    model = Model(inp, x)
    return model


# In[14]:


model_0 = get_model_simple((19,22))
model_0.compile(optimizer='adadelta', loss='mae')
model_0.summary()


# In[15]:


model_0.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 
            validation_split=0.2, callbacks=[es_callback])


# In[16]:


y_pred_0 = model_0.predict(X_test)
submission_0 = pd.DataFrame({'Id': test_ids, 'Expected': y_pred_0.reshape(-1)})
submission_0.to_csv('submission_0.csv', index=False)


# In[17]:


def get_model_seq(shape=(19,22)):
    inp = Input(shape)
    x = CuDNNLSTM(64, return_sequences=True)(inp)
    x = TimeDistributed(Dense(10))(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inp, x)
    return model


# In[18]:


model_1 = get_model_seq((19,22))
model_1.compile(optimizer='adadelta', loss='mae')
model_1.summary()


# In[19]:


model_1.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 
            validation_split=0.2, callbacks=[es_callback])


# In[20]:


y_pred_1 = model_1.predict(X_test)
submission_1 = pd.DataFrame({'Id': test_ids, 'Expected': y_pred_1.reshape(-1)})
submission_1.to_csv('submission_1.csv', index=False)


# In[21]:


def get_model_bilstm(shape=(19,22)):
    inp = Input(shape)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(inp)
    x = TimeDistributed(Dense(10))(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    model = Model(inp, x)
    return model


# In[22]:


model_2 = get_model_bilstm((19,22))
model_2.compile(optimizer='adadelta', loss='mae')
model_2.summary()


# In[23]:


model_2.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 
            validation_split=0.2, callbacks=[es_callback])


# In[24]:


y_pred_2 = model_2.predict(X_test)
submission_2 = pd.DataFrame({'Id': test_ids, 'Expected': y_pred_2.reshape(-1)})
submission_2.to_csv('submission_2.csv', index=False)


# In[25]:


def get_model_deep(shape=(19,22)):
    inp = Input(shape)
    x = Dense(16)(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = TimeDistributed(Dense(64))(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = TimeDistributed(Dense(1))(x)
    x = AveragePooling1D()(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inp, x)
    return model


# In[26]:


model_3 = get_model_deep((19,22))
model_3.compile(optimizer='adadelta', loss='mae')
model_3.summary()


# In[27]:


model_3.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 
            validation_split=0.2, callbacks=[es_callback])


# In[28]:


y_pred_3 = model_3.predict(X_test)
submission_3 = pd.DataFrame({'Id': test_ids, 'Expected': y_pred_3.reshape(-1)})
submission_3.to_csv('submission_3.csv', index=False)


# In[29]:


y_pred_avg = (y_pred_0 + y_pred_1 + y_pred_2 + y_pred_3) / 4
submission_avg = pd.DataFrame({'Id': test_ids, 'Expected': y_pred_avg.reshape(-1)})
submission_avg.to_csv('submission_avg.csv', index=False)

