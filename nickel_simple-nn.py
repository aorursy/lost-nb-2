#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (30, 15)
plt.rcParams['font.size'] = 25


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


path = ["../input/train.csv", "../input/test.csv"]
data = pd.concat(pd.read_csv(p, parse_dates=["activation_date"]) for p in path)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


control = pd.concat((data[c].value_counts() for c in ["param_1", "param_2", "param_3"]), axis=1)


# In[ ]:


control.sort_values("param_1", ascending=False)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.price.fillna(data.price.mean(), inplace=True)


# In[ ]:


data.dtypes


# In[ ]:


pd.concat([data.dtypes, data.apply(lambda x: x.unique().shape[0])], axis=1)


# In[ ]:


data.drop(["image", "title", "description"], axis=1, inplace=True)
data.set_index("item_id", inplace=True)
pd.concat([data.dtypes, data.apply(lambda x: x.unique().shape[0])], axis=1)


# In[ ]:


control.isnull().sum(axis=1).value_counts()


# In[ ]:


cat_cols = data.select_dtypes("object").columns.tolist() + ["image_top_1"]
for c in cat_cols:
    data[c].fillna("NA", inplace=True)
    data[c] = pd.factorize(data[c])[0]
cat_cols


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


from keras.layers import Input, Dense, Embedding, concatenate, Flatten, PReLU, Dropout, BatchNormalization
from keras.models import Model
from keras import optimizers


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Model')


# In[ ]:


inputs = []
embeddings = []
for c in cat_cols:
    size = data[c].unique().shape[0]
    inp = Input((1,), name=c)
    inputs.append(inp)
    emb = Flatten()(Embedding(size, output_dim=min(max(int(size ** 0.25), 1), 4))(inp))
    embeddings.append(emb)
numeric = Input((2,), name="numeric")
inputs.append(numeric)


# In[ ]:


final = concatenate(embeddings + [numeric])
# final = Dense(256, activation="relu")(final)
# final = Dense(128, activation="relu")(final)
final = Dense(64, activation="relu")(final)
final = Dense(32, activation="relu")(final)
final = Dense(1)(final)
model = Model(inputs=inputs, outputs=final)
model.compile(optimizer="rmsprop", loss='mean_squared_error')
model.summary()


# In[ ]:


cat_cols = [c for c in cat_cols if c != "user_id"]


# In[ ]:


cat_cols


# In[ ]:


inputs = []
embeddings = []
for c in cat_cols:
    size = data[c].unique().shape[0]
    inp = Input((1,), name=c)
    inputs.append(inp)
    emb = Flatten()(Embedding(size, output_dim=min(max(int(size ** 0.25), 1), 4))(inp))
    embeddings.append(emb)
numeric = Input((2,), name="numeric")
inputs.append(numeric)
final = concatenate(embeddings + [numeric])
# final = Dense(256, activation="relu")(final)
# final = Dense(128, activation="relu")(final)
final = Dense(64, activation="relu")(final)
final = Dense(32, activation="relu")(final)
final = Dense(1)(final)
model = Model(inputs=inputs, outputs=final)
model.compile(optimizer="rmsprop", loss='mean_squared_error')
model.summary()


# In[ ]:


train = data[data.deal_probability.notnull()]
test = data.drop(train.index)
X = {k: train[[k]].values for k in cat_cols}
numeric = train[["price", "item_seq_number"]]
numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)
X["numeric"] = numeric
y = train["deal_probability"].values

X_test = {k: test[[k]].values for k in cat_cols}
numeric = test[["price", "item_seq_number"]]
numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)
X_test["numeric"] = numeric


# In[ ]:


model.fit(X, y, batch_size=10000, epochs=30, validation_split=0.1)


# In[ ]:


inputs = []
embeddings = []
for c in cat_cols:
    size = data[c].unique().shape[0]
    inp = Input((1,), name=c)
    inputs.append(inp)
    emb = Flatten()(Embedding(size, output_dim=min(max(int(size ** 0.25), 1), 4))(inp))
    embeddings.append(emb)
numeric = Input((2,), name="numeric")
inputs.append(numeric)
final = concatenate(embeddings + [numeric])
# final = Dense(256, activation="relu")(final)
# final = Dense(128, activation="relu")(final)
final = BatchNormalization()(final)
final = Dropout(0.5)(final)
final = Dense(64)(final)
final = PReLU()(final)
final = BatchNormalization()(final)
final = Dropout(0.5)(final)
final = Dense(32)(final)
final = PReLU()(final)
final = BatchNormalization()(final)
final = Dropout(0.5)(final)
final = Dense(1)(final)
model = Model(inputs=inputs, outputs=final)
model.compile(optimizer="rmsprop", loss='mean_squared_error')
model.summary()


# In[ ]:


model.fit(X, y, batch_size=1000, epochs=30, validation_split=0.1)


# In[ ]:


preds = model.predict(X_test)[:, -1]
preds = pd.Series(preds, index=test.index, name="deal_probability").clip(0, 1)
preds.to_csv("preds.csv", index=True, header=True)


# In[ ]:


get_ipython().system('head preds.csv')


# In[ ]:




