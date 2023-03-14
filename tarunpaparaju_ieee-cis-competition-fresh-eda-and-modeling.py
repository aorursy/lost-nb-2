#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import SVG

import warnings
warnings.filterwarnings('ignore')

import lightgbm
import xgboost
import catboost

import keras
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler


# In[2]:


os.listdir('../input/ieee-fraud-detection')


# In[3]:


DATA_PATH = '../input/ieee-fraud-detection/'
TRAIN_PATH = DATA_PATH + 'train_transaction.csv'
TEST_PATH = DATA_PATH + 'test_transaction.csv'


# In[4]:


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# In[5]:


train_df = train_df.fillna(0.0)
train_df.head()


# In[6]:


plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13


# In[7]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.violinplot(x="isFraud", y="TransactionAmt", data=train_df.query("TransactionAmt < 1000"), palette=["limegreen", "dodgerblue"], linewidth=2
                      , ax=ax).set_title('TransactionAmt', fontsize=16)
plt.show()


# In[8]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.boxplot(x="isFraud", y="TransactionAmt", data=train_df.query("TransactionAmt < 500"), palette=["limegreen", "red"], ax=ax).set_title('TransactionAmt', fontsize=16)
plt.show()


# In[9]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="ProductCD", data=train_df, palette=reversed(['aquamarine', 'mediumaquamarine', 'mediumseagreen', 'seagreen', 'darkgreen'])).set_title('ProductCD', fontsize=16)
plt.show(plot)


# In[10]:


fig, ax = plt.subplots(figsize=(10, 10))

props = train_df.query("TransactionAmt < 500")                .groupby("ProductCD")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lightgreen', 'green'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[11]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.violinplot(x="ProductCD", y="TransactionAmt", hue="isFraud",
                      data=train_df.query('TransactionAmt < 500'), palette=['lightgreen', 'green'],
                      split=True, ax=ax).set_title('ProductCD vs. TransactionAmt', fontsize=16)

plt.show(plot)


# In[12]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="ProductCD", y="TransactionAmt", hue="isFraud",
                   data=train_df.query('TransactionAmt < 500'), palette=['lightgreen', 'green'],
                   ax=ax).set_title('ProductCD vs. TransactionAmt', fontsize=16)

plt.show(plot)


# In[13]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="P_emaildomain", data=train_df.query("P_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                     palette=['navy', 'darkblue', 'blue', 'dodgerblue', 'skyblue']).set_title('P_emaildomain vs. TransactionAmt', fontsize=16)
plt.show(plot)


# In[14]:


fig, ax = plt.subplots(figsize=(10, 10))

props = train_df.query("P_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500")                .groupby("P_emaildomain")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lightblue', 'darkblue'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[15]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.violinplot(x="P_emaildomain", y="TransactionAmt", hue="isFraud",
                      data=train_df.query("P_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                      palette=['lightblue', 'darkblue'], split=True, ax=ax).set_title('TransactionAmt vs. P_emaildomain', fontsize=16)

plt.show(plot)


# In[16]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="P_emaildomain", y="TransactionAmt", hue="isFraud",
                   data=train_df.query("P_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                   palette=['lightblue', 'darkblue'], ax=ax).set_title('TransactionAmt vs. P_emaildomain', fontsize=16)

plt.show(plot)


# In[17]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="R_emaildomain", data=train_df.query("R_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                     palette=['red', 'crimson', 'mediumvioletred', 'darkmagenta', 'indigo']).set_title('R_emaildomain', fontsize=16)
plt.show(plot)


# In[18]:


fig, ax = plt.subplots(figsize=(10, 10))

props = train_df.query("R_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500")                .groupby("R_emaildomain")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['pink', 'crimson'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[19]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.violinplot(x="R_emaildomain", y="TransactionAmt", hue="isFraud",
                      data=train_df.query("R_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                      palette=['pink', 'crimson'], split=True, ax=ax).set_title('TransactionAmt vs. R_emaildomain', fontsize=16)

plt.show(plot)


# In[20]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="R_emaildomain", y="TransactionAmt", hue="isFraud",
                   data=train_df.query("R_emaildomain in ['gmail.com', 'yahoo.com', 0.0, 'hotmail.com', 'anonymous.com']").query("TransactionAmt < 500"),
                   palette=['pink', 'crimson'], ax=ax).set_title('TransactionAmt vs. R_emaildomain', fontsize=16)

plt.show(plot)


# In[21]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="card4", data=train_df.query("TransactionAmt < 500"),
                     palette=reversed(['orangered', 'darkorange', 'orange', 'peachpuff', 'navajowhite'])).set_title('card4', fontsize=16)
plt.show(plot)


# In[22]:


fig, ax = plt.subplots(figsize=(10, 10))

props = train_df.query("TransactionAmt < 500")                .groupby("card4")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['peachpuff', 'darkorange'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[23]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.violinplot(x="card4", y="TransactionAmt", hue="isFraud",
                      data=train_df.query("TransactionAmt < 500"),
                      palette=['peachpuff', 'darkorange'], split=True, ax=ax).set_title('TransactionAmt vs. card4', fontsize=16)

plt.show(plot)


# In[24]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="card4", y="TransactionAmt", hue="isFraud",
                   data=train_df.query("TransactionAmt < 500"),
                   palette=['peachpuff', 'darkorange'], ax=ax).set_title('TransactionAmt vs. card4', fontsize=16)

plt.show(plot)


# In[25]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="card6", data=train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'"),
                     palette=reversed(['red', 'crimson', 'mediumvioletred', 'darkmagenta', 'indigo'])).set_title('card6', fontsize=16)
plt.show(plot)


# In[26]:


fig, ax = plt.subplots(figsize=(10, 10))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("card6")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['plum', 'purple'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[27]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.violinplot(x="card6", y="TransactionAmt", hue="isFraud",
                      data=train_df.query("TransactionAmt < 500"),
                      palette=['plum', 'purple'], split=True, ax=ax).set_title('TransactionAmt vs. card6', fontsize=16)

plt.show(plot)


# In[28]:


fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.boxplot(x="card6", y="TransactionAmt", hue="isFraud",
                   data=train_df.query("TransactionAmt < 500"),
                   palette=['plum', 'purple'], ax=ax).set_title('TransactionAmt vs. card6', fontsize=16)

plt.show(plot)


# In[29]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500")                .groupby("M1")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['coral', 'orangered'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[30]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M2")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lightsalmon', 'orangered'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[31]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M3")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['peachpuff', 'darkorange'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[32]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M4")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lemonchiffon', 'gold'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[33]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M5")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lightgreen', 'green'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[34]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M6")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['lightblue', 'darkblue'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[35]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M7")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['mediumslateblue', 'indigo'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[36]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M8")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['violet', 'darkviolet'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[37]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))

props = train_df.query("TransactionAmt < 500").query("card6 == 'credit' or card6 == 'debit'")                .groupby("M9")['isFraud'].value_counts(normalize=True).unstack()

sns.set_palette(['mediumorchid', 'purple'])
props.plot(kind='bar', stacked='True', ax=ax).set_ylabel('Proportion')
plt.show(plot)


# In[38]:


cat_cols = ['id_12', 'id_13', 'id_14', 'id_15',
            'id_16', 'id_17', 'id_18', 'id_19',
            'id_20', 'id_21','id_22', 'id_23',
            'id_24', 'id_25', 'id_26', 'id_27',
            'id_28', 'id_29', 'id_30', 'id_31',
            'id_32', 'id_33', 'id_34', 'id_35',
            'id_36', 'id_37', 'id_38',
            'DeviceType', 'DeviceInfo', 'ProductCD', 
            'card4', 'card6', 'M4', 'P_emaildomain',
            'R_emaildomain','card1', 'card2', 'card3', 'card5', 'addr1',
            'addr2', 'M1', 'M2', 'M3', 'M5', 'M6','M7', 'M8', 'M9', 
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
            'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

cat_cols = [col for col in cat_cols if col in train_df.columns]


# In[39]:


def prepare_data(df, cat_cols=cat_cols):
    cat_cols = [col for col in cat_cols if col in df.columns]
    for col in tqdm(cat_cols):        df[col] = pd.factorize(df[col])[0]
    return df


# In[40]:


train_data = prepare_data(train_df)
test_data = prepare_data(test_df)


# In[41]:


X = train_data.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train_data.sort_values('TransactionDT')['isFraud']
del train_data


# In[42]:


split = np.int32(0.8 * len(X))
X_train = X[:split]
y_train = np.int32(y)[:split]
X_val = X[split:]
y_val = np.int32(y)[split:]


# In[43]:


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data = lightgbm.Dataset(X_val, label=y_val)

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=val_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[44]:


plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13

plot = lightgbm.plot_importance(model, max_num_features=10, figsize=(20, 20), grid=False, color=sns.color_palette("husl", 20))
plt.show(plot)


# In[45]:


def get_neural_network():
    inputs = Input(shape=(X.shape[1],))
    dense_1 = Dense(10, activation='relu')(inputs)
    dense_2 = Dense(10, activation='relu')(dense_1)
    outputs = Dense(1, activation='sigmoid')(dense_2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


# In[46]:


model = get_neural_network()
model.summary()


# In[47]:


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[48]:


history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)


# In[49]:


fig, ax = plt.subplots(figsize=(7, 7))
plt.plot(history.history['acc'], color='blue')
plt.plot(history.history['val_acc'], color='orangered')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[50]:


fig, ax = plt.subplots(figsize=(7, 7))
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'], color='orangered')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[51]:


print("Thank you!")

