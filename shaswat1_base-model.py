#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.layers import Dense,Dropout,LayerNormalization,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['patch.force_edgecolor']=True
plt.rcParams['figure.figsize'] = (10,7)


# In[3]:


sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
df_spec = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
df_test = pd.read_csv("../input/data-science-bowl-2019/test.csv",parse_dates=['timestamp'])
df = pd.read_csv("../input/data-science-bowl-2019/train.csv",parse_dates=['timestamp'])
df_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df_labels.isnull().sum()


# In[8]:


df.describe().T


# In[9]:


plt.figure(figsize=(10,7 ))
sns.barplot(x=df.type.value_counts().index, y=df.type.value_counts())


# In[10]:


plt.figure(figsize=(10,7 ))
sns.barplot(x=df.world.value_counts().index, y=df.world.value_counts())


# In[11]:


df['date'] = df['timestamp'].apply(lambda date:date.date)


# In[12]:


df['year'] = df['date'].apply(lambda date:date.year)
df['month'] = df['date'].apply(lambda date:date.month)
df['day'] = df['date'].apply(lambda date:date.day)


# In[13]:


event_counts = df.groupby(['date'])['event_id'].agg('count')


# In[14]:


game_time_sum = df.groupby(['date'])['game_time'].agg('sum')


# In[15]:


plt.figure(figsize=(10,7))
sns.lineplot(event_counts.index,event_counts.values)
plt.title('Events Counts by Date')
plt.show()


# In[16]:


plt.figure(figsize=(10,7))
sns.lineplot(game_time_sum.index,game_time_sum.values)
plt.title('Total Game time by Date')
plt.ylabel('In Billions')
plt.show()


# In[17]:


df['Weekday'] = df['timestamp'].apply(lambda date: date.day_name())


# In[18]:


plt.figure(figsize=(10,7))
sns.countplot(df['Weekday'])


# In[19]:


gametime_wdays = df.groupby(['Weekday'])['game_time'].agg('sum')


# In[20]:


plt.figure(figsize=(10,7))
sns.barplot(gametime_wdays.index,gametime_wdays.values)
plt.title('Total Gametime by Day')
plt.ylabel('Count in Billions')
plt.show()


# In[21]:


plt.figure(figsize=(10,7))
sns.heatmap(df_labels.corr(),cmap='coolwarm',annot=True)


# In[22]:


sns.countplot(df_labels['accuracy_group'])


# In[23]:


df.groupby('installation_id')     .count()['event_id']     .apply(np.log1p)     .plot(kind='hist',
          bins=40,
          color='orange',
         figsize=(15, 5),
         title='Log(Count) of Observations by installation_id')
plt.show()


# In[24]:


df.groupby('title')['event_id'].count().sort_values().plot(kind='barh',
                                                           title='Count of Observation by Game/Video title',
                                                          color='orange',
                                                          figsize=(15,15))


# In[25]:


def group_and_reduce(df):
    # group1 and group2 are intermediary "game session" groups,
    # which are reduced to one record by game session. group1 takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    # group2 takes the total number of event_code of each type
    group1 = df.drop(columns=['event_id', 'event_code','timestamp']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    group2 = pd.get_dummies(
        df[['installation_id', 'event_code']], 
        columns=['event_code']
    ).groupby(['installation_id']).sum()

    # group3, group4 and group5 are grouped by installation_id 
    # and reduced using summation and other summary stats
    group3 = pd.get_dummies(
        group1.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world']
    ).groupby(['installation_id']).sum()

    group4 = group1[
        ['installation_id', 'event_count', 'game_time']
    ].groupby(
        ['installation_id']
    ).agg([np.sum, np.mean, np.std])

    return group2.join(group3).join(group4)


# In[26]:


get_ipython().run_line_magic('time', '')
train = group_and_reduce(df)
test = group_and_reduce(df_test)


# In[27]:


df.drop(df.index, inplace=True)


# In[28]:


small_labels = df_labels[['installation_id', 'accuracy_group']].set_index('installation_id')


# In[29]:


train_joined = train.join(small_labels).dropna()


# In[30]:


X = train_joined.drop(columns='accuracy_group').values


# In[31]:


y = train_joined['accuracy_group'].values.astype(np.int32)


# In[32]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[33]:


scaler = MinMaxScaler()


# In[34]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[35]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[36]:


X_train.shape


# In[37]:


# Early stopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1)


# In[38]:


#Model
model = Sequential()

model.add(Input(shape=X_train.shape[1]))
model.add(Dense(200,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(100,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(50,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(25,activation='relu'))
model.add(LayerNormalization())
model.add(Dropout(rate=0.1))

model.add(Dense(4,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[39]:


model.fit(x=X_train,y=y_train,
         validation_data=(X_test,y_test),
         batch_size=256,
         epochs=200,
         callbacks=[early_stop])


# In[40]:


loss_rep = pd.DataFrame(model.history.history)


# In[41]:


loss_rep.head()


# In[42]:


model.save('model.h5')


# In[43]:


loss_rep[['loss','val_loss']].plot()


# In[44]:


loss_rep[['accuracy','val_accuracy']].plot()


# In[45]:


a = model.predict_classes(X_test)


# In[46]:


np.unique(a)


# In[47]:


orig=np.argmax(y_test, axis=1)


# In[48]:


pred_df = pd.DataFrame(orig,columns=['Test Y'])


# In[49]:


test_predicition = pd.Series(a.reshape(4421,))


# In[50]:


pred_df = pd.concat([pred_df,test_predicition],axis=1)


# In[51]:


pred_df.columns = ['Test_Y','Predictions']


# In[52]:


print(confusion_matrix(orig,a))


# In[53]:


print(classification_report(orig,a))


# In[ ]:





# In[54]:


# ON test Dataset:


# In[55]:


df_test.head()


# In[56]:


df_test['date'] = df_test['timestamp'].apply(lambda date:date.date)


# In[57]:


df_test['year'] = df_test['date'].apply(lambda date:date.year)
df_test['month'] = df_test['date'].apply(lambda date:date.month)
df_test['day'] = df_test['date'].apply(lambda date:date.day)


# In[58]:


test = group_and_reduce(df_test)


# In[59]:


test.shape


# In[60]:


submission = pd.DataFrame(test.index)


# In[61]:


test = test.values


# In[62]:


# scaler1 = MinMaxScaler()
test_scaled = scaler.transform(test)


# In[63]:


a1 = model.predict_classes(test_scaled)


# In[64]:


submission = pd.concat([submission,pd.Series(a1.reshape(1000,))],axis=1)


# In[65]:


submission.columns = ['installation_id','accuracy_group']


# In[66]:


submission.head()


# In[67]:


submission.to_csv('submission.csv', index=None)


# In[68]:


submission['accuracy_group'].hist()

