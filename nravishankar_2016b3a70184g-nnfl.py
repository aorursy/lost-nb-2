#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np


# In[2]:


data=pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv',index_col=0)


# In[3]:


data = data.replace({'?': np.nan})
data = data.fillna(data.mean())
data.fillna(value = data.mode().loc[0], inplace = True)


# In[4]:


data.head(20)


# In[5]:


data = pd.get_dummies(data, columns= ['Size'], prefix = ['Size'])
data.head(100)


# In[6]:


data = data.astype('float64')


# In[7]:


data.dtypes


# In[8]:


# data = data.fillna(value = data.mean())


# In[9]:


# from sklearn.preprocessing import StandardScaler

# data_scaled = data.copy()
# scaler = StandardScaler()

# data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled), columns=data.columns)


# In[10]:



data.head()


# In[11]:


X = data.drop('Class', axis= 1)
#X = X.drop('ID', axis = 1)
y=data['Class']
y.value_counts()


# In[12]:


# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)
 
# # define baseline model
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(8, input_dim=13, activation='relu'))
# 	model.add(Dense(7, activation='softmax'))
# 	# Compile model
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model
 
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[13]:


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
 
# define baseline model

# create model
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(32, activation='tanh'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(rate = 0.2))
model.add(Dense(6, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X, dummy_y, validation_split=0.2, epochs=200,batch_size=15)
model.summary()

 
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[14]:


test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
test = test.replace({'?': np.nan})
test = test.fillna(data.mean())
test.fillna(value = test.mode().loc[0], inplace = True)


# In[15]:


test.head()


# In[16]:


test = pd.get_dummies(test, columns= ['Size'], prefix = ['size'])


# In[17]:


y2 = test['ID']
test = test.drop(['ID'], axis = 1)


# In[18]:



# test_scaled = test.copy()
# scaler = StandardScaler()

# test_scaled = pd.DataFrame(scaler.fit_transform(test_scaled), columns=test.columns)


# In[19]:


test.head()


# In[ ]:





# In[20]:


ans = model.predict(test)


# In[21]:


labels = ans.argmax(-1)


# In[22]:


len(labels)


# In[23]:


labels = pd.DataFrame(data = labels, index = y2)


# In[24]:


labels.head(20)


# In[25]:


labels.to_csv('nnfl4.csv')


# In[26]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64def create_download_link(df, title = "Download CSV file", filename = "data.csv"):    csv = df.to_csv(index=False)    b64 = base64.b64encode(csv.encode())    payload = b64.decode()html='<adownload="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'    html = html.format(payload=payload,title=title,filename=filename)    
return HTML(html)create_download_link(​<submission_DataFrame_name>​)


# In[ ]:




