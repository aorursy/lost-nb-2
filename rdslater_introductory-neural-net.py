#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
n=train.shape[0]
train['data_set']=1
test['data_set']=0
test.price_doc=np.nan
ids=test['id']
train.price_doc=np.log(train.price_doc)
target=train.price_doc
train=train.append(test)
train.drop(['id'],axis=1,inplace=True)
binary=[]
for i in train:
    if train[i].dtypes=='object':
        #print(train[i].value_counts())
        if train[i].value_counts().shape[0]==2:
            binary.append(i)
for i in binary:
    train[i]=pd.factorize(train[i])[0]
train.loc[train['ecology']=='no data','ecology_dat']=0
train.loc[train['ecology']!='no data','ecology_dat']=1
train.loc[train['ecology']=='no data','ecology']=np.nan
train.loc[train['ecology']=='poor','ecology']=1
train.loc[train['ecology']=='satisfactory','ecology']=2
train.loc[train['ecology']=='good','ecology']=3
train.loc[train['ecology']=='excellent','ecology']=4
train.ecology=pd.to_numeric(train.ecology)
train=pd.concat([train,pd.get_dummies(train.sub_area)],axis=1)

a=train.describe()
for i in a:
    train[i]=train[i].fillna((a.loc['min',i]-a.loc['max',i]*2))
    
train.drop(['timestamp','sub_area'],inplace=True,axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols=train.columns.tolist()

train = pd.DataFrame(scaler.fit_transform(train), columns=cols)
"""
for i in train:    
    train[i]=train[i]/(train[i].max()-train[i].min())
"""
test=train[train['data_set']==0]
train=train[train['data_set']==1]
print(test.shape,train.shape,n)
test.drop(['data_set','price_doc',],inplace=True,axis=1)
train.drop(['data_set','price_doc'],inplace=True,axis=1)


# In[2]:


from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping


# In[3]:


model = Sequential()

model.add(Dense(40, input_dim = train.shape[1], init = 'he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(40, init = 'he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())    
model.add(Dense(20, init = 'he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())    
model.add(Dense(1, init = 'he_normal'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
outputs=model.fit(train.as_matrix(),target.as_matrix() , batch_size=32, nb_epoch=5, verbose=1,validation_split=0.5)
preds=model.predict( test.as_matrix(), batch_size=32, verbose=0)


# In[4]:


y=np.reshape(preds,preds.shape[0])
y=np.exp(y)
subs=pd.DataFrame({'id':ids.as_matrix(),'price_doc':y})
subs.to_csv("test_smallnn.csv",index=False)


# In[5]:


"""
from sklearn.cross_validation import KFold
predict_x=np.zeros(train.shape[0])
kf=KFold(train.shape[0],n_folds=5)
outputs_all=[]

for train_index, target_index in kf:
    model = Sequential()
    
    model.add(Dense(40, input_dim = train.loc[train_index,].shape[1], init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Dense(40, init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())    
    #model.add(Dropout(0.2))
    model.add(Dense(20, init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())    
    #model.add(Dropout(0.2))

    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    
    ]
    outputs=model.fit(train.loc[train_index,].as_matrix(),target[train_index].as_matrix() , batch_size=32, 
                      nb_epoch=100, verbose=1,
                      validation_data=[train.loc[target_index,].as_matrix(),target[target_index].as_matrix()],
                      callbacks=callbacks
                     )
    outputs_all.append(outputs)
    preds=model.predict( train.loc[target_index,].as_matrix(), batch_size=32, verbose=0)
    predict_x[target_index]=preds
    """

