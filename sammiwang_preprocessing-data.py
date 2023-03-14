#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_data,news_data) = env.get_training_data()


# In[ ]:


# output the shape of the data
print(market_data.shape)
print(news_data.shape)


# In[ ]:


market_data.head()


# In[ ]:


news_data.head()


# In[ ]:


# look at missing data
import matplotlib.pyplot as plt
def mis_value_graph(data):
    data.isnull().sum().plot(kind="bar",figsize=(20,10),fontsize=20)
    plt.xlabel("Columns")
    plt.ylabel("Missing Value Count")
    plt.title("the number of null data of all column")
    plt.show()
mis_value_graph(market_data)
mis_value_graph(news_data)


# In[ ]:


# handle missing data about market_data
from sklearn.preprocessing import StandardScaler

ID_col = ['assetCode']
Target_col = ['returnsOpenNextMktres10']
Cat_col = ['assetCode']
Num_cols = ['volume','close','open','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1'
           ,'returnsOpenPrevMktres1','returnsClosePrevRaw1','returnsOpenPrevRaw10',
           'returnsClosePrevMktres10','returnsOpenPrevMktres10','returnsOpenNextMktres10']
# def HandleCatData(encoder,x):
#     len_encoder=len(encoder)
#     try:
#         id = encoder[x]
#     except KeyError:
#         return id
# encoders = [{} for cat in Cat_col]
# for i,cat in enumerate(Cat_col):
#     print('encoding %s ...'%cat,end='')
#     encoders[i] = {1:id for id,i in enumerate(market_data.loc[market_data,cat].astype(str).unique())}
#     market_data[cat] = market_data[cat].astype(str).apply(lambda x:HandleCatData(encoders[i],x))
#     print('Done')

# embed_sizes = [len(HandleCatData) +1 for encode in encoders]

# handle num data
market_data[Num_cols] = market_data[Num_cols].fillna(0)
print("scaling numerical columns")

scaler = StandardScaler()
market_data[Num_cols] = scaler.fit_transform(market_data[Num_cols])


# In[ ]:


market_data.head()


# In[ ]:


# create train model
# news data contact to marker data by using assetCode
assetCodeList = news_data['assetCodes']
CodeList = []
for item in assetCodeList:
    item = assetCodeList[0].replace('\'','')
    item = item[1:len(item)-1]
    arrList = item.split(',');
    CodeList.append(arrList)
print(len(CodeList))

# get News data by using assetCode
def getNewsData(assetCode):
    for i in range(len(CodeList)):
        for j in range(CodeList[i]):
            if(assetCode==CodeList[i][j]):
                return i


# In[ ]:


# handle fulldata list
cols = ['sentenceCount','wordCount','firstMentionSentenct','relevance','sentimentClass','sentimentNegative',
       'sentimentNeutral','sentimentPositive','sentimeWordCount','noveltyCount12H',
       'noveltyCount24H','noveltyCount5D','noveltyCount7D','volumeCounts12H',
       'volumeCounts3D','volumnCounts5D','volumeCounts7D']
def getFullData(market_data,news_data):
    news_data = news_data[cols]
    connactData = []
    for assetcode in market_data.assetCode:
        index = getNewsData(assetcode)
        connactData.append(news_data[index])
    np.hstack((market_data[Num_cols],connactData))
fullData = getFullData(market_data,news_data)


# In[ ]:


# split all data as train and test data
# the rate of the test data is 20%
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(fullData.index.values,test_size=0.2,random_state=0)
print(train_data.shape)


# In[ ]:


# show some descriptions of target variable
market_data.returnsOpenNextMktres10.describe()


# In[ ]:


# look at the distribution of target variable
plt.figure(figsize=(10,5))
print("skew",market_data.returnsOpenNextMktres10.skew())
import seaborn as sns
sns.distplot(market_data['returnsOpenNextMktres10'])


# In[ ]:


# create train model
# way 1 : NN model
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dropout,Dense
def  get_lstm(data):
    '''
    function: created an RNN forecasting model
    :param data:
    :return:
    '''
    model = Sequential()
    model.add(LSTM(data[1],input_shape=(data[0],1),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(data[3],activation='linear'))
    return model
model = get_lstm(fullData)


# In[ ]:


# train model
def train_model(model,x_train,y_train,config):
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])
    hist = model.fit(x_train,y_train,
                     batch_size=config["batch"],
                     epochs=config["epochs"],
                     validation_split=0.05)
    model.save('model/lstm.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/lstm loss.csv',encoding='utf-8',index=False)
config = {"batch":32,"epochs":10}
train_model(model,train_data-train_data.returnsOpenNextMktres10,train_data.returnsOpenNextMktres10,config)


# In[ ]:


# predict
def predict(X):
    lstm = load_model('model/lstm.h5')
    predictd = lstm.predict(X)
    return predictd
days = env.get_prediction_days()
predictd = []
for day in days:
    predictd.append(predict(days))
# look at the distribution of predict variable
plt.figure(figsize=(10,5))
print("skew",predictd.skew())
import seaborn as sns
sns.distplot(predictd)


# In[ ]:


# fourth : Evaluate model
def calculate_loss(x_test,y_test):
    from sklearn import linear_model
    lm = linear_model.LinearRegression()
    from sklearn.model_selection import cross_val_score
    scores = -cross_val_score(lm,x_train,y_train,cv=5,scoring='neg_mean_absolute_error')
    error = np.mean(scores)
    return error'
error = calculate_loss(test_data-test_data.returnsOpenNextMktres10,test_data.returnsOpenNextMktres10)
# look at the distribution of target variable
plt.figure(figsize=(10,5))
print("skew",error.skew())
import seaborn as sns
sns.distplot(error)


# In[ ]:


# write the result into submission.csv
sub = pd.DataFrame.from_dict(predictd)
# submit the final result
sub.to_csv('submission.csv',index=False)

