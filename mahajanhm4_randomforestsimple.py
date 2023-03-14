#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('../input/train_V2.csv')
dataset_test = pd.read_csv('../input/test_V2.csv')
dataset_train = reduce_mem_usage(dataset_train)
dataset_test = reduce_mem_usage(dataset_test)
dataset_train.dropna(inplace=True)
X = dataset_train.loc[:,'assists':'winPoints']
y = dataset_train.loc[:,'winPlacePerc']
X_test2 = dataset_test.loc[:,:]


# In[3]:


X_test2 = dataset_test.loc[:,'assists':]


# In[4]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[5]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x = LabelEncoder()
le_xt = LabelEncoder()
X_train.loc[:,'matchType'] = le_x.fit_transform(X_train.loc[:,'matchType'])
X_train = X_train.iloc[:,:].values
ohe =OneHotEncoder(categorical_features=[12])
X_train = ohe.fit_transform(X_train).toarray()
X_train = X_train[:,1:]

X_test.loc[:,'matchType'] = le_xt.fit_transform(X_test.loc[:,'matchType'])
X_test = X_test.iloc[:,:].values
ohext =OneHotEncoder(categorical_features=[12])
X_test = ohext.fit_transform(X_test).toarray()
X_test = X_test[:,1:]


# In[6]:


le_xt2 = LabelEncoder()
X_test2.loc[:,'matchType'] = le_xt2.fit_transform(X_test2.loc[:,'matchType'])
X_test2 = X_test2.iloc[:,:].values
ohext =OneHotEncoder(categorical_features=[12])
X_test2 = ohext.fit_transform(X_test2).toarray()
X_test2 = X_test2[:,1:]


# In[7]:


# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


# In[8]:


#X_test2 = sc.transform(X_test2)


# In[9]:


X_train


# In[10]:


from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


# In[11]:


'''import xgboost
XGBModel = xgboost.XGBRegressor()
XGBModel.fit(X_train, y_train, verbose = False)

XGB_pred1 = XGBModel.predict(X_test)
...


# In[12]:


from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=3, min_samples_leaf=3, max_features='sqrt')


# In[13]:


rfmodel.fit(X_train,y_train)


# In[14]:


pred1 = rfmodel.predict(X_test)
MAE = mean_absolute_error(y_test , pred1)
print("MAE = > {}".format(MAE))


# In[15]:


XGB_pred = rfmodel.predict(X_test2)


# In[16]:


ss = pd.read_csv("../input/sample_submission_V2.csv")


# In[17]:


ss.drop(['winPlacePerc'],axis = 1)


# In[18]:


ss['winPlacePerc'] = XGB_pred


# In[19]:


ss


# In[20]:


ss.to_csv('mysubmission2.csv', index = False)


# In[21]:




