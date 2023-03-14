#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.simplefilter('ignore')


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


data_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
data_train.head()


# In[5]:


data_train1 = data_train.drop(['Id', 'Province/State'] , axis = 1)
data_train1.head()


# In[6]:


data_train1_num = data_train1[['ConfirmedCases','Fatalities']]
data_train1_cat = data_train1[['Country/Region','Lat','Long','Date']]


# In[7]:


from sklearn.preprocessing import LabelEncoder
print("Done !!!")


# In[8]:


encoder1 = LabelEncoder()


# In[9]:


data_train2_cat= data_train1_cat.copy()
data_train2_cat.head()


# In[10]:


length = data_train2_cat.shape[1]
col = data_train2_cat.columns
for i in range(length):
    print(i)
    a = encoder1.fit_transform(data_train2_cat.iloc[:,i:i+1])
    a = pd.DataFrame(a, columns=[col[i]+'new'])
    data_train2_cat = data_train2_cat.join(a)


# In[11]:


data_train2_cat.tail()


# In[12]:


data_train3 =data_train2_cat.join(data_train1_num)
data_train3.tail()


# In[13]:


data_train_fix = data_train3.drop(['Country/Region','Lat','Long','Date'], axis = 1)
data_train_fix.head()


# In[14]:


plt.figure(figsize=(5,5))
sns.heatmap(data_train_fix.corr(),
            annot=True,
            linewidths=.10,
            fmt='.2f',
            cmap = 'Blues');


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = data_train_fix[['Country/Regionnew','Latnew','Longnew','Datenew']]  
y = data_train_fix[['ConfirmedCases','Fatalities']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


print("Training data :",X_train.shape)
print("Testing data  :",X_test.shape)


# In[17]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[18]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


# In[19]:


RFModel = RandomForestRegressor()
rf_tuned_params = {'n_estimators' : sp_randint(2, 100), 
                   'max_depth': sp_randint(10, 50),
                   'random_state' : sp_randint(0,10)}
n_iter_search = 5
random_search = RandomizedSearchCV(RFModel, param_distributions=rf_tuned_params,
                                   n_iter=n_iter_search, cv=5)

random_search.fit(X_train, y_train)
print("Best Params : ",random_search.best_params_)
print()
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


# In[20]:


DTReg = DecisionTreeRegressor()
DT_tuned_params = { 
                   'min_samples_leaf': sp_randint(2, 10),
                  'min_samples_split' : sp_randint(2, 10),
              'max_depth' : sp_randint(10, 100),}
n_iter_search = 10 
random_search = RandomizedSearchCV(DTReg, param_distributions=DT_tuned_params,
                                   n_iter=n_iter_search, cv=5)

random_search.fit(X_train, y_train)
print("Best Params : ",random_search.best_params_)
print()
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


# In[21]:


max_depth = 45
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=98,
                                                          max_depth=max_depth,
                                                          random_state=0))


# In[22]:


regr_multirf.fit(X_train, y_train)


# In[23]:


max_depth = 90
regr_multirf_DT = MultiOutputRegressor(DecisionTreeRegressor(min_samples_split=3,
                                                          max_depth=max_depth,
                                                          min_samples_leaf=3))


# In[24]:


regr_multirf_DT.fit(X_train, y_train)


# In[25]:


from sklearn.metrics import mean_squared_error


# In[26]:


print("R2 accuracy for training data is:",regr_multirf.score(X_train,y_train))


# In[27]:


print("R2 accuracy for training data is:",regr_multirf_DT.score(X_train,y_train))


# In[28]:


y_prediction = regr_multirf.predict(X_test)


# In[29]:


y_prediction_DT = regr_multirf_DT.predict(X_test)


# In[30]:


print("R2 accuracy for testing data is :",regr_multirf.score(X_test,y_test))
print("MSE for testing data is         :",mean_squared_error(y_test, y_prediction))


# In[31]:


print("R2 accuracy for testing data is :",regr_multirf_DT.score(X_test,y_test))
print("MSE for testing data is         :",mean_squared_error(y_test, y_prediction_DT))


# In[32]:


data_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
data_test.head()


# In[33]:


data_test1 = data_test.drop(['ForecastId', 'Province/State'] , axis = 1)
data_test1.head()


# In[34]:


data_test1_cat = data_test1[['Country/Region','Lat','Long','Date']]


# In[35]:


data_test1_cat= data_test1_cat.copy()
data_test1_cat.head()


# In[36]:


length = data_test1_cat.shape[1]
col = data_test1_cat.columns
for i in range(length):
    print(i)
    a = encoder1.fit_transform(data_test1_cat.iloc[:,i:i+1])
    a = pd.DataFrame(a, columns=[col[i]+'new'])
    data_test1_cat = data_test1_cat.join(a)


# In[37]:


data_test1_cat.tail()


# In[38]:


data_test_fix = data_test1_cat.drop(['Country/Region','Lat','Long','Date'], axis = 1)
data_test_fix.head()


# In[39]:


X_multirf = data_test_fix[['Country/Regionnew','Latnew','Longnew','Datenew']] 


# In[40]:


y_multirf = regr_multirf.predict(X_multirf)


# In[41]:


y_multirf_DT = regr_multirf_DT.predict(X_multirf)


# In[42]:


y_multirf


# In[43]:


y_multirf_DT


# In[44]:


multirf_Result = pd.DataFrame(y_multirf,columns = ['ConfirmedCases','Fatalities'])
multirf_Result.head()


# In[45]:


multirf_Result_DT = pd.DataFrame(y_multirf_DT,columns = ['ConfirmedCases','Fatalities'])
multirf_Result_DT.head()


# In[46]:


result = data_test[['ForecastId']]
result.head()


# In[47]:


multirf_Result_fix = result.join(multirf_Result)
multirf_Result_fix.head()


# In[48]:


multirf_Result_fix_DT = result.join(multirf_Result_DT)
multirf_Result_fix_DT.head()


# In[49]:


multirf_Result_fix.to_csv('submission.csv',index=False)


# In[50]:


multirf_Result_fix_DT.to_csv('submission.csv',index=False)


# In[ ]:




