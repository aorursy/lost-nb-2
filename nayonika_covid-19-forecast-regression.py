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

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")


# In[3]:


train_df.head(50)


# In[4]:


train_df.info()


# In[5]:


test_df.info()


# In[6]:


from pandas_profiling import ProfileReport
train_profile = ProfileReport(train_df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile


# In[7]:


#The Procince/State values are replaced with Country/Region to fill the nulls
train_df.Province_State.fillna(train_df.Country_Region, inplace=True)
test_df.Province_State.fillna(test_df.Country_Region, inplace=True)


# In[8]:


train_df.Date = train_df.Date.apply(pd.to_datetime)
test_df.Date = test_df.Date.apply(pd.to_datetime)

train_df['ReportDay_month'] = train_df['Date'].dt.month
train_df['ReportDay_week'] = train_df['Date'].dt.week
train_df['ReportDay_day'] = train_df['Date'].dt.day 

test_df['ReportDay_month'] = test_df['Date'].dt.month
test_df['ReportDay_week'] = test_df['Date'].dt.week
test_df['ReportDay_day'] = test_df['Date'].dt.day


# In[9]:


test_df.drop(['Date'], axis=1, inplace = True)
train_df.drop(['Date'], axis=1, inplace = True)


# In[10]:


le = LabelEncoder()

train_df.Country_Region = le.fit_transform(train_df.Country_Region)
train_df['Province_State'] = le.fit_transform(train_df['Province_State'])

test_df.Country_Region = le.fit_transform(test_df.Country_Region)
test_df['Province_State'] = le.fit_transform(test_df['Province_State'])
test_df.head(10)


# In[11]:


X_train = train_df.drop(["Id", "ConfirmedCases", "Fatalities"], axis = 1)
Y_train_CC = train_df["ConfirmedCases"] 
Y_train_Fat = train_df["Fatalities"] 
X_test = test_df.drop(["ForecastId"], axis = 1) 


# In[12]:


X_test.info()


# In[13]:



skfold = ShuffleSplit(random_state=7)


# In[14]:


'''#1. Ridge Regression

#train classifier
reg_CC = Ridge(alpha=1.0)
reg_Fat = Ridge(alpha=1.0)


#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())
#0.02542952540508501 0.011197579501272648'''


# In[15]:


'''#2.Lasso Regression

#train classifier
reg_CC = linear_model.Lasso(alpha=0.1)
reg_Fat = linear_model.Lasso(alpha=0.1)

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#rmsle_svm = test_model_r2(clf_svm, "CC")

#Print the scores
print (score_CC.mean(), score_Fat.mean())
#0.025431596423250158 0.01122396097530325''''''


# In[16]:


'''#3. SVM

#train classifier
reg_CC = svm.SVC()
reg_Fat = svm.SVC()

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())'''


# In[17]:


#4. ElasticNet

#train classifier
reg_CC = ElasticNet(alpha=100, copy_X=True, fit_intercept=True, l1_ratio=0.1,
           max_iter=2400, normalize=False, positive=False, precompute=False,
           random_state=45, selection='cyclic', tol=0.0001, warm_start=False)
reg_Fat = ElasticNet(alpha=100, copy_X=True, fit_intercept=True, l1_ratio=0.1,
           max_iter=2400, normalize=False, positive=False, precompute=False,
           random_state=45, selection='cyclic', tol=0.0001, warm_start=False)

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())
#0.025523931055995132 0.011306929750992378
#0.010042646409910571 0.0035125533790476117
#0.006609985459851919 0.0026637521539920163


# In[18]:


'''#5. LinearRegression

#train classifier
reg_CC = LinearRegression()
reg_Fat = LinearRegression()

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())
#0.02542909255578929 0.011196924912099948'''


# In[19]:


'''#6 Logistic Regression

#train classifier
reg_CC = LogisticRegression(random_state=0)
reg_Fat = LogisticRegression(random_state=0)

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())'''


# In[20]:


'''#7. XGBoost
import xgboost as xgb
#train classifier
reg_CC = xgb.XGBRegressor(n_estimators=10000)
reg_Fat =xgb.XGBRegressor(n_estimators=10000)

#Cross Validation to calculate the score
score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

#Print the scores
print (score_CC.mean(), score_Fat.mean())'''


# In[21]:


'''#8. Adaboost regressor


reg_CC = AdaBoostRegressor(random_state=51, n_estimators=1000)
reg_Fat = AdaBoostRegressor(random_state=51, n_estimators=1000)


score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

print (score_CC.mean(), score_Fat.mean())
#-0.28701899875717746 -0.07457296951523934'''


# In[22]:


'''#9. Bagging 
from sklearn.svm import SVR
reg_CC = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
reg_Fat = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)

score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

print (score_CC.mean(), score_Fat.mean())'''


# In[23]:


'''#10 Random Forest
reg_CC = RandomForestRegressor(max_depth=2, random_state=50)
reg_Fat = RandomForestRegressor(max_depth=2, random_state=50)

score_CC = cross_val_score(reg_CC, X_train, Y_train_CC, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

print (score_CC.mean(), score_Fat.mean())
#0.04355921589425451 0.021343671523731022'''


# In[24]:


'''#11 Decision Tree

reg_CC = DecisionTreeRegressor(max_depth=10, random_state=500)
reg_Fat = DecisionTreeRegressor(max_depth=10, random_state=500)

score_CC = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)
score_Fat = cross_val_score(reg_Fat, X_train, Y_train_Fat, cv = skfold)

print (score_CC.mean(), score_Fat.mean())
#0.6417006505631566 0.6417006505631566'''


# In[25]:


reg_CC.fit(X_train, Y_train_CC)
Y_pred_CC = reg_CC.predict(X_test) 

reg_Fat.fit(X_train, Y_train_Fat)
Y_pred_Fat = reg_Fat.predict(X_test) 


# In[26]:


df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
soln = pd.DataFrame({'ForecastId': test_df.ForecastId, 'ConfirmedCases': Y_pred_CC, 'Fatalities': Y_pred_Fat})
df_out = pd.concat([df_out, soln], axis=0)
df_out.ForecastId = df_out.ForecastId.astype('int')


# In[27]:


df_out.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

