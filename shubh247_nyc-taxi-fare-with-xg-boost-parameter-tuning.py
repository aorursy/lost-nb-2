#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np#linear algebra   
import pandas as pd #data preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train =  pd.read_csv('../input/train.csv', nrows = 100000, parse_dates=["pickup_datetime"])  # 55m rows,but we import 10m rows


# In[3]:


test = pd.read_csv('../input/test.csv')   #10k rows 


# In[4]:


train.head()  # first 5 record of train 


# In[5]:


train.describe() 


# In[6]:


train.columns


# In[7]:


train.info()


# In[8]:


print(train.isnull().sum())  # check anu null value is available or not .


# In[9]:


print('Old size: %d' % len(train))
train = train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train))
# if gives 20million data then NaN values comes.


# In[10]:


print(train.isnull().sum())


# In[11]:


sns.distplot(train['fare_amount']);


# In[12]:


train.loc[train['fare_amount']<0].shape


# In[13]:


train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0) | (train.dropoff_longitude==0)].shape


# In[14]:


sns.distplot(train['passenger_count'])


# In[15]:


train.describe()


# In[16]:


#clean up the train dataset to eliminate out of range values
train = train[train['fare_amount'] > 0]
train = train[train['pickup_longitude'] < -72]
train = train[(train['pickup_latitude'] > 40) &(train
                                               ['pickup_latitude'] < 44)]
train = train[train['dropoff_longitude'] < -72]
train = train[(train['dropoff_latitude'] >40) & (train
                                                ['dropoff_latitude'] < 44)]
train = train[(train['passenger_count']>0) &(train['passenger_count'] < 10)]


# In[17]:


train.describe()


# In[18]:


test.head()  # first 5 record of test 


# In[19]:


test.describe()


# In[20]:


test.info()


# In[21]:


print(test.isnull().sum())


# In[22]:


test[(test.pickup_latitude==0) | (test.pickup_longitude)==0 | (test.dropoff_latitude==0) | (test.dropoff_longitude==0)].shape


# In[23]:


print(test.isnull().sum())


# In[24]:


print('Old size: %d' % len(test))
test = test.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(test))


# In[25]:


#clean up the train dataset to eliminate out of range values
test = test[test['pickup_longitude'] < -72]
test = test[(test['pickup_latitude'] > 40) &(train
                                               ['pickup_latitude'] < 44)]
test = test[test['dropoff_longitude'] < -72]
test = test[(test['dropoff_latitude'] >40) & (train
                                                ['dropoff_latitude'] < 44)]
test = test[(test['passenger_count']>0) &(train['passenger_count'] < 10)]
train.head()


# In[26]:


#pickup_datetime 

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
type(train['pickup_datetime'].iloc[0])


# In[27]:


combine = [test, train]
for dataset in combine:
        # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear

    
#dataset['Year'] = dataset['pickup_datetime'].apply(lambda time: time.year)
#dataset['Month'] = dataset['pickup_datetime'].apply(lambda time: time.month)
#ataset['Day of Week'] = dataset['pickup_datetime'].apply(lambda time: time.dayofweek)
#dataset['Hour'] = dataset['pickup_datetime'].apply(lambda time: time.hour)

train.head()


# In[28]:


test.head()


# In[29]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train) 
train.head(1)


# In[30]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(test) 
test.head(1)


# In[31]:


# remove unnessary column that not requred for modeling.
train = train.drop(['key','pickup_datetime'],axis = 1) 
test = test.drop('pickup_datetime',axis = 1)
#train.info()


# In[32]:


x_train = train.drop(['fare_amount'], axis=1)
y_train = train['fare_amount']
x_test = test.drop('key', axis=1)


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[34]:


linmodel = LinearRegression()
linmodel.fit(x_train, y_train)


# In[35]:


linmodel_pred = linmodel.predict(x_test)  # prediction on train 


# In[36]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)


# In[37]:


from sklearn.model_selection import train_test_split
import xgboost as xgb

#Let's prepare the test set
x_pred = test.drop('key', axis=1)


# In[38]:


#feature selection
y = train['fare_amount']    
train_df = train.drop(['fare_amount'],axis = 1)


# In[39]:



# Let's run XGBoost and predict those fares
x_train,x_test,y_train,y_test = train_test_split(train_df,y,random_state=123,test_size=0.2)


# In[40]:


params = {
      #parameters that we are going to tune
    'max_depth' :8 ,#result of tuning with cv
    'eta' :.03, #result of tuning with cv
    'subsample' : 1, # result of tuning with cv
    'colsample_bytree' : 0.8, #result of tuning with cv
    #other parameter
    'objective': 'reg:linear',
    'eval_metrics':'rmse',
    'silent': 1
}


# In[41]:


#Block of code used for hypertuning parameters. Adapt to each round of parameter tuning.
CV=False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    # Define initial best params and RMSE
    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        # Update our parameters
        params['eta'] = eta

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        # Update best RMSE
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    #Print final params to use for the model
    params['silent'] = 0 #Turn on output
    print(params)


# In[42]:



def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params
                                  ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


# In[43]:


linmodel_pred, rfr_pred, xgb_pred


# In[44]:


# Assigning weights. More precise models gets higher weight.
linmodel_weight = 1
rfr_weight = 1
xgb_weight = 3
prediction = (linmodel_pred * linmodel_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (linmodel_weight + rfr_weight + xgb_weight)


# In[45]:


prediction


# In[46]:


# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)


# In[47]:


submission

