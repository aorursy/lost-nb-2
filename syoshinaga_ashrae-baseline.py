#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

target = 'meter_reading'


# In[2]:


building_count = 1

train_df = train_df[train_df['building_id'].isin(range(building_count))]
test_df = test_df[test_df['building_id'].isin(range(building_count))]


# In[3]:


[train_df, test_df] = [df.merge(building_metadata, on="building_id", how="left") for df in [train_df, test_df]]
train_df = train_df.merge(weather_train, on=["site_id", "timestamp"], how="left")
test_df = test_df.merge(weather_test, on=["site_id", "timestamp"], how="left")


# In[4]:


train_df


# In[5]:


test_df


# In[6]:


for df in [train_df, test_df]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].apply(lambda t: t.year)
    df['month'] = df['timestamp'].apply(lambda t: t.month)
    df['day'] = df['timestamp'].apply(lambda t: t.day)
    df['hour'] = df['timestamp'].apply(lambda t: t.hour)
    df['weekday'] = df['timestamp'].apply(lambda t: t.weekday())

numbers = train_df.select_dtypes(include=np.number).columns
objects = train_df.select_dtypes(include=np.object).columns
drops = ['timestamp']

for c in numbers:
    if c in [target] or c in drops:
        continue
    for df in [train_df, test_df]:
        median = df[c].median()
        if np.isnan(median):
            median = 0
        df[c] = df[c].fillna(median)

for c in objects:
    if c in drops:
        continue
    values = pd.concat([train_df[c], test_df[c]]).unique().tolist()
    for df in [train_df, test_df]:
        df[c] = df[c].fillna('NaN')
    le = LabelEncoder()
    le.fit(pd.concat([train_df[c], test_df[c]]))
    for df in [train_df, test_df]:
        df[c] = le.transform(df[c])

[train_df, test_df] = [df.drop(drops, axis=1) for df in [train_df, test_df]]


# In[7]:


train_df


# In[8]:


test_df


# In[9]:


X, y = train_df.drop([target], axis=1).values, np.log1p(train_df[target]).values
X_train, X_test, y_train, y_test =     train_test_split(X, y,
                     test_size=0.3,
                     random_state=0)

xgbr = XGBRegressor()

models = [
    xgbr,
]

results = []

for m in models:
    scores = np.sqrt(-cross_val_score(estimator=m,
                                      X=X_train,
                                      y=y_train,
                                      cv=10,
                                      n_jobs=-1,
                                      scoring='neg_mean_squared_error'))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    validation_score = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        'Model': m.__class__.__name__,
        'Training CV Accuracy': '%.4f +/- %.4f' % (np.mean(scores), np.std(scores)),
        'Validation Accuracy': validation_score
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('Validation Accuracy', ascending=True))


# In[10]:


name_list = train_df.drop([target], axis=1).columns.tolist()
fi_list = m.feature_importances_.tolist()
fi_df = pd.DataFrame(list(zip(name_list, fi_list)), columns=['Column', 'FeatureImportance'])    .sort_values('FeatureImportance', ascending=False)
fi_df


# In[11]:


y_pred = m.predict(test_df.drop(['row_id'], axis=1).values)

submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        target: np.expm1(y_pred)
    })
submission.to_csv('submission.csv', index=False)


# In[12]:


train_df['meter_reading'].plot()


# In[13]:


pd.DataFrame(np.expm1(y_pred), columns=['meter_reading']).plot()

