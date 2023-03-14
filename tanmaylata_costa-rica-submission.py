#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the dataset
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


#checking for null values
missing_values = train_df.isnull().sum().sort_values(ascending = False)
missing_values =(missing_values[missing_values > 0] / train_df.shape[0])
print(f'{missing_values *100} %')


# In[ ]:


#We need to take care of missing data
#We will first check for v18q1
train_df[['v18q','v18q1']].groupby(train_df['v18q'] == 0).count()


# In[ ]:


#We can update the null values in v18q1 by 0
train_df['v18q1'] = train_df['v18q1'].fillna(0)
test_df['v18q1'] = test_df['v18q1'].fillna(0)


# In[ ]:


#Lets replace the missisng rez_esc values by 0
train_df['rez_esc'] = train_df['rez_esc'].fillna(0)
test_df['rez_esc'] = test_df['rez_esc'].fillna(0)


# In[ ]:


#We can finish off with the meaneduc and SQBmeaned label by imputing them with the median of the columns.
median_meaneduc = train_df['meaneduc'].median()
median_SQBmeaned = train_df['SQBmeaned'].median()
train_df['meaneduc'] = train_df['meaneduc'].fillna(median_meaneduc)
train_df['SQBmeaned'] = train_df['SQBmeaned'].fillna(median_SQBmeaned)

median_meaneduc_test = test_df['meaneduc'].median()
median_SQBmeaned_test = test_df['SQBmeaned'].median()
test_df['meaneduc'] = test_df['meaneduc'].fillna(median_meaneduc_test)
test_df['SQBmeaned'] = test_df['SQBmeaned'].fillna(median_SQBmeaned_test)


# In[ ]:





# In[ ]:


#Check for households where The household population hs unequal target distribution
all_equal = train_df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print(len(not_equal))


# In[ ]:


#correcting the unqual households
for household in not_equal.index:
    true_target = int(train_df[(train_df['idhogar'] == household) & (train_df['parentesco1'] == 1.0)]['Target'])
    train_df.loc[train_df['idhogar'] == household, 'Target'] = true_target


# In[ ]:


#Replacing the values.
train_house_paid = train_df.loc[train_df['v2a1'].isnull() & (train_df['tipovivi1'] == 1)]
train_house_paid['v2a1'] = train_house_paid['v2a1'].fillna(0)
train_df.update(train_house_paid)


# In[ ]:


test_house_paid = test_df.loc[test_df['v2a1'].isnull() & (test_df['tipovivi1'] == 1)]
test_house_paid['v2a1'] = test_house_paid['v2a1'].fillna(0)
test_df.update(test_house_paid)


# In[ ]:


train_df.fillna(-1, inplace = True)
test_df.fillna(-1, inplace = True)


# In[ ]:


train_df['dependency'] = np.sqrt(train_df['SQBdependency'])
test_df['dependency'] = np.sqrt(test_df['SQBdependency'])


# In[ ]:


def mapping(data):
    if data == 'yes':
        return 1
    elif data == 'no':
        return 0
    else:
        return data
train_df['dependency'] = train_df['dependency'].apply(mapping).astype(float)
train_df['edjefa'] = train_df['edjefa'].apply(mapping).astype(float)
train_df['edjefe'] = train_df['edjefe'].apply(mapping).astype(float)

test_df['dependency'] = test_df['dependency'].apply(mapping).astype(float)
test_df['edjefa'] = test_df['edjefa'].apply(mapping).astype(float)
test_df['edjefe'] = test_df['edjefe'].apply(mapping).astype(float)


# In[ ]:


#converting into percentages
train_df['males_above_12'] = train_df['r4h2']/train_df['r4h3']
train_df['person_above_12'] = train_df['r4t2']/train_df['r4t3']
train_df['size_to_person_ratio'] = train_df['tamhog']/train_df['tamviv']

test_df['males_above_12'] = test_df['r4h2']/test_df['r4h3']
test_df['person_above_12'] = test_df['r4t2']/test_df['r4t3']
test_df['size_to_person_ratio'] = test_df['tamhog']/test_df['tamviv']


# In[ ]:


train_df['males-above_12'] = train_df['males_above_12'].fillna(0)


# In[ ]:


test_df['males-above_12'] = test_df['males_above_12'].fillna(0)


# In[ ]:


train_df = train_df.fillna(0)
test_df = test_df.fillna(0)


# In[ ]:


#dropping the useless columns
cols = ['Id','idhogar','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']
train_df.drop(cols, axis = 1, inplace = True)
test_df.drop(cols, axis = 1, inplace = True)


# In[ ]:


train_df.describe()


# In[ ]:


#creating the matrics of features
y = train_df.Target.values
train_df.drop('Target', axis =1, inplace = True)


# In[ ]:


X = train_df.iloc[:,:].values


# In[ ]:


X_test = test_df.iloc[:,:].values


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC(n_estimators =25 , random_state = 0)
classifier.fit(X,y)


# In[ ]:


# from xgboost import XGBClassifier as XGB
# model = XGB()
# model.fit(X,y)


# In[ ]:


predictions = classifier.predict(X_test).astype(int)


# In[ ]:


submission = pd.DataFrame({
    "Id" : submission['Id'],
    "Target" : predictions
})
submission.to_csv('sample_submission.csv', index =False, encoding = 'utf-8')

