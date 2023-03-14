#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#Data Visualization
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Loading Data
train= pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
#Displaying the first five rows of the dataset so as to get a feel of the data.
train.head()


# In[4]:


train.info()


# In[5]:


train['Target'].value_counts()


# In[6]:


train.describe()


# In[7]:


test.info()


# In[8]:


test.describe()


# In[9]:


#A plot to visualise the Target Distribution.
sns.countplot('Target',data=train)


# In[10]:


from collections import OrderedDict
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})
plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Target', y = 'meaneduc', data = train);
plt.xticks([0, 1, 2, 3], poverty_mapping.values())
plt.title('Average Schooling by Target')


# In[11]:


plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Target', y = 'overcrowding', data = train);
plt.xticks([0, 1, 2, 3], poverty_mapping.values())
plt.title('Overcrowding by Target');


# In[12]:


#We are doing this because the test doesn't have the Target column.
train2=train.drop('Target',axis=1)


# In[13]:


# Appending the data
data = train2.append(test,sort=True)


# In[14]:


data['dependency'].value_counts()


# In[15]:


mapping = {"yes": 1, "no": 0}

# Fill in the values with the correct mapping
data['dependency'] = data['dependency'].replace(mapping).astype(np.float64)
data['edjefa'] = data['edjefa'].replace(mapping).astype(np.float64)
data['edjefe'] = data['edjefe'].replace(mapping).astype(np.float64)

data[['dependency', 'edjefa', 'edjefe']].describe()


# In[16]:


#outlier in test set which rez_esc is 99.0
data.loc[data['rez_esc'] == 99.0 , 'rez_esc'] = 5


# In[17]:


# Number of missing in each column
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})

# Create a percentage missing
missing['percent'] = missing['total'] / len(data)

missing.sort_values('percent', ascending = False).head(10)


# In[18]:


data['v18q1'] = data['v18q1'].fillna(0)

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0
data['v2a1-missing'] = data['v2a1'].isnull()

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0
data['rez_esc-missing'] = data['rez_esc'].isnull()


# In[19]:


#electricity columns
elec = []

for i, row in data.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
data['elec'] = elec
data['elec-missing'] = data['elec'].isnull()


# In[20]:


#remove already present electricity columns
data = data.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])


# In[21]:


#walls ordinal
data['walls'] = np.argmax(np.array(data[['epared1', 'epared2', 'epared3']]),
                           axis = 1)
data = data.drop(columns = ['epared1', 'epared2', 'epared3'])


# In[22]:


#roof ordinal
data['roof'] = np.argmax(np.array(data[['etecho1', 'etecho2', 'etecho3']]),
                           axis = 1)
data = data.drop(columns = ['etecho1', 'etecho2', 'etecho3'])


# In[23]:


#floor ordinal
data['floor'] = np.argmax(np.array(data[['eviv1', 'eviv2', 'eviv3']]),
                           axis = 1)
data = data.drop(columns = ['eviv1', 'eviv2', 'eviv3'])


# In[24]:


#Flushing system
data['flush'] = np.argmax(np.array(data[["sanitario1",'sanitario5', 'sanitario2', 'sanitario3',"sanitario6"]]),
                           axis = 1)
data = data.drop(columns = ["sanitario1",'sanitario5', 'sanitario2', 'sanitario3',"sanitario6"])


# In[25]:


#Drop columns with squared variables
data = data[[x for x in data if not x.startswith('SQB')]]
data = data.drop(columns = ['agesq'])


# In[26]:


#waterprovision
data['waterprovision'] = np.argmax(np.array(data[['abastaguano', 'abastaguafuera', 'abastaguadentro']]),
                           axis = 1)
data = data.drop(columns = ['abastaguano', 'abastaguafuera', 'abastaguadentro'])


# In[27]:


#Education Level
data['inst'] = np.argmax(np.array(data[[c for c in data if c.startswith('instl')]]), axis = 1)
data = data.drop(columns = [c for c in data if c.startswith('instlevel')])


# In[28]:


#cooking
data['waterprovision'] = np.argmax(np.array(data[['energcocinar1','energcocinar4', 'energcocinar3', 'energcocinar2']]),
                           axis = 1)
data = data.drop(columns = ['energcocinar1','energcocinar4', 'energcocinar3', 'energcocinar2'])


# In[29]:


#meaneduc is defined as average years of education for adults (18+)
data.loc[pd.isnull(data['meaneduc']), 'meaneduc'] = data.loc[pd.isnull(data['meaneduc']), 'escolari']


# In[30]:


train2=data.iloc[0:9557,:]
test2=data.iloc[9557:33413,:]


# In[31]:


test2.drop(['Id','idhogar'],axis=1,inplace=True)


# In[32]:


X=train2.drop(['Id','idhogar'],axis=1)


# In[33]:


y=train['Target']


# In[34]:


import xgboost as xgb # Importing XGboost Library


# In[35]:


xg=xgb.XGBClassifier(n_estimators=200)


# In[36]:


xg.fit(X,y)


# In[37]:


preds = xg.predict(test2)


# In[38]:


def macro_f1_score(
    
    
    labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True


# In[39]:


# Libraries for LightGBM
import lightgbm as lgb
import sklearn.model_selection as model_selection
from sklearn.metrics import f1_score, make_scorer


# In[40]:


lgmodel = lgb.LGBMClassifier(metric = "",num_class = 4)


# In[41]:


hyp_OPTaaS = { 'boosting_type': 'dart',
             'colsample_bytree': 0.9843467236959204,
             'learning_rate': 0.11598629586769524,
             'min_child_samples': 44,
             'num_leaves': 49,
             'reg_alpha': 0.35397370408131534,
             'reg_lambda': 0.5904910774606467,
             'subsample': 0.6299872254632797,
             'subsample_for_bin': 60611}


# In[42]:


model = lgb.LGBMClassifier(**hyp_OPTaaS, class_weight = 'balanced',max_depth=-1,objective = 'multiclass', n_jobs = -1, n_estimators = 100)


# In[43]:


model.fit(X, y)


# In[44]:


pred=model.predict(test2)


# In[45]:


my_submission = pd.DataFrame({'Id': test.Id, 'Target': pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

