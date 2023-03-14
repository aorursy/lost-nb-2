#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


train = train.replace(-1, np.NaN)


# In[4]:


print ("Shape Of Train: ",train.shape)
print ("Shape Of Test: ",test.shape)


# In[5]:


train.head(5)


# In[6]:


dataTypeDf = pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sn.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax,color="#34495e")
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# In[7]:


missingValueColumns = train.columns[train.isnull().any()].tolist()
msno.bar(train[missingValueColumns],figsize=(20,8),color="#34495e",fontsize=12,labels=True,)


# In[8]:


msno.matrix(train[missingValueColumns],width_ratios=(10,1),            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=12,sparkline=True,labels=True)


# In[9]:


msno.heatmap(train[missingValueColumns],figsize=(10,10))


# In[10]:


from sklearn import model_selection, preprocessing
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

train_new = train.fillna(-999)
for f in train_new.columns:
    if train_new[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_new[f].values)) 
        train_new[f] = lbl.transform(list(train_new[f].values))
        
train_y = train_new.target.values
train_X = train_new.drop(["id",], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)


# In[11]:


featureImportance = model.get_fscore()
features = pd.DataFrame()
features['features'] = featureImportance.keys()
features['importance'] = featureImportance.values()
features.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
plt.xticks(rotation=90)
sn.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")


# In[12]:


topFeatures = features["features"].tolist()[:15]
corrMatt = train[topFeatures].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)


# In[13]:


# from statsmodels.stats.outliers_influence import variance_inflation_factor  
# import warnings
# warnings.filterwarnings("ignore")

# def calculate_vif_(X):
#     variables = list(X.columns)
#     vif = {variable:variance_inflation_factor(exog=X.values, exog_idx=ix) for ix,variable in enumerate(list(X.columns))}
#     return vif


# numericalCol = []
# for f in train.columns:
#     #print (f)
#     if train[f].dtype!='object' and f not in ["id", "target"]:
#         numericalCol.append(f)
# train_new = train[numericalCol].fillna(-999)
# vifDict = calculate_vif_(train_new)

# vifDf = pd.DataFrame()
# vifDf['variables'] = vifDict.keys()
# vifDf['vifScore'] = vifDict.values()
# vifDf.sort_values(by=['vifScore'],ascending=False,inplace=True)
# validVariables = vifDf[vifDf["vifScore"]<=5]
# variablesWithMC  = vifDf[vifDf["vifScore"]>5]

# fig,(ax1,ax2) = plt.subplots(ncols=2)
# fig.set_size_inches(20,8)
# sn.barplot(data=validVariables,x="vifScore",y="variables",ax=ax1,orient="h",color="#34495e")
# sn.barplot(data=variablesWithMC.head(5),x="vifScore",y="variables",ax=ax2,orient="h",color="#34495e")
# ax1.set(xlabel='VIF Scores', ylabel='Features',title="Valid Variables Without Multicollinearity")
# ax2.set(xlabel='VIF Scores', ylabel='Features',title="Variables Which Exhibit Multicollinearity")


# In[14]:


fig,ax= plt.subplots()
fig.set_size_inches(20,5)
sn.countplot(x= "target",data=train,ax= ax)


# In[15]:




