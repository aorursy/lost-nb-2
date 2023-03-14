#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn import model_selection, preprocessing
print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Any results you write to the current directory are saved as output.


# In[2]:


porp=pd.read_csv("../input/properties_2016.csv")
train=pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])


# In[3]:


porp.head(3)


# In[4]:


train.head(5)


# In[5]:


plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]),np.sort(train.logerror.values))


# In[6]:


train.logerror.describe()


# In[7]:


ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train['logerror'].ix[train['logerror']>ulimit] = ulimit
train['logerror'].ix[train['logerror']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(train.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()


# In[8]:


porp.head(5)


# In[9]:


porp.shape,train.shape


# In[10]:


train['transaction_month'] = train['transactiondate'].dt.month


# In[11]:


train.head(2)


# In[12]:


cnt_srs = train['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[13]:


train['transaction_year'] = train['transactiondate'].dt.year
cnt_srs_yr = train['transaction_year'].value_counts()
plt.figure(figsize=(2,2))
sns.barplot(cnt_srs_yr.index, cnt_srs_yr.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[14]:


(train['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# In[15]:


porp.isnull().sum().sort_values()


# In[16]:


missing_df = porp.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='green')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[17]:


plt.figure(figsize=(12,12))
sns.jointplot(x=porp.latitude.values, y=porp.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# In[18]:


train_df  = pd.merge(train, porp, on='parcelid', how='left')


# In[19]:


print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(porp, on='parcelid', how='left')


# In[20]:


train_df.shape


# In[21]:


# year and month #
#train_df["yearmonth"] = train_df["transactiondate"].dt.year*100 + train_df["transactiondate"].dt.month

# year and week #
#train_df["yearweek"] = train_df["transactiondate"].dt.year*100 + train_df["transactiondate"].dt.weekofyear

# week of year #
#train_df["week_of_year"] = train_df["transactiondate"].dt.weekofyear

# day of week #
#train_df["day_of_week"] = train_df["transactiondate"].dt.weekday


# In[22]:


train_df=train_df.drop(['transaction_year'],axis=1)


# In[23]:


train_df.head(3)


# In[24]:


sns.countplot(train_df['airconditioningtypeid'],data=train_df)


# In[25]:


plt.figure(figsize=(8,4))
sns.countplot(train_df['bathroomcnt'],data=train_df)


# In[26]:


plt.figure(figsize=(8,4))
sns.countplot(train_df['roomcnt'],data=train_df)


# In[27]:


pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[28]:


plt.figure(figsize=(12,8))
sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()


# In[29]:


train_df['bedroomcnt'].ix[train_df['bedroomcnt']>7] = 7
plt.figure(figsize=(12,8))
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
plt.xlabel('Bedroom count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[30]:


plt.figure(figsize=(15,6))
sns.violinplot(x='garagecarcnt', y='logerror', data=train_df)
plt.xlabel('Garag count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[31]:


plt.figure(figsize=(15,6))
sns.violinplot(x='roomcnt', y='logerror', data=train_df)
plt.xlabel('Room count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[32]:


plt.figure(figsize=(15,6))
sns.violinplot(x='fireplacecnt', y='logerror', data=train_df)
plt.xlabel('Fireplace count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[33]:


plt.figure(figsize=(15,6))
sns.violinplot(x='pooltypeid10', y='logerror', data=train_df)
plt.xlabel('Pool count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[34]:


col = "taxamount"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Tax Amount', fontsize=12)
plt.title("Tax Amount Vs Log error", fontsize=15)
plt.show()


# In[35]:


from ggplot import *
ggplot(aes(x='yearbuilt', y='logerror'), data=train_df) +     geom_point(color='steelblue', size=1) +     stat_smooth()


# In[36]:


ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) +     geom_point() +     scale_color_gradient(low = 'red', high = 'blue')


# In[37]:


plt.figure(figsize=(12,12))
sns.jointplot(x=train_df['lotsizesquarefeet'].values, y=train_df['logerror'].values, size=10, color='g')
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Lot area size', fontsize=12)
plt.title("Lot Area Vs Log error", fontsize=15)
plt.show()


# In[38]:


sns.barplot(x='numberofstories',y='logerror',data=train_df)


# In[39]:


sns.regplot(x='numberofstories',y='logerror',data=train_df)


# In[40]:


ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) +     geom_now_its_art()


# In[41]:


cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]


# In[42]:


train_df[cat_cols].head(3)


# In[43]:


sns.countplot(train_df['hashottuborspa'])


# In[44]:


parcelid=train_df['parcelid']
logerror=train['logerror']


# In[45]:


# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df.mean(axis=0)
train_df_new = train_df.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()


# In[46]:


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel


# In[47]:


cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[48]:


train_df=train_df.drop(['propertycountylandusecode'],axis=1)
df_test=df_test.drop(['propertycountylandusecode'],axis=1)


# In[49]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values.astype('str')) + list(df_test[f].values.astype('str')))
        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
        df_test[f] = lbl.transform(list(df_test[f].values.astype('str')))


# In[50]:


train_y = train_df['logerror'].values
#cat_cols = ["propertycountylandusecode"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month'], axis=1)
feat_names = train_df.columns.values


# In[51]:


from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)

## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[52]:


model.score(train_df, train_y)


# In[53]:


import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(train_df, train_y)
# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )


# In[54]:


num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model_xgb = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# In[55]:


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model_xgb, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[56]:


df_test=df_test.drop(['ParcelId','201610','201611','201612','201710','201711','201712','parcelid'],axis=1)


# In[57]:


df_test.head(3)


# In[58]:


train_df.head(3)


# In[59]:


df_test.shape,train_df.shape


# In[60]:


#df_test=df_test.drop(['ParcelId','201610','201611','201612','201710','201711','201712','parcelid'],axis=1)


# In[61]:


from sklearn import ensemble
ada_model = ensemble.AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=42)
ada_model.fit(train_df, train_y)
ada_model.score(train_df, train_y)


# In[62]:


df_test=df_test.drop()

