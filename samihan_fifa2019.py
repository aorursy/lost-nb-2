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


get_ipython().system('pip install fastai==0.7.0')


# In[3]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[4]:


get_ipython().system('pip install ggplot')


# In[5]:


import pathlib
PATH = '../input/fifa2019wages'
working_path = '/kaggle/working/'

path = pathlib.Path(PATH)
path_w = pathlib.Path(working_path)


# In[6]:


get_ipython().system('head -n 100000 {path}/FifaTrainNew.csv > {path_w}/FifaTrainNew.csv')


# In[7]:


df_raw = pd.read_csv(f'{working_path}/FifaTrainNew.csv', low_memory=False, 
                     parse_dates=["Joined",'Contract Valid Until'])


# In[8]:


df_raw.head()


# In[9]:


df_raw.tail().T


# In[10]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[11]:


display_all(df_raw.tail().T)


# In[12]:


df_raw['Contract Valid Until'].unique()


# In[13]:


display_all(df_raw.describe(include='all').T)


# In[14]:


df_raw = df_raw.drop('Ob' , axis = 1)


# In[15]:


train_cats(df_raw)


# In[16]:


df_raw.Club.unique()


# In[17]:


display_all(df_raw.head())


# In[18]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[19]:


c=0
for col in df_raw.columns:
    if(str(df_raw[col].dtype)!="category"):
        print("'"+col+"',")


# In[20]:


df_raw['LongPassing']


# In[21]:


add_datepart(df_raw, 'Joined')
add_datepart(df_raw, 'Contract Valid Until')


# In[22]:


df_trn, y_trn, nas= proc_df(df_raw,y_fld= 'WageNew')


# In[23]:


df_raw


# In[24]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df_trn, y)
m.score(df_trn,y)


# In[25]:


get_ipython().run_line_magic('pinfo', 'proc_df')


# In[26]:


len(df)


# In[27]:


def split_vals(a,n): return a[:n], a[n:]
n_valid = 6000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[28]:


def print_score(m,imp_cols=None):
    if(imp_cols is not None):
        res = [ m.score(X_train[imp_cols], y_train), m.score(X_valid[imp_cols], y_valid)]
    else:
        res = [ m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[29]:


len(df_raw)


# In[30]:


#set_rf_samples(6000)
reset_rf_samples()


# In[31]:


get_ipython().run_line_magic('pinfo2', 'set_rf_samples')


# In[32]:


m = RandomForestRegressor(n_estimators=60, min_samples_leaf=3,max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[33]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
np.mean(preds[:,0]), np.std(preds[:,0])


# In[34]:


def get_preds(t): return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:,0]), np.std(preds[:,0])


# In[35]:


display_all(df_trn)


# In[36]:


fi = rf_feat_importance(m, df_trn); fi[:10]


# In[37]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[38]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[39]:


plot_fi(fi[:30]);


# In[40]:


plot_fi(fi[:12]);


# In[41]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[42]:


to_keep


# In[43]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[44]:


m = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[45]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# In[46]:


fi


# In[47]:


df_trn2, y_trn, nas = proc_df(df_raw, 'WageNew', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[48]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25]);


# In[49]:


from scipy.cluster import hierarchy as hc


# In[50]:


get_ipython().run_line_magic('pinfo', 'scipy.stats.spearmanr')


# In[51]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[52]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=80, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[53]:


get_oob(df_keep)


# In[54]:


df_keep.columns


# In[55]:


for c in ( 'LCM', 'RCM', 'RAM', 'CAM', 'RW' , 'LM'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# In[56]:


to_drop = ['CM', 'CAM', 'RW']
get_oob(df_keep.drop(to_drop, axis=1))


# In[57]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[58]:


np.save('/kaggle/working/keep_cols.npy', np.array(df_keep.columns))


# In[59]:


keep_cols = np.load('/kaggle/working/keep_cols.npy' , allow_pickle=True)
df_keep = df_trn[keep_cols]


# In[60]:


reset_rf_samples()


# In[61]:


m = RandomForestRegressor(n_estimators=80, min_samples_leaf=4, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[62]:


from pdpbox import pdp
from plotnine import *


# In[63]:


reset_rf_samples()


# In[64]:


df_trn2, y_trn, nas = proc_df(df_raw, 'WageNew', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train);


# In[65]:


print_score(m)


# In[66]:


plot_fi(rf_feat_importance(m, df_trn2)[:10]);


# In[67]:


df_raw.plot('Reactions', 'JoinedElapsed', 'scatter', alpha=0.01, figsize=(10,8));


# In[68]:


sum(df_raw['Reactions']>45)


# In[69]:


x_all = get_sample(df_raw[df_raw.Reactions>40], 500)


# In[70]:


get_ipython().run_line_magic('pinfo2', 'get_sample')


# In[71]:


get_ipython().system('pip install scikit-misc')


# In[72]:


ggplot(x_all, aes('Reactions', 'WageNew'))+stat_smooth(se=True, method='loess')


# In[73]:


x = get_sample(X_train[X_train.Reactions>45], 500)
def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, x.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)


# In[74]:


plot_pdp('Reactions')


# In[75]:


feats = ['JoinedElapsed', 'Reactions']
p = pdp.pdp_interact(m, x, x.columns, feats)
pdp.pdp_interact_plot(p, feats)


# In[76]:


df_keep.describe


# In[77]:


df_keep.Age.describe


# In[78]:


get_ipython().system('pip install treeinterpreter')


# In[79]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:





# In[80]:


df_raw = pd.read_csv(f'{working_path}/FifaTrainNew.csv', low_memory=False, 
                     parse_dates=["Joined",'Contract Valid Until'])


# In[81]:


obj_cols = df_raw.dtypes[df_raw.dtypes == object].index.tolist()


# In[82]:


for col in obj_cols:
    print(f'{col}\t\t{df_raw[col].unique()}')


# In[83]:


df_raw.drop('Ob',axis=1,inplace=True)


# In[84]:


train_cats(df_raw)


# In[85]:


for col in obj_cols:
    print(f'{col}\t{df_raw[col].unique()}')


# In[86]:


add_datepart(df_raw, 'Joined')
add_datepart(df_raw, 'Contract Valid Until')


# In[87]:


def split_vals(a,n): return a[:n],a[n:]

df_trn, y_trn, nas = proc_df(df_raw,y_fld='WageNew', max_n_cat=10)

n_valid = 7500
n_train = len(df_trn) - n_valid

X_train, X_valid = split_vals(df_trn,n_train)
y_train, y_valid = split_vals(y_trn, n_train)
train_raw, valid_raw = split_vals(df_raw, n_train)


# In[88]:


set_rf_samples(5000)


# In[89]:


def print_score(m):
    res = [ m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[90]:


def print_score(m,imp_cols=None):
    if(imp_cols is not None):
        res = [ m.score(X_train[imp_cols], y_train), m.score(X_valid[imp_cols], y_valid)]
    else:
        res = [ m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[91]:


m = RandomForestRegressor(n_estimators=1000,min_samples_leaf=3,max_features=0.555,n_jobs=-1,warm_start=True,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[92]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# In[93]:


fi = rf_feat_importance(m,df_trn)
def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,10),legend=True)


# In[94]:


plot_fi(fi[fi['imp'] > 0.005])


# In[95]:


imps_cols = fi[fi['imp'] > 0.002]['cols'].tolist()
imps_cols


# In[96]:


len(imps_cols)


# In[97]:


X_train.columns.tolist()


# In[98]:


m = RandomForestRegressor(n_estimators=1000,min_samples_leaf=3,max_features=0.555,n_jobs=-1,warm_start=True,oob_score=True)
m.fit(X_train[imps_cols],y_train)
print_score(m,imps_cols)


# In[99]:


fi = rf_feat_importance(m,X_train[imps_cols])
def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,10),legend=True)
plot_fi(fi)


# In[100]:


plot_fi(fi[:25])


# In[101]:


imps_cols = fi[:25]['cols'].tolist()
df_trn[imps_cols].dtypes


# In[102]:


from scipy.cluster import hierarchy as hc


# In[103]:


corr = np.round(scipy.stats.spearmanr(X_train[imps_cols]).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_train[imps_cols].columns, orientation='left', leaf_font_size=16)
plt.show()


# In[104]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=1000,min_samples_leaf=3,max_features=0.555,n_jobs=-1,warm_start=True,oob_score=True)
    x,_ = split_vals(df,n_train)
    m.fit(x,y_train)
    return m.oob_score_


# In[105]:


get_oob(X_train[imps_cols])


# In[106]:


for col in ['RCM','LCM','CM','RDM','LDM','CDM']:
    print(col,get_oob(X_train[imps_cols].drop(col,axis=1)))


# In[107]:


to_drop = ['RCM','LCM','CM','RDM','LDM','CDM']
get_oob(X_train[imps_cols].drop(to_drop,axis=1))


# In[108]:


m = RandomForestRegressor(n_estimators=1000,min_samples_leaf=3,max_features=0.555,n_jobs=-1,warm_start=True,oob_score=True)
cols = X_train[imps_cols].drop(to_drop,axis=1).columns
m.fit(X_train[cols],y_train)
print_score(m,imp_cols=cols)


# In[109]:


df_test = pd.read_csv(f'{path}/FifaNoY.csv', low_memory=False, 
                     parse_dates=["Joined",'Contract Valid Until'])


# In[110]:


train_cats(df_test)


# In[111]:


add_datepart(df_test, 'Joined')
add_datepart(df_test, 'Contract Valid Until')


# In[112]:


df_test1, y_trn, nas = proc_df(df_test, max_n_cat=10)


# In[113]:


predictions = m.predict(df_test1[cols])


# In[114]:


get_ipython().run_line_magic('pinfo', 'proc_df')


# In[115]:


submission = pd.DataFrame({'Ob':df_test1['Ob'],'WageNew':predictions})


# In[116]:


submission


# In[117]:


filename = '/kaggle/working/FIFA2019Wages.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[118]:


ls


# In[119]:


pwd


# In[ ]:




