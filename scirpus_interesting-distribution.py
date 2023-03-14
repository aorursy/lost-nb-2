#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ls ../input


# In[3]:


df_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
df_train.shape


# In[4]:


df_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
df_test.shape


# In[5]:


df_test.day.unique()


# In[6]:


df_test.ord_3.unique()


# In[7]:


for c in df_train.columns[1:-1]:
    print(c)
    le = LabelEncoder()
    le.fit(list(df_train.loc[~df_train[c].isnull(),c])+list(df_test.loc[~df_test[c].isnull(),c]))
    df_train.loc[~df_train[c].isnull(),c] = le.transform(df_train.loc[~df_train[c].isnull(),c])
    df_test.loc[~df_test[c].isnull(),c] = le.transform(df_test.loc[~df_test[c].isnull(),c])


# In[8]:


df_train.head()


# In[9]:


df_train = df_train.set_index('id')
df_test = df_test.set_index('id')
y_train = df_train.target
del df_train['target']


# In[10]:


df_train.head()


# In[11]:


import category_encoders as ce
cat_feat_to_encode = df_train.columns[:].tolist()
smoothing=0.20

oof = pd.DataFrame([])
from sklearn.model_selection import StratifiedKFold
for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state= 1032, shuffle=True).split(df_train, y_train):
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(df_train.iloc[tr_idx, :], y_train.iloc[tr_idx])
    oof = oof.append(ce_target_encoder.transform(df_train.iloc[oof_idx, :]), ignore_index=False)
ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
ce_target_encoder.fit(df_train, y_train)
new_train = oof.sort_index()
new_test = ce_target_encoder.transform(df_test)


# In[12]:


for i,c in enumerate(new_train.columns):
    print(c, roc_auc_score(y_train,new_train[c]))


# In[13]:


new_train.head()


# In[14]:


new_test.head()


# In[15]:


new_train['target'] = y_train
new_test['target'] = -1


# In[16]:


alldata = pd.concat([df_train,df_test])
newalldata = pd.concat([new_train,new_test])


# In[17]:


for c in alldata.columns:
    newalldata.loc[alldata[c].isnull(),c] = np.nan 


# In[18]:


newalldata.head()


# In[19]:


for c in newalldata.columns[:-1]:
    newalldata.loc[~alldata[c].isnull(),c] -= newalldata.loc[~alldata[c].isnull(),c].mean()
    newalldata.loc[~alldata[c].isnull(),c] /= newalldata.loc[~alldata[c].isnull(),c].std()


# In[20]:


newalldata.head()


# In[21]:


def Output(p):
    return 1.0/(1.0+np.exp(-p))

def GPTEI(data):
    return (0.006038*np.tanh(np.real(((np.cosh((np.cosh((((np.cosh((data["day"]))) - (np.tanh((complex(0,1)*np.conjugate(data["day"])))))))))) + (((complex(8.0)) - (complex(0,1)*np.conjugate(np.tanh((np.tanh((complex(0,1)*np.conjugate(((np.cosh((((np.tanh((complex(11.26747322082519531)))) - (np.tanh((data["ord_3"]))))))) / 2.0))))))))))))))


# In[22]:


x_tr = pd.DataFrame()
x_tr['isTrain'] = np.ones(df_train.shape[0])
x_te = pd.DataFrame()
x_te['isTrain'] = np.zeros(df_test.shape[0])
x_al = pd.concat([x_tr,x_te])
x_al['gp'] = GPTEI(newalldata.astype(complex).fillna(complex(0,1)))


# In[23]:


colors = ['g','b']
plt.figure(figsize=(15,15))
plt.scatter(range(x_al.shape[0]),x_al.gp.values,s=1,color=[colors[int(c)] for c in x_al.isTrain.values])


# In[24]:


plt.hist(x_al[:df_train.shape[0]].gp,bins=20,alpha=.5)


# In[25]:


plt.hist(x_al[-df_test.shape[0]:].gp,bins=20,alpha=.5)


# In[ ]:




