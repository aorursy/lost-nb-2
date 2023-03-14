#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#key_1.csv
key = pd.read_csv('../input/web-traffic-time-series-forecasting/key_1.csv',index_col='Page')
print('key info',key.info())

key.head()


# In[3]:


data = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',
                   index_col='Page').T.unstack().reset_index().rename(
    columns={0:'Visits','level_1':'Date'}).dropna(subset=['Visits'])
print('data info',data.info())
data.head()


# In[4]:


def extractPageFeatures(df,key1='Page'):
    """
    Input df: pandas DataFrame/Series
    key: string, column name
    ==============
    returns pandas DataFrame X with feature columns
    
    Example Use:
    
    s = pd.Series(['2NE1_zh.wikipedia.org_all-access_spider',
    'AKB48_en.wikipedia.org_all-access_spider',
    'Angelababy_zh.wikipedia.org_all-access_mobile'])
    
    Xp = extractPageFeatures(s)
    print(Xp)
        Name        Language    Access      Agent
    --  ----------  ----------  ----------  -------
     0  2NE1        zh          all-access  spider
     1  AKB48       en          all-access  spider
     2  Angelababy  zh          all-access  mobile
    
    """
    fnames = ['Name','Language','Access','Agent']
    fnamedict = dict(zip(range(len(fnames)),fnames))
    if type(df) == pd.DataFrame:
        ser = df[key1]
    else:
        ser = df
    X = ser.str.extract(
    '(.+)_(\w{2})\.wiki.+_(.+)_(.+)',expand=True).rename(columns=fnamedict)
    return X
help(extractPageFeatures)


# In[5]:


data_s = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',
                     index_col='Page').rename(columns=pd.to_datetime)
data_s.info()


# In[6]:


# from tabulate import tabulate
# s = pd.Series(['2NE1_zh.wikipedia.org_all-access_spider',
#     'AKB48_en.wikipedia.org_all-access_spider',
#     'Angelababy_zh.wikipedia.org_all-access_mobile'])
# Xp = extractPageFeatures(s)
# print(tabulate(Xp,headers='keys'))


# In[7]:


##### Time idependant  learing model


# In[8]:


# fig, ax = plt.subplots(1,figsize=(8,4))
# data_s.iloc[-200:-195]#.apply(lambda row: row.hist(alpha=.7,normed=True,bins=10),axis=1);
# sns.kdeplot(data_s.iloc[-400:].fillna(0),legend=True,ax=ax);
# ax.axis([-10,20,-10,20]);
# ax.set_title('KDE plot of 400 pages');


# In[9]:


sample = data_s.sample(n=10,axis=0).rename(columns=lambda x: x.strftime('%Y-%m-%d')).reset_index()
fts = extractPageFeatures(sample)
sample['Language']=fts['Language']
sample.dropna(inplace=True)
sample.drop('Page',axis=1,inplace=True)
sample.drop_duplicates(subset=['Language'],inplace=True)
sample.set_index('Language',inplace=True)
sample.iloc[:,-6:]


# In[10]:


g = sns.pairplot(sample.T,kind='reg',diag_kind='kde',
                 diag_kws=dict(shade=True),palette="husl")


# In[11]:


from sklearn.preprocessing import LabelEncoder
X = extractPageFeatures(data_s.index)


# In[12]:


#Read language dictionary
lang_dict = pd.read_csv('../input/wikipedia-language-iso639/lang.csv',
                        index_col=0).iloc[:,0].to_dict()
print(lang_dict)


# In[13]:


lookup_c_i = dict(zip(X.columns.tolist(),range(X.shape[1])))
lookup_c_i


# In[14]:


le = LabelEncoder()
# X_l = X.apply(lambda col: pd.Series(le[lookup_c_i[col.name]].fit_transform(col.astype(str))))
X['encoded_Name'] = le.fit_transform(X['Name'].astype(str))
X['Language'] = X['Language'].map(lang_dict)
X.head()


# In[15]:


fig,ax = plt.subplots(1,figsize=(16,4))
sns.barplot(x="Language", y="encoded_Name",
            hue="Access", 
            data=X,capsize=.01,palette='husl',ax=ax);


# In[16]:


# sns.swarmplot(x="Access", y="encoded_Name",
#             hue="Language", 
#             data=X)
# # sns.swarmplot(x="Language", y="encoded_Name",
# #             hue="Access", 
# #             data=X, color="w", alpha=.5);


# In[17]:


import ipyparallel as ipp


# In[18]:




