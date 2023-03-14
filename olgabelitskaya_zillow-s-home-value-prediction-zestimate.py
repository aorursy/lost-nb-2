#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Akronim|Roboto&effect=3d|fire-animation');\na,h2 {color:#ff355e; font-family:Roboto;} \nspan {color:black; font-family:Roboto; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt,div.output_area pre {color:slategray;}\ndiv.input_prompt,div.output_subarea {color:#ff355e;}      \ndiv.output_stderr pre {background-color:gainsboro;}  \ndiv.output_stderr {background-color:slategrey;}       \n</style>")


# In[2]:


import warnings; warnings.filterwarnings('ignore')
import pandas as pd,numpy as np,os,sqlite3
import seaborn as sn,pylab as pl
import keras as ks,tensorflow as tf
pl.style.use('seaborn-whitegrid')
np.set_printoptions(precision=8)


# In[3]:


train_2016=pd.read_csv("../input/train_2016_v2.csv",
                       parse_dates=["transactiondate"])
properties_2016=pd.read_csv("../input/properties_2016.csv")
train_2017=pd.read_csv("../input/train_2017.csv",
                       parse_dates=["transactiondate"])
properties_2017=pd.read_csv("../input/properties_2017.csv")


# In[4]:


train_2016.describe(),train_2017.describe()


# In[5]:


properties_2016.head().T


# In[6]:


pl.figure(1,figsize=(12,6))
sn.distplot(train_2016['logerror'],
            color='#ff355e',bins=1000)
pl.ylabel("Distribution")
pl.xlabel("LogError"); pl.xlim(-.5,.5)
pl.suptitle('Zestimate Logerror',fontsize=15);


# In[7]:


data2016=train_2016['transactiondate'].dt.month.value_counts()
data2017=train_2017['transactiondate'].dt.month.value_counts()
pl.figure(1,figsize=(12,4)); pl.subplot(121)
sn.barplot(data2016.index,data2016.values,color='#ff355e',alpha=.7)
pl.ylabel("Number of Occurrences")
pl.xlabel("Month of Transactions 2016")
pl.subplot(122)
sn.barplot(data2017.index,data2017.values,color='crimson',alpha=.7)
pl.ylabel("Number of Occurrences")
pl.xlabel("Month of Transactions 2017")
pl.suptitle('Zestimate Logerror',fontsize=15);


# In[8]:


properties_2016.isnull().sum(axis=0).sort_values().plot(kind='bar',figsize=(12,4),color='#ff355e',title='2016');


# In[9]:


properties_2017.isnull().sum(axis=0).sort_values().plot(kind='bar',figsize=(12,4),color='crimson',title='2017');


# In[10]:


fig,ax=pl.subplots(ncols=2,nrows=1,figsize=(12,6))
properties_2016.plot(kind='scatter',ax=ax[0],
                     x='latitude',y='longitude',
                     color='#ff355e',s=.1)
properties_2017.plot(kind='scatter',ax=ax[1],
                     x='latitude',y='longitude',
                     color='crimson',s=.1);

