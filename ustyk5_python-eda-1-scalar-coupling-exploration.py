#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import ast, json

from datetime import datetime
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[3]:


max_scalar_coupling_constant = train.sort_values(by="scalar_coupling_constant", ascending=False).groupby('type').head(1)
min_scalar_coupling_constant = train.sort_values(by="scalar_coupling_constant", ascending=True).groupby('type').head(1)

min_sc = min_scalar_coupling_constant[['type', 'scalar_coupling_constant']]
max_sc = max_scalar_coupling_constant[['type', 'scalar_coupling_constant']]

sc_type = max_sc.join(min_sc.set_index('type'), on='type', lsuffix='_max', rsuffix='_min')
sc_type = sc_type.assign(delta=sc_type['scalar_coupling_constant_max']-sc_type['scalar_coupling_constant_min'])
sc_type.rename(columns = {"scalar_coupling_constant_min": "min", 
                     "scalar_coupling_constant_max":"max"}, 
                                 inplace = True) 
sc_type


# In[4]:


sc_type.set_index('type')      .reindex(sc_type.set_index('type').sum().sort_values().index, axis=1)      .T.plot(kind='bar', stacked=False,
              colormap=ListedColormap(sns.diverging_palette(145, 280, s=85, l=25, n=7)), 
              figsize=(18,9))
plt.xticks(rotation='horizontal')
plt.tick_params(labelsize=20)
plt.show()


# In[5]:


sc_type['j_number'] = sc_type['type'].astype(str).str[0]
sc_type.groupby(['j_number']).mean()
sc_type.groupby(
    ['j_number']
).agg(
    {
        'delta': ['mean'],
    }
)


# In[6]:


train['j_number'] = train['type'].astype(str).str[0]
j_number_group = train.groupby(
    ['j_number']
).agg(
    {
        # find the min, max, and sum of the duration column
        'scalar_coupling_constant': ['mean', 'max', 'min'],
         # find the number of network type entries
        'type': ["count"],
        # min, first, and number of unique dates per group
        'atom_index_1': ['nunique'],
        'atom_index_0': ['nunique']
    }
)
j_number_group


# In[7]:


j_number_group['scalar_coupling_constant'].plot(kind='bar',stacked=False,figsize=(8,8))
plt.xticks(rotation='horizontal')
plt.show()


# In[8]:


train['atom_0'] = train['type'].astype(str).str[2]
train['atom_1'] = train['type'].astype(str).str[3]
train.head()


# In[9]:


train.groupby(
    ['atom_0', 'atom_1']
).agg(
    {
        'atom_index_0': ['nunique'],
        'atom_index_1': ['nunique'],
        'j_number': ['nunique']
    }
)


# In[10]:


n_two_hh = train['type'][train['type']=='2JHH'].count()
n_three_hh = train['type'][train['type']=='3JHH'].count()
two_j_hh_mean = train['scalar_coupling_constant'][train['type']=='3JHH'].mean()
three_j_hh_mean = train['scalar_coupling_constant'][train['type']=='2JHH'].mean()
all_data = train['type'].count()
results = {}
for i in train['type'].unique():
    count = train['type'][train['type']==i].count()
    perc = (count/all_data)*100
    results[i] = perc
plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.figsize=(18,9)
plt.show()

