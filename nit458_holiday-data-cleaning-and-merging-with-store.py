#!/usr/bin/env python
# coding: utf-8

# In[1]:


from subprocess import check_output
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(check_output(["ls", "../input"]).decode("utf8"))
import re
from datetime import datetime as dt
from datetime import timedelta# Any results you write to the current directory are saved as output.


# In[2]:


holiday = pd.read_csv('../input/holidays_events.csv')
stores = pd.read_csv('../input/stores.csv')


# In[3]:


# converting date into datetime format
holiday['date'] = pd.to_datetime(holiday['date'])
print(holiday['description'].unique())


# In[4]:


# all bridge-type has puente (puente actually mean bridge), will remove it 
# all work day type has recupero (recupero means recovery), will remove it
# almost all Transfer have traslado (it actually means transfer), will remove it
# as all of these information are already contained in type
hol_groups = holiday.groupby('type')
for c,group in hol_groups:
    print(c,group['description'].unique())


# In[5]:


# for Regional and Local holidays, 
# name locale_name is part of description as well (will remove that)
# as this info is already contained in locale_name
hol_groups2 = holiday.groupby(['locale','locale_name'])
for c,group in hol_groups2:
    print(c,group['description'].unique())


# In[6]:


# number just mean consecutive events which is already given by there dates, 
# hence will remove them from description
navidadl = ['Navidad-4','Navidad-3','Navidad-2','Navidad-1','Navidad','Navidad+1']
print(holiday[holiday['description'].isin(navidadl)][['date','description']])
terr_hol = ['Terremoto Manabi','Terremoto Manabi+1','Terremoto Manabi+2',
            'Terremoto Manabi+3','Terremoto Manabi+4','Terremoto Manabi+5',
            'Terremoto Manabi+6','Terremoto Manabi+7','Terremoto Manabi+8',
            'Terremoto Manabi+9','Terremoto Manabi+10','Terremoto Manabi+11',
            'Terremoto Manabi+12','Terremoto Manabi+13','Terremoto Manabi+14',
            'Terremoto Manabi+15','Terremoto Manabi+16','Terremoto Manabi+17',
            'Terremoto Manabi+18','Terremoto Manabi+19','Terremoto Manabi+20',
            'Terremoto Manabi+21','Terremoto Manabi+22','Terremoto Manabi+23',
            'Terremoto Manabi+24','Terremoto Manabi+25','Terremoto Manabi+26',
            'Terremoto Manabi+27','Terremoto Manabi+28','Terremoto Manabi+29',
            'Terremoto Manabi+30']
print(holiday[holiday['description'].isin(terr_hol)][['date','description']])


# In[7]:


# de and del are just stop words, will remove them too
# seriously reduces dimensionality of description variable
holiday['description'] = holiday['description'].str.lower().replace(to_replace='[^a-z ]', value='', regex=True)
holiday['description'] = holiday.apply(lambda x: x['description'].replace(x['locale_name'].lower(),''),axis=1)
holiday['description'] = holiday['description']. replace(to_replace='traslado ',value='',regex=True).replace(to_replace='puente ',value='',regex=True). replace(to_replace='del ',value='',regex=True).replace(to_replace='de ',value='',regex=True). replace(to_replace='mundial futbol brasil',value='mfb',regex=True). replace(to_replace='recupero ',value='',regex=True).replace(to_replace='santo domingo',value='',regex=True)
holiday['description'] = holiday['description'].apply(lambda x: x.strip())
print(holiday['description'].unique())


# In[8]:


# now some holidays are city level and will impact only that city store (locale == 'Local')
# and some are state level and will only impact store in that state (locale == 'Regional')
# and national onces will impact all of stores in Ecuador
# and hence city level and state level will need us to merge store and date some how
## Creating date only dataframe
start_dt = dt(year=2013,month=1,day=1)
dt_only = pd.DataFrame([start_dt + timedelta(days=i) for i in range(1704)],columns=['date'])
print(dt_only.max())
print(dt_only.min())


# In[9]:


# taking a cross product of store and date
dt_only['dummy'] = 1
stores['dummy'] = 1
dt_store = pd.merge(dt_only,stores,on=['dummy']).drop('dummy',axis=1)
print(dt_store.head(10))


# In[10]:


# seperating out National, city and state level holidays
national_h = holiday[holiday['locale'] =='National'].drop(['locale','locale_name'],axis=1)
national_h.columns = ['date','ntype','ndescription','ntransfered']
city_h = holiday[holiday['locale'] == 'Local'].drop('locale',axis=1)
city_h.columns = ['date','ctype','city','cdescription','ctransfered']
state_h=holiday[holiday['locale'] == 'Regional'].drop('locale',axis=1)
state_h.columns = ['date','stype','state','sdescription','stransfered']
print(national_h.head())
print(city_h.head())
print(state_h.head())


# In[11]:


# merging with date store dataframe  
dt_store_nat = pd.merge(dt_store,national_h,on='date',how='left')
print(dt_store_nat.shape)
dt_store_nat_ct = pd.merge(dt_store_nat,city_h,on=['date','city'],how='left')
print(dt_store_nat_ct.shape)
dt_store_nat_ct_st = pd.merge(dt_store_nat_ct,state_h,on=['date','state'],how='left')
print(dt_store_nat_ct_st.shape)
print(dt_store_nat_ct_st.head())


# In[12]:


# You can use holiday features now after on hot encoding or whatever you would like to do
# you can choose to keep only one of (national,city,state) features by using some hirachial ordering
# now you can use this dataframe to merge with main data set on ['date','store_nbr']
# Note: the increase in number of records on merging city_h
# happening because of transfer and Additional on falling on same date
# My suggestion is to remove record with ctype = 'Transfer' and city = 'Guayaquil'
citygroups = city_h.groupby(['date','city'])
for c,group in citygroups:
    if group.shape[0] > 1:
        print(group)


# In[13]:




