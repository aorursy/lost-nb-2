#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


label = pd.read_csv('../input/app_labels.csv')
cat = pd.read_csv('../input/label_categories.csv')
app_cat = pd.merge(label, cat, how='left', on='label_id')
print(app_cat.head())
del label, cat


# In[3]:


app_cat['category'].value_counts()


# In[4]:


def to_Finance(x):
    if re.search('([iI]ncome)|([pP]rofitabil)|([lL]iquid)|([rR]isk)|([bB]ank)|([fF]uture)|([fF]und)|([sS]tock)|([sS]hare)',
                 x) is not None:
        return('Finance')
    if re.search('([fF]inanc)|([pP]ay)|(P2P)|([iI]nsura)|([lL]oan)|([cC]ard)|([mM]etal)|'
                 '([cC]ost)|([wW]ealth)|([bB]roker)|([bB]usiness)|([eE]xchange)', x) is not None:
        return('Finance')
    if x in ['High Flow', 'Housekeeping', 'Accounting', 'Debit and credit', 'Recipes', 'Heritage Foundation', 'IMF',]:
        return('Finance')
    else:
        return(x)

app_cat['general_cat'] = app_cat['category'].apply(to_Finance)


# In[5]:


app_finance = app_cat[app_cat['general_cat']=='Finance']
print(app_finance.head())
del app_cat


# In[6]:


app_ev = pd.read_csv('../input/app_events.csv')
ev = pd.read_csv('../input/events.csv')
events = pd.merge(app_ev, ev, how='inner', on='event_id')
del app_ev, ev

print(events.head())


# In[7]:


finance_events = events[events['app_id'].isin(app_finance['app_id'])]
del events, app_finance
print(finance_events.head())


# In[8]:


print(finance_events.shape)


# In[9]:


# Sample it down to only the China region
lon_min, lon_max = 75, 135
lat_min, lat_max = 15, 55

idx_china = (finance_events["longitude"] > lon_min) &            (finance_events["longitude"] < lon_max) &            (finance_events["latitude"] > lat_min) &            (finance_events["latitude"] < lat_max)

china_finance = finance_events[idx_china]

print (china_finance.shape)


# In[10]:


plt.figure(1, figsize=(12,6))

m_1 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='i')

m_1.drawmapboundary(fill_color='#000000')                # black background
m_1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m_1(china_finance["longitude"].tolist(), china_finance["latitude"].tolist())
m_1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=0.05, zorder=5)

plt.title("China view of finance events")
plt.show()


# In[11]:


# Sample it down to only the Beijing region
lon_min, lon_max = 116, 117
lat_min, lat_max = 39.75, 40.25

idx_beijing = (finance_events["longitude"]>lon_min) &              (finance_events["longitude"]<lon_max) &              (finance_events["latitude"]>lat_min) &              (finance_events["latitude"]<lat_max)

beijing_finance = finance_events[idx_beijing]

# Mercator of Beijing
plt.figure(2, figsize=(12,6))

m_2 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='c')

m_2.drawmapboundary(fill_color='#000000')                # black background
m_2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m_2(beijing_finance["longitude"].tolist(), beijing_finance["latitude"].tolist())
m_2.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)

plt.title("Beijing view of events")
plt.show()


# In[12]:


# Load the train data and join on the events
df_train = pd.read_csv("../input/gender_age_train.csv")

bj_finance_demo = pd.merge(df_train, beijing_finance, how="inner", on="device_id")

df_m = bj_finance_demo[bj_finance_demo["gender"]=="M"]
df_f = bj_finance_demo[bj_finance_demo["gender"]=="F"]

print(df_m.shape, df_f.shape)


# In[13]:


df_m['is_active'].value_counts()[1] / len(df_m['is_active'])


# In[14]:


df_f['is_active'].value_counts()[1] / len(df_f['is_active'])


# In[15]:


def bj_map():
    bj_map= Basemap(projection='merc',
                 llcrnrlat=lat_min,
                 urcrnrlat=lat_max,
                 llcrnrlon=lon_min,
                 urcrnrlon=lon_max,
                 lat_ts=35,
                 resolution='c')
    bj_map.drawmapboundary(fill_color='#000000')              
    bj_map.drawcountries(linewidth=0.1, color="w")        
    return bj_map


# In[16]:


plt.figure(3, figsize=(12,6))

# Male/female plot
# df_m and df_f 
plt.subplot(321)
m3a = bj_map()
mxy = m3a(df_m["longitude"].tolist(), df_m["latitude"].tolist())
m3a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male events in Beijing")

plt.subplot(322)
m3b = bj_map()
mxy = m3b(df_f["longitude"].tolist(), df_f["latitude"].tolist())
m3b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female events in Beijing")


# Active Male/female plot
df_m_active = df_m[df_m['is_active']==1]
df_f_active = df_f[df_f['is_active']==1]

plt.subplot(323)
m4a = bj_map()
mxy = m4a(df_m_active["longitude"].tolist(), df_m_active["latitude"].tolist())
m4a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male active events in Beijing")

plt.subplot(324)
m4b = bj_map()
mxy = m4b(df_f_active["longitude"].tolist(), df_f_active["latitude"].tolist())
m4b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female active events in Beijing")


# Inactive Male/female plot
df_m_inactive = df_m[df_m['is_active']==0]
df_f_inactive = df_f[df_f['is_active']==0]

plt.subplot(325)
m5a = bj_map() 
mxy = m5a(df_m_inactive["longitude"].tolist(), df_m_inactive["latitude"].tolist())
m5a.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.1, zorder=5)
plt.title("Male inactive events in Beijing")

plt.subplot(326)
m5b = bj_map()
mxy = m5b(df_f_inactive["longitude"].tolist(), df_f_inactive["latitude"].tolist())
m5b.scatter(mxy[0], mxy[1], s=5, c="#fd3096", lw=0, alpha=0.1, zorder=5)
plt.title("Female inactive events in Beijing")

plt.show()


# In[17]:


# plot time of events

plt.figure(4, figsize=(12,18))
plt.subplot(611)
plt.hist(df_m['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male")

plt.subplot(612)
plt.hist(df_f['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female")

plt.subplot(613)
plt.hist(df_m_active['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male active")

plt.subplot(614)
plt.hist(df_f_active['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female active")

plt.subplot(615)
plt.hist(df_m_inactive['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("male inactive")

plt.subplot(616)
plt.hist(df_f_inactive['timestamp'].map(lambda x: pd.to_datetime(x).hour), bins=24)
plt.xticks(np.arange(0, 24, 1.0))
plt.title("female inactive")

plt.subplots_adjust(hspace=.8)
plt.show()

