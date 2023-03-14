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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


events = pd.read_csv('../input/events.csv')
events.head()


# In[4]:


events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str})
events.head()


# In[5]:


Plot maps showing the locations of events


# In[6]:


# Set up plot
events_sample = events.sample(n=100000)
plt.figure(1, figsize=(12, 6))

# Merator of Wold
m1 = Basemap(projection='merc',
            llcrnrlat=-60,
            urcrnrlat=65,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=0,
            resolution='c')

m1.fillcontinents(color='#191919', lake_color='#000000')
m1.drawmapboundary(fill_color='#000000')
m1.drawcountries(linewidth=0.1, color='w')

# Plot the date
mxy = m1(events_sample['longitude'].tolist(), events_sample['latitude'].tolist())
m1.scatter(mxy[0], mxy[1], c='#1292db', s=3,lw=0, alpha=1, zorder=5)

plt.title('Global view of events')
plt.show()


# In[7]:


Plot the events in china


# In[8]:


lon_min, lon_max = 75, 135
lat_min, lat_max = 15, 55

idx_china = (events['longitude'] > lon_min) &            (events['longitude'] < lon_max) &            (events['latitude'] > lat_min) &            (events['latitude'] < lat_max) 
           
events_china = events[idx_china].sample(n=100000)

plt.figure(2, figsize=(12,6))
m2 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=35,
             resolution='i')            

m2.fillcontinents(color='#191919', lake_color='#000000')
m2.drawmapboundary(fill_color='#000000')
m2.drawcountries(linewidth=0.1, color='w')

# Plot the date
mxy = m2(events_sample['longitude'].tolist(), events_sample['latitude'].tolist())
m2.scatter(mxy[0], mxy[1], c='#1292db', s=3,lw=0, alpha=1, zorder=5)

plt.title('Events in china')
plt.show()


# In[9]:




