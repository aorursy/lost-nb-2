#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import os
get_ipython().system('ls -GFlash --color ../input/')


# In[2]:


X_test = pd.read_csv('../input/X_test.csv')
X_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[3]:


X_train.head()


# In[4]:


y_train['count'] = 1
y_train.groupby('surface').sum()['count']     .sort_values(ascending=True)     .plot(kind='barh', color='grey', figsize=(15, 5), title='Count of Surface Type')
plt.show()


# In[5]:


from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
output_notebook()


# In[6]:


for surface in y_train['surface'].unique():
    first = y_train.loc[y_train['surface'] == surface].index[0]
    series = X_train.loc[X_train['series_id'] == first]
    p = figure(width=1000, height=200, title='{}- Angular Velocity'.format(surface))
    source = ColumnDataSource(series)
    avX = p.line(x='measurement_number', y='angular_velocity_X', source=source, color='red')
    p.add_tools(HoverTool(tooltips='angular_velocity_X', renderers=[avX]))
    avY = p.line(x='measurement_number', y='angular_velocity_Y', source=source, color='blue')
    p.add_tools(HoverTool(tooltips='angular_velocity_Y', renderers=[avY]))
    avZ = p.line(x='measurement_number', y='angular_velocity_Z', source=source, color='orange')
    p.add_tools(HoverTool(tooltips='angular_velocity_Z', renderers=[avZ]))
    show(p)


# In[7]:


for surface in y_train['surface'].unique():
    first = y_train.loc[y_train['surface'] == surface].index[0]
    series = X_train.loc[X_train['series_id'] == first]
    p = figure(width=1000, height=200, title='{}- Orientation'.format(surface))
    source = ColumnDataSource(series)
    avX = p.line(x='measurement_number', y='orientation_X', source=source, color='red')
    p.add_tools(HoverTool(tooltips='orientation_X', renderers=[avX]))
    avY = p.line(x='measurement_number', y='orientation_Y', source=source, color='blue')
    p.add_tools(HoverTool(tooltips='orientation_Y', renderers=[avY]))
    avZ = p.line(x='measurement_number', y='orientation_Z', source=source, color='orange')
    p.add_tools(HoverTool(tooltips='orientation_Z', renderers=[avZ]))
    show(p)


# In[8]:


for surface in y_train['surface'].unique():
    first = y_train.loc[y_train['surface'] == surface].index[0]
    series = X_train.loc[X_train['series_id'] == first]
    p = figure(width=1000, height=200, title='{}- linear acceleration'.format(surface))
    source = ColumnDataSource(series)
    avX = p.line(x='measurement_number', y='linear_acceleration_X', source=source, color='red')
    p.add_tools(HoverTool(tooltips='linear_acceleration_X', renderers=[avX]))
    avY = p.line(x='measurement_number', y='linear_acceleration_Y', source=source, color='blue')
    p.add_tools(HoverTool(tooltips='linear_acceleration_Y', renderers=[avY]))
    avZ = p.line(x='measurement_number', y='linear_acceleration_Z', source=source, color='orange')
    p.add_tools(HoverTool(tooltips='linear_acceleration_Z', renderers=[avZ]))
    show(p)


# In[9]:




