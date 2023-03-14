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


get_ipython().run_cell_magic('bash', '', 'wc -l ../input/train.csv ../input/test.csv ../input/members.csv ../input/songs.csv ../input/song_extra_info.csv')


# In[3]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
members = pd.read_csv('../input/members.csv')
songs = pd.read_csv('../input/songs.csv')
song_extra_info = pd.read_csv('../input/song_extra_info.csv')


# In[4]:


pd.read_csv('../input/songs.csv', quoting=3, error_bad_lines=False)


# In[5]:


# add 1 for header to compare with raw counts
print('train: ',train.shape[0]+1)
print('test: ',test.shape[0]+1)
print('members: ',members.shape[0]+1)
print('songs: ',songs.shape[0]+1)
print('song_extra_info: ',song_extra_info.shape[0]+1)


# In[6]:


songs[songs['artist_name'].str.len() == songs['artist_name'].str.len().max()]['artist_name'].values


# In[7]:


song_extra_info[song_extra_info['name'].str.len() == song_extra_info['name'].str.len().max()]['name'].values


# In[8]:




