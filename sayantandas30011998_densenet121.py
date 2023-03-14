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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


from fastai.vision import *


# In[3]:


df = pd.read_csv("../input/train_labels.csv")
df.head()


# In[4]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[5]:


np.random.seed(42)
src = (ImageItemList.from_csv('../input', 'train_labels.csv', folder='train', suffix='.tif')
       .random_split_by_pct(0.2)
       .label_from_df(label_delim=' '))


# In[6]:


data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))


# In[7]:


data.show_batch(rows=3, figsize=(12,9))


# In[8]:


arch = models.resnet50


# In[9]:


df['label'].value_counts()


# In[10]:


PATH = "../input/"
print("training data")
get_ipython().system('ls -l {PATH}train | grep ^[^dt] | wc -l')


# In[11]:


print("test data")
get_ipython().system('ls -l {PATH}test | grep ^[^dt] | wc -l')


# In[12]:


from os.path import *
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


# In[13]:


get_ipython().system('ls ~/.torch/models')


# In[14]:


import pathlib
arch.path = pathlib.Path('.')


# In[15]:


get_ipython().system('cp ../input/models/resnet50-19c8e357.pth ~/.torch/models/resnet50-19c8e357.pth')


# In[16]:


learn = create_cnn(data,arch, path='.')


# In[17]:


get_ipython().run_line_magic('pinfo2', 'create_cnn')


# In[18]:


learn.lr_find()


# In[19]:


learn.recorder.plot()


# In[20]:


lr = 7.6e-3
learn.fit_one_cycle(5,slice(lr))


# In[21]:


learn.save('stage-1-rn50')


# In[22]:


learn.unfreeze()


# In[23]:


learn.lr_find()


# In[24]:


learn.recorder.plot()


# In[25]:


learn.fit_one_cycle(5,slice(5e-6,lr/5))


# In[26]:


learn.save('stage-2-rn50')


# In[27]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[28]:


learn.freeze()


# In[29]:


learn.lr_find()
learn.recorder.plot()


# In[30]:




