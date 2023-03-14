#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Depedencies
import numpy as np 
import pandas as pd 
from collections import Counter
import os
print(os.listdir("../input"))


# In[2]:


train_df = pd.read_csv('../input/train_relationships.csv')
train_df.head()


# In[3]:


train_df.shape


# In[4]:


# Number of families in the train folder
train = get_ipython().getoutput('ls ../input/train/')
len(train)


# In[5]:


# Number of unlabelled images
test = get_ipython().getoutput('ls ../input/test/')
len(test)


# In[6]:


# fastai and torch imports
import torch
from fastai.vision import *
from fastai.metrics import *

np.random.seed(7)
torch.cuda.manual_seed_all(7)


# In[7]:


# Looking at the naming conventions
train_folder = Path('../input/train')
train_folder.ls()


# In[8]:


# Looking at specific folder
specific_folder = train_folder/'F0768'
specific_folder.ls()


# In[9]:


# Looking at individual images belonging to a particular folder
more_specific = train_folder/'F0768/MID4'
more_specific.ls()


# In[10]:


sample1 = open_image('../input/train/F0768/MID4/P08113_face1.jpg')
show_image(sample1)


# In[11]:


sample2 = open_image('../input/train/F0768/MID4/P08114_face1.jpg')
show_image(sample2)


# In[12]:


sample3 = open_image('../input/train/F0768/MID4/P12113_face2.jpg')
show_image(sample3)


# In[13]:


a = []
for i in train_df.p1:
    try:
        i2=i
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        index_to_drop = train_df.p1[train_df.p1==i2].index.tolist()
        # print(index_to_drop)
        train_df.drop(train_df.index[index_to_drop], inplace=True)

len(train_df), len(a)


# In[14]:


first_person = pd.DataFrame(train_df.p1)
second_person = pd.DataFrame(train_df.p2)
len(first_person)==len(second_person)


# In[15]:


print(first_person.head(3))
print('\n')
print(second_person.head(3))


# In[16]:


# Features DataFrame
a = []
for i in first_person.p1:
    # Suspicious code block since there should not be any FileNotFoundError now
    try:
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        pass

b = []
for i in a:
    for ii in i:
        b.append(ii)
        
features = pd.DataFrame()
features['Path'] = b
features.head()


# In[17]:


# Labels DataFrame
a = []
for i in second_person.p2:
    try:
        i = Path('../input/train/'+i)
        a.append(i.ls())
    except:
        pass

b = []
for i in a:
    for ii in i:
        b.append(ii)        
labels = pd.DataFrame()
labels['Labels'] = b
labels.head()


# In[18]:


len(features), len(labels)


# In[19]:


features_new = features[:16307]
features_new['Labels'] = labels['Labels']
features_new.head()


# In[20]:


features_databunch = ImageImageList.from_df(features_new, path='.')
len(features_databunch)


# In[21]:


img = open_image(features_databunch.items[0])
img.shape


# In[22]:


open_image(features_databunch.items[0])


# In[23]:


databunch = features_databunch.split_by_rand_pct(0.1, seed=7)        .label_from_df(cols='Labels')        .transform(get_transforms(), size=224, tfm_y=True)        .databunch(bs=64).normalize(imagenet_stats, do_y=True)


# In[24]:


databunch.show_batch(rows=4, figsize=(8,8))


# In[25]:


learner = unet_learner(databunch, models.resnet34, wd=1e-3, blur=True, norm_type=NormType.Weight,
                            y_range=(-3.,3.), loss_func=MSELossFlat()).to_fp16()
learner.lr_find()
learner.recorder.plot()


# In[26]:


learner.fit_one_cycle(2, pct_start=0.8, max_lr=slice(1e-05, 1e-03))


# In[27]:


learner.unfreeze()
learner.fit_one_cycle(2, slice(1e-5,1e-3))


# In[28]:


learner.validate()

