#!/usr/bin/env python
# coding: utf-8

# In[1]:


__author__ = "imflash217"
__copyright__ = "FlashAI Labs, 2019"


# In[2]:


# magic commands
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# libraries
import os
import numpy as np
import pandas as pd
import matplotlib 
import scipy
import torch
from fastai.vision import *
from fastai.metrics import error_rate


# In[4]:


# setting paths
x_path = untar_data(URLs.PETS)
print(x_path)
print(x_path.ls())                               #support function by fastai library to list items in a path

x_path_annotations = x_path/'annotations'
x_path_images = x_path/'images'


# In[5]:


# exploring the data
x_fnames = get_image_files(x_path_images)
print(len(x_fnames))
print(type(x_fnames))
x_fnames[0:5]


# In[6]:


# creating the databunch for training
np.random.seed(217)                      # setting random seed for numpy
x_pattern = r'/([^/]+)_\d+.jpg$'         # regex pattern
x_batch_size = 6                           # batch size for loading data for training

# creating databunch
x_data = ImageDataBunch.from_name_re(x_path_images, x_fnames, x_pattern, ds_tfms=get_transforms(), size=224, bs=x_batch_size)         .normalize(imagenet_stats)


# In[7]:


x_data.show_batch(rows=2, figsize=(7,6))


# In[8]:


print(x_data.classes)
print(type(x_data.classes))
print(len(x_data.classes), x_data.c)


# In[9]:


# Transfer Learning from pretrained architecture
x_learn = create_cnn(data=x_data, arch=models.resnet34, metrics=error_rate)
print(x_learn.model)


# In[10]:


# Training the network
x_learn.fit_one_cycle(cyc_len=2)


# In[11]:


# saving the trained model
x_learn.save('stage1')


# In[12]:


x_interp = ClassificationInterpretation.from_learner(x_learn)
x_losses, x_idxs = x_interp.top_losses()
print(len(x_data.valid_ds)==len(x_losses)==len(x_idxs))
x_interp.plot_top_losses(k=9, figsize=(15,15))


# In[13]:


x_interp.plot_confusion_matrix(figsize=(10,10), dpi=90)


# In[14]:


x_interp.most_confused(min_val=3)


# In[15]:


x_learn.unfreeze()                    # unfreezing our model to train it a bit more
x_learn.fit_one_cycle(cyc_len=2)      # training the unfrozen model for another 2 epochs


# In[16]:


x_learn.load('stage1');


# In[17]:


x_learn.lr_find()


# In[18]:


x_learn.recorder.plot()


# In[19]:


x_learn.unfreeze()
x_learn.fit_one_cycle(cyc_len=2, max_lr=slice(1e-6,1e-4))


# In[20]:





# In[20]:





# In[20]:





# In[20]:





# In[20]:





# In[20]:





# In[20]:





# In[20]:





# In[20]:


help(x_data.c)


# In[21]:


get_ipython().run_line_magic('pinfo2', 'ImageDataBunch')


# In[22]:




