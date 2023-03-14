#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from fastai.vision import *


# In[3]:


data_path = Path('/kaggle/input/aerial-cactus-identification')
data_path.ls()


# In[4]:


np.random.seed(4)
data = ImageList.from_csv(data_path, 'train.csv', folder='train/train', cols='id').split_by_rand_pct(0.1).label_from_df(cols='has_cactus').transform(get_transforms(), size=224).databunch().normalize(imagenet_stats)


# In[5]:


# data.show_batch(rows=3, figsize=(9,9))


# In[6]:


print(data.classes)
print(data.c)


# In[7]:


learner = cnn_learner(data, models.resnet34, metrics=error_rate)
learner.model_dir = '/kaggle/working/'


# In[8]:


learner.lr_find()
learner.recorder.plot()


# In[9]:


# find appropriate weights only for newly added layers
learner.freeze()


# In[10]:


learner.fit_one_cycle(5)


# In[11]:


# update weights as per the new layer weights
learner.unfreeze()
learner.fit_one_cycle(3)


# In[12]:


learner.export('/kaggle/working/export.pkl')


# In[13]:


# test using the model
test_imgs = ImageList.from_folder(data_path/'test')


# In[14]:


learn = load_learner('/kaggle/working/', test=test_imgs)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds[:5]


# In[15]:


# information for submissions.csv
img_names = [os.path.basename(os.fspath(fname)) for fname in test_imgs.items]
img_names[:5]


# In[16]:


df = pd.DataFrame({'id': img_names, 'has_cactus': preds.numpy()[:,0]})
df.to_csv('submission.csv', index=None)
df.head()


# In[ ]:




