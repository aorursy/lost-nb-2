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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from fastai import *
from fastai.vision import * 


# In[3]:


PATH = Path('../input')


# In[4]:


PATH


# In[5]:


PATH = Path('../input/aptos2019-blindness-detection')

df_train = pd.read_csv(PATH/'train.csv')
df_test = pd.read_csv(PATH/'test.csv')

# if is_interactive():
#     df_train = df_train.sample(800)

_ = df_train.hist()


# In[6]:


cd ..


# In[7]:


kappa = KappaScore()
kappa.weights = "quadratic"
aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])
data = ImageDataBunch.from_df(df=df_train,
                              path=PATH, folder='train_images', suffix='.png',
                              valid_pct=0.1,
                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.1, max_zoom=1.15, max_rotate=45.),
                              test='test_images',
                              size=224,
                              bs=32, 
                              num_workers=os.cpu_count()
#                              )
                             ).normalize(aptos19_stats)


# In[8]:


learn =cnn_learner(data, models.resnet34,metrics=[error_rate,kappa])


# In[9]:


learn.unfreeze()
learn.fit_one_cycle(1)


# In[10]:


interp=ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(15,11))


# In[11]:


interp.plot_confusion_matrix()


# In[12]:


learn.save('stage-basic')


# In[13]:


learn.unfreeze()
learn.fit_one_cycle(1)


# In[14]:


learn.lr_find()
learn.recorder.plot()


# In[15]:


learn.fit_one_cycle(4, slice(1e-5 ,1e-3))


# In[16]:


learn.fit_one_cycle(20, slice(1e-5 ,1e-3))


# In[17]:


learn.freeze()
learn.save('stage-2')


# In[18]:


learn1 =cnn_learner(data, models.resnet50,metrics=[error_rate,kappa,accuracy])


# In[19]:


learn1.fit_one_cycle(4)


# In[20]:


learn1.unfreeze()
# learn1.fit_one_cycle(1)


# In[21]:


cd working/


# In[22]:


learn1.fit_one_cycle(4, slice(1e-5 ,1e-3))


# In[23]:


learn1.fit_one_cycle(4, slice(1e-5 ,1e-3))


# In[24]:


learn1.fit_one_cycle(10, slice(1e-5 ,1e-3))


# In[25]:


# learn1.save('stage50-3')


# In[26]:


# learn1.freeze()
learn1.save('stage50-4')


# In[27]:


PATH


# In[28]:


img = open_image(PATH + '\test_images\006efc72b638.png')
pred_class,pred_idx,output =learn1.predict(img)
pred_class


# In[29]:


preds,y, loss = learner.get_preds(with_loss=True)


# In[30]:


tta_params = {'beta':0.12, 'scale':1.0}
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[31]:


# learn1.data.add_test(ImageList.from_df(
#     sample_df, PATH,
#     folder='test_images',
#     suffix='.png'
# ))


# In[32]:


preds,y = learn1.TTA(ds_type=DatasetType.Test, **tta_params)


# In[33]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[34]:


sample_df.hist()


# In[35]:


sample_df.to_csv('submission.csv',index=False)
_ = sample_df.hist()


# In[36]:


ls

