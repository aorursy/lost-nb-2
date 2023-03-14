#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm


# In[2]:


data_path = Path('../input/prostate-cancer-grade-assessment/')
os.listdir(data_path)


# In[3]:


get_ipython().system('cd ../input/prostate-cancer-grade-assessment/; du -h')


# In[4]:


os.listdir(data_path/'train_images')


# In[5]:


import pandas as pd
train_df = pd.read_csv(data_path/'train.csv')
train_df.head(10)


# In[6]:


print('Number of whole-slide images in training set: ', len(train_df))


# In[7]:


sample_image = train_df.iloc[np.random.choice(len(train_df))].image_id
print(sample_image)


# In[8]:


import openslide


# In[9]:


openslide_image = openslide.OpenSlide(str(data_path/'train_images'/(sample_image+'.tiff')))


# In[10]:


openslide_image.properties


# In[11]:


img = openslide_image.read_region(location=(0,0),level=2,size=(openslide_image.level_dimensions[2][0],openslide_image.level_dimensions[2][1]))
img


# In[12]:


Image.fromarray(np.array(img.resize((512,512)))[:,:,:3])


# In[13]:


get_ipython().run_line_magic('pinfo', 'Image.save')


# In[14]:


for i in tqdm(train_df['image_id'],total=len(train_df)):
    openslide_image = openslide.OpenSlide(str(data_path/'train_images'/(i+'.tiff')))
    img = openslide_image.read_region(location=(0,0),level=2,size=(openslide_image.level_dimensions[2][0],openslide_image.level_dimensions[2][1]))
    Image.fromarray(np.array(img.resize((256,256)))[:,:,:3]).save(i+'.jpeg')
    

