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


pip install -U git+https://github.com/lyft/nuscenes-devkit


# In[3]:


get_ipython().system('pip install moviepy')
import pdb
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from moviepy.editor import ImageSequenceClip
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:



get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')


# In[5]:


lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)


# In[6]:


lyftdata.list_scenes()


# In[7]:



my_scene = lyftdata.scene[0]
my_scene


# In[8]:


my_sample_token = my_scene["first_sample_token"]
# my_sample_token = level5data.get("sample", my_sample_token)["next"]  # proceed to next sample

lyftdata.render_sample(my_sample_token)


# In[9]:


my_sample = lyftdata.get('sample', my_sample_token)
my_sample


# In[10]:


lyftdata.list_sample(my_sample['token'])


# In[11]:



lyftdata.render_pointcloud_in_image(sample_token = my_sample["token"],
                                      dot_size = 1,
                                      camera_channel = 'CAM_FRONT')


# In[12]:


my_sample['data']


# In[13]:



sensor_channel = 'CAM_FRONT'  # also try this e.g. with 'LIDAR_TOP'
my_sample_data = lyftdata.get('sample_data', my_sample['data'][sensor_channel])
my_sample_data


# In[14]:


lyftdata.render_sample_data(my_sample_data['token'])


# In[15]:


my_annotation_token = my_sample['anns'][16]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
my_annotation


# In[16]:


lyftdata.render_annotation(my_annotation_token)


# In[17]:



my_instance = lyftdata.instance[100]
my_instance
print("First annotated sample of this instance:")
lyftdata.render_annotation(my_instance['first_annotation_token'])


# In[18]:


print("Last annotated sample of this instance")
lyftdata.render_annotation(my_instance['last_annotation_token'])


# In[19]:


lyftdata.list_categories()


# In[20]:


lyftdata.category[2]

