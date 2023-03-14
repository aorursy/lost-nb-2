#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -qqq -U git+https://github.com/stalkermustang/nuscenes-devkit.git')


# In[2]:


from pathlib import Path
from PIL import Image


# In[3]:


# dir with all input data from Kaggle
INP_DIR = Path('/kaggle/input/3d-object-detection-for-autonomous-vehicles/')


# In[4]:


# dir with index json tables (scenes, categories, logs, etc...)
TABLES_DIR = INP_DIR.joinpath('train_data')


# In[5]:


# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps
get_ipython().system('ln -s {INP_DIR}/train_images images')
get_ipython().system('ln -s {INP_DIR}/train_maps maps')
get_ipython().system('ln -s {INP_DIR}/train_lidar lidar')


# In[6]:


DATA_DIR = Path().absolute() 
# Empty init equals '.'.
# We use this because we link train dirs to current dir (cell above)


# In[7]:


# dir to write KITTY-style dataset
STORE_DIR = DATA_DIR.joinpath('kitti_format')


# In[8]:


get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti nuscenes_gt_to_kitti -h')


# In[9]:


# convertation to KITTY-format
get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti nuscenes_gt_to_kitti         --lyft_dataroot {DATA_DIR}         --table_folder {TABLES_DIR}         --samples_count 20         --parallel_n_jobs 2         --get_all_detections True         --store_dir {STORE_DIR}')


# In[10]:


# check created (converted) files. velodyne = LiDAR poinclouds data (in binary)
get_ipython().system('ls {STORE_DIR}/velodyne | head -2')


# In[11]:


# render converted data for check. Currently don't support multithreading :(
get_ipython().system('python -m lyft_dataset_sdk.utils.export_kitti render_kitti         --store_dir {STORE_DIR}')


# In[12]:


# Script above write images to 'render' folder
# in store_dir (where we have converted dataset)
RENDER_DIR = STORE_DIR.joinpath('render')


# In[13]:


# get all rendered files
all_renders = list(RENDER_DIR.glob('*'))
all_renders.sort()


# In[14]:


# render radar data (bird view) and camera data with bboxes


# In[15]:


Image.open(all_renders[0])


# In[16]:


Image.open(all_renders[1])


# In[17]:


get_ipython().system('rm -rf {STORE_DIR}')

