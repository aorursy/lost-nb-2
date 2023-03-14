#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import gc
import warnings
import pydicom
import cv2
from tqdm import tqdm
warnings.simplefilter(action = 'ignore')


# In[2]:


get_ipython().system('ls ../input/*')


# In[3]:


ls -R ../input/rsnasample/*


# In[4]:


train_labels = pd.read_csv('../input/rsnasample/stage_1_train.csv')
train_labels.head()


# In[5]:


train_labels = train_labels.drop_duplicates()
train_labels.info()


# In[6]:


train_labels['ID'].value_counts(sort=True).head(10)


# In[7]:


train_labels['Label'].plot.hist()


# In[8]:


pd.DataFrame(train_labels.groupby('Label')['ID'].count())


# In[9]:


plt.style.use('ggplot')
plot = train_labels.groupby('Label')     .count()['ID']     .plot(kind='bar', figsize=(10,4), rot=0)


# In[10]:


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print()
    
    print("Patient id..........:", dataset.PatientID )
    print("Patient's Age.......:", dataset.SOPInstanceUID )
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
            
def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


# In[11]:


get_ipython().system('ls ../input/rsnasample/stage_1_train_images/stage_1_train_images')


# In[12]:


root_path = '../input/rsnasample/stage_1_train_images/stage_1_train_images/'
for r, d, files in os.walk(root_path):
    for dcm_file in files:
        file_path = os.path.join(root_path, dcm_file)
        dataset = pydicom.dcmread(file_path)
        print(dataset)


# In[13]:


for r, d, files in os.walk(root_path):
    for dcm_file in files:
        file_path = os.path.join(root_path, dcm_file)
        dataset = pydicom.dcmread(file_path)
        show_dcm_info(dataset)
        plot_pixel_array(dataset)

