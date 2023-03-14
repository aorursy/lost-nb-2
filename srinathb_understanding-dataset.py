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


import os
import dicom
INPUT_FOLDER = '../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# In[3]:


patients


# In[4]:


type(patients)


# In[5]:


len(patients)


# In[6]:


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[7]:


s = dicom.read_file('../input/sample_images'+ '/' +'00cba091fa4ad62cc3200a657aeb957e')


# In[8]:


'../input/sample_images'+ '/' +'00cba091fa4ad62cc3200a657aeb957e'


# In[9]:


for s in os.listdir(INPUT_FOLDER)
   print(s)


# In[10]:


patients[0]


# In[11]:


slices = [dicom.read_file(INPUT_FOLDER + patients[0] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[0])]


# In[12]:


len(slices)


# In[13]:


slices


# In[14]:


type(slices[0])


# In[15]:


slices[0]


# In[16]:


slices[0].SliceLocation


# In[17]:


slices[1].SliceLocation


# In[18]:


slices[0].ImagePositionPatient[2]


# In[19]:


slices[0].dir('setup')


# In[20]:


slices[0].PatientName


# In[21]:


type(slices[1].PixelData)


# In[22]:


slices[0].PixelData


# In[23]:


slices[0].pixel_data


# In[24]:




