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
print(os.listdir("../input/sample images"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Convert to JPEG2K via Pillow
# https://github.com/python-pillow/Pillow
 
import os
import pydicom
import glob
from PIL import Image

inputdir = '../input/sample images/'
outdir = './'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
for f in test_list:  
    ds = pydicom.read_file( inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface
    
#   There is an exception in Kaggle kernel about "encoder jpeg2k not available", please test following code on your local workstation
#   img_mem.save(outdir + f.replace('.dcm','.jp2'))


# In[3]:


# Convert to JPEG2K via imageio
# https://github.com/imageio/imageio

import os
import pydicom
import glob
import imageio

inputdir = '../input/sample images/'
outdir = './'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
for f in test_list:  
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    
#   There is an exception in Kaggle kernel about "encoder jpeg2k not available", please test following code on your local workstation
#   imageio.imwrite(outdir + f.replace('.dcm','.jp2'), img)


# In[4]:


# DCMTK 
DCMTK (https://dicom.offis.de/dcmtk.php.en) also provides DICOM conversion to JPEG 2000

Install guide can be found at Quick installation section (https://dicom.offis.de/dcmtk.php.en). 

To perform DCMTK, please use executable binary software "dcmj2pnm" and run (e.g. Linux user)

> dcmj2pnm /PATH_TO_SAMPLE_DICOM/sample.dcm /PATH_TO_SAVE_JPEG2K/sample.jpg

The create "*.jpg" file should be value lossless. Please do double check.  


# In[5]:


# Convert DICOM to PNG via openCV
import cv2
import os
import pydicom
import glob

inputdir = '../input/sample images/'
outdir = './'
#os.mkdir(outdir)

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]

for f in test_list:   
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image


# In[6]:


# Convert to PNG via PIL 
# https://github.com/python-pillow/Pillow
import os
import pydicom
import glob
from PIL import Image

inputdir = '../input/sample images/'
outdir = './'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
#glob.glob(inputdir + './*.dcm')
for f in test_list:   
    ds = pydicom.read_file( inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface
    img_mem.save(outdir + f.replace('.dcm','.png'))

