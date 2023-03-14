#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imutils')


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

import imageio
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from kaggle_datasets import KaggleDatasets

from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
import tensorflow as tf
from functools import partial

import glob
import numpy as np
import cv2
from skimage import filters as skifilters
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
import tqdm
from sklearn.utils import shuffle
import pandas as pd

import os
import h5py
import time
import json
import warnings
from PIL import Image

from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import models
import pdb
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import matplotlib.pyplot as plt

import pickle 
import os

from tqdm.notebook import tqdm

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


def list_files(path:Path):
    return [o for o in path.iterdir()]


# In[4]:


path = Path('../input/jpeg-melanoma-768x768/')
df_path = Path('../input/jpeg-melanoma-768x768/')
im_sz = 256
bs = 16


# In[5]:


train_fnames = list_files(path/'train')
df = pd.read_csv(df_path/'train.csv')
df.head()


# In[6]:


df.target.value_counts(),df.shape


# In[ ]:





# In[7]:


GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-768x768')


# In[8]:


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


# In[9]:


def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


# In[10]:


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


# In[11]:



BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [768, 768]
TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
    tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'),
    test_size=0.2, random_state=5
)
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
print('Train TFRecord Files:', len(TRAINING_FILENAMES))
print('Validation TFRecord Files:', len(VALID_FILENAMES))
print('Test TFRecord Files:', len(TEST_FILENAMES))


# In[12]:




def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    #dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# In[13]:


train_dataset = get_training_dataset()


# In[14]:


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(15,15))
    for n in range(8):
        ax = plt.subplot(8,8,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("MALIGNANT(1)")
        else:
            plt.title("BENIGN(0)")
        plt.axis("off")


# In[15]:


"""
%%time
for i in range(0,10):
    image_batch, label_batch = next(iter(train_dataset))
    for j in range(0,8):
        var = label_batch[j].numpy()
        if(var!=0):
            show_batch(image_batch.numpy(), label_batch.numpy())
"""


# In[16]:


"""
print("Samples with Melanoma")
imgs = df[df.target==1]['image_name'].values
_, axs = plt.subplots(2, 3, figsize=(20, 8))
axs = axs.flatten()
for f_name,ax in zip(imgs[10:20],axs):
    img = Image.open(path/f'train/{f_name}.jpg')
    ax.imshow(img)
    ax.axis('off')    
plt.show()
"""


# In[17]:


path


# In[18]:


import glob
train_list = glob.glob("../input/jpeg-melanoma-768x768/train/*.jpg")
test_list = glob.glob("../input/jpeg-melanoma-768x768/test/*.jpg")


# In[19]:



# Usage: This script will measure different objects in the frame using a reference object 



# Function to show array of images (intermediate results)

def show_images(images):
    for i, img in enumerate(images):
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        plt.show()

def get_size_of_rectangle(f_name, train=True):
    
    if train:
        im1 = Image.open(path/f'train/{f_name}.jpg')
        #print(path/f'train/{f_name}.jpg')
    else:
        if '../input/jpeg-melanoma-768x768/test/' + f_name + ".jpg" in test_list:
            im1 = Image.open(path/f'test/{f_name}.jpg')
        else:
            im1 = Image.open(path/f'train/{f_name}.jpg')
        #print(path/f'train/{f_name}.jpg')
        
    im1.save('./a.png')
    img_path = '../working/a.png'



    '''load our image from disk, convert it to grayscale, and then smooth it using a Gaussian filter.
    We then perform edge detection along with a dilation + erosion to close any gaps 
    in between edges in the edge map
    '''

    # Read image and preprocess
    image = cv2.imread(img_path)
 
    #image = img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    #show_images([blur, edged])

    '''find contours (i.e., the outlines) that correspond to the objects in our edge map.'''
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    try:
        '''These contours are then sorted from left-to-right (allowing us to extract our reference object)'''
        (cnts, _) = contours.sort_contours(cnts)
         # Remove contours which are not large enough
        for k in range(0,20):
            try:
                cnts = [x for x in cnts if cv2.contourArea(x) > k]
                # Reference object dimensions
                # Here for reference I have used a 2cm x 2cm square
                mid = len(cnts)//2
                ref_object = cnts[mid]
            except:
                #pass
                return [(0, 0), (0, 0)]
    except:
        #print("An exception occurred") 
        return [(0, 0), (0, 0)]
    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)

    #show_images([image, edged])
    #print(len(cnts))

    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    
    box = perspective.order_points(box)
    
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm
    largestht = []
    largestwid = []
    # Draw remaining contours
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        largestht.append(ht)
        largestwid.append(wid)
       
        #cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        #cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    #show_images([image])   
    if(len(largestht)>0):
        a = largestht.index(max(largestht))
        b = largestwid.index(max(largestwid))
        largestht1 = largestht[b]
        largestwid1 = largestwid[b]
        largestht = largestht[a]
        largestwid = largestwid[a]
        

        #print("Rectangle 1  has : HEIGHT = ",largestht,"and WIDTH = ",largestwid)
        #print("Rectangle 2  has : HEIGHT = ",largestht1,"and WIDTH = ",largestwid1)
        return [(largestht, largestwid), (largestht1, largestwid1)]
    else:
        return [(0, 0), (0, 0)]


# In[20]:


test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
test_image_name = test.image_name.values

train_image_name = df.image_name.values

print(len(train_image_name))
print(len(test_image_name))


# In[21]:


train_melanoma_sizes = []
for f_name in tqdm(train_image_name):
    size = get_size_of_rectangle(f_name)
    train_melanoma_sizes.append((f_name, size))
    
test_melanoma_sizes = []
for f_name in tqdm(test_image_name):
    size = get_size_of_rectangle(f_name, train=False)
    test_melanoma_sizes.append((f_name, size))


# In[22]:


unpacked_train =[(name, sizes[0][0], sizes[0][1], sizes[1][0], sizes[1][1]) for (name, sizes) in train_melanoma_sizes]

unpacked_test =[(name, sizes[0][0], sizes[0][1], sizes[1][0], sizes[1][1]) for (name, sizes) in test_melanoma_sizes]


# In[23]:


pd.DataFrame(unpacked_train).to_csv("train_melanoma_size.csv", index=False)
pd.DataFrame(unpacked_test).to_csv("test_melanoma_size.csv", index=False)


# In[ ]:




