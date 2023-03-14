#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.misc import imread
from scipy.misc import imshow
from scipy import sum, average
from skimage import feature
from skimage.transform import resize
from sklearn import datasets, svm, metrics, mixture
from skimage.color import rgb2gray
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
import cv2
import math
from sklearn.utils import shuffle
import numpy as np
import pywt

from glob import glob
basepath = '../input/train/'


all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)


# In[2]:


image_name = all_cervix_images['imagepath'].values[0]
img = np.flipud(plt.imread(image_name))
plt.imshow(img,cmap=plt.cm.gray,interpolation='nearest')


# In[3]:


img_clean = img[1000:2400, :]
plt.imshow(img_clean,cmap=plt.cm.gray,interpolation='nearest')


# In[4]:


img_med = ndi.median_filter(img_clean, size=5)
plt.imshow(img_med,cmap=plt.cm.gray,interpolation='nearest')


# In[5]:


plt.hist(img_med.flat,bins=40,range=(0,250));


# In[6]:


bubbles = (img_med <= 45)
sand = (img_med > 45) & (img_med <= 150)
whiter =(img_med > 150) & (img_med <= 220)
glass = (img_med > 220)

def plot_images(cmap=plt.cm.gray):
    for n, (name, image) in         enumerate([('Original', img_med),
                   ('Bubbles', bubbles),
                   ('Sand', sand),
                   ('white',whiter),
                   ('Glass', glass)]):
    
        plt.subplot(3, 2, n + 1)
        plt.imshow(np.float64(image), cmap=cmap)
        plt.title(name)
        plt.axis('off')
        
plot_images()


# In[ ]:


for img in (sand, bubbles, glass, whiter):
    img[:] = ndi.binary_opening(img, np.ones((5,5))
    img[:] = ndi.binary_closing(img, np.ones((5,5)))

plot_images()


# In[ ]:


image_name = all_cervix_images['imagepath'].values[2]
img = np.flipud(plt.imread(image_name))
plt.imshow(img,cmap=plt.cm.gray,interpolation='nearest')


# In[ ]:


img_clean = img[1000:2000, :]
plt.imshow(img_clean,cmap=plt.cm.gray,interpolation='nearest')

img_med = ndi.median_filter(img_clean, size=5)
plt.imshow(img_med,cmap=plt.cm.gray,interpolation='nearest')

plt.hist(img_med.flat,bins=100,range=(0,150));


# In[ ]:


def w2d(img, mode='haar', level=1):
    #imArray = cv2.imread(img)
    imArray = cv2.resize(imread(image_name), dsize=(256,256))
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    #Display result
    return imArray_H


# In[ ]:



fig = plt.figure()

image_name = all_cervix_images['imagepath'].values[2]
w2d(image_name,'db1',8)


# In[ ]:


train_data = []
train_target = []
test_data = []
test_target = []
raw_data = []
feature_names = ['image_array']

all_samples = 50
train_samples = 40

for t in all_cervix_images['type'].unique():
    for a in range(all_samples):
        image_name = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[a]
        #image = resize(imread(image_name), (200, 200))
        
        #gray_image = w2d(image_name,'db1',8) #.6
        #gray_image = gabfn(image_name)
        #gray_image = crop_image(rgb2gray(new_image)) #.45
        #gray_image = crop_image(rgb2gray(image)) #.5
        
           
        
        image_array = gray_image #resize(gray_image, (200, 200))
    
        if a > train_samples:
            test_data.append(image_array.flatten())
            test_target.append(t)
        else:
            train_data.append(image_array.flatten())
            train_target.append(t)
    
print(len(train_data))
print(len(test_data))

random_forest = RandomForestClassifier(n_estimators=30)
random_forest.fit(train_data, train_target)

random_forest_predicted = random_forest.predict(test_data)
random_forest_probability = random_forest.predict_proba(test_data)

print(metrics.classification_report(test_target, random_forest_predicted))
print(metrics.confusion_matrix(test_target, random_forest_predicted))
print(test_target)
print(random_forest_predicted)
print(random_forest_probability)


# In[ ]:


def gabfn(image_name):
 g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
 image_name = cv2.resize(imread(image_name), dsize=(256,256)) 
 image_name = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
 filtered_img = cv2.filter2D(image_name, cv2.CV_8UC3, g_kernel)
 cv2.imshow('image', image_name)
 cv2.imshow('filtered image', filtered_img)
 h, w = g_kernel.shape[:2]
 g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
 #cv2.imshow('gabor kernel (resized)', g_kernel)
 return g_kernal


# In[ ]:


fig = plt.figure()

image_name = all_cervix_images['imagepath'].values[2]
gabfn(image_name)

