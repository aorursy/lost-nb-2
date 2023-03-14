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


x = check_output(["ls", "../input/TrainDotted"]).decode("utf8").split("\n")
np.shape(x)[0]


# In[3]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("../input/TrainDotted/10.jpg"), cv2.COLOR_BGR2RGB)
plt.imshow(img)


# In[4]:


img2 = cv2.resize(img, (300, 400))
plt.imshow(img2)


# In[5]:


# Create search color masks for given circle colors
red = cv2.inRange(img, np.array([160, 0, 0]), np.array([255, 50, 50]))
magenta = cv2.inRange(img, np.array([128, 0, 128]), np.array([255, 0, 255]))
brown = cv2.inRange(img, np.array([139, 69, 16]), np.array([222,184,135]))
blue = cv2.inRange(img, np.array([0, 0, 128]), np.array([50, 50, 255]))
green = cv2.inRange(img, np.array([0, 128, 0]), np.array([50, 255, 50]))

colors = [red, magenta, brown, blue, green]


# In[6]:


coord = np.zeros((2, np.shape(circles)[1])) # Array containing seal boundaries for image cropping
resizer = np.zeros((1,4))
cropCoordinates = np.zeros((1,4))
margin = 150 # makes sure border seals are not deleted

# Get images
imgList = check_output(["ls", "../input/TrainDotted"]).decode("utf8").split("\n")

# Loop over images and crop them
for k in range(0, np.shape(imgList)[0]):
    
    img = cv2.cvtColor(cv2.imread(imgList[k]), cv2.COLOR_BGR2RGB)
    
    # Loop over color masks to find seal boundaries
    for j in range(0, np.shape(circles)[0]):
        cmsk = colors[j]
        circles = cv2.HoughCircles(cmsk,cv2.HOUGH_GRADIENT,1,50, 
                                   param1=40,param2=1,minRadius=0,maxRadius=25)
        for i in range(0, np.shape(circles)[1]):
            coord[0,i] = circles[0][i][0]
            coord[1,i] = circles[0][i][1]

        # Find limits in x and y for ONE color
        ymin_color = int(round(np.min(coord[1,:])))
        ymax_color = int(round(np.max(coord[1,:])))
        xmin_color = int(round(np.min(coord[0,:])))
        xmax_color = int(round(np.max(coord[0,:]))) 
        
    if ymin_color 
    croppedImage = img[ymin - margin:ymax + margin,
                      xmin - margin:xmax + margin]
        
    # Find largest and smallest pictures for resizing
    yDiff = ymax-ymin
    xDiff = xmax-xmin

    # Give largest and smallest height and width of cropped image
    if yDiff > resizer[0][0]:
        resizer[0][0] = yDiff
    if yDiff < resizer[0][1]:
        resizer[0][1] = yDiff
    if xDiff > resizer[0][2]:
        resizer[0][2] = xDiff
    if xDiff > resizer[0][3]:
        resizer[0][3] = xDiff
    
    
    


# In[7]:


cropCoordinates = img[int(round(np.min(coord[1,:]))) - margin:
           int(round(np.max(coord[1,:]))) + margin, 
           int(round(np.min(coord[0,:]))) - margin:
           int(round(np.max(coord[0,:]))) + margin]


# In[8]:


plt.imshow(crop)

