#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pydicom

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from IPython.display import HTML


# In[2]:


get_ipython().run_line_magic('pinfo', 'np.sort')


# In[3]:


patient_id = "ID00035637202182204917484"

dicom_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"

files = np.array([f.replace(".dcm","") for f in os.listdir(f"{dicom_path}/{patient_id}/")])
files = -np.sort(-files.astype("int"))
dicoms = [f"{dicom_path}/{patient_id}/{f}.dcm" for f in files]


# In[4]:


images = []
for dcm in dicoms:
    tmp = pydicom.dcmread(dcm)
    slope = tmp.RescaleSlope
    intercept = tmp.RescaleIntercept
    final = tmp.pixel_array*slope + intercept
    images.append(final)
    
images = np.array(images) 


# In[5]:


fig = plt.figure()

ims = []
for image in range(0,images.shape[0],10):
    im = plt.imshow(images[image,:,:], 
                    animated=True, cmap=plt.cm.bone)
    plt.axis("off")
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                repeat_delay=1000)

plt.close()


# In[6]:


HTML(ani.to_jshtml())


# In[7]:


HTML(ani.to_html5_video())


# In[8]:


fig = plt.figure()

ims = []
for image in range(0,images.shape[1],5):
    im = plt.imshow(images[:,image,:], animated=True, cmap=plt.cm.bone)
    plt.axis("off")
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                repeat_delay=1000)

plt.close()

HTML(ani.to_jshtml())


# In[9]:


fig = plt.figure()

ims = []
for image in range(0,images.shape[2],5):
    im = plt.imshow(images[:,:,image], animated=True, cmap=plt.cm.bone)
    plt.axis("off")
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                repeat_delay=1000)

plt.close()

HTML(ani.to_jshtml())


# In[10]:


plt.hist(np.array(images).reshape(-1,), bins=50)
plt.show()


# In[11]:


from ipywidgets import interact
import ipywidgets as widgets


# In[12]:


get_ipython().run_line_magic('matplotlib', 'notebook')

fig = plt.figure(figsize=(5,5))

img_plot = plt.imshow(images[0], cmap="Greys")
plt.axis("off")

@interact(slice = widgets.IntSlider(min=0, max=len(images), step=1, value=0))
def update(slice):
    global img_plot
    img_plot.set_data(images[int(slice)])
    plt.draw()


# In[ ]:




