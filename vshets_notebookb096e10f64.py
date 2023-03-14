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


import keras, theano


# In[3]:


from matplotlib import pyplot as plt
from matplotlib import image as mpimg


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from glob import glob


# In[6]:


from random import choice


# In[7]:


files = glob('../input/train/c9/*.*')


# In[8]:


plt.imshow(mpimg.imread(choice(files)))


# In[9]:


get_ipython().run_line_magic('pinfo', 'plt.imshow')


# In[10]:




