#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = os.listdir("../input/train")
print("Number of image files in the train dataset:", len(train))

test = os.listdir("../input/test")
print("Number of image files in the test dataset:", len(test))


# In[ ]:


train_df = pd.read_csv("../input/train_ship_segmentations.csv")
print("Train data:\n", train_df.head())

print("\n\n" + "*"*80 + "\n\n")

test_df = pd.read_csv("../input/test_ship_segmentations.csv")
print("Test data:\n", test_df.head())


# In[ ]:


def rle_decode(mask_rle, shape=(768, 768)):
    pass


# In[ ]:


ImageId = 


# In[ ]:





# In[ ]:




