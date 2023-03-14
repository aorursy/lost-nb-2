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
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# In[2]:


# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
train = pd.read_csv("../input/labeledTrainData.tsv", header=0,                     delimiter="\t", quoting=2,dtype=str)


# In[3]:


print (train["review"][0])


# In[4]:


# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison

print (train["review"][0])
print (example1.get_text())


# In[5]:


import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)


# In[6]:


lower_case = letters_only.lower().split() # Convert to lower case
print(lower_case)        # Convert to lower case


# In[7]:


import nltk
nltk.download()  # ストップワード（is am など）が含まれる集合体をダウンロード


# In[8]:


from nltk.corpus import stopwords # ストップのみをインポート
print stopwords.words("english") 

