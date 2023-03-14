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


import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string


# In[3]:


Datasets loading


# In[4]:


dataframes = {
    "cooking" : pd.read_csv("../input/cooking.csv"),
    "crypto"  : pd.read_csv("../input/crypto.csv"),
    "robotics": pd.read_csv("../input/biology.csv"),
    "travel"  : pd.read_csv("../input/travel.csv"),
    "diy"     : pd.read_csv("../input/diy.csv"),
}


# In[5]:


print(dataframes["robotics"].iloc[1])


# In[6]:


dataframes["robotics"]


# In[7]:


print(dataframes["robotics"].iloc[1])


# In[8]:


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'


# In[9]:


def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text = soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""


# In[10]:


# This could take a while
for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)


# In[11]:


print(dataframes["robotics"].iloc[1])


# In[12]:


def removePunctuation(x):
    # Lowrcasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r"[^\x00-\x7f]", r" ", x)
    # Removing (replacing with empty spaces acutually) all the punctuations
    return re.sub("[" + string.punctuation+"]", " ", x)


# In[13]:


for df in dataframes.values():
    df["title"]   = df["title"].map(removePunctuation)
    df["content"] = df["content"].map(removePunctuation)


# In[14]:


print(dataframes["robotics"].iloc[1])


# In[15]:


stops = set(stopwords.words("english"))
def removeStopwords(x):
    #Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


# In[16]:


for df in dataframes.values():
    df["title"]   = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)
    


# In[17]:


print(dataframes["robotics"].iloc[1])


# In[18]:


Splitting tags string in a list of tags


# In[19]:


for df in dataframes.values():
    # From a string sequence of tags to a list of tags
    df["tags"] = df["tags"].map(lambda x: x.split())
    


# In[20]:


print(dataframes["robotics"].iloc[1])


# In[21]:


for name, df in dataframes.items():
    # Saving to file
    df.to_csv(name + "_light.csv", index=False)


# In[22]:





# In[22]:





# In[22]:





# In[22]:





# In[22]:





# In[22]:




