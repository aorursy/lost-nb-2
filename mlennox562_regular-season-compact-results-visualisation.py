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


# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
RegularSeasonCompactResults = pd.read_csv("../input/RegularSeasonCompactResults.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
RegularSeasonCompactResults.head()

# Press shift+enter to execute this cell


# In[3]:


# Let's see how many examples we have for the Wloc variable. 
RegularSeasonCompactResults["Wloc"].value_counts()


# In[4]:


RegularSeasonCompactResults["Wloc"].value_counts().plot(kind='barh')


# In[5]:


# Let's see how many examples we have for the Wloc variable. 
RegularSeasonCompactResults["Numot"].value_counts()


# In[6]:


RegularSeasonCompactResults["Numot"].value_counts().plot(kind='barh')


# In[7]:


# Let's see how many examples we have for the Wloc variable. 
RegularSeasonCompactResults["Wteam"].value_counts()


# In[8]:


RegularSeasonCompactResults["Wteam"].value_counts()[:20].plot(kind='barh')


# In[9]:


# Let's see how many examples we have for the Wloc variable. 
RegularSeasonCompactResults["Lteam"].value_counts()


# In[10]:


RegularSeasonCompactResults["Lteam"].value_counts()[:20].plot(kind='barh')


# In[11]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the RegularSeasonCompactResults features.
RegularSeasonCompactResults.plot(kind="scatter", x="Lscore", y="Wscore")


# In[12]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="Lscore", y="Wscore", data=RegularSeasonCompactResults, size=5)


# In[13]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the RegularSeasonCompactResults features.
RegularSeasonCompactResults.plot(kind="scatter", x="Daynum", y="Wscore")


# In[14]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="Daynum", y="Wscore", data=RegularSeasonCompactResults, size=5)


# In[15]:


sns.FacetGrid(RegularSeasonCompactResults, hue="Season", size=5)    .map(plt.scatter, "Daynum", "Wscore")    .add_legend()


# In[16]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Season", y="Wscore", data=RegularSeasonCompactResults)


# In[17]:


# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(RegularSeasonCompactResults, hue="Wloc", size=6)    .map(sns.kdeplot, "Wscore")    .add_legend()


# In[18]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Season", y="Lscore", data=RegularSeasonCompactResults)


# In[19]:


# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(RegularSeasonCompactResults, hue="Wloc", size=6)    .map(sns.kdeplot, "Lscore")    .add_legend()


# In[20]:


pd.pivot_table(RegularSeasonCompactResults,index=["Season"])


# In[21]:


pd.pivot_table(RegularSeasonCompactResults,index=["Wloc"])


# In[22]:


pd.pivot_table(RegularSeasonCompactResults,index=["Wteam"])


# In[23]:


pd.pivot_table(RegularSeasonCompactResults,index=["Lteam"])


# In[24]:




