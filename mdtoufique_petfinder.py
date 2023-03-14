#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Standard libraries
import os
import json
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


trainDf = pd.read_csv("../input/train/train.csv")
testDf = pd.read_csv("../input/test/test.csv")


# In[ ]:


trainDf.describe()


# In[ ]:


trainDf.head(5)


# In[ ]:


trainDf.columns     ## So here are the Columns 


# In[ ]:


sns.pairplot(trainDf)


# In[ ]:


trainDf.columns


# In[ ]:


features=['Type','Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State']    ## here Gender 1= male 2= female.     Choose some random feature. Check for it.


# In[ ]:


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


def metric(y1,y2,labels='AdoptionSpeed', weights='quadratic', sample_weight=None):
    return cohen_kappa_score(y1,y2,labels=None,weights=None,sample_weight=None)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lrclf=LogisticRegression(multi_class='multinomial',solver ='newton-cg')


# In[ ]:


lrclf.fit(trainDf[features], trainDf['AdoptionSpeed'])


# In[ ]:


metric(lrclf.predict(trainDf[features])[:50], trainDf['AdoptionSpeed'][:50])  ## OK this is all with the  logistic regression


# In[ ]:


# Get and store predictions
predictions = lrclf.predict(testDf[features])
submissionLR = pd.DataFrame(data={"PetID" : testDf["PetID"], 
                                   "AdoptionSpeed" : predictions})
submissionLR.to_csv("submission.csv", index=False)


# In[ ]:


submissionLR.head()


# In[ ]:




