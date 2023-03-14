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


train = pd.read_csv("../input/train.csv", index_col = "id")
train.head()


# In[3]:


test = pd.read_csv("../input/test.csv", index_col = "id")
test.head()


# In[4]:


sample = pd.read_csv("../input/sample_submission.csv", index_col = "id")
sample.head()


# In[5]:


list(train.columns.values)


# In[6]:


X = train.drop(['target'], axis = 1)
y = train.target
X.shape


# In[7]:


from sklearn.feature_selection import chi2

features, chi2s, pvals = [],[],[]

for col_name in X:
    col = X[col_name]
    present_x = (col >= 0)
    ch2_result = chi2(col[present_x].values.reshape(-1, 1), y[present_x])
    features.append(col_name)
    chi2s.append(ch2_result[0][0])
    pvals.append(ch2_result[1][0])

ch2_df = pd.DataFrame({"feature": features, "chi2": chi2s, "pval": pvals})
ch2_df[['feature', 'chi2', 'pval']].sort_values('pval', axis = 0, ascending = True).head()


# In[8]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

kbest = SelectKBest(k="all")
results = kbest.fit(X, y)
ch2_df['kbest_score'] = results.scores_
ch2_df.sort_values('pval', axis = 0, ascending = True).head()


# In[9]:


ch2_df.to_csv("chi2_features.csv")


# In[10]:


sig_features = ch2_df.pval < .05
ch2_df[sig_features]


# In[11]:


print(y.describe())
print(y[:20])
print(y.value_counts())
zeros = y.value_counts()[0]
ones = y.value_counts()[1]

print("Upsample magnitude to 50% would be {}".format(zeros / ones))


# In[12]:


corr = train.corr()
corr


# In[13]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

size = 18
fig, ax = plt.subplots(figsize = (size, size))
leg = ax.matshow(corr)
fig.colorbar(leg)
plt.xticks(range(len(corr.columns)), corr.columns, rotation = 45, horizontalalignment = 'left', fontsize = 8)
plt.yticks(range(len(corr.columns)), corr.columns)
("")

