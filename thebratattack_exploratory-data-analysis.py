#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[2]:


get_ipython().run_cell_magic('capture', '', 'import numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport pandas_profiling as pp\n\n# Input data files are available in the "../input/" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n\nimport os\nprint(os.listdir("../input"))\n\n# Any results you write to the current directory are saved as output.')


# In[3]:


train = pd.read_csv("../input/train.csv", encoding='UTF-8', parse_dates = ['project_submitted_datetime'])


# In[4]:


pp.ProfileReport(train)


# In[5]:


resources = pd.read_csv("../input/resources.csv", encoding='UTF-8')


# In[6]:


pp.ProfileReport(resources)

