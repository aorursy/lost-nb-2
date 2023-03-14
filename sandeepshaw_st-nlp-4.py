#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import io


# In[3]:


pip install simpletransformers


# In[4]:


pip install pytorch-transformers


# In[5]:


a=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
b=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
c=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")


# In[6]:


a.head()
a['text'][a['text'].isnull()]='0'
a['selected_text'][a['selected_text'].isnull()]='0'
a.isnull().sum()


# In[7]:


def f(x):
    return(x[1].find(x[2]))
    
a['str_len']=a.apply(f, axis=1) #Imp

def f1(x):
    return({'context':x[1], 'qas':[{'question':x[3],'id':x[0],'is_impossible':False,'answers':[{'answer_start':x[4],'text':x[2]}]}]})

a['dict']=a.apply(f1, axis=1)

a.loc[0]

train=a['dict']
train[:3]

outer_list=[]
len(train)

train[0]

for i in range(len(train)):
    outer_list.append(train[i])

outer_list[:3]

train=outer_list

train[:3]


# In[8]:


pip install pytorch-transformers


# In[9]:


pip install simpletransformers


# In[10]:


from simpletransformers.question_answering import QuestionAnsweringModel


# In[11]:


model=QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=True)


# In[12]:


train


# In[13]:


import os
import json
import numpy as np

os.makedirs('data', exist_ok = True)

with open('data/train.json', 'w') as f:
    json.dump(train, f)
    f.close()


# In[14]:


type(train)


# In[15]:


train[:3]


# In[17]:


import os, sys, shutil
import time
import gc
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np, pandas as pd
from tqdm import tqdm, tqdm_notebook

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

USE_APEX = True

if USE_APEX:
            with timer('install Nvidia apex'):
                # Installing Nvidia Apex
                os.system('git clone https://github.com/NVIDIA/apex; cd apex; pip install -v --no-cache-dir' + 
                          ' --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
                os.system('rm -rf apex/.git') # too many files, Kaggle fails
                from apex import amp


# In[18]:


model.train_model('data/train.json')


# In[19]:


#for test
b.columns
b.isnull().sum()

def ft(x):
    return({'context':x[1], 'qas':[{'question':x[2],'id':x[0],'is_impossible':False,'answers':[{'answer_start':1000000,'text':'__None__'}]}]})

b['dict']=b.apply(ft, axis=1)

test=b['dict']
test[:3]

outer_test=[]
len(test)

test[0]

for i in range(len(test)):
    outer_test.append(test[i])

outer_test[:3]


# In[20]:


test=outer_test


# In[21]:


with open('data/test.json', 'w') as f:
    json.dump(test, f)
    f.close()


# In[23]:


pred_df = model.predict(test)
pred_df = pd.DataFrame.from_dict(pred_df)

