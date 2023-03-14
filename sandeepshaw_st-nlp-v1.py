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


# In[3]:


a=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
b=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
c=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")


# In[4]:


a.head()
a['text'][a['text'].isnull()]='0'
a['selected_text'][a['selected_text'].isnull()]='0'
a.isnull().sum()


# In[5]:


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


# In[6]:


pip install pytorch-transformers


# In[7]:


pip install simpletransformers


# In[8]:


from simpletransformers.question_answering import QuestionAnsweringModel


# In[9]:


model=QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=True)


# In[10]:


train


# In[11]:


import os
import json
import numpy as np

os.makedirs('data', exist_ok = True)

with open('data/train.json', 'w') as f:
    json.dump(train, f)
    f.close()


# In[12]:


type(train)


# In[13]:


train[:3]


# In[14]:


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


# In[15]:


model.train_model('data/train.json')


# In[16]:


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


# In[17]:


test=outer_test


# In[18]:


with open('data/test.json', 'w') as f:
    json.dump(test, f)
    f.close()


# In[19]:


pred_df = model.predict(test)
pred_df = pd.DataFrame.from_dict(pred_df)


# In[20]:


pred_df.head()


# In[21]:


c.head()
d=pred_df
d.head()
d1=d
d1.head()

e=d1.merge(c,left_on='id', right_on='textID')
e.head()
e['ck']=e['id']==e['textID']
e['ck'].value_counts()

f=e[[e.columns.values[0],e.columns.values[1]]]
f.head()
f.columns.values[0]=c.columns.values[0]
f.columns.values[1]=c.columns.values[1]


# In[22]:


f.head()


# In[23]:


f.to_csv("submission.csv")

