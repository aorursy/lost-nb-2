#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('bash', '', 'pip install transformers')


# In[ ]:





# In[2]:


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

#!pip install transformers     
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig , DistilBertModel
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
import gc
gc.collect()        

# Any results you write to the current directory are saved as output.


# In[3]:


df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t",usecols = ['Phrase','Sentiment'])
df = df.rename(columns={'Phrase': 0 , 'Sentiment' : 1})
df.head()


# In[4]:


## Subset 
batch_1 = df[:2000]
batch_1[1].value_counts()


# In[5]:


### Loading the Pre-trained BERT model¶


# In[6]:


# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# In[7]:


### Tokenization¶
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[8]:


## Padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


# In[9]:


## Masking
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# In[10]:


## Model - The model() function runs our sentences through BERT. The results of the processing will be returned into last_hidden_states.


# In[11]:


input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)


# In[12]:


#We'll save those in the features variable, as they'll serve as the features to our logitics regression model.
features = last_hidden_states[0][:,0,:].numpy()

#lables - Target 
labels = batch_1[1]


# In[13]:


# Input to the Logistic Regression
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


# In[14]:


lr_clf = LogisticRegression(multi_class = 'ovr', C=1, solver='sag')
lr_clf.fit(train_features, train_labels)


# In[15]:


# Model Evaluation 
lr_clf.score(test_features, test_labels)


# In[ ]:




