#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import gc
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
get_ipython().run_line_magic('matplotlib', 'inline')
seed = 2390


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# In[3]:


for message_no, message in enumerate(train['comment_text'][:10]):
    print(message_no, message)
    print('\n')


# In[4]:


'''
Messtype1 = train[['comment_text','toxic']]
Messtype2 = train[['comment_text','severe_toxic']]
Messtype3 = train[['comment_text','obscene']]
Messtype4 = train[['comment_text','threat']]
Messtype5 = train[['comment_text','insult']]
Messtype6 = train[['comment_text','identity_hate']]
''';


# In[5]:


cols= ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(2,3)
for i, cn in enumerate(cols):
    ax = plt.subplot(gs[i])
    sns.countplot(y = cn , data = train)
    ax.set_xlabel('')
    ax.set_title(str(cn))
    ax.set_ylabel(' ')


# In[6]:


plt.figure(figsize=(14,6))
sns.heatmap(train[['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");


# In[7]:


Messtype1 = train[['comment_text','toxic']]
#Messtype1['length'] = Messtype1['comment_text'].apply(len)
Messtype1['length'] = Messtype1['comment_text'].str.split().apply(len)
Messtype1.head()


# In[8]:


Messtype1['length'].plot(bins=50, kind='hist');


# In[9]:


Messtype1.length.describe()


# In[10]:


Messtype1[Messtype1['length'] == 1411]['comment_text'].iloc[0]


# In[11]:


Messtype1.hist(column='length', by='toxic', bins=50,figsize=(12,4));


# In[12]:


def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[13]:


Messtype1['comment_text'].head(5).apply(text_process)


# In[14]:


Messtype1.head()


# In[15]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(Messtype1['comment_text'], Messtype1['toxic'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[16]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[17]:


pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)


# In[18]:


print(classification_report(predictions,label_test))

