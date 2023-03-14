#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np #
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        if filename.find("rain")>0:
            print('reading train')            
            train=pd.read_csv(os.path.join(dirname, filename) )
        if filename.find("est")>0:
            print('reading test')
            test=pd.read_csv(os.path.join(dirname, filename) )


# In[2]:


trainr=train
train


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = TfidfVectorizer()#CountVectorizer()
train_tf=cv.fit_transform(train.ItemId.fillna(''))
count_df=pd.DataFrame(cv.transform(train['ItemId']).toarray(), columns=cv.get_feature_names())
words=count_df.sum()
words=pd.DataFrame(words,columns=['pos'])
words


# In[4]:


from sklearn.metrics.pairwise import cosine_similarity

item_item=cosine_similarity(train_tf.T,train_tf.T)
item_item


# In[5]:


np.asarray(train.iloc[3]['ItemId'].split(' '))


# In[6]:


recommended=pd.DataFrame(item_item.T*cv.transform(train.iloc[0:1].ItemId).T,index=cv.get_feature_names()).sort_values(0,ascending=False)[:100]
#1938 490 128 1197 2893 2983 1861 1307 2547 231...
recommended=[xi for xi in recommended.index if xi not in np.asarray(train.iloc[0]['ItemId'].split(' '))]
' '.join(recommended)


# In[7]:


for xi in range(len(train)):
    if xi/100==int(xi/100):
        print(xi)
    recommended=pd.DataFrame(item_item.T*cv.transform(train.iloc[xi:xi+1].ItemId).T,index=cv.get_feature_names()).sort_values(0,ascending=False)[:100]
    recommended=[ri for ri in recommended.index if ri not in np.asarray(train.loc[xi].ItemId.split(' '))]
    predi=' '.join(recommended)
    train.iat[xi,1]=predi
    
train


# In[8]:


train.to_csv('submit.csv',index=False)


# In[9]:


user_user=cosine_similarity(train_tf,train_tf)
user_user


# In[10]:


train_tf[:,1]


# In[11]:


sameuser=pd.DataFrame(user_user*train_tf[:,1]).sort_values(0,ascending=False)[:3]
sameuser


# In[12]:


user_user[0][1:].max()


# In[13]:


comarket=(user_user*train_tf)+(item_item.T*train_tf.T).T


# In[14]:


comarket.shape


# In[15]:


for xi in range(train_tf.shape[0]):
    if xi/100==int(xi/100):
        print(xi)
    
    recommended=pd.DataFrame(comarket[xi],index=cv.get_feature_names()).sort_values(0,ascending=False)[:100]
    recommended=[ri for ri in recommended.index if ri not in np.asarray(trainr.loc[xi]['ItemId'].split(' '))]
    predi=' '.join(recommended)
    train.iat[xi,1]=predi
    


# In[16]:


train.to_csv('submit2.csv',index=False)

