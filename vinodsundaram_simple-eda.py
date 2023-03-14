#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# import os
# print(os.listdir("../input"))


# In[2]:


# print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resources = pd.read_csv('../input/resources.csv')


# In[ ]:





# In[ ]:





# In[3]:


print("The size of the dataset:",train.shape)
train.head()


# In[4]:


print(train.dtypes.value_counts())
train.columns.dtype


# In[5]:


print(train.project_is_approved.value_counts())
df = train.project_is_approved.value_counts()
print(100*df[1]/(df[0]+df[1]),"% of the projects are approved")


# In[6]:


print('The number of states present in the dataset:',train.school_state.value_counts().shape[0],'\n')


# In[7]:


state_df=train.school_state.value_counts()
y_pos = np.arange(len(state_df[0:].tolist()))
plt.figure(figsize=(20,5))
plt.bar(y_pos,state_df[0:].tolist(),align='center',alpha=1)
plt.xticks(y_pos,state_df.index.tolist())
plt.xlabel("States")
plt.ylabel("No. of projects")
plt.title('No. of projects across different states')
plt.show()


# In[8]:


df=pd.crosstab(train.school_state,train.project_is_approved)
df['total']=df[0]+df[1]
df['percent']=df[1]/df['total']
df=df.sort_values(['total'],ascending=0)
x_pos=np.arange(len(df.index.tolist()))
plt.figure(figsize=(20,5))
p1=plt.bar(x_pos,df[1],align='center',alpha=1)
p2=plt.bar(x_pos,df[0],align='center',bottom=df[1], alpha=1)
plt.xticks(x_pos,df.index.tolist())
plt.xlabel("States")
plt.ylabel("No. of projects")
plt.title('No. of projects approved & notapproved across different states')
plt.legend((p1,p2),('Approved','Not Approved'))
plt.show()


# In[9]:


df=pd.crosstab(train.teacher_prefix,train.project_is_approved)
df['total']=df[0]+df[1]
df['percent']=df[1]/df['total']
df=df.sort_values(['total'],ascending=0)
x_pos=np.arange(len(df.index.tolist()))
plt.figure(figsize=(20,5))
p1=plt.bar(x_pos,df[1],align='center',alpha=1)
p2=plt.bar(x_pos,df[0],align='center',bottom=df[1], alpha=1)
plt.xticks(x_pos,df.index.tolist())
plt.xlabel("Prefix")
plt.ylabel("No. of projects")
plt.title('No. of projects approved & notapproved across teachers')
plt.legend((p1,p2),('Approved','Not Approved'))
plt.show()


# In[10]:


df['percent']


# In[11]:


df=pd.crosstab(train.project_grade_category,train.project_is_approved)
df['total']=df[0]+df[1]
df['percent']=df[1]/df['total']
df=df.sort_values(['total'],ascending=0)
x_pos=np.arange(len(df.index.tolist()))
plt.figure(figsize=(20,5))
p1=plt.bar(x_pos,df[1],align='center',alpha=1)
p2=plt.bar(x_pos,df[0],align='center',bottom=df[1], alpha=1)
p3=plt.plot(x_pos,df['percent'].tolist())
plt.xticks(x_pos,df.index.tolist())
plt.xlabel("Grade Category")
plt.ylabel("No. of projects")
plt.title('No. of projects approved & notapproved across different grades')
plt.legend((p1,p2,p3),('Approved','Not Approved','percent'))
plt.show()


# In[12]:


df['percent']


# In[13]:


df=pd.crosstab(train.project_subject_categories,train.project_is_approved)
df['total']=df[0]+df[1]
df['percent']=df[1]/df['total']
df=df.sort_values(['total'],ascending=0)
x_pos=np.arange(len(df.index.tolist()))
plt.figure(figsize=(20,5))
p1=plt.bar(x_pos,df[1],align='center',alpha=1)
p2=plt.bar(x_pos,df[0],align='center',bottom=df[1], alpha=1)
plt.xticks(x_pos,df.index.tolist())
plt.xlabel("Subject categories")
plt.xticks(rotation='vertical')
plt.ylabel("No. of projects")
plt.title('No. of projects approved & notapproved across different subjects')
plt.legend((p1,p2),('Approved','Not Approved'))
plt.show()


# In[29]:


import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS

# words=train.project_title.tolist()
words = ' '.join(train.project_title.tolist())
cleaned_word = " ".join([word for word in words.split() ])


# In[52]:


type(STOPWORDS)
STOPWORDS.add('Help')
STOPWORDS.add('Need')
STOPWORDS.add('Learn')


# In[53]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      width=400,height=200,
                     max_words=100,min_font_size=1,mode='RGB',background_color='white',colormap='viridis').generate(cleaned_word)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


To be continued..


# In[ ]:





# In[ ]:




