#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sparse
from scipy.sparse import hstack, csr_matrix

import warnings
warnings.simplefilter(action='ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')


# In[3]:


# we have six types of labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

train_data.head()


# In[4]:


test_data.head()


# In[5]:


print(train_data.shape, test_data.shape)


# In[6]:


# take only comments from test and train sets

train_text = train_data['comment_text']
test_text = test_data['comment_text']

texts_data = pd.concat([train_text, test_text]).to_frame()
texts_data.head()


# In[7]:


train_data['total_length'] = train_data['comment_text'].apply(len)
train_data['uppercase'] = train_data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
train_data['exclamation_punctuation'] = train_data['comment_text'].apply(lambda comment: comment.count('!'))
train_data['num_punctuation'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:?'))
train_data['num_symbols'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
train_data['num_words'] = train_data['comment_text'].apply(lambda comment: len(comment.split()))
train_data['num_happy_smilies'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
train_data['num_sad_smilies'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-(', ':(', ';-(', ';(')))


# In[8]:


test_data['total_length'] = test_data['comment_text'].apply(len)
test_data['uppercase'] = test_data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
test_data['exclamation_punctuation'] = test_data['comment_text'].apply(lambda comment: comment.count('!'))
test_data['num_punctuation'] = test_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:?'))
test_data['num_symbols'] = test_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
test_data['num_words'] = test_data['comment_text'].apply(lambda comment: len(comment.split()))
test_data['num_happy_smilies'] = test_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
test_data['num_sad_smilies'] = test_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-(', ':(', ';-(', ';(')))


# In[9]:


train_data.head()


# In[10]:


# Spearman correlation. 

corr = train_data.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, annot = True)


# In[11]:


# let's drop one of the features which have a high correlation with another feature
# in our case, num_words have a very high correlation with the total length

train_data = train_data.drop(['num_words'], axis = 1)


# In[12]:


# Spearman correlation. 

corr = train_data.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, annot = True)


# In[13]:


train_features_data = train_data
train_features_data.head()


# In[14]:


train_features_data['toxic_type'] = ''
train_features_data['toxic_type'].loc[train_features_data['toxic'] == 1] += 'toxic '
train_features_data['toxic_type'].loc[train_features_data['severe_toxic'] == 1] += 'severe_toxic '
train_features_data['toxic_type'].loc[train_features_data['obscene'] == 1] += 'obscene '
train_features_data['toxic_type'].loc[train_features_data['threat'] == 1] += 'threat '
train_features_data['toxic_type'].loc[train_features_data['insult'] == 1] += 'insult '
train_features_data['toxic_type'].loc[train_features_data['identity_hate'] == 1] += 'identity_hate '


# In[15]:


table_top = train_features_data['toxic_type'].value_counts().to_frame()[:10].style.background_gradient(cmap=cmap)
table_top


# In[16]:


train_features_data.head()


# In[17]:


df_toxic = train_features_data.loc[train_features_data['toxic'] == 1]
df_severe_toxic = train_features_data.loc[train_features_data['severe_toxic'] == 1]
df_obscene = train_features_data.loc[train_features_data['obscene'] == 1]
df_threat = train_features_data.loc[train_features_data['threat'] == 1]
df_insult = train_features_data.loc[train_features_data['insult'] == 1]
df_identity_hate = train_features_data.loc[train_features_data['identity_hate'] == 1]
df_normal = train_features_data.loc[train_features_data['toxic_type'] == '']


# In[18]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Histogram(x=df_toxic.total_length, name='toxic'))
fig.add_trace(go.Histogram(x=df_severe_toxic.total_length, name='severe_toxic'))
fig.add_trace(go.Histogram(x=df_obscene.total_length, name='obscene'))
fig.add_trace(go.Histogram(x=df_threat.total_length, name='threat'))
fig.add_trace(go.Histogram(x=df_insult.total_length, name='insult'))
fig.add_trace(go.Histogram(x=df_identity_hate.total_length, name='identity hate'))
fig.add_trace(go.Histogram(x=df_normal.total_length, name='normal'))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.5)
fig.show()


# In[19]:


train_features_data['list_toxic_type'] = train_features_data['toxic_type'].apply(lambda row: row.split(' '))
train_features_data['list_toxic_type'] =train_features_data['list_toxic_type'].apply(lambda row: len(row)-1)
table_types = train_features_data['list_toxic_type'].value_counts().to_frame().style.background_gradient(cmap=cmap)
table_types


# In[20]:


train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==0] = 'normal comment'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==1] = 'has one type of toxic'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==2] = 'has two types of toxic'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==3] = 'has three types of toxic'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==4] = 'has four types of toxic'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==5] = 'has five types of toxic'
train_features_data['list_toxic_type'].loc[train_features_data['list_toxic_type']==6] = 'has six types of toxic'


# In[21]:


types = ['normal comment', 'has one type of toxic', 'has two types of toxic', 'has three types of toxic',
         'has four types of toxic', 'has five types of toxic', 'has six types of toxic']

columns = ['total_length', 'uppercase', 'exclamation_punctuation',
                                'num_punctuation', 'num_symbols', 'num_happy_smilies', 'num_sad_smilies']

df_mean = pd.DataFrame(columns=columns)


# In[22]:


for i, toxic_type in enumerate(types):
    for col in columns:    
        df_mean.at[i, col] = train_features_data[col].loc[train_features_data['list_toxic_type'] == toxic_type].mean()


# In[23]:


df_mean['toxic_types'] = types 


# In[24]:


df_mean


# In[25]:


df_mean['total_length'] = pd.to_numeric(df_mean['total_length'])
df_mean['uppercase'] = pd.to_numeric(df_mean['uppercase'])
df_mean['exclamation_punctuation'] = pd.to_numeric(df_mean['exclamation_punctuation'])
df_mean['num_punctuation'] = pd.to_numeric(df_mean['num_punctuation'])
df_mean['num_symbols'] = pd.to_numeric(df_mean['num_symbols'])
df_mean['num_happy_smilies'] = pd.to_numeric(df_mean['num_happy_smilies'])
df_mean['num_sad_smilies'] = pd.to_numeric(df_mean['num_sad_smilies'])


# In[26]:


import plotly.express as px


px.scatter(df_mean, x="exclamation_punctuation", y="num_punctuation",
           size="total_length", color="toxic_types", hover_name="toxic_types",
           size_max=55)


# In[27]:


types = ['toxic', 'severe_toxic', 'obscene', 'threat',
         'insult', 'identity_hate']

df_types_mean = pd.DataFrame(columns=columns)


# In[28]:


for i, toxic_type in enumerate(types):
    for col in columns:    
        df_types_mean.at[i, col] = train_features_data[col].loc[train_features_data[toxic_type] == 1].mean()


# In[29]:


for col in columns:    
        df_types_mean.at[6, col] = train_features_data[col].loc[train_features_data['toxic'] == 0]                                                         .loc[train_features_data['severe_toxic'] == 0]                                                         .loc[train_features_data['obscene'] == 0]                                                         .loc[train_features_data['threat'] == 0]                                                         .loc[train_features_data['insult'] == 0]                                                         .loc[train_features_data['identity_hate'] == 0]                                                         .mean()


# In[30]:


df_types_mean


# In[31]:


types.append('normal comment')

df_types_mean['type'] = types
df_types_mean


# In[32]:


df_types_mean['total_length'] = pd.to_numeric(df_types_mean['total_length'])
df_types_mean['uppercase'] = pd.to_numeric(df_types_mean['uppercase'])
df_types_mean['exclamation_punctuation'] = pd.to_numeric(df_types_mean['exclamation_punctuation'])
df_types_mean['num_punctuation'] = pd.to_numeric(df_types_mean['num_punctuation'])
df_types_mean['num_symbols'] = pd.to_numeric(df_types_mean['num_symbols'])
df_types_mean['num_happy_smilies'] = pd.to_numeric(df_types_mean['num_happy_smilies'])
df_types_mean['num_sad_smilies'] = pd.to_numeric(df_types_mean['num_sad_smilies'])


# In[33]:


px.scatter(df_types_mean, x="exclamation_punctuation", y="num_punctuation",
           size="total_length", color="type", hover_name="type",
           size_max=55)


# In[34]:


fig = px.scatter_3d(df_types_mean, x='exclamation_punctuation', y='num_punctuation', z='uppercase', size='total_length', color='type',
                    hover_data=['type'])
fig.update_layout(scene_zaxis_type="log")
fig.show()


# In[35]:


px.scatter(df_types_mean, x="uppercase", y="num_punctuation",
           size="total_length", color="type", hover_name="type",
           size_max=55)


# In[36]:


px.scatter(df_types_mean, x="uppercase", y="num_symbols",
           size="total_length", color="type", hover_name="type",
           size_max=55)


# In[37]:


px.scatter(df_types_mean, x="uppercase", y="exclamation_punctuation",
           size="total_length", color="type", hover_name="type",
           size_max=55)


# In[38]:


tvec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000
)

tvec.fit(train_data['comment_text'])

train_texts = tvec.transform(train_data['comment_text'])
test_texts = tvec.transform(test_data['comment_text'])


# In[39]:


# If we want to work with frames from pandas for algorithms in Scikit-Learn, we could use a special module for that - Sklearn-pandas
# This module provides a bridge between Scikit-Learn's machine learning methods and pandas-style Data Frames.

mapper = DataFrameMapper([
      (['uppercase'], StandardScaler()),
      (['exclamation_punctuation'], StandardScaler()),
      (['num_punctuation'], StandardScaler()),
      (['num_symbols'], StandardScaler()),
      (['num_happy_smilies'],StandardScaler()),
      (['num_sad_smilies'],StandardScaler()),
      (['total_length'],StandardScaler())
], df_out=True)


# In[40]:


numeric_features_train = train_data.iloc[:, 8:15]
numeric_features_train.head()


# In[41]:


x_train = np.round(mapper.fit_transform(numeric_features_train.copy()), 2).values
x_train


# In[42]:


x_train_features = sparse.hstack((csr_matrix(x_train), train_texts))


# In[43]:


numeric_features_test = test_data.iloc[:, 1:]
numeric_features_test.head()


# In[44]:


x_test = np.round(mapper.fit_transform(numeric_features_test.copy()), 2).values
x_test


# In[45]:


x_test_features = sparse.hstack((csr_matrix(x_test), test_texts))


# In[46]:


train_features = x_train_features
test_features  = x_test_features


# In[47]:


scores = []
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for class_name in class_names:
  train_target = train_data[class_name]
  classifier = LogisticRegression(C=0.1, solver='sag')
  cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
  scores.append(cv_score)
  print('CV score for class {} is {}'.format(class_name, cv_score))

  classifier.fit(train_features, train_target)

print('Total CV score is {}'.format(np.mean(scores)))

