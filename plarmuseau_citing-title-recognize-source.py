#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


direct='3c-shared-task-influence'

train=pd.read_csv('../input/'+direct+'/train.csv')
train


# In[3]:


test=pd.read_csv('../input/'+direct+'/test.csv')
test


# In[4]:


total=train.append(test)
total['Stxt']=total['cited_title']+' '+total['cited_author']+' '+total['citation_context']
total['Qtxt']=total['citing_title']+' '+total['citing_author']


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = TfidfVectorizer(sublinear_tf=True,ngram_range=(1, 2))

Qtf=cv.fit_transform(total['Qtxt'])
Stf=cv.transform(total['Stxt'])
Qtf,Stf


# In[6]:


from sklearn.decomposition import FastICA 
ICA = FastICA(n_components=3, random_state=12) 
Qi=ICA.fit_transform(Qtf.todense())
Si=ICA.fit_transform(Stf.todense())


# In[7]:


SQica=np.linalg.inv(np.dot(Si,Si.T)).dot(np.dot(Qi,Qi.T) )


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifier = LogisticRegression(C=0.1, solver='sag',max_iter=3000)
cv_score = np.mean(cross_val_score(classifier, SQica[:len(train)],train['citation_influence_label'].values, cv=3, scoring='roc_auc'))# train_features, train_target, cv=3, scoring='roc_auc'))
print('CV score for class {} is {}'.format('logis', cv_score))

classifier.fit(SQica[:3000], train['citation_influence_label'].values)
predictions = classifier.predict(SQica[3000:])
pred=pd.DataFrame(predictions,columns=['citation_influence_label'])
pred['unique_id']=test['unique_id']
pred.to_csv('submission3.csv', index=False)
pred.groupby('citation_influence_label').count       



# In[9]:


QStf=np.dot(Qtf,Stf.T)
QStf


# In[10]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=2000)
logreg.fit(QStf[:3000], train['citation_influence_label'].values)
predictions = logreg.predict(QStf[3000:])
pred=pd.DataFrame(predictions,columns=['citation_influence_label'])
pred['unique_id']=test['unique_id']
pred.to_csv('submission.csv', index=False)
pred.groupby('citation_influence_label').count()


# In[ ]:





# In[11]:


from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
QStf=cosine_distances(Qtf,Stf)


# In[12]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=2000)
logreg.fit(QStf[:3000], train['citation_influence_label'].values)
predictions = logreg.predict(QStf[3000:])
pred=pd.DataFrame(predictions,columns=['citation_influence_label'])
pred['unique_id']=test['unique_id']
pred.to_csv('submission2.csv', index=False)
pred.groupby('citation_influence_label').count()


# In[13]:


QSi=np.linalg.pinv(np.dot(Qtf,Stf.T).todense())
QSi.shape


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifier = LogisticRegression(C=0.1, solver='sag',max_iter=3000)
cv_score = np.mean(cross_val_score(classifier, QSi[:len(train)],train['citation_influence_label'].values, cv=3, scoring='roc_auc'))# train_features, train_target, cv=3, scoring='roc_auc'))
print('CV score for class {} is {}'.format('logis', cv_score))

classifier.fit(QSi[:3000], train['citation_influence_label'].values)
predictions = classifier.predict(QSi[3000:])
pred=pd.DataFrame(predictions,columns=['citation_influence_label'])
pred['unique_id']=test['unique_id']
pred.to_csv('submission3.csv', index=False)
pred.groupby('citation_influence_label').count       



# In[15]:


QQi=np.linalg.pinv(np.dot(Stf,Stf.T).todense())
QSQ=np.dot(QQi,np.dot(Qtf,Qtf.T).todense())


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifier = LogisticRegression(C=0.1, solver='sag',max_iter=3000)
cv_score = np.mean(cross_val_score(classifier, QSQ[:len(train)],train['citation_influence_label'].values, cv=3, scoring='roc_auc'))# train_features, train_target, cv=3, scoring='roc_auc'))
print('CV score for class {} is {}'.format('logis', cv_score))

classifier.fit(QSQ[:3000], train['citation_influence_label'].values)
predictions = classifier.predict(QSQ[3000:])
pred=pd.DataFrame(predictions,columns=['citation_influence_label'])
pred['unique_id']=test['unique_id']
pred.to_csv('submission4.csv', index=False)
pred.groupby('citation_influence_label').count       


