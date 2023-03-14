#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#PMR3508-2018-66f39f7a58 TAREFA2
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/ep22018"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd


# In[ ]:


trainep2 = pd.read_csv("../input/ep22018/train_data.csv", engine='python')


# In[ ]:


trainep2.head()


# In[ ]:


trainep2.info()


# In[ ]:


trainep2.describe()


# In[ ]:


trainep2_nospam = trainep2.query('ham == 1')
trainep2_spam = trainep2.query('ham == 0')


# In[ ]:


trainep2_nospam.head()


# In[ ]:


trainep2_nospam.describe()


# In[ ]:


trainep2_spam.head()


# In[ ]:


trainep2_spam.describe()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
(trainep2[trainep2['ham'] == True].mean() - trainep2[trainep2['ham'] == False].mean())[trainep2.columns[:54]].plot(kind = 'bar')


# In[ ]:


trainep2_nospam_cifrao = np.mean(trainep2_nospam['char_freq_$'])
trainep2_spam_cifrao = np.mean(trainep2_spam['char_freq_$'])

locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de $ por tipo de email')
plt.ylabel('Taxa de $')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_exclamacao = np.mean(trainep2_nospam['char_freq_!'])
trainep2_spam_exclamacao = np.mean(trainep2_spam['char_freq_!'])

locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de ! por tipo de email')
plt.ylabel('Taxa de !')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_free = np.mean(trainep2_nospam['word_freq_free'])
trainep2_spam_free = np.mean(trainep2_spam['word_freq_free'])
locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de free por tipo de email')
plt.ylabel('Taxa de free')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_free = np.mean(trainep2_nospam['word_freq_you'])
trainep2_spam_free = np.mean(trainep2_spam['word_freq_you'])
locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de you por tipo de email')
plt.ylabel('Taxa de you')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_free = np.mean(trainep2_nospam['word_freq_credit'])
trainep2_spam_free = np.mean(trainep2_spam['word_freq_credit'])
locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de credit por tipo de email')
plt.ylabel('Taxa de credit')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_free = np.mean(trainep2_nospam['word_freq_your'])
trainep2_spam_free = np.mean(trainep2_spam['word_freq_your'])
locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de your por tipo de email')
plt.ylabel('Taxa de your')
plt.xlabel('Tipo')


# In[ ]:


trainep2_nospam_free = np.mean(trainep2_nospam['word_freq_000'])
trainep2_spam_free = np.mean(trainep2_spam['word_freq_000'])
locations = [1, 2]
heights = [trainep2_nospam_cifrao, trainep2_spam_cifrao]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa de your por tipo de email')
plt.ylabel('Taxa de your')
plt.xlabel('Tipo')


# In[ ]:


Xtrainep2 = trainep2[["char_freq_$","char_freq_!","word_freq_free","word_freq_you","word_freq_credit","word_freq_your","word_freq_000"]]


# In[ ]:


Ytrainep2 = trainep2['ham']


# In[ ]:


porcentagemacerto =0
melhorvizinhomelhorneighbors = 0
for vizinhosTemp in range(3,20,1):
    knn = KNeighborsClassifier(n_neighbors=vizinhosTemp)
    scores = cross_val_score(knn, Xtrainep2, Ytrainep2, cv=10)
    scores
    if scores.mean() > porcentagemacerto:
        porcentagemacerto = scores.mean()
        melhorvizinho = vizinhosTemp


# In[ ]:


porcentagemacerto


# In[ ]:


melhorvizinho


# In[ ]:


gnb = GaussianNB()
scores = cross_val_score(gnb, Xtrainep2, Ytrainep2, cv=10, n_jobs=-1, scoring='roc_auc')


# In[ ]:


print('Gaussian: {0:.2f}%'.format(np.mean(scores)*100))


# In[ ]:


gnb = BernoulliNB()
scores = cross_val_score(gnb, Xtrainep2, Ytrainep2, cv=10, n_jobs=-1, scoring='roc_auc')


# In[ ]:


print('Bernoulli: {0:.2f}%'.format(np.mean(scores)*100))


# In[ ]:


gnb = MultinomialNB()
scores = cross_val_score(gnb, Xtrainep2, Ytrainep2, cv=10, n_jobs=-1, scoring='roc_auc')


# In[ ]:


print('Multinomial: {0:.2f}%'.format(np.mean(scores)*100))


# In[ ]:


gnb = ComplementNB()
score = cross_val_score(gnb, Xtrainep2, Ytrainep2, cv=10, n_jobs=-1, scoring='roc_auc')


# In[ ]:


print('Complement: {0:.2f}%'.format(np.mean(scores)*100))


# In[ ]:


plt.bar(['Gaussian', 'Complement', 'Multinomial', 'Bernoulli'], height=[scores[0], scores[1], scores[2], scores[3]])


# In[ ]:


testeep2 = pd.read_csv("../input/ep22018/test_features.csv")
testeep2.head()


# In[ ]:


Xtestep2 = testeep2[["char_freq_$","char_freq_!","word_freq_free","word_freq_you","word_freq_credit","word_freq_your","word_freq_000"]]


# In[ ]:


gnb = BernoulliNB()
scores = cross_val_score(gnb, Xtrainep2, Ytrainep2, cv=10, n_jobs=-1, scoring='roc_auc')
scores.mean()


# In[ ]:


gnb.fit(Xtrainep2, Ytrainep2)


# In[ ]:


Ytestep2 = gnb.predict(Xtestep2)


# In[ ]:


RES = pd.DataFrame({"id":testeep2.Id, "ham":Ytestep2})
RES.to_csv("submission-66f39f7a58-TAREFA2.csv", index=False)
RES

