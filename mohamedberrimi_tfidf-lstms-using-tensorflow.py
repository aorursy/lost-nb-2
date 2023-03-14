#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


train=pd.read_csv('/kaggle/input/fake-news/train.csv')
test=pd.read_csv('/kaggle/input/fake-news/test.csv')
submit=pd.read_csv('/kaggle/input/fake-news/submit.csv')


# In[4]:


train.head()


# In[5]:


df=train.dropna() # Removing every missing value existing on the dataframe 


# In[6]:


X=df['title'] # Make Text column in Variable X
y=df['label'] # Make the labels on variable Y


# In[7]:


# Firstly, fill all the null spaces with a space
train = train.fillna(' ')
train['total'] = train['title'] + ' ' + train['author'] + ' ' +  train['text']


# In[8]:


pip install nltk 


# In[9]:


# cleaning our dataset 
import nltk 
from nltk.corpus import stopwords # to remove stop words such as ' the , they , it, a ...'
from nltk.stem import WordNetLemmatizer # for lemmatization task 


# In[10]:


stop_words = stopwords.words('english')


# In[11]:



word_data = "It originated from the idea that there are readers who prefer learning new skills from the comforts of their drawing rooms"
nltk_tokens = nltk.word_tokenize(word_data)
print(nltk_tokens)


# In[12]:


lemmatizer = WordNetLemmatizer()


# In[13]:


import re
for index, row in train.iterrows():
    filter_sentence = ''
    sentence = row['total']
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenization
    words = nltk.word_tokenize(sentence)
    # Stopwords removal
    words = [w for w in words if not w in stop_words]
    # Lemmatization
    for words in words:
        filter_sentence = filter_sentence  + ' ' +  str(lemmatizer.lemmatize(words)).lower()
    train.loc[index, 'total'] = filter_sentence
train = train[['total', 'label']]


# In[14]:


X_train = train['total']
Y_train = train['label']


# In[15]:


# this could take a while.
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix,  Y_train, random_state=0)


# In[17]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Accuracy = logreg.score(X_test, y_test)
print( 'LogisticRegression Accuracy :  ',Accuracy )


# In[18]:


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)
Accuracy = NB.score(X_test, Y_test)
print( 'Multinomial NB Accuracy :  ',Accuracy )


# In[ ]:





# In[ ]:





# In[19]:


import tensorflow as tf # Import Latest tensorflow version 
tf.__version__


# In[20]:


from tensorflow.keras.layers import Embedding, LSTM, Dense ## Neural networks layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot # to encode the depending variable 
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[21]:


voc_size=5000 # max num words to take into consideration while training your model


# In[22]:


X=[i.lower() for i in X] # lowercase each text 


# In[23]:


onehot=[one_hot(words,voc_size) for words in X] 


# In[24]:


sen_len=30
embedded_doc=pad_sequences(onehot, padding='pre', maxlen=sen_len) # pad sequence your texts
print(embedded_doc)


# In[25]:


embedding_vector_feature=40 
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_feature, input_length=sen_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
# sigmoid : to handle the output ( binary case )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# binary_crossentropy : because we have a binaray classification task 
# Adam : Stochastic gradient decenet optimizatiion 
print(model.summary())


# In[26]:


X_final=np.array(embedded_doc)
y_final=np.array(y)


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.20, random_state=0)


# In[28]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


# In[ ]:




