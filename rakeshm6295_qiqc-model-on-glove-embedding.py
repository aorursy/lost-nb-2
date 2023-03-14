#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)


# In[3]:


train.head()


# In[4]:


import nltk


# In[5]:


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[6]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

question_lines = list()
lines = train['question_text'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    question_lines.append(words)


# In[7]:


len(question_lines)


# In[8]:


max_length = 100
EMBEDDING_DIM = 300
max_features = 50000


# In[9]:


import gensim


# In[10]:


pip install paramiko


# In[11]:


model = gensim.models.Word2Vec(sentences=question_lines,size=EMBEDDING_DIM,window=5,workers=4,min_count=1)
words = list(model.wv.vocab)


# In[12]:


len(words)


# In[13]:


model.wv.most_similar('hate')


# In[14]:


filename = 'quora_embedding_word2vec.text'
model.wv.save_word2vec_format(filename,binary=False)


# In[15]:


import os

embeddings_index={}
f = open(os.path.join('','quora_embedding_word2vec.text'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()


# In[16]:


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(question_lines)
sequences = tokenizer_obj.texts_to_sequences(question_lines)

word_index = tokenizer_obj.word_index
print(len(word_index))
question_pad = pad_sequences(sequences,maxlen=max_length)
target = train['target'].values
print(question_pad.shape)
print(target.shape)


# In[17]:


num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word, i in word_index.items():
    if i>num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[18]:


from keras.models import Sequential
from keras.layers  import Dense,Embedding,LSTM,GRU
from keras.initializers import Constant
model1 = Sequential()
model1.add(Embedding(num_words, EMBEDDING_DIM, input_length=max_length, weights=[embedding_matrix], trainable=False))
model1.add(GRU(units=32,dropout=0.2,recurrent_dropout=0.2))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# In[19]:


VALIDATION_SPLIT = 0.2

indices = np.arange(question_pad.shape[0])
np.random.shuffle(indices)
question_pad = question_pad[indices]
target = target[indices]
num_validation_samples = int(VALIDATION_SPLIT*question_pad.shape[0])

X_train_pad = question_pad[:-num_validation_samples]
y_train = target[:-num_validation_samples]
X_test_pad = question_pad[-num_validation_samples:]
y_test = target[-num_validation_samples:]


# In[20]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# In[21]:


model1.fit(X_train_pad,y_train,batch_size=128,epochs=25,validation_data=(X_test_pad,y_test),verbose=1,callbacks=[es])


# In[22]:


test.head()


# In[23]:


question_lines_test = list()
lines_test = test['question_text'].values.tolist()

for line in lines_test:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    question_lines_test.append(words)


# In[24]:


len(question_lines_test)


# In[25]:


tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(question_lines_test)
sequences_test = tokenizer_obj.texts_to_sequences(question_lines_test)

word_index_test = tokenizer_obj.word_index
print(len(word_index_test))
question_pad_test = pad_sequences(sequences_test,maxlen=max_length)
print(question_pad_test.shape)


# In[26]:


target_test = model1.predict(question_pad_test)


# In[27]:


target_test = (target_test>0.35).astype(int)
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = target_test
out_df.to_csv("submission.csv", index=False)

