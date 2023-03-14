#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import json
import numpy as np


# In[2]:


df = pd.read_csv('../input/train.csv')


# In[3]:


#Basic Preprocessing
#df['question_text'] = df['question_text'].str.replace(r'\_',' ')
#df['question_text'] = df['question_text'].str.replace(r'[^\w]+',' ')
#df['question_text'] = df['question_text'].str.replace(r'\b\d+\b',' ')
#df['question_text'] = df['question_text'].str.replace(r' +',' ')
#df['question_text'] = df['question_text'].str.replace(r'[^A-Za-z]+', ' ')
#df['question_text'] = df['question_text'].str.strip()
df['question_text'] = df['question_text'].str.lower()


# In[4]:


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
glove_model = dict(get_coefs(*o.split(" ")) for o in open(glove))


# In[5]:


#glove_file = datapath('C:/Users/187403/Downloads/quora-insincere-questions-classification/glove.840B.300d/glove.840B.300d.txt')
#tmp_file = get_tmpfile("/quora-insincere-questions-classification/glove.840B.300d/glove_word2vec_tmp.txt")
#print('glove_file', glove_file)
#print('tmp_file:', tmp_file)

#_ = glove2word2vec(glove_file, tmp_file)
#glove_model = KeyedVectors.load_word2vec_format(tmp_file)
#word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)


# In[6]:


X = df['question_text']


# In[7]:


import re
def tokenize_input(row):
    return re.findall(r"[\w]+|[^\s\w]", row)

def make_tokenizer(texts):
    from keras.preprocessing.text import Tokenizer
    t = Tokenizer(filters='')
    t.fit_on_texts(texts)
    return t

splitted_input = X.apply(lambda row : ' '.join(tokenize_input(row)))
tokenizer = make_tokenizer(splitted_input)


# In[8]:


tokenized_input = tokenizer.texts_to_sequences(X)


# In[9]:


def form_embed_matrix(vocab_size, _tokenizer):
    embed_layer = np.zeros((vocab_size, 300))
    print("embed shape:", embed_layer.shape)

    for word, i in tokenizer.word_index.items():
        #print(word, i)
        try:
            if glove_model[word] is not None:
                embed_layer[i] = glove_model[word]            
        except KeyError:
            embed_layer[i] = glove_model['unk']
    return embed_layer


# In[10]:


vocab_size = len(tokenized_input)
embed_layer = form_embed_matrix(vocab_size, tokenizer)


# In[11]:


from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout


# In[12]:


input_layer = Input(shape = (105,))


# In[13]:


embedding_layer = Embedding(vocab_size, 300, weights=[embed_layer], trainable=False)(input_layer)


# In[14]:


bidirectional_lstm_layer = Bidirectional(LSTM(50))(embedding_layer)


# In[15]:


hidden_drop_1 = Dropout(0.3)(bidirectional_lstm_layer)


# In[16]:


output_layer = Dense(1, activation='sigmoid')(hidden_drop_1)


# In[17]:


model = Model(inputs = input_layer, outputs = output_layer)


# In[18]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[19]:


print(model.summary())


# In[20]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        


# In[21]:


from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(tokenized_input)


# In[22]:


#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X, df['target'], test_size = 0.999, random_state = 0)
#print('\nGiven data_dump has been split into 80:20 ratio, as train and test data respectively')


# In[23]:


model.fit(X, df['target'], 
          #validation_data = (X_valid, y_valid),
          batch_size = 1000,
          epochs=20, 
          callbacks =[es,mc])


# In[24]:


test_df = pd.read_csv("../input/test.csv")
X_test = df['question_text']
test_splitted_input = X_test.apply(lambda row : ' '.join(tokenize_input(row)))
test_tokenizer = make_tokenizer(test_splitted_input)
test_tokenized = test_tokenizer.texts_to_sequences(X_test)
#X_test_pad = pad_sequences(test_tokenized)


# In[25]:


y_prob = model.predict(X_test_pad)


# In[26]:


y_predict = [1 for prob in y_prob if prob>=0.5 else 0]


# In[27]:


predict_df = pd.DataFrame(y_predict, columns=['prediction'])
submission_df = pd.read_csv("../input/sample_submission.csv")
summary = np.where(predict_df['prediction']==submission_df['prediction'], 1, 0)
acc = summary/submission_df.shape[0]

