#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler # Randomize Data

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,Embedding,Dropout,Reshape,Flatten,LSTM,Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[2]:


def text_to_wordlist(text , remove_stopwords = False , stem_words = False):
    
    # Clean The text , with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()
    
    # Optionally , remove stop words
    
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        text= [w for w in text if not w in stops]
        
    text = ' '.join(text)
    
    # Clean the text
    
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    
    # Optionally , shorten words to their stems
    
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = ' '.join(stemmed_words)
        
    # Return a list of words
    return(text)


# In[3]:


def read_train_data(file):
    texts = []
    labels = []
    df_train = pd.read_csv(file)
    line_num = 0
    for idx in range(len(df_train)):
        
        # if Line_num < 8000: # For test purpose
        
        texts.append(text_to_wordlist(df_train['question_text'][idx]))
        labels.append(df_train['target'][idx])
        line_num += 1
    return texts,labels


# In[4]:


def read_test_data(file):
    texts = []
    ids = []
    df_test = pd.read_csv(file)
    line_num  = 0
    for idx in range(len(df_test)):
        # if line_num < 200 : # for test purpose
        texts.append(text_to_wordlist(df_test['question_text'][idx]))
        ids.append(df_test['qid'][idx])
        line_num += 1 
    return texts , ids


# In[5]:


def preprocess_data(train_data_file , test_data_file , max_seq_len , split_ratio):
    
    # 1) load train and test datasets
    texts , labels = read_train_data(train_data_file)
    
    print('Finished loading train.csv: %s samples '%len(texts))
    
    test_texts , test_ids = read_test_data(test_data_file)
    
    print('Finished loading test.csv: %s samples '%len(test_texts))
    
    # 2) train the tokenizer
    
    tokenizer = Tokenizer(num_words = 200000)
    tokenizer.fit_on_texts(texts + test_texts)
    word_index = tokenizer.word_index
    print('%s tokens in total' % len(word_index))
    
    # 3) sentences to sequences
    
    train_sequences = tokenizer.texts_to_sequences(texts)
    test_sequence = tokenizer.texts_to_sequences(test_texts)
    x = pad_sequences(train_sequences , maxlen = max_seq_len , padding='post',truncating='post')
    test_x = pad_sequences(test_sequence , maxlen=max_seq_len , padding='post',truncating = 'post')
    
    # 4) final step
    
    num_samples = len(x)
    perm = np.random.permutation(num_samples)
    idx = int(num_samples * split_ratio)
    idx_train = perm[:idx]
    idx_val = perm[idx:]
    
    train_x = x[idx_train]
    val_x = x[idx_val]
    
    y = np.array(labels)
    train_y = y[idx_train]
    val_y = y[idx_val]
    
    print('shape of taining data: {}'.format(train_x.shape))
    print('shape of training label: {}'.format(train_y.shape))
    print('shape of val data: {}'.format(val_x.shape))
#     print('shape of test data : {}'.format(test_x.shape))
    
    return train_x , train_y,val_x , val_y , test_x , test_ids , word_index


# In[6]:


# Generating embedding matrix

def load_embeddings_index(file , embedding_dim):
    embeddings_index = {} # dict
    f = open(file)
    for line in tqdm(f):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:],dtype = 'float32')
        # Ex ~> word : 123 233 133 232
        embeddings_index[word] = coefs
        
        # embeddings for all words in glove are contained here
    
    print('Found %s word vectors' % len(embeddings_index))
        


# In[7]:


def generate_embedding_layer(word_index , embeddings_index):
    
        nb_words = len(word_index) + 1
        embeddings_matrix = np.zeros((nb_words , embedding_dim)) #embedding matrix for all words
        
        word_out_netword = []
        
        for word , i in word_index.items():
            embedding_vector = embeddings_index.get(word) # Get embedding vector for a given word
            
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
                
            else:
                word_out_netword.append(word)
        
        percent = round(100 * len(word_out_network) / len(word_index),1)
        print('%s precent of words out of network' % percent)
        
        return embeddings_matrix


# In[8]:


# Function for building the model

def build_model(max_seq_len , word_index , embedding_dim , embedding_matrix):
    
    # 1) Embedding Layer
    
    inp = Input(shape=(max_seq_len,),dtype = 'int32')
    
    x = Embedding(len(word_index) + 1,
                 embedding_dim,
                 input_length= max_seq_len,
                 weights = [embedding_matrix],
                 trainable = False)(inp)
    
    # 2) LSTM Layer
    
    x = LSTM(64 , dropout = 0.2 , recurrent_dropout=0.2)(x)
    x = Dropout(0,2)(x)
    x = BatchNormalization()(x)
    
    # 3) Dense Layer
    
    x = Dense(32 , activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    
    # 4) Output layer
    
    preds = Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs = inp , outputs = preds)
    return model
    


# In[9]:


# --- step 1) preprocessing texts (texts to numberical values)

max_seq_len = 30
split_ratio = 0.8
train_file = '../input/quora-insincere-questions-classification/train.csv'
test_file = '../input/quora-insincere-questions-classification/test.csv'

train_x , train_y , val_x , val_y , test_x , test_idx , word_index = preprocess_data(train_file , test_file , max_seq_len , split_ratio)


# In[10]:


# --- Step 2) Prepare embedding matrix


embedding_dim = 300
embedding_matrix = np.zeros((max(list(word_index.values())) + 1 , embedding_dim),dtype = 'float')
embedding_file = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
f = open(embedding_file)
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    if word not in word_index:
        continue
    embedding_matrix[word_index[word]] = np.asarray(values[1:],dtype = 'float32')
    f.close


# In[11]:


# -- Step 3) Build The model

# keras.backend_clear_session()

model = build_model(max_seq_len , word_index , embedding_dim , embedding_matrix)
model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['acc'])
model.summary()


# In[12]:


#-- Step 4) Train The Model

nb_epoches = 200

early_stopping = EarlyStopping(monitor = 'val_loss',patience = 5)
model_name = 'model_best.h5'
model_checkpoint = ModelCheckpoint(model_name , save_best_only = True)

hist = model.fit(train_x , train_y ,                validation_data = (val_x , val_y),                epochs = nb_epoches , batch_size = 2048,
                shuffle = True , verbose = 2 ,\
                callbacks = [early_stopping , model_checkpoint])

# Load Model : 
model.load_weights(model_name)
best_val_score = min(hist.history['val_loss'])
print('min val loss is' , best_val_score)


# In[13]:


# -- Final Step: Submission

preds = model.predict(test_x, batch_size=1024, verbose=1)
preds = (preds > 0.35).astype(int)

sub = pd.DataFrame({'qid':test_idx, 'prediction':preds.ravel()})
sub.to_csv('submission.csv', index=False)

