#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# !pip install /kaggle/input/kerasselfattention/keras-self-attention-0.42.0
# import tensorflow as tf
import os
# from tensorflow.keras.layers import Layer
# import tensorflow.keras.backend as K
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow_hub as hub
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# import config
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import re
import numpy as np
import nltk
import keras.backend as K
# from nltk.probability import FreqDist
from nltk.corpus import stopwords
# import string
from keras.preprocessing.sequence import pad_sequences
# # nltk.download('stopwords')
# # nltk.download('punkt')
eng_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
import gc, os, pickle
from nltk import word_tokenize, sent_tokenize

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD

def plot_len(df, col_name, i):
    plt.figure(i)
    sns.distplot(df[col_name].str.len())
    plt.ylabel("length of string")
    plt.show()

def plot_cnt_words(df, col_name, i):
    plt.figure(i)
    vals = df[col_name].apply(lambda x: len(x.strip().split()))
    sns.distplot(vals)
    plt.ylabel("count of words")
    plt.show()


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

    return df

def plot_freq_dist(train_data):
    freq_dist = FreqDist([word for text in train_data['question_body'].str.replace('[^a-za-z0-9^,!.\/+-=]',' ') for word in text.split()])
    plt.figure(figsize=(20, 7))
    plt.title('Word frequency on question title (Training Data)').set_fontsize(25)
    plt.xlabel('').set_fontsize(25)
    plt.ylabel('').set_fontsize(25)
    freq_dist.plot(60,cumulative=False)
    plt.show()

def get_tfidf_features(data, dims=256):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tsvd = TruncatedSVD(n_components = dims, n_iter=5)
    tfquestion_title = tfidf.fit_transform(data["question_title"].values)
    tfquestion_title = tsvd.fit_transform(tfquestion_title)

    tfquestion_body = tfidf.fit_transform(data["question_body"].values)
    tfquestion_body = tsvd.fit_transform(tfquestion_body)

    tfanswer = tfidf.fit_transform(data["answer"].values)
    tfanswer = tsvd.fit_transform(tfanswer)

    return tfquestion_title, tfquestion_body, tfanswer

def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / r_den


# In[3]:


from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation, Average, Maximum
from keras.layers import Concatenate, GRU, Maximum, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, Dot
from keras.models import Model
# from keras_self_attention import SeqSelfAttention

tokens = []
def get_words(col):
  global tokens
  toks = []
  for x in sent_tokenize(col):
    tokens += word_tokenize(x)
    toks += word_tokenize(x)
  return toks

def convert_to_indx(col, word2idx, vocab_size):
  return [word2idx[word] if word in word2idx else vocab_size for word in col]

def LSTM_model_initial(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=64, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, epochs=2):
    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    Eqt = Embedding(vocab_size, embedding_size, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size, embedding_size, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size, embedding_size, input_length=maxlen_an)(inpan)
    if(rnn_type=="LSTM"):
        BLqt = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqt)
        BLqb = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqb)
        BLan = Bidirectional(LSTM(rnn_units, return_sequences=True))(Ean)
    elif(rnn_type=="GRU"):
        BLqt = Bidirectional(GRU(rnn_units))(Eqt)
        BLqb = Bidirectional(GRU(rnn_units))(Eqb)
        BLan = Bidirectional(GRU(rnn_units))(Ean)
        
    BLqt1 = GlobalAveragePooling1D()(BLqt)
    BLqb1 = GlobalAveragePooling1D()(BLqb)
    BLan1 = GlobalAveragePooling1D()(BLan)
    BLqt2 = GlobalMaxPooling1D()(BLqt)
    BLqb2 = GlobalMaxPooling1D()(BLqb)
    BLan2 = GlobalMaxPooling1D()(BLan)
    
    BLqt = Concatenate()([BLqt1,BLqt2])
    BLqb = Concatenate()([BLqb1,BLqb2])
    BLan = Concatenate()([BLan1,BLan2])
    
    Dqt = Dropout(dropout_rate)(BLqt)
    Dqb = Dropout(dropout_rate)(BLqb)
    Dan = Dropout(dropout_rate)(BLan)
    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
    df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    target_columns = df_submission.columns
    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(1,len(target_columns)):
        outp[target_columns[i]] = y_test[:, i-1]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
def LSTM_model_stacked(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=64, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, num_stacks=2, epochs=2):
    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    Eqt = Embedding(vocab_size, embedding_size, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size, embedding_size, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size, embedding_size, input_length=maxlen_an)(inpan)
    if(rnn_type=="LSTM"):
        BLqt = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqt)
        BLqb = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqb)
        BLan = Bidirectional(LSTM(rnn_units, return_sequences=True))(Ean)
        for i in range(num_stacks-1):
            BLqt = Bidirectional(LSTM(rnn_units, return_sequences=True))(BLqt)
            BLqb = Bidirectional(LSTM(rnn_units, return_sequences=True))(BLqb)
            BLan = Bidirectional(LSTM(rnn_units, return_sequences=True))(BLan)
    elif(rnn_type=="GRU"):
        BLqt = Bidirectional(GRU(rnn_units))(Eqt)
        BLqb = Bidirectional(GRU(rnn_units))(Eqb)
        BLan = Bidirectional(GRU(rnn_units))(Ean)
    Dqt = Dropout(dropout_rate)(Lambda(lambda x: x[:,-1,:], output_shape=(128,))(BLqt))
    Dqb = Dropout(dropout_rate)(Lambda(lambda x: x[:,-1,:], output_shape=(128,))(BLqb))
    Dan = Dropout(dropout_rate)(Lambda(lambda x: x[:,-1,:], output_shape=(128,))(BLan))
    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
    df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    target_columns = df_submission.columns
    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(1,len(target_columns)):
        outp[target_columns[i]] = y_test[:, i-1]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
# from keras.layers import Lambda, Dot, Activation, Average

def attention_3d_block_self(hidden_states, rnn_units=64):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False)(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,))(hidden_states)
    score = Dot([2, 1])([score_first_part, h_t])
    attention_weights = Activation('softmax')(score)
    context_vector = Dot([1, 1])([hidden_states, attention_weights])
    pre_activation = Concatenate()([context_vector, h_t])
    attention_vector = Dense(rnn_units*2, use_bias=False, activation='tanh')(pre_activation)
    return attention_vector
    
    
def attention_3d_block_another(hidden_states1, hidden_state2,rnn_units=64):
    hidden_size = int(hidden_states1.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False)(hidden_states1)
    score = Dot([2, 1])([score_first_part, hidden_state2])
    attention_weights = Activation('softmax')(score)
    context_vector = Dot([1, 1])([hidden_states1, attention_weights])
    pre_activation = Concatenate()([context_vector, hidden_state2])
    attention_vector = Dense(rnn_units*2, use_bias=False, activation='tanh')(pre_activation)
    return attention_vector

def LSTM_model_modified_with_attention_self(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=64, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, epochs=2):
    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    
    Eqt = Embedding(vocab_size, embedding_size, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size, embedding_size, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size, embedding_size, input_length=maxlen_an)(inpan)
    
    if(rnn_type=="LSTM"):
        BLqt = Bidirectional(LSTM(rnn_units, return_state=True))(Eqt)
        BLqb = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqb, initial_state=BLqt[1:])
        BLan = Bidirectional(LSTM(rnn_units, return_sequences=True))(Ean)
    elif(rnn_type=="GRU"):
        BLqt = Bidirectional(GRU(rnn_units))(Eqt)
        BLqb = Bidirectional(GRU(rnn_units))(Eqb)
        BLan = Bidirectional(GRU(rnn_units))(Ean)
    
    AtQ = attention_3d_block_self(BLqb, rnn_units)
    AtAn = attention_3d_block_self(BLan, rnn_units)
#     Dqt = Dropout(dropout_rate)(BLqt[0])
    qbin = GlobalAveragePooling1D()(BLqb)
    anin = GlobalAveragePooling1D()(BLan)
#     qbin = Lambda(lambda x: x[:,-1,:], output_shape=(rnn_units*2,), name="lambda_layer1")(BLqb_out)
#     anin = Lambda(lambda x: x[:,-1,:], output_shape=(rnn_units*2,), name="lambda_layer2")(BLan_out)
#     attn_out, attn_states = AttentionLayer()([BLqb_out, BLan_out], verbose=True)
    Dqb = Dropout(dropout_rate)(qbin)
    Dan = Dropout(dropout_rate)(anin)
#     print(Dqb.shape, Dan.shape, attn_out.shape)
    Concatenated = Concatenate()([Dqb, Dan, AtQ, AtAn])
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
    df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    target_columns = df_submission.columns
    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(1,len(target_columns)):
        outp[target_columns[i]] = y_test[:, i-1]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
def LSTM_model_modified_with_attention_a2q(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=64, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, epochs=2):
    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    
    Eqt = Embedding(vocab_size, embedding_size, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size, embedding_size, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size, embedding_size, input_length=maxlen_an)(inpan)
    
    if(rnn_type=="LSTM"):
        BLqt = Bidirectional(LSTM(rnn_units, return_state=True))(Eqt)
        BLqb = Bidirectional(LSTM(rnn_units, return_sequences=True))(Eqb, initial_state=BLqt[1:])
        BLan = Bidirectional(LSTM(rnn_units, return_sequences=True))(Ean)
    elif(rnn_type=="GRU"):
        BLqt = Bidirectional(GRU(rnn_units))(Eqt)
        BLqb = Bidirectional(GRU(rnn_units))(Eqb)
        BLan = Bidirectional(GRU(rnn_units))(Ean)
    
    list1 = [attention_3d_block_another(BLqb, Lambda(lambda x: x[:,i,:], output_shape=(rnn_units*2,))(BLan), rnn_units) for i in range(maxlen_an)]
    AtA2Qm = Maximum()(list1)
    AtA2Qa = Average()(list1)
    
    Dqbin1 = GlobalAveragePooling1D()(BLqb)
    Dqbin2 = GlobalMaxPooling1D()(BLqb)
    Dqbin = Concatenate()([Dqbin1, Dqbin2])
    Dqb = Dropout(dropout_rate)(Dqbin)
    
    Danin1 = GlobalAveragePooling1D()(BLan)
    Danin2 = GlobalMaxPooling1D()(BLan)
    Danin = Concatenate()([Danin1, Danin2])
    Dan = Dropout(dropout_rate)(Danin)
    
    Concatenated = Concatenate()([Dqb, Dan, AtA2Qm, AtA2Qa])
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
    df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    target_columns = df_submission.columns
    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(1,len(target_columns)):
        outp[target_columns[i]] = y_test[:, i-1]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
    
def LSTM_model_modified_concatenated_qa(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=64, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, epochs=2):
    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    
    Eqt = Embedding(vocab_size, embedding_size, input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size, embedding_size, input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size, embedding_size, input_length=maxlen_an)(inpan)
    
    if(rnn_type=="LSTM"):
        BLqt = Bidirectional(LSTM(rnn_units, return_state=True))(Eqt)
        BLqb = Bidirectional(LSTM(rnn_units, return_state=True))(Eqb, initial_state=BLqt[1:])
        BLan = Bidirectional(LSTM(rnn_units, return_state=True))(Ean, initial_state=BLqb[1:])
    elif(rnn_type=="GRU"):
        BLqt = Bidirectional(GRU(rnn_units, return_state=True))(Eqt)
        BLqb = Bidirectional(GRU(rnn_units, return_state=True))(Eqb, initial_state=BLqt[1:])
        BLan = Bidirectional(GRU(rnn_units, return_state=True))(Ean, initial_state=BLqb[1:])
        
    Dqt = Dropout(dropout_rate)(BLqt[0])
    Dqb = Dropout(dropout_rate)(BLqb[0])
    Dan = Dropout(dropout_rate)(BLan[0])
    
    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
    df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    target_columns = df_submission.columns
    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(1,len(target_columns)):
        outp[target_columns[i]] = y_test[:, i-1]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)

df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
# LSTM_model_modified_with_attention_self(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
#                        rnn_units=64, maxlen_qt = 40, maxlen_qb = 260, maxlen_an = 210,
#                       dropout_rate=0.2, dense_hidden_units=50, epochs=5)
# LSTM_model_initial(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
#                        rnn_units=128, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
#                       dropout_rate=0.2, dense_hidden_units=60, epochs=6)
# LSTM_model_stacked(df_train, df_test, df_submission, rnn_type="LSTM")
# LSTM_model_modified_with_attention_self_with_lib(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
#                        rnn_units=64, maxlen_qt = 40, maxlen_qb = 260, maxlen_an = 210,
#                       dropout_rate=0.2, dense_hidden_units=40, epochs=3)
# y_test = model.predict(X_test)
LSTM_model_modified_with_attention_a2q(df_train, df_test, df_submission, rnn_type="LSTM", embedding_size=200, 
                       rnn_units=128, maxlen_qt = 26, maxlen_qb = 260, maxlen_an = 210,
                      dropout_rate=0.2, dense_hidden_units=60, epochs=6)


# In[4]:


def dan_model(df_train, df_test, df_submission, batch_size=8, epochs=4, hidden_layers=[120]):
  if(len(hidden_layers)<1):
    print("Non-Empty Hidden Layers List Required!")
    return
  module_url = "/kaggle/input/sent-embed-model"
  model = hub.load(module_url)

  def embed(input):
    return model(input)

  qt_train = np.array(embed(df_train["question_title"]))
  qb_train = np.array(embed(df_train["question_body"]))
  an_train = np.array(embed(df_train["answer"]))

  X_train = np.concatenate([qt_train, qb_train, an_train], axis=1)

  qt_test = np.array(embed(df_test["question_title"]))
  qb_test = np.array(embed(df_test["question_body"]))
  an_test = np.array(embed(df_test["answer"]))

  X_test = np.concatenate([qt_test, qb_test, an_test], axis=1)

  target_columns = df_submission.columns[1:]
  y_train = df_train[target_columns].values

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(hidden_layers[0], activation="relu"))
  model.add(tf.keras.layers.Dropout(0.2))  
  for h in hidden_layers[1:]:
    model.add(tf.keras.layers.Dense(h, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))  
  model.add(tf.keras.layers.Dense(30, activation="sigmoid"))

  model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, batch_size=batch_size,
            epochs=epochs, validation_split=0.1)
  print(model.summary())
  y_test = model.predict(X_test)

  outp = {}
  outp["qa_id"] = df_test["qa_id"]
  for i in range(len(target_columns)):
      outp[target_columns[i]] = y_test[:, i]
  my_submission = pd.DataFrame(outp)
  my_submission.to_csv('submission.csv', index=False)

def pad_seq_custom(original_series, maxlen=50):
    n = original_series.size
    for i in range(n):
        ll = len(original_series[i])
        for kk in range(maxlen-ll):
            np.append(original_series[i],[0 for rr in range(512)])
    return original_series
    
    
def dan_model_with_words(df_train, df_test, df_submission, batch_size=8,
                         epochs=4, hidden_layers=[120],
                         rnn_units=64, maxlen_qt = 26,
                         maxlen_qb = 260, maxlen_an = 210,
                         dropout_rate=0.2, dense_hidden_units=50,
                         embedding_size=512):
    if(len(hidden_layers)<1):
        print("Non-Empty Hidden Layers List Required!")
        return
    module_url = "/kaggle/input/sent-embed-model"
    model = hub.load(module_url)

    def embed(input):
        return model([input])

    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    embedding_matrix = np.zeros((vocab_size+1, embedding_size))
    for word, i in word2idx.items():
        embedding = np.array(embed(word))
        if embedding is not None:
            embedding_matrix[i] = embedding


    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    
    Eqt = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_qt, trainable=False)(inpqt)
    Eqb = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_qb, trainable=False)(inpqb)
    Ean = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_an, trainable=False)(inpan)

    BLqt = Bidirectional(LSTM(rnn_units))(Eqt)
    BLqb = Bidirectional(LSTM(rnn_units))(Eqb)
    BLan = Bidirectional(LSTM(rnn_units))(Ean)
    
    Dqt = Dropout(dropout_rate)(BLqt)
    Dqb = Dropout(dropout_rate)(BLqb)
    Dan = Dropout(dropout_rate)(BLan)

    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    
    Ds = Dense(dense_hidden_units, activation='elu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=8, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(len(target_columns)):
        outp[target_columns[i]] = y_test[:, i]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
# dan_model_with_words(df_train, df_test, df_submission, epochs=4)


# In[5]:


# from gensim.models import KeyedVectors

# df_train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
# df_test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
# df_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")

def glove_model_with_words(df_train, df_test, df_submission, batch_size=8,
                         epochs=4, rnn_units=64, maxlen_qt = 26,
                         maxlen_qb = 260, maxlen_an = 210,
                         dropout_rate=0.2, dense_hidden_units=50,
                         embedding_size=300):
    filename = '/kaggle/input/googlenewsvectors/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    def embed(model, input):
        if input not in model.wv.vocab:
            return np.array([0 for _ in range(300)])
        return model.wv[input]

    columns = ['question_title','question_body','answer']
    df_train = clean_data(df_train, columns)
    df_test = clean_data(df_test, columns)
    # columns = ["question_title", "question_body", "answer"]
    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: get_words(x))
      df_test[col] = df_test[col].apply(lambda x: get_words(x))
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(vocab):
      word2idx[word] = idx
      idx2word[idx] = word

    for col in columns:
      df_train[col] = df_train[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))
      df_test[col] = df_test[col].apply(lambda x: convert_to_indx(x,word2idx,vocab_size))

    X_train_question_title = pad_sequences(df_train["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_train_question_body = pad_sequences(df_train["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_train_answer = pad_sequences(df_train["answer"], maxlen=maxlen_an, padding='post', value=0)

    X_test_question_title = pad_sequences(df_test["question_title"], maxlen=maxlen_qt, padding='post', value=0)
    X_test_question_body = pad_sequences(df_test["question_body"], maxlen=maxlen_qb, padding='post', value=0)
    X_test_answer = pad_sequences(df_test["answer"], maxlen=maxlen_an, padding='post', value=0)

    target_columns = df_submission.columns[1:]
    y_train = df_train[target_columns]

    embedding_matrix = np.zeros((vocab_size+1, embedding_size))
    for word, i in word2idx.items():
        embedding = np.array(embed(model, word))
        if embedding is not None:
            embedding_matrix[i] = embedding


    inpqt = Input(shape=(maxlen_qt,),name='inpqt')
    inpqb = Input(shape=(maxlen_qb,),name='inpqb')
    inpan = Input(shape=(maxlen_an,),name='inpan')
    
    Eqt = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_qt)(inpqt)
    Eqb = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_qb)(inpqb)
    Ean = Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=maxlen_an)(inpan)

    BLqt = Bidirectional(LSTM(rnn_units))(Eqt)
    BLqb = Bidirectional(LSTM(rnn_units))(Eqb)
    BLan = Bidirectional(LSTM(rnn_units))(Ean)

    Dqt = Dropout(dropout_rate)(BLqt)
    Dqb = Dropout(dropout_rate)(BLqb)
    Dan = Dropout(dropout_rate)(BLan)
    
    Concatenated = Concatenate()([Dqt, Dqb, Dan])
    
    Ds = Dense(dense_hidden_units, activation='relu')(Concatenated)
    Dsf = Dense(30, activation='sigmoid')(Ds)

    model = Model(inputs=[inpqt, inpqb, inpan], outputs=Dsf)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit({'inpqt': X_train_question_title, 'inpqb': X_train_question_body, 'inpan': X_train_answer}, y_train, batch_size=32, epochs=epochs, validation_split=0.1)

    y_test = model.predict({'inpqt': X_test_question_title, 'inpqb': X_test_question_body, 'inpan': X_test_answer})

    outp = {}
    outp["qa_id"] = df_test["qa_id"]
    for i in range(len(target_columns)):
        outp[target_columns[i]] = y_test[:, i]
    my_submission = pd.DataFrame(outp)
    my_submission.to_csv('submission.csv', index=False)
    
# glove_model_with_words(df_train, df_test, df_submission, epochs=5)


# In[6]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# import tensorflow as tf
# import tensorflow.keras.backend as K
# import os
# from scipy.stats import spearmanr
# from math import floor, ceil
# from transformers import *

BERT_PATH = '/kaggle/input/bert-base-uncased-huggingface-transformer/'
BERT_LARGE_PATH = '/kaggle/input/bert-large-uncased-huggingface-transformer/'

def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, length, is_pair = True):
        if is_pair:
            inputs = tokenizer.encode_plus(str1, str2,
                add_special_tokens=True,
                max_length=length,
                truncation_strategy="only_second")
        else:
            inputs = tokenizer.encode_plus(str1, None,
                add_special_tokens=True,
                max_length=length,
                truncation_strategy="only_first")
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        title, question, max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, max_sequence_length, False)
    
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a =         _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]

def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def create_model():
    q_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
    # pretrained model has been downloaded manually and uploaded to kaggle. 
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn,], outputs=x)
    
    return model


# In[7]:


# from datetime import datetime

def create_model_large():
    q_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
    # pretrained model has been downloaded manually and uploaded to kaggle. 
    bert_model = TFBertModel.from_pretrained(
        BERT_LARGE_PATH+'bert-large-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn,], outputs=x)
    
    return model

def bert_uncased():
    PATH = '/kaggle/input/google-quest-challenge/'

    BERT_PATH = '/kaggle/input/bert-base-uncased-huggingface-transformer/'
    # !cp /kaggle/input/bert-base-uncased-huggingface-transformer/bert-base-uncased-vocab.txt ./vocab.txt
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH + "bert-base-uncased-vocab.txt")

    MAX_LEN = 90

    df_train = pd.read_csv(PATH+'train.csv')
    df_test = pd.read_csv(PATH+'test.csv')
    df_submission = pd.read_csv(PATH+'sample_submission.csv')

    output_categories = list(df_submission.columns)
    input_categories = ["question_title", "question_body", "answer"]
    
    outputs = df_train[output_categories[1:]].values
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_LEN)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_LEN)
    
    test_preds = []
    train_inputs = [inputs[i] for i in range(len(inputs))]
    train_outputs = outputs

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    logdir = "/kaggle/working/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.fit(train_inputs, train_outputs, epochs=1, batch_size=16, validation_split=0.5, callbacks=[tensorboard_callback])

    test_preds.append(model.predict(test_inputs))
    df_submission.iloc[:, 1:] = np.average(test_preds, axis=0)
    df_submission.to_csv('submission.csv', index=False)
    
def bert_large_uncased():
    PATH = '/kaggle/input/google-quest-challenge/'

    BERT_PATH = '/kaggle/input/bert-large-uncased-huggingface-transformer/'
    # !cp /kaggle/input/bert-base-uncased-huggingface-transformer/bert-base-uncased-vocab.txt ./vocab.txt
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH + "bert-large-uncased-vocab.txt")

    MAX_LEN = 90

    df_train = pd.read_csv(PATH+'train.csv')
    df_test = pd.read_csv(PATH+'test.csv')
    df_submission = pd.read_csv(PATH+'sample_submission.csv')

    output_categories = list(df_submission.columns)
    input_categories = ["question_title", "question_body", "answer"]
    
    outputs = df_train[output_categories[1:]].values
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_LEN)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_LEN)
    
    test_preds = []
    train_inputs = [inputs[i] for i in range(len(inputs))]
    train_outputs = outputs

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    logdir = "/kaggle/working/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.fit(train_inputs, train_outputs, epochs=1, batch_size=16, validation_split=0.5, callbacks=[tensorboard_callback])

    test_preds.append(model.predict(test_inputs))
    df_submission.iloc[:, 1:] = np.average(test_preds, axis=0)
    df_submission.to_csv('submission.csv', index=False)


# In[8]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras.backend as K
# import os
# from scipy.stats import spearmanr
# from math import floor, ceil
# from transformers import *

# PATH = '/content/drive/My Drive/NLPDATA/'
# BERT_PATH = '/kaggle/input/bert-base-uncased-huggingface-transformer/'
# MAX_LEN = 360

def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, length, is_pair = True):
        if is_pair:
            inputs = tokenizer.encode_plus(str1, str2,
                add_special_tokens=True,
                max_length=length,
                truncation_strategy="only_second")
        else:
            inputs = tokenizer.encode_plus(str1, None,
                add_special_tokens=True,
                max_length=length,
                truncation_strategy="only_first")
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        title, question, max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        answer, None, max_sequence_length, False)
    
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a =         _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]

def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def create_model():
    q_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config = XLNetConfig()
    config.d_inner = 3072
    config.n_head = 12
    config.d_model = 768
    config.n_layer = 12
    config.output_hidden_states = False
    
    bert_model = TFXLNetModel.from_pretrained(
        '/kaggle/input/xlnet-base-tf/xlnet-base-cased-tf_model.h5', config=config)
    
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn,], outputs=x)
    
    return model


# from datetime import datetime

def xlnet_cased():
    PATH = '/kaggle/input/google-quest-challenge/'

    BERT_PATH = '/kaggle/input/xlnet-base-tf/'
    # !cp /kaggle/input/bert-base-uncased-huggingface-transformer/bert-base-uncased-vocab.txt ./vocab.txt
#     tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    tokenizer = XLNetTokenizer("/kaggle/input/xlnet-base-tf/xlnet-base-cased-spiece.model")

    df_train = pd.read_csv(PATH+'train.csv')
    df_test = pd.read_csv(PATH+'test.csv')
    df_submission = pd.read_csv(PATH+'sample_submission.csv')

    output_categories = list(df_submission.columns)
    input_categories = ["question_title", "question_body", "answer"]
    
    outputs = df_train[output_categories[1:]].values
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_LEN)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_LEN)
    
    test_preds = []
    train_inputs = [inputs[i] for i in range(len(inputs))]
    train_outputs = outputs

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.fit(train_inputs, train_outputs, epochs=5, batch_size=6, validation_split=0.1)

    test_preds.append(model.predict(test_inputs))
    df_submission.iloc[:, 1:] = np.average(test_preds, axis=0)
    df_submission.to_csv('submission.csv', index=False)
    
# xlnet_cased()


# In[ ]:




