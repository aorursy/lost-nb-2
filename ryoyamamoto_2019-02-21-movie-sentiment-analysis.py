#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)


# In[2]:


train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")


# In[3]:


train.head(10)


# In[4]:


train.loc[train.SentenceId == 10]


# In[5]:


# Average count of phrases per sentence in train is:
train.groupby('SentenceId')['Phrase'].count().mean()


# In[6]:


# Average count of phrases per sentence in test is:
test.groupby('SentenceId')['Phrase'].count().mean()


# In[7]:


# Number of phrases in train:
train.shape[0]


# In[8]:


# Number of sentences in train:
len(train.SentenceId.unique())


# In[9]:


# Number of phrases in test:
test.shape[0]


# In[10]:


# Number of sentences in test:
len(test.SentenceId.unique())


# In[11]:


# Average word length of phrases in train is:
train.Phrase.apply(lambda x: x.count(" ") + 1).mean()
# or train.Phrase.apply(lambda x: len(x.split())).mean()


# In[12]:


# Average word length of phrases in test is:
test.Phrase.apply(lambda x : len(x.split())).mean()


# In[13]:


text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
text


# In[14]:


ngrams(text.split(), 3)


# In[15]:


list(ngrams(text.split(), 3))[:5]


# In[16]:


text_trigrams = list(ngrams(text.split(), 3))
text_trigrams


# In[17]:


Counter(text_trigrams)


# In[18]:


Counter(text_trigrams).most_common(10)


# In[19]:


text_ = [i for i in text.split() if i not in stopwords.words('english')]
text_trigrams = list(ngrams(text_, 3))
Counter(text_trigrams).most_common(10)


# In[20]:


stopwords.words('english')


# In[21]:


tokenizer = TweetTokenizer()


# In[22]:


tokenizer


# In[23]:


tokenizer.tokenize


# In[24]:


tokenizer.tokenize('Hello world, Mr. Smith! I am new to this field.')


# In[25]:


vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
vectorizer


# In[26]:


full_text = list(train.Phrase.values) + list(test.Phrase.values)
full_text

# QUESTION: Is it OK to tokenize phrases in test data?
# Preprocessing is outside the training?


# In[27]:


vectorizer.fit(full_text)


# In[28]:


train_vectorized = vectorizer.transform(train.Phrase)
test_vectorized = vectorizer.transform(test.Phrase)


# In[29]:


y = train.Sentiment


# In[30]:


logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg) # LEARN


# In[31]:


get_ipython().run_cell_magic('time', '', 'ovr.fit(train_vectorized, y)')


# In[32]:


scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
scores


# In[33]:


# Cross-validation mean accuracy
scores.mean(), scores.std()


# In[34]:


print('accuracy: {0:.2f}%, std: {1:.2f}%pt'.format(scores.mean() * 100, scores.std() * 100))


# In[35]:


get_ipython().run_cell_magic('time', '', "svc = LinearSVC(dual=False)\nsvc.fit(train_vectorized, y)\nscores = cross_val_score(svc, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)\nprint('CV mean accuracy: {0:.2f}%, std: {1:.2f}%pt'.format(scores.mean()*100, scores.std()*100))")


# In[36]:


# LEARN: toxic competition


# In[37]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


# In[38]:


tk = Tokenizer(lower=True, filters='')
tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train.Phrase)
test_tokenized = tk.texts_to_sequences(test.Phrase)

# NOTE: tokenize is to convert each word into integer code (frequent words only)


# In[39]:


tk


# In[40]:


full_text[:10]


# In[41]:


len(train.Phrase)


# In[42]:


len(train_tokenized), train_tokenized
# NOTE: number of train_tokenized's rows equals to that of train.Phase


# In[43]:


max_len = 50
X_train = pad_sequences(train_tokenized, maxlen=max_len) # NOTE: justified each row to right, padding zeros on the left
X_test = pad_sequences(test_tokenized, maxlen=max_len)


# In[44]:


X_train[:10]


# In[45]:


embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

# NOTE: serialized matrix of shape (2M rows, 300 dimention embeddings), 
# each row has 300-dimentional embeddings for the preceding word

# LEARN: FastText, from where this embeddings data comes


# In[46]:


embed_size = 300
max_features = 3e4


# In[47]:


list(o for o in open(embedding_path))


# In[48]:


list(o.split()[0] for o in open(embedding_path))


# In[49]:


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
# NOTE: 'word, *arr' corresponds to each text row in the embedding file
# this function converts only the array part into numpy array
# in preparation for creating dictionary


# In[50]:


get_ipython().run_cell_magic('time', '', 'embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))')


# In[51]:


embedding_index # array of embeddings for words included in FastText file (not yet adjusted for the dataset)


# In[52]:


word_index = tk.word_index
word_index


# In[53]:


nb_words = min(max_features, len(word_index)) # NOTE: 'nb' means 'number'
embedding_matrix = np.zeros((nb_words + 1, embed_size)) 
embedding_matrix


# In[54]:


for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[55]:


embedding_matrix # array of embeddings for most frequent words in full_text


# In[56]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1)) 
# NOTE: fit/transform done together for one-hot encoding


# In[57]:


get_ipython().run_line_magic('pinfo', 'np.reshape')


# In[58]:


get_ipython().run_line_magic('pinfo', 'y.values.reshape')


# In[59]:


y.values.reshape(-1, 1)


# In[60]:


y_ohe # one-hot encoded version of labels


# In[61]:


def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, 
                 kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
    file_path = 'best_model.hdf5'
    check_point = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    
    inp = Input(shape=(max_len, )) 
    
    
    x = Embedding(19479, embed_size, weights=[embedding_matrix], trainable=False)(inp) # QUESTION: what is 19479 ??
    x1 = SpatialDropout1D(spatial_dr)(x)
    
    x_gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)    
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)    
    avg_pool3_gru = GlobalAveragePooling1D()(x1)
    max_pool3_gru = GlobalMaxPooling1D()(x1)
    
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)
    
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)    
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)    
    avg_pool3_lstm = GlobalAveragePooling1D()(x1)
    max_pool3_lstm = GlobalMaxPooling1D()(x1)
    
    
    x = concatenate([
        avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
        avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm,
    ])
    x = BatchNormalization()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dr)(x)
    x = BatchNormalization()(x)
    x = Dense(int(dense_units / 2), activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(5, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=lr, decay=lr_d), 
        metrics=['accuracy']
    )
    history = model.fit(X_train, y_ohe, 
        batch_size=128, epochs=10, validation_split=0.1, verbose=1, 
        callbacks=[check_point, early_stop]
    ) # execute fitting
    model = load_model(file_path) # load best model obtained
    return model


# In[62]:


model1 = build_model1(lr=1e-3, lr_d=1e-10, units=64, spatial_dr=0.3, 
                      kernel_size1=3, kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)


# In[63]:


pred1 = model1.predict(X_test, batch_size = 1024, verbose = 1)
pred1


# In[64]:


pred = pred1


# In[65]:


predictions = np.round(np.argmax(pred, axis=1)).astype(int)
predictions


# In[66]:


sub['Sentiment'] = predictions
sub.to_csv("blend.csv", index=False)

