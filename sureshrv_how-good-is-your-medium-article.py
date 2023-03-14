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


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error


# In[3]:


from nltk.corpus import stopwords
import gensim
from scipy import sparse
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import seaborn as sns
import pyLDAvis.gensim


# In[4]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# In[5]:


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# In[6]:


def preprocess(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm_notebook(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            output_list.append(content_no_html_tags)
    return output_list


# In[7]:


get_ipython().run_cell_magic('time', '', "train_raw_content = preprocess(path_to_inp_json_file=os.path.join('../input', \n                                                                  'train.json'),)")


# In[8]:


train_raw_content[0]


# In[9]:


get_ipython().run_cell_magic('time', '', "test_raw_content = preprocess(path_to_inp_json_file=os.path.join('../input', \n                                                                  'test.json'),)")


# In[10]:


# custom stop_words for job search 
# custom_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
#                      'aa','aaa','bb','bbb','c','ccc','d','ddd','e','eee','f','fff','ummm','hmmm','xiii','xxiii','http','https','sooo','orc','mmm',
#                      'ets'
           
#                     ]


# In[11]:


# from nltk import word_tokenize          
# from nltk.stem import WordNetLemmatizer 
# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#     def __call__(self, articles):
#         return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# In[12]:


# cv = CountVectorizer(max_features=80000,stop_words=custom_stop_words, analyzer='word',min_df=1,max_df=0.7,token_pattern=r'\b[a-zA-Z]{3,}\b'
#                      )


# In[13]:


# %%time
# X_train = cv.fit_transform(train_raw_content)


# In[14]:


# cv.vocabulary_.items()= 


# In[15]:


#cv.get_feature_names()


# In[16]:


# %%time
# X_test = cv.transform(test_raw_content)


# In[17]:


# print(X_train[2,:])


# In[18]:


# X_train.shape, X_test.shape


# In[19]:


# train_target = pd.read_csv(os.path.join('../input', 'train_log1p_recommends.csv'), 
#                            index_col='id')


# In[20]:


# train_target.shape


# In[21]:


# y_train = train_target['log_recommends'].values


# In[22]:


# train_part_size = int(0.7 * train_target.shape[0])
# X_train_part = X_train[:train_part_size, :]
# y_train_part = y_train[:train_part_size]
# X_valid =  X_train[train_part_size:, :]
# y_valid = y_train[train_part_size:]


# In[23]:


# from sklearn.linear_model import Ridge


# In[24]:


# ridge = Ridge(random_state=17,alpha=0.21)


# In[25]:


# %%time
# ridge.fit(X_train_part, y_train_part);


# In[26]:


# ridge_pred = ridge.predict(X_valid)


# In[27]:


# plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
# plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
# plt.legend();


# In[28]:


# valid_mae = mean_absolute_error(y_valid, ridge_pred)
# valid_mae, np.expm1(valid_mae)


# In[29]:


# %%time
# ridge.fit(X_train, y_train);


# In[30]:


# %%time
# ridge_test_pred = ridge.predict(X_test)


# In[31]:


# def write_submission_file(prediction, filename,
#     path_to_sample=os.path.join('../input', 'sample_submission.csv')):
#     submission = pd.read_csv(path_to_sample, index_col='id')
    
#     submission['log_recommends'] = prediction
#     submission.to_csv(filename)


# In[32]:


# write_submission_file(prediction=ridge_test_pred, 
#                       filename='first_ridge4.csv')


# In[33]:


# full_sparse_data =  sparse.vstack([X_train, X_test])


# In[34]:


# #Transform our sparse_data to corpus for gensim
# corpus_data_gensim = gensim.matutils.Sparse2Corpus(full_sparse_data, documents_columns=False)


# In[35]:


# #Create dictionary for LDA model
# vocabulary_gensim = {}
# for key, val in cv.vocabulary_.items():
#     vocabulary_gensim[val] = key
    
# dict = Dictionary()
# dict.merge_with(vocabulary_gensim)


# In[36]:


# lda = LdaModel(corpus_data_gensim, num_topics = 30 )


# In[37]:


# data_ =  pyLDAvis.gensim.prepare(lda, corpus_data_gensim, dict)


# In[38]:


# pyLDAvis.display(data_)


# In[39]:


# def document_to_lda_features(lda_model, document):
#     topic_importances = lda.get_document_topics(document, minimum_probability=0)
#     topic_importances = np.array(topic_importances)
#     return topic_importances[:,1]

# lda_features = list(map(lambda doc:document_to_lda_features(lda, doc),corpus_data_gensim))


# In[40]:


# data_pd_lda_features = pd.DataFrame(lda_features)
# data_pd_lda_features.head()


# In[41]:


# data_pd_lda_features_train = data_pd_lda_features.iloc[:y_train.shape[0]]
# data_pd_lda_features_train['target'] = y_train

# fig, ax = plt.subplots()
# # the size of A4 paper
# fig.set_size_inches(20.7, 8.27)
# sns.heatmap(data_pd_lda_features_train.corr(method = 'spearman'), cmap="RdYlGn", ax = ax)


# In[42]:


# X_tr = sparse.hstack([X_train, data_pd_lda_features_train.drop('target', axis = 1)]).tocsr()


# In[43]:


# X_test1 = sparse.hstack([X_test, data_pd_lda_features.iloc[y_train.shape[0]:]]).tocsr()


# In[44]:


# ridge = Ridge(random_state=17)
# ridge.fit(X_tr,y_train)


# In[45]:


# ridge_test_pred1 = ridge.predict(X_test1)


# In[46]:


# plt.hist(ridge_test_pred1, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
# plt.legend();


# In[47]:


# write_submission_file(prediction=ridge_test_pred1, 
#                       filename='first_ridge_lda.csv')


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[49]:


# train_raw_content.


# In[50]:


max_fatures = 30000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_raw_content)
X1 = tokenizer.texts_to_sequences(train_raw_content)


# In[51]:


tokenizer.fit_on_texts(test_raw_content)
X2_test = tokenizer.texts_to_sequences(test_raw_content)


# In[52]:


X1 = pad_sequences(X1,maxlen=900)


# In[53]:


X2_test = pad_sequences(X2_test,maxlen=900)


# In[54]:


train_target = pd.read_csv(os.path.join('../input', 'train_log1p_recommends.csv'), 
                           index_col='id')
Y1 = train_target['log_recommends'].values


# In[55]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)


# In[56]:


embed_dim = 150
lstm_out = 200
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(1,kernel_initializer='normal'))
model.compile(loss = 'mean_squared_error', optimizer='adam')
print(model.summary())


# In[57]:


batch_size = 80
model.fit(X1_train, Y1_train, nb_epoch = 5, batch_size=batch_size, verbose = 2)


# In[58]:


lstm_test_pred = model.predict(X2_test)


# In[59]:


def write_submission_file(prediction, filename,
   path_to_sample=os.path.join('../input', 'sample_submission.csv')):
   submission = pd.read_csv(path_to_sample, index_col='id')
   
   submission['log_recommends'] = prediction
   submission.to_csv(filename)


# In[60]:


lstm_test_pred.shape


# In[61]:


write_submission_file(prediction=lstm_test_pred, 
                      filename='first_lstm.csv')


# In[62]:




