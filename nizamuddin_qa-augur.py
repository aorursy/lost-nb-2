#!/usr/bin/env python
# coding: utf-8

# In[1]:


#most of the credits goes to: https://www.kaggle.com/mobassir
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import os
import warnings
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tqdm.notebook import tqdm
import transformers
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from gensim import corpora, models
import pyLDAvis#for interactive topic model visualization
import pyLDAvis.gensim
from keras.preprocessing.text import Tokenizer

pyLDAvis.enable_notebook()
np.random.seed(2018)
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import spearmanr
from math import floor, ceil
from tensorflow.keras.models import load_model
import math

import re

import pickle  
import random
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import glob
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet
import torch
np.set_printoptions(suppress=True)


# In[2]:


input_columns = ['question_title', 'question_body', 'answer']
targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]


# In[3]:


train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv',index_col='qa_id')
test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv',index_col='qa_id')
submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
train.shape,test.shape,submission.shape


# In[4]:


# a=train.groupby(by=['category'])
# train=pd.concat([train.iloc[a.indices['STACKOVERFLOW']],train.iloc[a.indices['TECHNOLOGY']],train.iloc[a.indices['SCIENCE']],train.iloc[a.indices['LIFE_ARTS']],train.iloc[a.indices['CULTURE']]])
# b=test.groupby(by=['category'])
# test=pd.concat([test.iloc[b.indices['STACKOVERFLOW']],test.iloc[b.indices['TECHNOLOGY']],test.iloc[b.indices['SCIENCE']],test.iloc[b.indices['LIFE_ARTS']],test.iloc[b.indices['CULTURE']]])


# In[5]:


# train.shape,test.shape


# In[6]:


# for colname in tqdm(input_columns):
#     preprocess(colname)


# In[7]:


# from sklearn.model_selection import train_test_split
# train, _ = train_test_split(train, test_size=0.3,random_state=42,shuffle=True)
# train.shape,test.shape,submission.shape


# In[8]:


##checking the distributions of targets(all 30)
# import matplotlib.pyplot as plt
# %matplotlib inline
# for col in targets:
#     plt.hist(train[col])
#     plt.title(col)
#     plt.show()
#plt.hist(train['question_body_critical'])


# In[9]:


# %%time
# import torch
# import sys
# def fetch_vectors(string_list, batch_size=64):
#     # credits: https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
#     DEVICE = torch.device("cuda")
#     tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
#     model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
#     model.to(DEVICE)
#     fin_features = []
#     for data in chunks(string_list, batch_size):
#         tokenized = []
#         for x in data:
#             x = " ".join(x.strip().split()[:300])
#             tok = tokenizer.encode(x, add_special_tokens=True)
#             tokenized.append(tok[:512])

#         max_len = 512
#         padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
#         attention_mask = np.where(padded != 0, 1, 0)
#         input_ids = torch.tensor(padded).to(DEVICE)
#         attention_mask = torch.tensor(attention_mask).to(DEVICE)

#         with torch.no_grad():
#             last_hidden_states = model(input_ids, attention_mask=attention_mask)

#         features = last_hidden_states[0][:, 0, :].cpu().numpy()
#         fin_features.append(features)

#     fin_features = np.vstack(fin_features)
#     return fin_features

# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]
# sys.path.insert(0, "../input/transformers/transformers-master/")
# train_question_body_dense = fetch_vectors(train.question_body.values)
# train_answer_dense = fetch_vectors(train.answer.values)
# test_question_body_dense = fetch_vectors(test.question_body.values)
# test_answer_dense = fetch_vectors(test.answer.values)


# In[10]:


get_ipython().run_cell_magic('time', '', 'import torch\nimport sys\n\ndef fetch_vectors(string_list, batch_size=64):\n    # credits: https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/\n    DEVICE = torch.device("cuda")\n    tokenizer = transformers.BertTokenizer.from_pretrained("../input/bertbaseuncased/bert-base-uncased/")\n    model = transformers.BertModel.from_pretrained("../input/bertbaseuncased/bert-base-uncased/")\n    model.to(DEVICE)\n    fin_features = []\n    for data in chunks(string_list, batch_size):\n        tokenized = []\n        for x in data:\n            x = " ".join(x.strip().split()[:300])\n            tok = tokenizer.encode(x, add_special_tokens=True)\n            tokenized.append(tok[:512])\n\n        max_len = 512\n        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])\n        attention_mask = np.where(padded != 0, 1, 0)\n        input_ids = torch.tensor(padded).to(DEVICE)\n        attention_mask = torch.tensor(attention_mask).to(DEVICE)\n\n        with torch.no_grad():\n            last_hidden_states = model(input_ids, attention_mask=attention_mask)\n\n        features = last_hidden_states[0][:, 0, :].cpu().numpy()\n        fin_features.append(features)\n\n    fin_features = np.vstack(fin_features)\n    return fin_features\n\ndef chunks(l, n):\n    """Yield successive n-sized chunks from l."""\n    for i in range(0, len(l), n):\n        yield l[i:i + n]\nsys.path.insert(0, "../input/transformers/transformers-master/")\ntrain_question_body_dense = fetch_vectors(train.question_body.values)\ntrain_answer_dense = fetch_vectors(train.answer.values)\ntest_question_body_dense = fetch_vectors(test.question_body.values)\ntest_answer_dense = fetch_vectors(test.answer.values)')


# In[11]:


get_ipython().run_cell_magic('time', '', 'import re\nfrom urllib.parse import urlparse\nfrom sklearn.preprocessing import OneHotEncoder\nfind = re.compile(r"^[^.]*")\n\ntrain[\'netloc\'] = train[\'url\'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])\ntest[\'netloc\'] = test[\'url\'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])\n\nfeatures = [\'netloc\', \'category\']\nmerged = pd.concat([train[features], test[features]])\nohe = OneHotEncoder()\nohe.fit(merged)\n\nfeatures_train = ohe.transform(train[features]).toarray()\nfeatures_test = ohe.transform(test[features]).toarray()')


# In[12]:


# train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv',index_col='qa_id')
# test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv',index_col='qa_id')
# submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
# train.shape,test.shape,submission.shape


# In[13]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nimport tensorflow_hub as hub\nimport sys\nsys.path.insert(0, "/kaggle/input/tftext/tensorflow_text/")\nembed = hub.load("/kaggle/input/useqa3/USEQA3/")')


# In[14]:


get_ipython().run_cell_magic('time', '', 'embeddings_train = {}#sentence_embeddings\nembeddings_test = {}#sentence_embeddings\nprint("preparing embeddings for train data....")\ntrain[\'question_title\']=train[\'question_title\'].apply(lambda x:x.strip(\'\\n\'))\ntrain[\'question_body\']=train[\'question_body\'].apply(lambda x:x.strip(\'\\n\'))\ntrain[\'answer\']=train[\'answer\'].apply(lambda x:x.strip(\'\\n\'))\ntrain_ans_emb = []\ntrain_questitle_emb = []\ntrain_quesbody_emb = []\nfor q_t,q_b,a in tqdm(zip(list(train[\'question_title\']),list(train[\'question_body\']),list(train[\'answer\']))):\n    question_title=[q_t]\n    question_body=[q_b]\n    responses=[a]\n    response_contexts = responses\n    question_title_embeddings = embed.signatures[\'question_encoder\'](tf.constant(question_title))[\'outputs\']\n    question_body_embeddings = embed.signatures[\'question_encoder\'](tf.constant(question_body))[\'outputs\']\n    response_embeddings = embed.signatures[\'response_encoder\'](input=tf.constant(responses),context=tf.constant(response_contexts))[\'outputs\']\n    train_ans_emb.append(response_embeddings.numpy())\n    train_questitle_emb.append(question_title_embeddings.numpy())\n    train_quesbody_emb.append(question_body_embeddings.numpy())\n#Stacking the sentence embeddings for all len(train)\nembeddings_train[\'answer_embedding\'] = np.vstack(train_ans_emb)\nembeddings_train[\'question_body_embedding\'] = np.vstack(train_quesbody_emb)\nembeddings_train[\'question_title_embedding\'] = np.vstack(train_questitle_emb)\n\nprint("preparing embeddings for test data....")\ntest[\'question_title\']=test[\'question_title\'].apply(lambda x:x.strip(\'\\n\'))\ntest[\'question_body\']=test[\'question_body\'].apply(lambda x:x.strip(\'\\n\'))\ntest[\'answer\']=test[\'answer\'].apply(lambda x:x.strip(\'\\n\'))\ntest_ans_emb = []\ntest_questitle_emb = []\ntest_quesbody_emb = []\nfor q_t,q_b,a in tqdm(zip(list(test[\'question_title\']),list(test[\'question_body\']),list(test[\'answer\']))):\n    question_title=[q_t]\n    question_body=[q_b]\n    responses=[a]\n    response_contexts = responses\n    question_title_embeddings = embed.signatures[\'question_encoder\'](tf.constant(question_title))[\'outputs\']\n    question_body_embeddings = embed.signatures[\'question_encoder\'](tf.constant(question_body))[\'outputs\']\n    response_embeddings = embed.signatures[\'response_encoder\'](input=tf.constant(responses),context=tf.constant(response_contexts))[\'outputs\']\n    test_ans_emb.append(response_embeddings.numpy())\n    test_questitle_emb.append(question_title_embeddings.numpy())\n    test_quesbody_emb.append(question_body_embeddings.numpy())\n#Stacking the sentence embeddings for all len(test)\nembeddings_test[\'answer_embedding\']  = np.vstack(test_ans_emb)\nembeddings_test[\'question_body_embedding\'] = np.vstack(test_quesbody_emb)\nembeddings_test[\'question_title_embedding\'] = np.vstack(test_questitle_emb)\n\n\ndel embed\nK.clear_session()\ngc.collect()')


# In[15]:


# %%time
# import torch
# model = torch.load("../input/mt-dnn-largept/mt_dnn_large.pt")


# In[16]:


# %%time
# module_url = "../input/universalsentenceencoderlarge4/"
# embed = hub.load(module_url)
# embeddings_train = {}#sentence_embeddings
# embeddings_test = {}#sentence_embeddings

# for text in tqdm(['question_title']):
#     print(text)
#     train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
#     test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()

#     curr_train_emb = []
#     curr_test_emb = []
#     batch_size = 4
#     ind = 0
#     while ind*batch_size < len(train_text):
#         curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
#         ind += 1
        
#     ind = 0
#     while ind*batch_size < len(test_text):
#         curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
#         ind += 1    
        
#     embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
#     embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)
    
# embeddings_train['question_body_embedding'] = train_quesbody_emb_stacked
# embeddings_train['answer_embedding']        = train_ans_emb_stacked
# embeddings_test['question_body_embedding']  = test_quesbody_emb_stacked
# embeddings_test['answer_embedding']         = test_ans_emb_stacked

# del embed
# K.clear_session()
# gc.collect()


# In[17]:


l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)
#embeddings_train is a sentence embedding dictionary
dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
]).T

dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
]).T
X_train = np.hstack([item for k, item in embeddings_train.items()] + [features_train,dist_features_train])
X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test,dist_features_test])
y_train = train[targets].values


# In[18]:


# #Universal sentence encoder embedding
try:
    print(embeddings_train['question_title_embedding'].shape,embeddings_train['question_body_embedding'].shape,embeddings_train['answer_embedding'].shape)
    print(embeddings_test['question_title_embedding'].shape,embeddings_test['question_body_embedding'].shape,embeddings_test['answer_embedding'].shape)
except:
    print("Error due to print statement")


# In[19]:


X_train = np.hstack((X_train, train_question_body_dense, train_answer_dense))
X_test = np.hstack((X_test, test_question_body_dense, test_answer_dense))
X_train.shape,X_test.shape,y_train.shape


# In[20]:


pd.DataFrame(X_train).to_csv('X_train_USEQA_BERTuncased.csv')
pd.DataFrame(y_train).to_csv('y_train_USEQA_BERTuncased.csv')
pd.DataFrame(X_test).to_csv('X_test_USEQA_BERTuncased.csv')


# In[21]:


class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
            #self.model.save_weights(self.model_name)
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
def create_model(n_dense1=256,dropout1=0.30,lr_rate=0.00003):
    model = Sequential()
    model.add(Dense(n_dense1, input_dim=X_train.shape[1], activation='elu'))
    model.add(Dropout(dropout1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_rate),loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'])
    model.summary()
    return model


# In[23]:


get_ipython().run_cell_magic('time', '', "all_predictions = []\nkf = KFold(n_splits=5, random_state=42, shuffle=True)\nfor ind, (tr, val) in enumerate(kf.split(X_train)):\n    X_tr = X_train[tr]\n    y_tr = y_train[tr]\n    X_vl = X_train[val]\n    y_vl = y_train[val]\n    model = create_model()\n    print( X_tr.shape,y_tr.shape,X_vl.shape,y_vl.shape)\n    model.fit(\n        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=False, \n        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),\n                                       patience=5, model_name=f'best_model_batch{ind}.h5')]\n    )\n    \n    all_predictions.append(model.predict(X_test))")


# In[24]:


model = create_model()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))


# In[25]:


get_ipython().run_cell_magic('time', '', 'kf = KFold(n_splits=5, random_state=2019, shuffle=True)\nfor ind, (tr, val) in enumerate(kf.split(X_train)):\n    X_tr = X_train[tr]\n    y_tr = y_train[tr]\n    X_vl = X_train[val]\n    y_vl = y_train[val]\n    model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)\n    model.fit(X_tr, y_tr)\n    all_predictions.append(model.predict(X_test))')


# In[26]:


for i in range(len(all_predictions)):
    print(i+1,":",all_predictions[i].shape)


# In[27]:


get_ipython().run_cell_magic('time', '', 'model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)\nmodel.fit(X_train, y_train)\nall_predictions.append(model.predict(X_test))\nlen(all_predictions)')


# In[28]:


get_ipython().run_cell_magic('time', '', 'test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)\nmax_val = test_preds.max() + 1\ntest_preds = test_preds/max_val + 1e-12')


# In[29]:


'''
Expansion of 
test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)

temp1=[]
for p in all_predictions:
    temp2=[]
    for c in p.T:
        temp2.append(rankdata(c))
    temp1.append(np.array(temp2).T)
test_preds2=np.array(temp1).mean(axis=0)
max_val2 = test_preds2.max() + 1
test_preds2 = test_preds2/max_val2 + 1e-12
'''


# In[30]:


import collections
import re
import unicodedata
import six
def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# In[31]:


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks, segments for BERT"""
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]#including special BERT tokens
    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)
    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[32]:


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        rho_val = compute_spearmanr(self.valid_outputs, np.average(self.valid_predictions, axis=0))
        print("\nvalidation rho: %.4f" % rho_val)
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        self.test_predictions.append(self.model.predict(self.test_inputs, batch_size=self.batch_size))


# In[33]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


# In[34]:


def train_bert_model(input_units,output_units):
    input_word_ids = tf.keras.layers.Input((input_units,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input((input_units,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input((input_units,), dtype=tf.int32, name='input_segments')
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(output_units, activation="sigmoid",name="dense_output")(x)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    return model    
        

def train_and_predict(model, train_data, valid_data, test_data, learning_rate, epochs, batch_size, loss_function, fold):  
    custom_callback = CustomCallback((valid_data[0], valid_data[1]),test_data,batch_size,fold=None)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss_function,optimizer)
    model.fit(train_data[0], train_data[1],epochs,batch_size, callbacks=[custom_callback])
    return custom_callback


# In[35]:


# BERT_PATH = '/kaggle/input/berthub'
# tokenizer = FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
# target_cols = list(submission.columns)
# design_cols = list(set(train.columns)-set(target_cols))#totally 10 in nos of which many are gratuitous.
# Y_cols = target_cols
# X_cols = ['question_title','question_body','answer']
# len(X_cols),len(Y_cols)


# In[36]:


# gkf = GroupKFold(n_splits=5).split(X=train.question_body, groups=train.question_body)
# bert_dimension=512
# #outputs = compute_output_arrays(train, Y_cols)
# #inputs = compute_input_arays(train, X_cols, tokenizer, bert_dimension)
# test_inputs = compute_input_arays(test, X_cols, tokenizer, bert_dimension)


# In[37]:


# %%time
# #Using pretrained models
# from tqdm import tqdm
# tqdm.pandas()
# models = []
# bert_dimension = 512#dimension of bert embedding
# for i in tqdm(range(5)):
#     weights_path = f'../input/bertuned-f{i}/bertuned_f{i}.h5'
#     model = train_bert_model(input_units=bert_dimension,output_units=len(Y_cols)-1)
#     model.load_weights(weights_path)
#     models.append(model)
# weights_path = f'../input/bertf1e15/Full-0.h5'
# model = train_bert_model(input_units=bert_dimension,output_units=len(Y_cols)-1)
# model.load_weights(weights_path)
# models.append(model)
# len(models)


# In[38]:


# %%time
test_predictions = []
# BATCH_SIZE_FOR_INFERENCE = 8#was 8 earlier
# from tqdm import tqdm
# tqdm.pandas()
# test_predictions = []
# for model in tqdm(models):
#     test_predictions.append(model.predict(test_inputs, batch_size=BATCH_SIZE_FOR_INFERENCE))
test_predictions.append(test_preds)#appending DistillBert,USEs,handfeatured
#print("test_predictions shape",test_predictions[i].shape)
final_predictions = np.mean(test_predictions, axis=0)
final_predictions.shape


# In[39]:


submission.iloc[:,1:] = final_predictions
submission.to_csv('submission.csv', index=False)
submission.head()


# In[40]:


# output1 = pd.read_csv("../input/output1/output1_USEDistBertoof.csv",index_col='qa_id')
# output1.drop(columns='Unnamed: 0',inplace=True)
# output2 = pd.read_csv("../input/2layerednn/2layeredNN.csv",index_col='qa_id')


# submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
# submission.iloc[:,1:] =(output2.to_numpy()+final_predictions)/2
# submission.head()
# submission.to_csv('submission.csv', index=False)


# In[41]:


# submission.iloc[:, 1:] = final_predictions
# if len(submission)==476:
#     submission.to_csv("submission.csv",index=False)
# else:
#     temp = submission[~submission.qa_id.isin(submission.qa_id.values)]
#     sub = pd.concat([submission,temp],ignore_index=True)
#     sub.to_csv("submission.csv",index=False)

