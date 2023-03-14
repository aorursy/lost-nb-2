#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

data_path = os.path.join('..', 'input')
trn = pd.read_csv(os.path.join(data_path, 'train.csv'), low_memory=True)
sub = pd.read_csv(os.path.join(data_path, 'test.csv'), low_memory=True)

# Preprocess data
trn['comment_text'] = trn.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)
sub['comment_text'] = sub.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import regex
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    max_features=20000
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
all_vocab = vectorizer.vocabulary_

trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]

common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(np.round(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]), 3))

