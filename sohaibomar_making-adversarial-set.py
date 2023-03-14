#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np
import pandas as pd
trn = pd.read_csv("../input/train.csv", encoding="utf-8")
sub = pd.read_csv("../input/test.csv", encoding="utf-8")


# In[107]:


#assign target if set is test set or not
trn['is_test'] = 0
sub['is_test'] = 1


# In[108]:


orginal_train = trn.copy()


# In[109]:


train = pd.concat([trn, sub], axis=0)


# In[110]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer\nimport regex\nvectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    tokenizer=lambda x: regex.findall(r'[^\\p{P}\\W]+', x),\n    analyzer='word',\n    token_pattern=None,\n    stop_words='english',\n    ngram_range=(1, 1), \n    max_features=20000\n)\ntrn_idf = vectorizer.fit_transform(trn.comment_text)\ntrn_vocab = vectorizer.vocabulary_\nsub_idf = vectorizer.fit_transform(sub.comment_text)\nsub_vocab = vectorizer.vocabulary_\nall_idf = vectorizer.fit_transform(train.comment_text.values)\nall_vocab = vectorizer.vocabulary_")


# In[111]:


trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]


# In[112]:


common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# In[113]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\n#predictions to save each fold predictions results\npredictions = np.zeros(train.shape[0])\n\n# Create target where all train samples are ones and all test samples are zeros\ntarget = train.is_test.values\n# Shuffle samples to mix zeros and ones\nidx = np.arange(all_idf.shape[0])\nnp.random.seed(1)\nnp.random.shuffle(idx)\nall_idf = all_idf[idx]\ntarget = target[idx]\n# Train a Logistic Regression\nfolds = StratifiedKFold(5, True, 1)\nfor trn_idx, val_idx in folds.split(all_idf, target):\n    lr = LogisticRegression()\n    lr.fit(all_idf[trn_idx], target[trn_idx])\n    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))\n    predictions[val_idx] = lr.predict_proba(all_idf[val_idx])[:, 1]')


# In[115]:



#seperate train rows which have been misclassified as test and use them as validation
train["predictions"] = predictions
predictions_argsort = predictions.argsort()
train_sorted = train.iloc[predictions_argsort]

#select only trains set because we need to find train rows which have been misclassified as test set and use them for validation
train_sorted = train_sorted.loc[train_sorted.is_test == 0]

#Why did I chose 0.7 as thereshold? just a hunch, but you should try different thresholds i.e 0.6, 0.8 and see the difference in validation score and please report back. :) 
train_as_test = train_sorted.loc[train_sorted.predictions > 0.7]
#save the indices of the misclassified train rows to use as validation set
adversarial_set_ids = train_as_test.index.values
adversarial_set = pd.DataFrame(adversarial_set_ids, columns=['adversial_set_ids'])
#save adversarial set index
adversarial_set.to_csv('adversarial_set_ids.csv', index=False)


# In[ ]:




