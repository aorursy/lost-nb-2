#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier


# In[2]:


trn = pd.read_csv("../input/train.csv")
target = trn.target
del trn["target"]


# In[3]:


clf1 = LGBMClassifier(n_estimators=100, n_jobs=2)
clf2 = LGBMClassifier(n_estimators=100, reg_alpha=1, reg_lambda=1, min_split_gain=2, n_jobs=2)


# In[4]:


# Choose seeds for each 2-fold iterations
seeds = [13, 51, 137, 24659, 347]
# Initialize the score difference for the 1st fold of the 1st iteration 
p_1_1 = 0.0
# Initialize a place holder for the variance estimate
s_sqr = 0.0
# Initialize scores list for both classifiers
scores_1 = []
scores_2 = []
diff_scores = []
# Iterate through 5 2-fold CV
for i_s, seed in enumerate(seeds):
    # Split the dataset in 2 parts with the current seed
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    # Initialize score differences
    p_i = np.zeros(2)
    # Go through the current 2 fold
    for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        # Split the data
        trn_x, trn_y = trn.iloc[trn_idx], target.iloc[trn_idx]
        val_x, val_y = trn.iloc[val_idx], target.iloc[val_idx]
        # Train classifiers
        clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
        clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
        # Compute scores
        preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)[:, 1]
        score_1 = roc_auc_score(val_y, preds_1)
        preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)[:, 1]
        score_2 = roc_auc_score(val_y, preds_2)
        # keep score history for mean and stdev calculation
        scores_1.append(score_1)
        scores_2.append(score_2)
        diff_scores.append(score_1 - score_2)
        print("Fold %2d score difference = %.6f" % (i_f + 1, score_1 - score_2))
        # Compute score difference for current fold  
        p_i[i_f] = score_1 - score_2
        # Keep the score difference of the 1st iteration and 1st fold
        if (i_s == 0) & (i_f == 0):
            p_1_1 = p_i[i_f]
    # Compute mean of scores difference for the current 2-fold CV
    p_i_bar = (p_i[0] + p_i[1]) / 2
    # Compute the variance estimate for the current 2-fold CV
    s_i_sqr = (p_i[0] - p_i_bar) ** 2 + (p_i[1] - p_i_bar) ** 2 
    # Add up to the overall variance
    s_sqr += s_i_sqr
    
# Compute t value as the first difference divided by the square root of variance estimate
t_bar = p_1_1 / ((s_sqr / 5) ** .5) 

print("Classifier 1 mean score and stdev : %.6f + %.6f" % (np.mean(scores_1), np.std(scores_1)))
print("Classifier 2 mean score and stdev : %.6f + %.6f" % (np.mean(scores_2), np.std(scores_2)))
print("Score difference mean + stdev : %.6f + %.6f" 
      % (np.mean(diff_scores), np.std(diff_scores)))


# In[5]:


"t_value for the current test is %.6f" % t_bar


# In[6]:


n_splits = 10 
scores_1 = []
scores_2 = []
oof_1 = np.zeros(len(trn))
oof_2 = np.zeros(len(trn))
diff_scores = []
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)
p_i = np.zeros(2)
for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_x, trn_y = trn.iloc[trn_idx], target.iloc[trn_idx]
    val_x, val_y = trn.iloc[val_idx], target.iloc[val_idx]
    # Train classifiers
    clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    # Compute scores
    preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)[:, 1]
    oof_1[val_idx] = preds_1
    score_1 = roc_auc_score(val_y, preds_1)
    preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)[:, 1]
    score_2 = roc_auc_score(val_y, preds_2)
    oof_2[val_idx] = preds_2
    # keep score history for mean and stdev calculation
    scores_1.append(score_1)
    scores_2.append(score_2)
    diff_scores.append(score_1 - score_2)
    print("Fold %2d score difference = %.6f" % (i_f + 1, diff_scores[i_f]))
# Compute t value
centered_diff = np.array(diff_scores) - np.mean(diff_scores)
t = np.mean(diff_scores) * (n_splits ** .5) / (np.sqrt(np.sum(centered_diff ** 2) / (n_splits - 1)))
print("OOF score for classifier 1 : %.6f" % roc_auc_score(target, oof_1))
print("OOF score for classifier 2 : %.6f" % roc_auc_score(target, oof_2))
print("t statistic for %2d-fold CV = %.6f" % (n_splits, t))


# In[7]:




