#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd

from catboost import CatBoostRegressor, Pool
from catboost.eval.catboost_evaluation import *

from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=False)

import os
print(os.listdir("../input"))
print(os.listdir("../input/LANL-Earthquake-Prediction"))
print(os.listdir("../input/lanl-features"))


# In[2]:


X = pd.read_csv('../input/lanl-features/train_features.csv')
X_test = pd.read_csv('../input/lanl-features/test_features.csv')
y = pd.read_csv('../input/lanl-features/y.csv')
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')


# In[3]:


df_train = y.join(X)
df_train.to_csv('train.csv', header=False, index=False)
df_train


# In[4]:


learn_params = {'iterations': 10, 
                'random_seed': 0, 
                'logging_level': 'Silent',
                'loss_function': 'MAE',
                # You could set learning process to GPU
                #'devices': '1',  
                'task_type': 'GPU',                
                'boosting_type': 'Ordered', 
                # For feature evaluation learning time is important and we need just the relative quality
                'max_ctr_complexity' : 4}


# In[5]:


features_to_evaluate = [i for i in range(50)]
features_to_evaluate


# In[6]:


from catboost.utils import create_cd

feature_names = dict()
for column, name in enumerate(df_train):
    if column == 0:
        continue
    feature_names[column - 1] = name
    
create_cd(
    label=0, 
    cat_features=[477, 804, 981],
    feature_names=feature_names,
    output_path='train.cd'
)
get_ipython().system("cat 'train.cd'")


# In[7]:


fold_size = X.shape[0]//2
fold_offset = 0
folds_count = 5
random_seed = 1

evaluator = CatboostEvaluation('train.csv',
                               fold_size,
                               folds_count,
                               delimiter=',',
                               column_description='train.cd',
                               partition_random_seed=random_seed,
                               #working_dir=...  — working directory, we will need to create temp files during evaluation, 
                               #so ensure you have enough free space. 
                               #By default we will create unique temp dir in system temp directory
                               #group_column=... — set it if you have column which should be used to split 
)


# In[8]:


get_ipython().run_cell_magic('time', '', 'result = evaluator.eval_features(learn_config=learn_params,\n                                 eval_metrics=["MAE"],\n                                 features_to_eval=features_to_evaluate)')


# In[9]:


MAE_result = result.get_metric_results("MAE")


# In[10]:


#MAE_result.get_baseline_comparison()
MAE_result.


# In[11]:


iplot(MAE_result.create_fold_learning_curves(0))


# In[12]:


baseline_case = MAE_result.get_baseline_case()


# In[13]:


baseline_case


# In[14]:


baseline_result = MAE_result.get_case_result(baseline_case)


# In[15]:


iplot(baseline_result.create_learning_curves_plot())


# In[16]:


learning_rate_params = learn_params


# In[17]:


baseline_case = ExecutionCase(label="Step {}".format(0.03),
                              params=learning_rate_params, 
                              learning_rate=0.03)


# In[18]:


other_learning_rate_cases = [ExecutionCase(label="Step {}".format(step), 
                                           params=learning_rate_params, 
                                           learning_rate=step) for step in [0.05, 0.015]]


# In[19]:


evaluator = CatboostEvaluation('train.csv',
                               fold_size, 
                               fold_count=1,  #For learning rate estimation we need just 1 fold
                               delimiter=',',
                               column_description='train.cd',
                               partition_random_seed=random_seed)


# In[20]:


evaluator.get_working_dir()


# In[21]:


learning_rates_result = evaluator.eval_cases(baseline_case, 
                                             other_learning_rate_cases,
                                             eval_metrics="MAE")


# In[22]:


MAE_learning_rate_search_results = learning_rates_result.get_metric_results("MAE")


# In[23]:


tmp = MAE_learning_rate_search_results.create_fold_learning_curves(fold=0, offset=200)


# In[24]:


iplot(tmp)


# In[25]:




