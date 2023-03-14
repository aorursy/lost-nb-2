#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '-f')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
dirToInclude=parentdir +'/features/'
sys.path.insert(0,dirToInclude)

import IeegConsts
from IeegConsts import *
from IeegFeatures import *

import pandas
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 100)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 6)
    
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train_dir=TRAIN_DATA_FOLDER_IN_ALL
test_dir=TEST_DATA_FOLDER_IN_ALL    


ieegFeatures= IeegFeatures(train_dir, True)
df_cols_train=ieegFeatures.ieegGenCols()
print len(df_cols_train)
# F_NAME_TRAIN= TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL +'-feat_TRAIN_df.hdf'
# X_df_train=pandas.read_hdf(F_NAME_TRAIN, engine='python')

X_df_train= pd.read_hdf(TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL 
                  + 'X_df_train.hdf', 'data',format='fixed',complib='blosc',complevel=9)

# X_df_train.drop('Unnamed: 0', axis=1, inplace=True)


n=16
last_cols=list()
for i in range(1, n_psd + 1):
    last_cols.append('psd_{}'.format(i))    
for i in range(1, 16 + 1):
    last_cols.append('var_{}'.format(i))    
for i in range(1, 16 + 1):
    last_cols.append('kurt_{}'.format(i))
for i in range(1, n_corr_coeff + 1):
    last_cols.append('corcoef_{}'.format(i))
for i in range(1, n + 1):
    last_cols.append('hurst_{}'.format(i))
# for i in range(1,  n_plv+ 1):
#     last_cols.append('plv_{}'.format(i))    
# for i in range(1, n + 1):
#     last_cols.append('mean_{}'.format(i))
# for i in range(1, n + 1):
#     last_cols.append('median_{}'.format(i))
# for i in range(1, n + 1):
#     last_cols.append('std_{}'.format(i))

X_df_train_SINGLE=X_df_train


X_df_train_SINGLE.drop('id', axis=1, inplace=True)
X_df_train_SINGLE.drop('file', axis=1, inplace=True)
X_df_train_SINGLE.drop('patient_id', axis=1, inplace=True)

X_df_train_SINGLE = X_df_train_SINGLE.loc[X_df_train_SINGLE['file_size'] > 100000]
X_df_train_SINGLE.drop('file_size', axis=1, inplace=True)
X_df_train_SINGLE.drop('sequence_id', axis=1, inplace=True)
X_df_train_SINGLE.drop('segment', axis=1, inplace=True)

answers_1_SINGLE = list (X_df_train_SINGLE[singleResponseVariable].values)
X_df_train_SINGLE = X_df_train_SINGLE.drop(singleResponseVariable, axis=1)

X_df_train_SINGLE=X_df_train_SINGLE[last_cols]
X_df_train_SINGLE=X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
X_df_train_SINGLE.head(5)


# In[2]:


from scipy.optimize import fmin_powell
niter=30000
with logistic_model:
#     start_MAP = pm.find_MAP(fmin=fmin_powell, disp=True)
#     print start_MAP
#     start_MAP=lr_coeefs
    step = pm.NUTS()
    step=pm.Metropolis()
    trace_logistic_model = pm.sample(niter, step=step, progressbar=True)y=answers_1_SINGLE
X=X_df_train_SINGLE

lr_best_params = {'penalty': 'l2', 'C': 100, 'solver': 'newton-cg', 'fit_intercept': False}
lr = LogisticRegression(**lr_best_params)
lr.fit(X, y)
#Store LR coeefs
lr_coeefs=lr.coef_        

k = (X_df_train_SINGLE.shape[1])


import theano.tensor as tt
invlogit = lambda x: 1/(1 + tt.exp(-x))


with pm.Model() as logistic_model:        
    b = pm.Normal('b', 0.0, sd=10000, shape=k)    
    p = invlogit(tt.dot(X, b))    
    likelihood = pm.Bernoulli('likelihood', p, observed=y)


# In[3]:


ax = pm.traceplot(trace_logistic_model[-1000:], figsize=(12,len(trace_logistic_model.varnames)*1.5),  
    lines={k: v['mean'] for k, v in pm.df_summary(trace_logistic_model[-1000:]).iterrows()})# predict
# last_cols=X_df_train_SINGLE.columns

df_trace_logistic_model = pm.trace_to_dataframe(trace_logistic_model[niter//2:])
df_trace_logistic_model.columns=last_cols
w_theta = df_trace_logistic_model[last_cols].mean(0)
# df_trace_logistic_model.to_csv("df_trace_logistic_model.csv")
# w_theta.to_csv("w_theta.csv")
# w_intercept=df_trace_logistic_model['Intercept'].mean(0)
# pm.summary(trace_logistic_model[-1000:])


# In[4]:


# --------------------------------------------------------
    #       PATIENT ID
    # --------------------------------------------------------
def getIdFromFileName(id_str):
    arr = id_str.split("_")
#     print arr
    patient = int(arr[1])
#     print patient
    p_id_str = str(arr[2])
#     print p_id_str
    p_id = int((p_id_str)[:-4])
#     print p_id
    new_id = [patient * 100000 + p_id]
    return new_id
    
from scipy.special import expit

def fastPredict(new_observation, theta): 
    v =  np.einsum('j,j->',new_observation, theta)    
    return expit(v)


test_dir=TEST_DATA_FOLDER_IN_ALL
ieegFeatures= IeegFeatures(test_dir, False)
df_cols_test=ieegFeatures.ieegGenCols()
print len(df_cols_test)
F_NAME_TEST= TEST_FEAT_BASE + TEST_PREFIX_ALL +'-feat_TEST_df.csv'
X_df_TEST=pandas.read_csv(F_NAME_TEST, engine='python') 
X_df_TEST.drop('Unnamed: 0', axis=1, inplace=True)
# X_df_TEST.drop('id', axis=1, inplace=True)
X_df_TEST.drop('file', axis=1, inplace=True)
X_df_TEST.drop('patient_id', axis=1, inplace=True)
# X_df_TEST.drop('file_size', axis=1, inplace=True)
# X_df_TEST.drop('sequence_id', axis=1, inplace=True)
X_df_TEST.head(3)

#------------------------------------------------------------------------------#
now = datetime.now()
import dis
sub_file = 'submission' + '_mcmc_' + str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv'

r= pandas.DataFrame.from_csv('sample_submission.csv')
print('Writing submission: ', sub_file)
f = open(sub_file, 'w') # append mode
f.write('File,Class\n')
total = 0

for index, row in r.iterrows():            
    id_str= index     
    arr = id_str.split("_")
#     print str(arr)
#     print str(arr[0])
#     print str(arr[1])
#     print str(arr[2])
    patient = int(arr[1])        
    new_id= getIdFromFileName(id_str) 
#     print str(new_id)
    
    X_df_single_row=X_df_TEST.loc[X_df_TEST['id'] == new_id]
    X_df_single_row.drop('id', axis=1, inplace=True)
    X_df_single_row= X_df_single_row[last_cols]        
#     X_df_single_row.drop('file', axis=1, inplace=True)
#     X_df_single_row.drop('patient_id', axis=1, inplace=True)                    
    X_df_single_row = np.asarray(X_df_single_row)        
    c_pred= 1.0- fastPredict( (tuple (X_df_single_row)[0]), w_theta)
    str1 = id_str + ',' + str(c_pred) + '\n'  
#     print str1
    
    f.write(str1)
    
f.close()

print('Done writing submission: ', sub_file)

In [ ]:



# In[5]:





# In[5]:





# In[5]:





# In[5]:





# In[5]:




