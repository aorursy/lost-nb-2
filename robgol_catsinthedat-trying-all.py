#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

## basic imports
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

# Keras imports 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils


# In[2]:


# loading data and getting shape
df_train = pd.read_csv("../input/cat-in-the-dat/train.csv") 
df_test = pd.read_csv("../input/cat-in-the-dat/test.csv")
df_sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
print(df_train.shape, df_test.shape, df_sample_submission.shape)


# In[3]:


## analysing the binary features
binary = [ 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
for i in binary:
    print(df_train[i].value_counts())


# In[4]:


nominal = ['nom_0', 'nom_1','nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for i in nominal:
    print(f'{i}: {df_train[i].value_counts()}')


# In[5]:


## Do not worth work with 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9' because the processing will be very hard
nominal = ['nom_0', 'nom_1','nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for i in nominal:
    print(f'The diferent values in {i} feature is: {df_train[i].value_counts().count()}')


# In[6]:


ordinal = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
for i in ordinal:
    print(f'{i}: {df_train[i].value_counts()}')


# In[7]:


## Do not worth work with 'ord_3', 'ord_4', 'ord_5' because the processing will be very hard
ordinal = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
for i in ordinal:
    print(f'The diferent values in {i} feature is: {df_train[i].value_counts().count()}')


# In[8]:


## Slicing for validate the best model 
## using only 5000 lines to validate ... to spend my time
X = df_train.drop(['id', 'target'], axis = 1).head(10000)
X_teste= df_test.drop('id', axis = 1).head(10000)
y = df_train['target'].values


# In[9]:


X.shape, X_teste.shape


# In[10]:


## let's transform to categorical data
columns = ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0','ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
for i in columns:
    X[i] = pd.Categorical(X[i]).codes
    X_teste[i] = pd.Categorical(X_teste[i]).codes
    
X = X.values
X_teste = X_teste.values    

ohe = OneHotEncoder(categorical_features=[5, 6,7,8,9])
X = ohe.fit_transform(X).toarray()
X_teste = ohe.fit_transform(X_teste).toarray()


# In[11]:


X.shape, X_teste.shape


# In[12]:


get_ipython().run_cell_magic('time', '', "## Trying to validade the model here... testing with some algoritms and verifying the best perform\nkfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)\nmodelos = {'Logistic': LogisticRegression(solver = 'lbfgs', C =  0.1, max_iter = 5000), 'naive': GaussianNB(), 'random': RandomForestClassifier(n_estimators=100), 'tree': DecisionTreeClassifier(), 'knn': KNeighborsClassifier(), 'svc': SVC(gamma='scale'),\n          'neural': MLPClassifier(learning_rate='adaptive', max_iter=2000)}\nresultados = {}\ndesvio_padrao = {}\nmatrizes = []\nfor i, j in modelos.items():\n    for ind_treinamento, ind_teste in kfold.split(X, np.zeros(shape = (X.shape[0], 1))):\n        modelo = j\n        modelo.fit(X[ind_treinamento], y[ind_treinamento])\n        previsoes = modelo.predict(X[ind_teste])\n        precisao = accuracy_score(y[ind_teste], previsoes)\n        matrizes.append(confusion_matrix(y[ind_teste], previsoes))        \n        resultados[i] = np.mean(precisao)\n        desvio_padrao[i] =  np.std(precisao)\n    matriz_final = np.mean(matrizes, axis = 0)\n    print(f'Resultados para o modelo {i}: {resultados[i]}')\n    print(f'Desvio padrão para o modelo {i}: {desvio_padrao[i]}')\n    print(matriz_final)")


# In[13]:


## analysing better approach
## not good results until here and the processing is very hard
sns.barplot(list(modelos.keys()), list(resultados.values()))

plt.grid()


# In[14]:


modelo_svc = SVC(gamma='scale', verbo)
modelo_svc.fit(X, y)
predicao = modelo_svc.predict(X_teste)
df_submission = pd.DataFrame({'id': df_test['id'], 'target': predicao})
df_submission.to_csv('predict_svc.csv', index = False)


# In[15]:


modelo_random = RandomForestClassifier(n_estimators=100,  verbose = True)
modelo_random.fit(X, y)
predicao = modelo_random.predict(X_teste)
df_submission = pd.DataFrame({'id': df_test['id'], 'target': predicao})
df_submission.to_csv('predict_random.csv', index = False)


# In[16]:


# Organizing data
target = df_train['target']
train_id = df_train['id']
test_id = df_test['id']
df_train.drop(['target', 'id'], axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

print(df_train.shape)
print(df_test.shape)


# In[17]:


get_ipython().run_cell_magic('time', '', '\n# using Dummies variables in the model and expand a lot the dimensions  - sparse matrix\ntraintest = pd.concat([df_train, df_test])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\ndf_train_ohe = dummies.iloc[:df_train.shape[0], :]\ndf_test_ohe = dummies.iloc[df_train.shape[0]:, :]\n\nprint(df_train_ohe.shape)\nprint(df_test_ohe.shape)')


# In[18]:


dummies.shape


# In[19]:


get_ipython().run_cell_magic('time', '', '## using tocoo() to transform in coordinate format and tocsr() to compress matrix\nX = df_train_ohe.sparse.to_coo().tocsr()\nX_teste = df_test_ohe.sparse.to_coo().tocsr()\ny = target.values')


# In[20]:


X, X_teste


# In[21]:


get_ipython().run_cell_magic('time', '', "## functions to create the models - LogisticRegression and XgbootClassifier\n\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score as auc\nfrom sklearn.linear_model import LogisticRegression\n\n# Model\ndef run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):\n    kf = KFold(n_splits=5)\n    fold_splits = kf.split(train, target)\n    cv_scores = []\n    pred_full_test = 0\n    pred_train = np.zeros((train.shape[0]))\n    i = 1\n    for dev_index, val_index in fold_splits:\n        print('Started ' + label + ' fold ' + str(i) + '/5')\n        dev_X, val_X = train[dev_index], train[val_index]\n        dev_y, val_y = target[dev_index], target[val_index]\n        params2 = params.copy()\n        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)\n        pred_full_test = pred_full_test + pred_test_y\n        pred_train[val_index] = pred_val_y\n        if eval_fn is not None:\n            cv_score = eval_fn(val_y, pred_val_y)\n            cv_scores.append(cv_score)\n            print(label + ' cv score {}: {}'.format(i, cv_score))\n        i += 1\n    print('{} cv scores : {}'.format(label, cv_scores))\n    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))\n    print('{} cv std score : {}'.format(label, np.std(cv_scores)))\n    pred_full_test = pred_full_test / 5.0\n    results = {'label': label,\n              'train': pred_train, 'test': pred_full_test,\n              'cv': cv_scores}\n    return results\n\n\ndef runLR(train_X, train_y, test_X, test_y, test_X2, params):\n    print('Train LR')\n    model = LogisticRegression(**params)\n    model.fit(train_X, train_y)\n    print('Predict 1/2')\n    pred_test_y = model.predict_proba(test_X)[:, 1]\n    print('Predict 2/2')\n    pred_test_y2 = model.predict_proba(test_X2)[:, 1]\n    return pred_test_y, pred_test_y2\n\n\ndef runXGB(train_X, train_y, test_X, test_y, test_X2, params):\n    print('Train XGB')\n    model = XGBoostClassifier()\n    model.fit(train_X, train_y)\n    print('Predict 1/2')\n    pred_test_y = model.predict_proba(test_X)[:, 1]\n    print('Predict 2/2')\n    pred_test_y2 = model.predict_proba(test_X2)[:, 1]\n    return pred_test_y, pred_test_y2\n\n### You can change the function when you call the function. In this case I call using runLR\n\nlr_params = {'solver': 'lbfgs', 'C': 0.1}\nresults = run_cv_model(X, X_teste, target, runXGB, lr_params, auc, 'lr')")


# In[22]:


modelo_logistic = LogisticRegression(solver = 'lbfgs', C=0.1)
modelo_logistic.fit(X, y)
predicao = modelo_logistic.predict(X_teste)
df_submission = pd.DataFrame({'id': test_id, 'target': predicao})
df_submission.to_csv('predict_logistic.csv', index = False)


# In[ ]:




