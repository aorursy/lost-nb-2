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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


ghosts_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

ghosts_df.head()
#All data seems normalized. Should probably do the same for colors / types


# In[3]:


ghosts_df.info()
print("--------------------")
test_df.info()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

sns.pairplot(ghosts_df, hue="type")


# In[5]:


ghosts_df = ghosts_df.drop("id",axis=1)
# convert string-values to floats for predicting on them
def remap_as_int(dataframe, indice, value, newValue):
    print("value={} reassigned to id={}".format(value, newValue))
    dataframe.loc[dataframe[indice] == value, indice] = newValue
    
def enumerate_list_unique(dataframe, indice):
    return list(enumerate(np.unique(dataframe[indice])))

monsters_list = enumerate_list_unique(ghosts_df, 'type')
monsters_list = [(0, 'Ghost'),(1, 'Ghoul'),(2, 'Goblin')]
print("All known types of monsters = {}".format(np.unique(ghosts_df['type'])))
for index, monster in monsters_list:
      remap_as_int(ghosts_df, 'type', monster, index)

#colors_list = enumerate_list_unique(ghosts_df, 'color')
colors_list = [(0, 'black'),(0.2, 'blood'),(0.4, 'blue'),(0.6, 'clear'),(0.8, 'green'),(1, 'white')]
print("All known colors of monsters = {}".format(np.unique(ghosts_df['color'])))
for index, color in colors_list:
      remap_as_int(ghosts_df, 'color', color, index)


# In[6]:


def create_features(dataframe):
    #Create some new variables.
    dataframe['hair_soul'] = dataframe['hair_length'] * dataframe['has_soul']
    dataframe['bone_soul'] = dataframe['bone_length'] * dataframe['has_soul']
    dataframe['hair_bone'] = dataframe['hair_length'] * dataframe['bone_length']
    dataframe['rotting_hair'] = dataframe['rotting_flesh'] * dataframe['hair_length']
    dataframe['rotting_soul'] = dataframe['rotting_flesh'] * dataframe['has_soul']
    
create_features(ghosts_df)

sns.lmplot("hair_soul", "type", data=ghosts_df, hue='type', fit_reg=False)
sns.lmplot("bone_soul", "type", data=ghosts_df, hue='type', fit_reg=False)
sns.lmplot("hair_bone", "type", data=ghosts_df, hue='type', fit_reg=False)
sns.lmplot("rotting_hair", "type", data=ghosts_df, hue='type', fit_reg=False)
sns.lmplot("rotting_soul", "type", data=ghosts_df, hue='type', fit_reg=False)


# In[7]:


sns.factorplot('color','type', data=ghosts_df,size=4,aspect=3)

#Change types from object to int to compute on these indices
ghosts_df['type'] = ghosts_df['type'].astype(float)
ghosts_df['color'] = ghosts_df['color'].astype(float)

#It seems like color-distribution is fairly even across all types.
# However blood type is slightly higher for ghosts, and clear is more prominent for the other two.

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='color', data=ghosts_df, ax=axis1)
sns.countplot(x='type', hue="color", data=ghosts_df, order=[monsters_list[0][0],monsters_list[1][0],monsters_list[2][0]], ax=axis2)

color_perc = ghosts_df[["color", "type"]].groupby(['color'],as_index=False).mean()
sns.barplot(x='color', y='type', data=color_perc, ax=axis3)


# In[8]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
    

    
remove_indexes = ["color", "hair_length", "has_soul", "rotting_flesh"]
variant = 'all'
test = False
show_test_result = False
alphas = [0.00001, 0.01, 0.1, 0.5, 1, 10]
if test:
    X_train, X_test, y_train, y_test = train_test_split(ghosts_df.drop(remove_indexes + ["type"],axis=1), ghosts_df["type"], test_size=0.2, random_state=0)
    variant = 'all'
else:
    X_train = ghosts_df.drop(remove_indexes + ["type"],axis=1)
    y_train = ghosts_df["type"]

create_features(test_df)

def print_results(grid_search):
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    if show_test_result:
        y_pred = grid_search.predict(X_test)
        print('Test-accuracy={}'.format(accuracy_score(y_test, y_pred)))

#remap colors to ints for test-set
for index, color in colors_list:
    remap_as_int(test_df, 'color', color, index)
test_df['color'] = test_df['color'].astype(float)
test_df = test_df.drop(remove_indexes, axis=1)

X_submission = test_df.drop("id",axis=1).copy()


# In[9]:


if variant is 'lr' or variant is 'all':
    logreg = LogisticRegression()
    parameter_grid = {'C': alphas,
        'solver' :  ['newton-cg', 'lbfgs', 'sag'],
        'multi_class' : ['multinomial', 'ovr']}
    
    grid_search_logit = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(4))
    grid_search_logit.fit(X_train, y_train)

    print_results(grid_search_logit)


# In[10]:


if variant is 'svc' or variant is 'all':
    svc = SVC()
    parameter_grid = {'C': alphas,
    'kernel' :  ['linear', 'rbf', 'sigmoid'],
    'gamma' : alphas}
    grid_search_svc = GridSearchCV(svc, param_grid=parameter_grid, cv=StratifiedKFold(4))
    grid_search_svc.fit(X_train, y_train)
    print_results(grid_search_svc)


# In[11]:


if variant is 'rf' or variant is 'all':
   random_forest = RandomForestClassifier(random_state=1)

   parameter_grid = {'min_samples_leaf': [1, 5, 10],
                     'n_estimators': [10, 30, 50, 100],
                     'min_samples_split': [2,3]}
   grid_search_rf = GridSearchCV(random_forest, param_grid=parameter_grid, cv=StratifiedKFold(4))
   grid_search_rf.fit(X_train, y_train)
   
   print_results(grid_search_rf)


# In[12]:


if variant is 'knn' or variant is 'all':
    knn = KNeighborsClassifier()
    parameter_grid = {'n_neighbors': [3, 5, 10],
                     'leaf_size': [5, 10, 30, 50],
                     'metric': ['minkowski', 'euclidean']}
    
    grid_search_knn = GridSearchCV(knn, param_grid=parameter_grid, cv=StratifiedKFold(4))
    grid_search_knn.fit(X_train, y_train)
    
    print_results(grid_search_knn)


# In[13]:


if variant is 'nn' or variant is 'all':
    neural_net = MLPClassifier(solver='lbfgs', random_state=1)
    
    parameter_grid = {'hidden_layer_sizes': [(15,1), (20,1), (15,3), (20,3), (15,6), (20,6)],
                     'alpha': alphas}
    
    grid_search_nn = GridSearchCV(neural_net, param_grid=parameter_grid, cv=StratifiedKFold(4))
    grid_search_nn.fit(X_train, y_train)
    
    print_results(grid_search_nn)


# In[14]:


Y_pred = grid_search_svc.predict(X_submission)
#remap to monster labels
Y_pred = Y_pred.astype(object)
Y_pred[Y_pred == monsters_list[0][0]] = monsters_list[0][1]
Y_pred[Y_pred == monsters_list[1][0]] = monsters_list[1][1]
Y_pred[Y_pred == monsters_list[2][0]] = monsters_list[2][1]


# In[15]:



submission = pd.DataFrame({
        "id": test_df["id"],
        "type": Y_pred
    })
submission.to_csv('new.csv', index=False)

