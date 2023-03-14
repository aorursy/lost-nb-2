#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm_notebook
import string
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# In[2]:


# Train Data Preparation
data_path = '/kaggle/input/whats-cooking-kernels-only/'
json_train_path = os.path.join(data_path, 'train.json')
json_train = pd.read_json(json_train_path)

json_train


# In[3]:


# Step 1: Make a list with all unique ingredients

total_ingredients_list = []
print('Phase 1 of 2')
tq = tqdm_notebook(total=json_train.shape[0])
for ing in json_train['ingredients']:
    total_ingredients_list = total_ingredients_list + ing
    tq.update(1)
tq.close()

print('Total number of pre-clean unique ingredients: {}'.format(len(np.unique(total_ingredients_list))))

# Remove a lot of things in the unique ingredients list to clean the data:
# - Remove (number  oz.)
# - Remove all spaces in the beggining or in the end
# - Remove all special characters

import string
from nltk.corpus import stopwords

chars = re.escape(string.punctuation)
clean_ingredients_list = [re.sub(r'['+chars+']', '', 
                                 re.sub('[0-9]+','', c.replace("oz",""))).strip(' ').lower()
                    for c in total_ingredients_list]

stop_words = set(stopwords.words('english'))
print('Phase 2 of 2')
tq = tqdm_notebook(total=len(clean_ingredients_list))
for i, ingredients in enumerate(clean_ingredients_list):
    cleaned_ingredients = [c for c in ingredients.split(' ') if c not in stop_words]
    cleaned_ingredients = (' '.join(cleaned_ingredients)).strip(' ')
    clean_ingredients_list[i] = cleaned_ingredients
    tq.update(1)
tq.close()
# See some of the cleaned data
# BagOfWords
clean_unique_ingredients_list = [c for c in list(np.unique(clean_ingredients_list)) if len(c)>0]
print('Total number of cleaned unique ingredients: ', len(clean_unique_ingredients_list))


# In[4]:


# Step 2: Clean the recipes.

tq = tqdm_notebook(total=json_train.shape[0])
for i in range(json_train.shape[0]):
    recipe = json_train['ingredients'][i]
    clean_recipe = [re.sub(r'['+chars+']', '',
                           re.sub('[0-9]+','', c.replace("oz",""))).strip(' ').lower()
                    for c in recipe]
    # delete stop words
    for k, ingredients in enumerate(clean_recipe):
        cleaned_ingredients = [c for c in ingredients.split(' ') if c not in stop_words]
        cleaned_ingredients = (' '.join(cleaned_ingredients)).strip(' ')
        clean_recipe[k] = cleaned_ingredients
    json_train['ingredients'][i] = ' '.join(clean_recipe)
    tq.update(1)
    
tq.close()

json_train


# In[5]:


# Previous Step: Codificate the train data in a sparse matrix to save memory.

vectorizer = TfidfVectorizer(binary=False)
vectorizer.fit(json_train['ingredients'])

X_train = vectorizer.transform(json_train['ingredients']) 

lb = LabelEncoder()
y_train = lb.fit_transform(json_train['cuisine'])


# In[6]:


# Model 1: Neuronal Network

classifier_1 = MLPClassifier(hidden_layer_sizes=300,
                             early_stopping=True,
                             random_state=7,
                             learning_rate_init=0.0001)

from sklearn.multiclass import OneVsRestClassifier
classifier_1 = OneVsRestClassifier(classifier_1, n_jobs=-1)

model_1 = classifier_1.fit(X_train,y_train)


# In[7]:


# Model 2: SVMachine Classifier (This model alone have an 0.78761 score under the test set in Kaggle)

classifier_2 = SVC(C=100, # penalty parameter
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value
                 gamma=1, # kernel coefficient
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=True, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)

classifier_2 = OneVsRestClassifier(classifier_2, n_jobs=1)

model_2 = classifier_2.fit(X_train, y_train)


# In[8]:


json_test_path = json_train_path = os.path.join(data_path, 't.json')
json_test = pd.read_json(os.path.join(json_test_path)

# Clean the recipes using the unique ingredients list calculated before.

tq = tqdm_notebook(total=json_test.shape[0])
for i in range(json_test.shape[0]):
    recipe = json_test['ingredients'][i]
    clean_recipe = [re.sub(r'['+chars+']', '',
                           re.sub('[0-9]+','', c.replace("oz",""))).strip(' ').lower()
                    for c in recipe]
    # delete stop words
    for k, ingredients in enumerate(clean_recipe):
        cleaned_ingredients = [c for c in ingredients.split(' ') if c not in stop_words]
        cleaned_ingredients = (' '.join(cleaned_ingredients)).strip(' ')
        clean_recipe[k] = cleaned_ingredients
    json_test['ingredients'][i] = ' '.join(clean_recipe)
    tq.update(1)
    
tq.close()

json_test


# In[9]:


X_test = vectorizer.transform(json_test['ingredients'])

y_pred_1 = model_1.predict_proba(X_test)

print(y_pred_1.shape) #shape=[n_examples, n_classes]

y_pred_2 = model_2.predict_proba(X_test)

print(y_pred_2.shape)


# In[10]:


# Ensembling:

y_pred = (0.3*y_pred_1 + 0.7*y_pred_2)  #0.3*MLP+0.7*SVC = 0.82019
y_pred = np.argmax(y_pred, axis=1)
y_pred = lb.inverse_transform(y_pred)
y_pred


# In[11]:


# Submission:

print ("Generate Submission File ... ")
test_id = json_test['id']
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)


# In[ ]:




