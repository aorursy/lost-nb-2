#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# HANDLE DATA SIZE
# Downcaster les colonnes int64 

def handle_data(data):

    list_int64 = data.select_dtypes(include=['int64']).columns.tolist()

    for col in list_int64 :
        if data[col].max() <= 255 and data[col].min() > 0  :
            data[col] = data[col].astype('uint8')  # les uint8 rassemble les entiers positifs 0 à 255 
        elif data[col].max() <= 65535 and data[col].min() > 0  :
            data[col] = data[col].astype('uint16')
        elif data[col].max() <= 4294967295 and data[col].min() > 0  :
            data[col] = data[col].astype('uint32')
        elif data[col].max() <= 127 and data[col].min() >= -128  :
            data[col] = data[col].astype('int8')
        elif data[col].max() <= 32767 and data[col].min() >= -32768  :
            data[col] = data[col].astype('int16')

    # transfomer les colonnes str en variables catégorielles
    list_str = data.select_dtypes(include=['object']).columns.tolist()
    for col in list_str :
        data[col] = data[col].astype('category') 

    dict_dtypes = data.dtypes.to_dict()
    return dict_dtypes


# In[3]:


train_rows = 3767000
test_rows = 2530000

data_train_init = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', nrows = train_rows*0.001)
data_test_init = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/test.csv', nrows = test_rows*0.001)

dtypes_train = handle_data(data_train_init)
dtypes_test =handle_data(data_test_init)


# In[4]:


dtypes_train


# In[5]:


import random
    
def get_random_idx(liste_idx) :
    random.seed(42)
    sample = random.sample(liste_idx, k=int(len(liste_idx)*0.7))
    print(int(len(liste_idx)*0.7))
    return sample


# In[6]:


train_rows = 3767000
test_rows = 2530000

liste_idx_train = range(3767000)
liste_idx_test = range(2530000)

list_idx_random_train = get_random_idx(liste_idx_train)
list_idx_random_test = get_random_idx(liste_idx_test)


# In[7]:


data_train = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', dtype=dtypes_train, nrows = 300_000)
#data_test = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/test.csv', dtype=dtypes_test, skiprows = list_idx_random_test)


# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


X = data_train.iloc[:, 18].values  #train on is_booking -> booké ou non
y = data_train.iloc[:, -1].values  #test on hotel_cluster -> quel type d'hotel


# In[10]:


X = X.reshape(-1,1)
y


# In[11]:


np.unique(y)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)


# In[14]:


y_pred = classifier.predict(X_test)


# In[15]:


classifier.score(X_test, y_pred)


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[17]:


from sklearn.metrics import f1_score

f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("weighted :", f1_weighted, "\n--------------------------------\n","macro :", f1_macro)


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[19]:


from sklearn.tree import export_graphviz


# In[20]:


pip install pydotplus


# In[21]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

