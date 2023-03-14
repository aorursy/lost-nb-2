#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn import preprocessing
from sklearn import ensemble
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train= pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
train.head(5)


# In[3]:


train.head(5)


# In[4]:


#Take categorical columns#
categorical_column=train.columns[1:117]


# In[5]:


train_1=train.copy()
test_1=test.copy()


# In[6]:


#Filler with missing Value#

train_1=train_1.fillna(-999)
test_1=test_1.fillna(-999)
id=train_1["id"]


# In[7]:


#Encoding the data for processing#

for var in categorical_column:
    lb=preprocessing.LabelEncoder()
    full_var_data = pd.concat((train_1[var],test_1[var]),axis=0).astype('str')
    lb.fit(full_var_data)
    train_1[var] = lb.transform(train_1[var].astype('str'))
    test_1[var]=lb.transform(test_1[var].astype('str'))
    


# In[8]:


train_1['log_target']=np.log(train_1['loss'])


# In[9]:


train_1.head()


# In[10]:





# In[10]:





# In[10]:


X=np.array(X)
Y=np.array(Y)


# In[11]:


val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)


# In[12]:


#Scoring parameter
from sklearn.metrics import mean_absolute_error


# In[13]:


alpha =0.5
seed = 0
from sklearn.linear_model import Ridge
model = Ridge(alpha=alpha,random_state=seed)
model.fit(X_train,Y_train)


# In[14]:


import numpy    
result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val)))
print(result)

  


# In[15]:




X = numpy.concatenate((X_train,X_val),axis=0)
#del X_train
#del X_val
Y = numpy.concatenate((Y_train,Y_val),axis=0)
#del Y_train
#del Y_val


# In[16]:


from xgboost import XGBRegressor
import pandas

#X = numpy.concatenate((X_train,X_val),axis=0)
#del X_train
#del X_val
#Y = numpy.concatenate((Y_train,Y_val),axis=0)
#del Y_train
#del Y_val

n_estimators = 1000

#Best model definition
best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
best_model.fit(X,Y)
#del X
#del Y


# In[17]:


#Read test dataset
dataset_test = pandas.read_csv("../input/test.csv")
#Drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id',axis=1,inplace=True)

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

del cats

#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

del encoded_cats
del dataset_test

#Make predictions using the best model
predictions = numpy.expm1(best_model.predict(X_test))
del X_test
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))


# In[18]:


#Pairplot to see the relation between veriable#

sns.pairplot(data = train_1[["log_target","cont1","cont2","cont3","cont4","cont5"]].sample(1000),dropna=True)


# In[19]:


#Get the veriable name #

nolcolumn=[]
for col in train_1.columns:
    if col.startswith('cont'):
        nolcolumn.append(col)
    

nolcolumn.append('loss') 


# In[20]:


#Corelation plot to see relation among veriable#

d = train[nolcolumn]
corrmat = d.corr().abs()

# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(12, 6))
plt.subplots(figsize=(13, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8,annot=True, square=True)

