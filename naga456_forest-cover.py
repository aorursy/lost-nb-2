#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
​
import os
print(os.listdir("../input"))
​
# Any results you write to the current directory are saved as output.

#WL - Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
​

# The following two lines determines the number of visible columns and 
#the number of visible rows for dataframes and that doesn't affect the code
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#WL - Separate X_train, y_train
y_train = train.iloc[:,-1]
#y_train.head()
X_train = train.iloc[:,:-1]
#X_train.head()
X_train.shape

#WL - Analyze the data
train.head()
X_train.describe()
#X_train.shape

#WL - Plot some data
#X_train['Elevation'].value_counts().head(20).plot.bar()
#X_train['Elevation'].value_counts().sort_index().plot.line()
#X_train['Elevation'].sort_index().plot.line()
X_train['Elevation'].plot.hist()
​
​
​
​

X_train['Aspect'].plot.hist()

X_train['Slope'].plot.hist()

X_train['Horizontal_Distance_To_Hydrology'].plot.hist()

X_train['Vertical_Distance_To_Hydrology'].plot.hist()

X_train['Wilderness_Area1'].plot.hist()

X_train['Soil_Type1'].plot.hist()

import seaborn as sns
sns.jointplot(x='Elevation',y='Aspect',data=X_train,kind='hex',gridsize=20)

import seaborn as sns
sns.jointplot(x='Elevation',y='Aspect',data=X_train,kind='hex',gridsize=20)

sns.jointplot(x='Elevation',y='Id',data=X_train,kind='hex',gridsize=20)

sns.jointplot(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',data=X_train,kind='hex',gridsize=20)

sns.jointplot(x='Hillshade_9am',y='Hillshade_Noon',data=X_train,kind='hex',gridsize=20)

#g = sns.FacetGrid(X_train,col='Hillshade_3pm',col_wrap=6)
#g.map(sns.kdeplot,"Slope")

#WL - Select col to use and build the model
#col_to_use = ['Elevation', 'Slope','Vertical_Distance_To_Hydrology']
#df = X_train[col_to_use]
​
​

#WL - build model(tree)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
X_train.shape

#WL - make the test data
#X_test = test.iloc[:,:-1]
#y_test = test.iloc[:,-1]
#X_test.shape
#X_test.head()
#test.shape

prediction = clf.predict(test)
​

output = pd.DataFrame({'Id':test.Id,'Cover_Type':prediction})
output = pd.DataFrame({'Id':test.Id,'Cover_Type':prediction})

#output.head()
output = to_csv('submission.csv',index=False)

​

