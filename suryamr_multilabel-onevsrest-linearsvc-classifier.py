#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting 
import seaborn as sns 
color = sns.color_palette() 
import warnings 
warnings.filterwarnings('ignore') #Supress unnecessary warnings for readabilit y and cleaner presentation 
pd.set_option('display.float_format', lambda x: '%.3f' % x) #Limiting floats o utput to 3 decimal points 
train = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\X_train.csv') 
train.head(5)
train.isnull().sum()
test.isnull().sum()

#correlation plot
corr = train.corr()
sns.heatmap(corr)

train1 = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\y_train.csv') 
train1.head(5)

alldata = train.append(train1)
alldata.shape

test = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\X_test.csv') 
test.head(5)

test1 = pd.read_csv('C:\\Users\\Surya\\Desktop\\career-con-2019\\y_train.csv') 
test1.head(5)

alldatatest = test.append(test1)
alldatatest.shape

te = alldatatest['series_id']
te.shape

alldata.drop(["series_id", "group_id", "measurement_number", "row_id"], axis=1, inplace=True)
alldatatest.drop(["series_id", "group_id", "measurement_number", "row_id"], axis=1, inplace=True)

alldatatest.drop(["surface"], axis=1, inplace=True)
alldata.drop(["surface"], axis=1, inplace=True)

from sklearn.preprocessing import scale
x_scale = scale(alldata)
x_scale= pd.DataFrame(x_scale, columns=alldata.columns)
x_scale.head()

from sklearn.preprocessing import scale
x_scal = scale(alldatatest)
x_scal= pd.DataFrame(x_scal, columns=alldatatest.columns)
x_scal.head()

y = train1["surface"]
y.dtype

obj_train1 = train1.select_dtypes(include=['object']).copy()
obj_train1.head()

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
obj_train1["surfaces"] = lb_make.fit_transform(obj_train1["surface"])
obj_train1[["surface", "surfaces"]]

y1 =obj_train1[["surfaces"]]
y1

append = x_scale.append(obj_train1)
print(append.shape)

append1 = x_scal.append(obj_train1)
print(append1.shape)

append.drop(["surface"], axis=1, inplace=True)
append1.drop(["surface"], axis=1, inplace=True)

append["angular_velocity_X"].fillna(0, inplace=True)
append["angular_velocity_Y"].fillna(0, inplace=True)
append["angular_velocity_Z"].fillna(0, inplace=True)
append["linear_acceleration_X"].fillna(0, inplace=True)
append["linear_acceleration_Y"].fillna(0, inplace=True)
append["linear_acceleration_Z"].fillna(0, inplace=True)
append["orientation_W"].fillna(0, inplace=True)
append["orientation_X"].fillna(0, inplace=True)
append["orientation_Z"].fillna(0, inplace=True)
append["orientation_Y"].fillna(0, inplace=True)
append["surfaces"].fillna(0, inplace=True)

append1["angular_velocity_X"].fillna(0, inplace=True)
append1["angular_velocity_Y"].fillna(0, inplace=True)
append1["angular_velocity_Z"].fillna(0, inplace=True)
append1["linear_acceleration_X"].fillna(0, inplace=True)
append1["linear_acceleration_Y"].fillna(0, inplace=True)
append1["linear_acceleration_Z"].fillna(0, inplace=True)
append1["orientation_W"].fillna(0, inplace=True)
append1["orientation_X"].fillna(0, inplace=True)
append1["orientation_Z"].fillna(0, inplace=True)
append1["orientation_Y"].fillna(0, inplace=True)
append1["surfaces"].fillna(0, inplace=True)
append.drop(["surfaces", "orientation_W", "orientation_X", "orientation_Y", "orientation_Z"], axis=1, inplace=True)
append1.drop(["surfaces","orientation_W", "orientation_X", "orientation_Y", "orientation_Z"], axis=1, inplace=True)

a = append.append(y1)
b = append1.append(y1)

a["angular_velocity_X"].fillna(0, inplace=True)
a["angular_velocity_Y"].fillna(0, inplace=True)
a["angular_velocity_Z"].fillna(0, inplace=True)
a["linear_acceleration_X"].fillna(0, inplace=True)
a["linear_acceleration_Y"].fillna(0, inplace=True)
a["linear_acceleration_Z"].fillna(0, inplace=True)

a["surfaces"].fillna(0, inplace=True)

b["angular_velocity_X"].fillna(0, inplace=True)
b["angular_velocity_Y"].fillna(0, inplace=True)
b["angular_velocity_Z"].fillna(0, inplace=True)
b["linear_acceleration_X"].fillna(0, inplace=True)
b["linear_acceleration_Y"].fillna(0, inplace=True)
b["linear_acceleration_Z"].fillna(0, inplace=True)

b["surfaces"].fillna(0, inplace=True)


XTEST = b.iloc[:499878,1:6]
YTEST = b.iloc[:, -1]
XTRAIN = a.iloc[:499878,1:6]
YTRAIN = a.iloc[:, -1]
#from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(XTRAIN,YTRAIN)
PREDI = OVR.predict(XTEST) 
 
print('Accuracy of SVC linear test set: {:.2f}'.format(OVR.score(XTEST, YTEST)))
Accuracy of SVC linear test set: 0.99
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(YTEST, PREDI))


('Accuracy:', 0.9927562325207351)

















# In[2]:




