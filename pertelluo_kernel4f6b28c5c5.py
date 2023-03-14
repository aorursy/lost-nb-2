#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[2]:


train_data=pd.read_csv('../input/train.csv',sep=',')


# In[3]:


train_data.shape


# In[4]:


train_data.columns


# In[5]:


train_data.isnull().sum()#检查是否有空值，根据结果查看是没有的


# In[6]:


train_data['Cover_Type'].value_counts() #查看预测值的分布


# In[7]:


test_data=pd.read_csv('../input/test.csv',sep=',')


# In[8]:


test_data.shape


# In[9]:


test_data.head()


# In[10]:


#3 
corr = train_data.corr()
f, ax = plt.subplots(figsize=(10, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)

#4
train_data.boxplot(rot = 90, figsize=(12,10))
plt.scatter(x = train_data.loc[:,'Hillshade_3pm'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()
train_data.drop(train_data[(train_data.loc[:,'Hillshade_3pm']< 10)&(train_data.iloc[:,-1]==1)].index, inplace=True)

plt.scatter(x = train_data.loc[:,'Hillshade_Noon'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()
train_data.drop(train_data[(train_data.loc[:,'Hillshade_Noon']< 150)&(train_data.iloc[:,-1]==2)].index, inplace=True)
train_data.drop(train_data[(train_data.loc[:,'Hillshade_Noon']< 120)&(train_data.iloc[:,-1]==1)].index, inplace=True)
train_data.drop(train_data[(train_data.loc[:,'Hillshade_Noon']< 100)&(train_data.iloc[:,-1]==7)].index, inplace=True)

plt.scatter(x = train_data.loc[:,'Slope'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()

train_data.drop(train_data[(train_data.loc[:,'Slope']> 45)&(train_data.iloc[:,-1]==6)].index, inplace=True)
train_data.drop(train_data[(train_data.loc[:,'Slope']> 40)&(train_data.iloc[:,-1]==5)].index, inplace=True)

plt.scatter(x = train_data.loc[:,'Vertical_Distance_To_Hydrology'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()

train_data.drop(train_data[(train_data.loc[:,'Vertical_Distance_To_Hydrology']> 500)].index, inplace=True)
train_data.drop(train_data[(train_data.loc[:,'Vertical_Distance_To_Hydrology']<-150)].index, inplace=True)

plt.scatter(x = train_data.loc[:,'Horizontal_Distance_To_Hydrology'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()

plt.scatter(x = train_data.loc[:,'Horizontal_Distance_To_Roadways'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()

train_data.drop(train_data[(train_data.loc[:,'Horizontal_Distance_To_Roadways']> 4000)&(train_data.iloc[:,-1]==5)].index, inplace=True)

plt.scatter(x = train_data.loc[:,'Horizontal_Distance_To_Fire_Points'], y = train_data.iloc[:,-1])
plt.grid(True)
plt.show()

train_data.drop(train_data[(train_data.loc[:,'Horizontal_Distance_To_Fire_Points']> 3600)&(train_data.iloc[:,-1]==5)].index, inplace=True)


# In[11]:


train_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
test_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
train_x,val_x,train_y,val_y=train_test_split(train_data.iloc[:,1:-1].values,train_data.iloc[:,-1].values,test_size=0.2)


# In[12]:


train_y=train_y.reshape(-1,1)
val_y=val_y.reshape(-1,1)


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


model=RandomForestClassifier(max_depth=20,random_state=0,n_estimators=500)
model.fit(train_x,train_y)


# In[15]:


train_y_pred=model.predict(train_x)


# In[16]:


print(classification_report(train_y,train_y_pred))


# In[17]:


val_y_pred=model.predict(val_x)


# In[18]:


print(classification_report(val_y,val_y_pred))


# In[19]:


test_y_pred=model.predict(test_data.iloc[:,1:].values)


# In[20]:


test_data['Cover_Type']=test_y_pred


# In[21]:


test_submission=test_data[["Id",'Cover_Type']]


# In[22]:


test_submission.to_csv('../working/test_submission.csv',index=False)


# In[23]:




