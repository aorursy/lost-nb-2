#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading essential python packages

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


#loading files

df=pd.read_csv("../input/properties_2016.csv")
#df.info()

#Checking basic info i.e. Coulumns


# In[3]:


#Checkinh list of columns with >50% null values
#Using-> df.isnull().sum()/df.shape[0]
#Drop columns with >50% NaN values

index=0
for i in df.columns:
    count=df[i].isnull().sum()
    if count/df.shape[0]>=0.500000:
        df.drop(i,1,inplace=True)
        print (index,"Dropping:",i,": Count=",count/df.shape[0])
        index+=1


# In[4]:


#Now % null values in our df <0.5
df.isnull().sum()/df.shape[0]
#Now we have to fill remaining values.
#I will be using mean()/median() to fill NA. Just cuz i am still a n00b.


# In[5]:


#Filling NA values
#Too long. I know.It sucks.
#Had to view categorical values of all columns.

df['bathroomcnt']=df['bathroomcnt'].fillna(2)
df['bedroomcnt']=df['bedroomcnt'].fillna(3)
df['buildingqualitytypeid']=df['buildingqualitytypeid'].fillna(7)
df['calculatedbathnbr']=df['calculatedbathnbr'].fillna(2)
df['calculatedfinishedsquarefeet']=df['calculatedfinishedsquarefeet'].fillna(df['calculatedfinishedsquarefeet'].mean())
df['finishedsquarefeet12']=df['finishedsquarefeet12'].fillna(df['finishedsquarefeet12'].mean())
df['fips']=df['fips'].fillna(df['fips'].mean())
df['fullbathcnt']=df['fullbathcnt'].fillna(2)
df['heatingorsystemtypeid']=df['heatingorsystemtypeid'].fillna(2)
df['latitude']=df['latitude'].fillna(df['latitude'].mean())
df['longitude']=df['longitude'].fillna(df['longitude'].mean())
df['lotsizesquarefeet']=df['lotsizesquarefeet'].fillna(df['lotsizesquarefeet'].mean())
df['propertylandusetypeid']=df['propertylandusetypeid'].fillna(261)
df['regionidcity']=df['regionidcity'].fillna(df['regionidcity'].mean())
df['regionidcounty']=df['regionidcounty'].fillna(3101)
df['regionidzip']=df['regionidzip'].fillna(df['regionidzip'].mean())
df['roomcnt']=df['roomcnt'].fillna(0)
df['unitcnt']=df['unitcnt'].fillna(1)
df['yearbuilt']=df['yearbuilt'].fillna(1963)
df['structuretaxvaluedollarcnt']=df['structuretaxvaluedollarcnt'].fillna(df['structuretaxvaluedollarcnt'].mean())
df['taxvaluedollarcnt']=df['taxvaluedollarcnt'].fillna(df['taxvaluedollarcnt'].median())
df['assessmentyear']=df['assessmentyear'].fillna(2015)
df['landtaxvaluedollarcnt']=df['landtaxvaluedollarcnt'].fillna(df['landtaxvaluedollarcnt'].median())
df['taxamount']=df['taxamount'].fillna(df['taxamount'].median())
df['censustractandblock']=df['censustractandblock'].fillna(df['censustractandblock'].median())


df.drop(['propertycountylandusecode','propertyzoningdesc','rawcensustractandblock'],1,inplace=True)


# In[6]:


#loading Training set.
train=pd.read_csv("../input/train_2016_v2.csv")
#Storing ParselID 
id=df['parcelid']

df.set_index('parcelid',inplace=True)
del df.index.name

train.set_index('parcelid',inplace=True)
del train.index.name

combined = pd.concat([train,df], axis=1,join='inner')


# In[7]:


combined.info()
#Everything looks good. Ahha! Finally


# In[8]:


combined.head()


# In[9]:


#Looks like i'l have to remove transaction date. (Why i did that?)
combined['tyear']=combined['transactiondate'].map(lambda x: x.split('-')[0].strip())
combined['tmonth']=combined['transactiondate'].map(lambda x: x.split('-')[1].strip())
combined['tmonth']=combined['tmonth'].astype(float)
combined['tyear']=combined['tyear'].astype(float)
                                                  
combined.reset_index(inplace=True)

#Dropping logerror, since we need it in training.
target=combined['logerror']
combined.drop(['logerror','transactiondate','index'],1,inplace=True)


# In[10]:


#combined.info()
#df.info()


# In[11]:


#Adding Dates in training set as well.
#201610,201611,201612,201710,201711,201712

df['tyear']=0.00
df['tmonth']=0.00


# In[12]:


#Needs tuning
X,x,Y,y=train_test_split(combined,target,train_size=0.95)
import sklearn.ensemble as ske
from xgboost import XGBRegressor
clf=ske.GradientBoostingRegressor(n_estimators=40)
clf.fit(combined,target)


# In[13]:


clf.score(x,y)


# In[14]:


""" 
df['tyear']=2016
df['tmonth']=10
pred201610=clf.predict(df)
#----
df['tyear']=2016
df['tmonth']=11
pred201611=clf.predict(df)
#-----

df['tyear']=2016
df['tmonth']=12
pred201612=clf.predict(df)
#-----
df['tyear']=2017
df['tmonth']=10
pred201710=clf.predict(df)
#---
df['tyear']=2017
df['tmonth']=11
pred201711=clf.predict(df)
#-----
df['tyear']=2017
df['tmonth']=12
pred201712=clf.predict(df)
#----
""" 


# In[15]:


"""output=pd.DataFrame()

output['parcelId']=id
output['201610']=pred201610
output['201611']=pred201611
output['201612']=pred201612
output['201710']=pred201710
output['201711']=pred201711
output['201712']=pred201712

output=output.round(4)
""" 


# In[16]:


#output[['parcelId','201610','201611','201612','201710','201711','201712']].to_csv("output.csv",index=False)


# In[17]:


#output.head()
#Time for submission!


# In[18]:


#output.info()

