#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
from pydicom import dcmread

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


path = "/kaggle/input/osic-pulmonary-fibrosis-progression/"


# In[3]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/


# In[4]:


train_df  = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
subm = pd.read_csv(path + "sample_submission.csv")


# In[5]:


train_df=pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")


# In[6]:


train_df.dtypes


# In[7]:


train_df.head(1)


# In[8]:


train_df['Sex']=="Female"


# In[9]:


train_df.describe()


# In[10]:


print("Total Number of data Points are \n",str(len(train_df)))
print("Details of Unique and Mising value of patients \n")


for col in train_df.columns:
    print('{} : {} unique values, {} missing.'.format(col, 
                                                          str(len(train_df[col].unique())), 
                                                          str(train_df[col].isna().sum())))


# In[11]:


unique_patient_df = train_df.drop(['Weeks', 'FVC', 'Percent'],
                                  axis=1).drop_duplicates().reset_index(drop=True)
unique_patient_df['Times_Visits'] = [train_df['Patient'].
                                 value_counts().loc[pid] for pid in unique_patient_df['Patient']]

print('Number of data points: ' + str(len(unique_patient_df)))
print("\n")
for col in unique_patient_df.columns:
    print('{} : {} unique values, {} missing.'.format(col,
                                                      
                                                     str(len(unique_patient_df[col].unique())), 
                                                     str(unique_patient_df[col].isna().sum())))
unique_patient_df.head()


# In[12]:



plt.figure(figsize=(10,10))
sns.countplot('Age',data=unique_patient_df)


# In[13]:


sns.countplot(x='Sex',data=unique_patient_df)
plt.figure(figsize=(5,2))


# In[14]:


sns.countplot(x='SmokingStatus',data=unique_patient_df)
plt.figure(figsize=(5,2))


# In[15]:


plt.figure(figsize=(20,8))
sns.countplot(x='Weeks',data=train_df)


plt.figure(figsize=(8,3))
sns.countplot(x='Times_Visits',data=unique_patient_df)


# In[16]:


plt.figure(figsize=(10,2))
sns.countplot('FVC',data=train_df)


# In[17]:


plt.figure(figsize=(10,5))
sns.distplot(train_df['FVC'])


# In[18]:


plt.figure(figsize=(10,5))
sns.distplot(train_df['Percent'])


# In[19]:


train_df['Expected_FVC']=train_df['FVC']+(train_df['Percent']/100)*train_df['FVC']
plt.figure(figsize=(10,5))
sns.distplot(train_df['Expected_FVC'])


# In[20]:


train_df['diff_FVC']=train_df['Expected_FVC']-train_df["FVC"]
sns.distplot(train_df['diff_FVC'])


# In[21]:


pd.crosstab(train_df.Sex,train_df.SmokingStatus,margins=True)


# In[22]:


unique_patient_df.head()


# In[23]:


corr=unique_patient_df.corr()
features=corr.index
plt.figure(figsize=(10,5))
sns.heatmap(unique_patient_df[features].corr(),annot=True)


# In[24]:


corr=train_df.corr()
features=corr.index
plt.figure(figsize=(10,5))
sns.heatmap(train_df[features].corr(),annot=True)


# In[25]:


# CT Scan Images


# In[26]:


ls/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00422637202311677017371/


# In[27]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/


# In[28]:


fig, axs = plt.subplots(5, 6,figsize=(20,20))
for n in range(0,30):
    image = dcmread("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/" + str(n+1) + ".dcm")
    axs[int(n/6),np.mod(n,6)].imshow(image.pixel_array);


# In[29]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/


# In[30]:


ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00026637202179561894768/


# In[31]:


test_df.head()


# In[ ]:




