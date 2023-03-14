#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# linear algebra
import numpy as np 
# data processing
import pandas as pd 
#import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import imagesize

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import os
print(os.listdir("../input/siim-isic-melanoma-classification"))


# In[4]:


# Reading the dataset
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
print("{} images in train set.".format(train.shape[0]))
print("{} images in test set.".format(test.shape[0]))


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.isnull().any()


# In[8]:


train.info()


# In[9]:


train.describe()


# In[10]:


train.shape


# In[11]:


train.nunique()


# In[12]:


sns.countplot(train['benign_malignant'])
plt.show()


# In[13]:


fig,ax = plt.subplots(figsize=(15,5))
sns.countplot(train['age_approx'],hue=train['benign_malignant'],ax=ax)
plt.xlabel('age_approx')
plt.ylabel('Counts')
plt.xticks(rotation=45)


# In[14]:


sns.countplot(x='benign_malignant',hue='sex',data=train)


# In[15]:


ax = train["age_approx"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train["age_approx"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[16]:


train.age_approx.hist()
plt.title('Histogram of age_approx')
plt.xlabel('age_approx')
plt.ylabel('Frequency')
plt.savefig('age_approx')


# In[17]:


multi_target_count = train.groupby("patient_id").target.sum()

fig, ax = plt.subplots(1,2,figsize=(15,5))

sns.countplot(train.target, ax=ax[0], palette="Reds")
ax[0].set_xlabel("Binary target")
ax[0].set_title("How often do we observe a positive label?");

sns.countplot(multi_target_count, ax=ax[1])
ax[1].set_xlabel("Numer of targets per image")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Multi-Hot occurences")


# In[18]:


gbSub = train.groupby('anatom_site_general_challenge').sum()
gbSub
sns.barplot(y=gbSub.index, x=gbSub.target, palette="deep")


# In[19]:


fig=plt.figure(figsize=(10, 8))

sns.countplot(x="anatom_site_general_challenge", hue="target", data=train)

plt.title("Total Images by Subtype")


# In[20]:


np.mean(train.target)


# In[21]:


# Showing a sample image
image = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_5766923.jpg')
plt.imshow(image)


# In[22]:


w = 10
h = 10
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 4

# ax enables access to manipulate each of subplots
ax = []

for i in range(columns*rows):
    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'][i]+'.jpg')
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    # Hide grid lines
    ax[-1].grid(False)

      # Hide axes ticks
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    ax[-1].set_title(train['benign_malignant'][i])  # set title
    plt.imshow(img)


plt.show()  # finally, render the plot


# In[23]:


w = 10
h = 10
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 4

# ax enables access to manipulate each of subplots
ax = []

for i in range(columns*rows):
    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train.loc[train['target'] == 1]['image_name'].values[i]+'.jpg')
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    # Hide grid lines
    ax[-1].grid(False)

      # Hide axes ticks
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    ax[-1].set_title(train.loc[train['target'] == 1]['benign_malignant'].values[i])  # set title
    plt.imshow(img)



plt.show()  # finally, render the plot


# In[24]:


train['benign_malignant'] = train['benign_malignant'].replace('malignant',np.nan)
train['benign_malignant'] = train['benign_malignant'].fillna(1)
train['benign_malignant'] = train['benign_malignant'].replace('benign',np.nan)
train['benign_malignant'] = train['benign_malignant'].fillna(0)

train['diagnosis'] = train['diagnosis'].replace('nevus',np.nan)
train['diagnosis'] = train['diagnosis'].fillna(1)
train['diagnosis'] = train['diagnosis'].replace('unknown',np.nan)
train['diagnosis'] = train['diagnosis'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].replace('head/neck',np.nan)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna(1)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].replace('upper extremity',np.nan)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].replace('lower extremity',np.nan)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna(2)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].replace('torso',np.nan)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna(3)

train['sex'] = train['sex'].replace('male',np.nan)
train['sex'] = train['sex'].fillna(1)
train['sex'] = train['sex'].replace('female',np.nan)
train['sex'] = train['sex'].fillna(0)


# In[25]:


df3=train.drop(['patient_id','image_name','anatom_site_general_challenge','diagnosis'], axis=1)
df3.head()


# In[26]:


df3= train.dropna()


# In[27]:


X=df3[['age_approx','benign_malignant','sex']]  
y= df3[['target']]


# In[28]:


X


# In[29]:


y


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[32]:


X_train


# In[33]:


y_train


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model= RandomForestClassifier(n_estimators= 10)
model.fit(X_train,y_train)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[36]:


model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)


# In[37]:


model.fit(X_train, y_train)


# In[38]:


model.score(X_test, y_test)


# In[ ]:




