#!/usr/bin/env python
# coding: utf-8

# In[1]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
         import os
    os.chdir(r'kaggle/working')

# Any results you write to the current directory are saved as output.


# In[2]:


import h2o
from h2o.automl import H2OAutoML
h2o.init()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly as py
#plotly.offline doesn't push your charts to the clouds
import plotly.offline as pyo
import plotly.graph_objs as go
pyo.offline.init_notebook_mode()
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier ,AdaBoostClassifier,VotingClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
# roc curve and auc score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

pd.set_option('display.max_columns', None)


# In[3]:


# loading dataset 
df= pd.read_csv("../input/widsdatathon2020/training_v2.csv")
test=pd.read_csv("..//input/widsdatathon2020/unlabeled.csv")


# In[4]:


df.sample(5)


# In[5]:


#By using the above code, maybe I can check how missing values vary by thresholds?
for x in range(30):
    df_check = df.dropna(thresh=x)
    print(x," variables = ",df_check.shape)


# In[6]:



# Drop columns based on threshold limit
threshold = len(df) * 0.60
df_thresh=df.dropna(axis=1, thresh=threshold)

# View columns in the dataset
#We can see that 74 columns have been dropped as they cant be used for predictions as they are missing lots of data
df_thresh.shape


# In[7]:


# Drop columns based on threshold limit
threshold = len(test) * 0.50
test=test.dropna(axis=1, thresh=threshold)

# View columns in the dataset
#We can see that 74 columns have been dropped as they cant be used for predictions as they are missing lots of data
test.shape


# In[8]:


#Getting the numerical data from the dataset
df_thresh._get_numeric_data()


# In[9]:


#Checking for missing values in IDs
df_thresh[['encounter_id','hospital_id','patient_id','icu_id']].isnull().sum()


# In[10]:


#Applying skelarn imputer for numerical values 
#Using mean as the imputation value as most of the numerical data are clinical data that can be approximated to mean
imputer_skdf = df_thresh._get_numeric_data()
colNames=imputer_skdf.columns;
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit and transform to the parameters
imputer_skdf = pd.DataFrame(imputer.fit_transform(imputer_skdf))
imputer_skdf.columns = colNames;
# Checking for any null values
imputer_skdf.isna().sum()
imputer_skdf.info()


# In[11]:


#Applying skelarn imputer for numerical values 
#Using mean as the imputation value as most of the numerical data are clinical data that can be approximated to mean
imputer_sktest =test._get_numeric_data()
colNames1=imputer_sktest.columns;
from sklearn.impute import SimpleImputer
imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit and transform to the parameters
imputer_sktest = pd.DataFrame(imputer.fit_transform(imputer_skdf))
imputer_sktest.columns = colNames1;
# Checking for any null values
imputer_sktest.isna().sum()
imputer_sktest.info()


# In[12]:


#categorical values
categ_df=df_thresh.select_dtypes(exclude=['int','float'])
column_names=categ_df.columns
column_names


# In[13]:


#categorical values for test
categ_df1=test.select_dtypes(exclude=['int','float'])
column_names1=categ_df1.columns
column_names1


# In[14]:


# Replacing null values in categorical data with most frequent value for test set
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
categ_df1 = pd.DataFrame(imputer2.fit_transform(categ_df1))
categ_df1.columns =column_names1;


# In[15]:


# Replacing null values in categorical data with most frequent value
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
categ_df = pd.DataFrame(imputer.fit_transform(categ_df))
categ_df.columns =column_names;


# In[16]:


#merging Categorical and numerical data to a single dataset for test set
categ_df1['encounter_id']=imputer_sktest.encounter_id

test=pd.merge(imputer_sktest,categ_df1,on='encounter_id')
test.shape


# In[17]:


#merging Categorical and numerical data to a single dataset 
categ_df['encounter_id']=imputer_skdf.encounter_id

df1=pd.merge(imputer_skdf,categ_df,on='encounter_id')
df1.shape


# In[18]:


#Exploratory data anlaysis and feature engineering 
#Checking the gender distribution using a pie chart
labels = df1['gender'].value_counts().index
values = df1['gender'].value_counts().values

colors = ['#eba796', '#96ebda']

fig = {'data' : [{'type' : 'pie',
                  'name' : "Patients by Gender: Pie chart",
                 'labels' : df1['gender'].value_counts().index,
                 'values' : df1['gender'].value_counts().values,
                 'direction' : 'clockwise',
                 'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'Patients by Gender'}}

pyo.iplot(fig)


# In[19]:


#Visualizing the ICU admit source Distribution 
#we could see that accident and emergency, operating room and floor are major contributors for ICU admissions
colors = ['#eba796', '#96ebda']


fig = {'data' : [{'type' : 'pie',
                  'name' : "ICU admit source",
                 'labels' : df1['icu_admit_source'].value_counts().index,
                 'values' : df1['icu_admit_source'].value_counts().values,
                 'direction' : 'clockwise',
                 'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'ICU admit source distribution'}}

pyo.iplot(fig)


# In[20]:


#Visualizing the Hospital admit source Distribution 
#we could see that accident and emergency are major contributors for ICU admissions also floor constitute 11% of the hospital admissions 
colors = ['#eba796', '#96ebda']


fig = {'data' : [{'type' : 'pie',
                  'name' : "Hospital admit source",
                 'labels' : df1['hospital_admit_source'].value_counts().index,
                 'values' : df['hospital_admit_source'].value_counts().values,
                 'direction' : 'clockwise',
                 'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'hospital admit source distribution'}}

pyo.iplot(fig)


# In[21]:


#Visualizing the Hospital admit source Distribution 
#we could see that accident and emergency are major contributors for ICU admissions also floor constitute 11% of the hospital admissions 
colors = ['#eba796', '#96ebda']


fig = {'data' : [{'type' : 'pie',
                  'name' : "Hospital admit source",
                 'labels' : df1['hospital_admit_source'].value_counts().index,
                 'values' : df['hospital_admit_source'].value_counts().values,
                 'direction' : 'clockwise',
                 'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'hospital admit source distribution'}}

pyo.iplot(fig)


# In[22]:


#taking a subset of the data to understand the effect of features on dataset
df1['hospital_death'].value_counts(normalize=True)



# In[23]:


#Checking the distribution of the icu admit source
df1['icu_admit_source'].value_counts()


# In[24]:


#Checking the distribution of icu_type
df1['icu_type'].value_counts()


# In[25]:


#Visualizing the age 
#We can see that most hospital death occur for patients in the 60 -70 age group
fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df1['age'],bins = 7) # Here you can play with number of bins

plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Patient')
plt.show()


# In[26]:


#gender and #age distrubution
import seaborn as sns 
sns.violinplot(df1['age'], df1['gender']) #Variable Plot
sns.despine()


# In[27]:


var = df1.groupby('gender').hospital_death.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('gender')
ax1.set_ylabel('Sum of deaths')
ax1.set_title("Gender wise Sum of deaths")
var.plot(kind='bar')


# In[28]:


#Visualizing ethnicity
var = df1.groupby('ethnicity').hospital_death.sum() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('ethnicity')
ax1.set_ylabel('Sum of deaths')
ax1.set_title("Ethnicity wise Sum of deaths")
var.plot(kind='bar')


# In[29]:


var = df1.groupby('bmi').hospital_death.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('bmi')
ax1.set_ylabel('Sum of deaths')
ax1.set_title("BMI wise Sum of deaths")
var.plot(kind='line')


# In[30]:


df1['apache_3j_diagnosis'].value_counts()


# In[31]:


df1['apache_2_bodysystem'].value_counts()


# In[32]:



#Visualizing hospital death by apache_2_bodySystem
#We could see that cardio vascular conditions account for the most hospital deaths
var = df1.groupby('apache_2_bodysystem').hospital_death.sum() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('apache_2_bodysystem')
ax1.set_ylabel('Sum of deaths')
ax1.set_title("apache bodysystem wise Sum of deaths")
var.plot(kind='bar')


# In[33]:


test.columns=df1.columns


# In[34]:


df_encoded=pd.get_dummies(df1, columns=['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',
       'icu_stay_type', 'icu_type', 'apache_3j_bodysystem',
       'apache_2_bodysystem'])
df_encoded.columns


# In[35]:


# creating independent features X and dependent feature Y
y =df_encoded['hospital_death']
X = df_encoded
X = df_encoded.drop(columns=['hospital_death'],axis='columns')
test=pd.get_dummies(test, columns=['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',
       'icu_stay_type', 'icu_type', 'apache_3j_bodysystem',
       'apache_2_bodysystem'])


# In[36]:


# Split into training and validation set
X_train, valid_features, Y_train, valid_y = train_test_split(X, y, test_size = 0.25, random_state = 1)


# In[37]:


# Gradient Boosting Classifier
GBC = GradientBoostingClassifier(random_state=1)


# In[38]:


# Random Forest Classifier
RFC = RandomForestClassifier(n_estimators=100)


# In[39]:


# Voting Classifier with soft voting 
votingC = VotingClassifier(estimators=[('rfc', RFC),('gbc',GBC)], voting='soft')
votingC = votingC.fit(X_train, Y_train)


# In[40]:


predict_y = votingC.predict(valid_features)


# In[41]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[42]:


probs = votingC.predict_proba(valid_features)
probs = probs[:, 1]
auc = roc_auc_score(valid_y, probs)
fpr, tpr, thresholds = roc_curve(valid_y, probs)
plot_roc_curve(fpr, tpr)
print("AUC-ROC :",auc)


# In[43]:


test1 = test.copy()

test1["hospital_death"] = votingC.predict(test)
test1.hospital_death =test1.hospital_death.astype(int)
test1.encounter_id =test1.encounter_id.astype(int)
test1[["encounter_id","hospital_death"]].to_csv("..//output/kaggle/working/submission5.csv",index=False)
test1[["encounter_id","hospital_death"]].head()
from IPython.display import FileLink
FileLink(r'submission5.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




