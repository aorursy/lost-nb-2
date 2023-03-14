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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Restrict minor warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:





# Import test and train data
df_train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
df_Test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
df_test = df_Test


# In[4]:


df_train.head()


# In[5]:


df_test.head()


# In[6]:


df_train.dtypes


# In[7]:


pd.set_option('display.max_columns', None) # we need to see all the columns
df_train.describe()


# In[8]:


# frequency count of column A 
count_train = df_train['Cover_Type'].value_counts() 
print(count_train)


# In[9]:


plt.figure(figsize=(12,5))
plt.title("Distribution of forest categories(Target Variable)")
ax = sns.distplot(df_train["Cover_Type"])


# In[10]:


# Drop 'Id'
df_train = df_train.iloc[:,1:]
df_test = df_test.iloc[:,1:]


# In[11]:


# Drop Soil_Type7 and Soil_Type15
df_train =df_train.drop(['Soil_Type7','Soil_Type15'], axis = 1)
df_test =df_test.drop(['Soil_Type7','Soil_Type15'], axis = 1)


# In[12]:


no_of_continuous_attributes = 10
correlation_matrix =df_train.iloc[:,:no_of_continuous_attributes].corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(correlation_matrix,vmax=0.8,square=True);


# In[13]:


#Correlation values

data = df_train.iloc[:,:no_of_continuous_attributes]





# In[14]:


# Get name of the continuous attributes
cols = data.columns
print(cols)


# In[15]:


# Calculate the pearson correlation coefficients for all combinations
data_corr = data.corr()

# Threshold ( only highly correlated ones matter)
threshold = 0.5
corr_list = []


# In[16]:


# Get correlation matrix
data_corr


# In[17]:


#Bubble sorting of correlation array
             
                
    for i in range(0, no_of_continuous_attributes):
        for j in range(i+1, no_of_continuous_attributes):
            if data_corr.iloc[i,j]>= threshold and data_corr.iloc[i,j]<1                or data_corr.iloc[i,j] <0 and data_corr.iloc[i,j]<=-threshold:
                    corr_list.append([data_corr.iloc[i,j],i,j])


# In[18]:


#sort the correlation values

s_corr_list = sorted(corr_list,key= lambda x: -abs(x[0]))

# print the higher values
for v,i,j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i], cols[j], v))


# In[19]:


df_train.iloc[:,:10].skew()


# In[20]:


for v,i,j in s_corr_list:
    sns.pairplot(data = df_train, hue='Cover_Type', size= 10, x_vars=cols[i], y_vars=cols[j])
    plt.show()


# In[21]:


print(df_train.columns)


# In[22]:


columns = df_train.columns
#target value is not needed
size = len(columns)-1

# x-axis has target attributes to distinguish between classes
x = columns[size]
y = columns[0:size]

for i in range(0, size):
    sns.violinplot(data=df_train, x=x, y=y[i])
    plt.show()


# In[23]:


print(df_train.columns)


# In[24]:


df_train.Wilderness_Area1.value_counts()


# In[25]:


df_train.Wilderness_Area2.value_counts()


# In[26]:


df_train.Wilderness_Area3.value_counts()


# In[27]:


df_train.Wilderness_Area4.value_counts()


# In[28]:


#grouping of one-hot encoded variables into a single variable
cols = df_train.columns
r,c = df_train.shape


# In[29]:


# Create a new dataframe with r rows, one column for each encoded category[Wilderness_Area(1-4),Soil_type[1-40], and target in the end
new_data = pd.DataFrame(index= np.arange(0,r), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])


# In[30]:



# Make an entry in data for each r for category_id, target_value
for i in range(0,r):
    p = 0;
    q = 0;
    # Category1_range
    for j in range(10,14):
        if (df_train.iloc[i,j] == 1):
            p = j-9 # category_class
            break
    # Category2_range
    for k in range(14,54):
        if (df_train.iloc[i,k] == 1):
            q = k-13 # category_class
            break
    # Make an entry in data for each r
    new_data.iloc[i] = [p,q,df_train.iloc[i, c-1]]


# In[31]:


# plot for category1
sns.countplot(x = 'Wilderness_Area', hue = 'Cover_Type', data = new_data)
plt.show()


# In[32]:


# Plot for category2
plt.rc("figure", figsize = (25,10))
sns.countplot(x='Soil_Type', hue = 'Cover_Type', data= new_data)
plt.show()


# In[33]:


# calculating the count of soil_type[1-40] as well as  Wilderness_Area[1-4] 
for i in range(10,df_train.shape[1]-1):
    j = df_train.columns[i]
    print (df_train[j].value_counts())



# In[34]:


# Let's drop them
df_train = df_train.drop(['Soil_Type8', 'Soil_Type25'], axis=1)
df_test = df_test.drop(['Soil_Type8', 'Soil_Type25'], axis=1)
df_train1 = df_train # To be used for algos like SVM where we need normalization and StandardScaler
df_test1 = df_test # To be used under normalization and StandardScaler


# In[35]:


df_train.iloc[:,:10].skew()


# In[36]:


#Horizontal_Distance_To_Hydrology
from scipy import stats
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)


# In[37]:


fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)


# In[38]:


df_train2=df_train1


# In[39]:


df_train1['Horizontal_Distance_To_Hydrology'] = np.sqrt(df_train1['Horizontal_Distance_To_Hydrology'])


# In[40]:


# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)


# In[41]:


df_train1.head()

Both the transformations are working properly for this data but squared one gives better result 
# In[ ]:





# In[42]:


plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)


# In[43]:


df_train1['Horizontal_Distance_To_Roadways'] = np.sqrt(df_train1['Horizontal_Distance_To_Roadways'])


# In[44]:


# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)


# In[45]:


# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train2['Horizontal_Distance_To_Roadways'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train2['Horizontal_Distance_To_Roadways'], plot=plt)


# In[46]:


#Hillshade_9am
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_9am'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_9am'],plot=plt)


# In[47]:


df_train1['Hillshade_9am'] = np.square(df_train1['Hillshade_9am'])


# In[48]:


# Plot again after square transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_9am'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_9am'], plot=plt)


# In[49]:


# Hillshade_Noon
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)


# In[50]:


df_train1.head()


# In[51]:


df_train1['Hillshade_Noon'] = np.square(df_train1['Hillshade_Noon'])


# In[52]:


# Hillshade_Noon
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)


# In[53]:


plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Fire_Points'], fit=stats.norm)
plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Fire_Points'],plot=plt)


# In[54]:


df_train1['Horizontal_Distance_To_Fire_Points'] = np.sqrt(df_train1['Horizontal_Distance_To_Fire_Points'])


# In[55]:


# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Fire_Points'], fit=stats.norm)
plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Fire_Points'],plot=plt)


# In[56]:


#Vertical_Distance_To_Hydrology
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Vertical_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Vertical_Distance_To_Hydrology'], plot=plt)


# In[57]:


plt.figure(figsize=(8,6))
sns.distplot(df_train1['Vertical_Distance_To_Hydrology'], fit=stats.norm)
plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Vertical_Distance_To_Hydrology'],plot=plt)


# In[58]:


# performing same transformation in test dataset 
df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Fire_Points'        ,'Horizontal_Distance_To_Roadways']] = np.sqrt(df_test1[['Horizontal_Distance_To_Hydrology',        'Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Roadways']])


# In[59]:


df_test1[['Hillshade_9am','Hillshade_Noon']] = np.square(df_test1[['Hillshade_9am','Hillshade_Noon']])


# In[60]:


from sklearn.preprocessing import StandardScaler


# In[61]:


# Taking only non-categorical values
Size = 10
X_temp = df_train.iloc[:,:Size]
X_test_temp = df_test.iloc[:,:Size]
X_temp1 = df_train1.iloc[:,:Size]
X_test_temp1 = df_test1.iloc[:,:Size]
a = df_train.iloc[:,:Size]


X_temp1 = StandardScaler().fit_transform(X_temp1)
X_test_temp1 = StandardScaler().fit_transform(X_test_temp1)


# In[62]:


r,c = df_train.shape
print(df_train.shape)


# In[63]:


r,c = df_train.shape
X_train = np.concatenate((X_temp,df_train.iloc[:,Size:c-1]),axis=1)
X_train1 = np.concatenate((X_temp1, df_train1.iloc[:,Size:c-1]), axis=1) # to be used for SVM
y_train = df_train.Cover_Type.values


# In[64]:


from sklearn import svm as svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[65]:


# Setting parameters
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train1,y_train,test_size=0.2, random_state=123)
svm_para = [{'kernel':['rbf'],'C': [1,10,100,100]}]


# In[66]:


# Parameters optimized using the code in above cell
C_opt = 10 # reasonable option
clf = svm.SVC(C=C_opt,kernel='rbf')
clf.fit(X_train1,y_train)


# In[67]:


clf.score(X_train1,y_train)


# In[68]:


#y_pred = clf.predict(X_test1)


# In[69]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


# In[70]:


# setting parameters
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train,y_train,test_size= 0.3, random_state=0)
etc_para = [{'n_estimators':[20,30,100], 'max_depth':[5,10,15], 'max_features':[0.1,0.2,0.3]}] 


# In[71]:


#ETC = GridSearchCV(ExtraTreesClassifier(),param_grid=etc_para, cv=3, n_jobs=-1)
#ETC.fit(x_data, y_data)
#ETC.best_params_
#ETC.grid_scores_


# In[72]:


# setting parameters
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train,y_train,test_size= 0.3, random_state=0)
etc_para = [{'n_estimators':100, 'max_depth':15, 'max_features':0.3}] 
# Default number of features is sqrt(n)
# Default number of min_samples_leaf is 1


# In[73]:


# Classification Report
Y_pred = ETC.predict(x_test_data)
target = ['class1', 'class2','class3','class4','class5','class6','class7' ]
print (classification_report(y_test_data, Y_pred, target_names=target))


# In[74]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(model,title, X, y,n_jobs = 1, ylim = None, cv = None,train_sizes = np.linspace(0.1, 1, 5)):
    
    # Figrue parameters
    plt.figure(figsize=(10,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    
    train_sizes, train_score, test_score = learning_curve(model, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)

    # Calculate mean and std
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std,                    alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std,                    alpha = 0.1, color = 'g')
    
    plt.plot(train_sizes, train_score_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_score_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc = "best")
    return plt


# In[75]:


# 'max_features': 0.3, 'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf: 1'
etc = ExtraTreesClassifier(bootstrap=True, oob_score=True, n_estimators=100, max_depth=10, max_features=0.3,                            min_samples_leaf=1)

etc.fit(X_train, y_train)
# yy_pred = etc.predict(X_test)
etc.score(X_train, y_train)


# In[76]:


# Plotting learning curve
title = 'Learning Curve (ExtraTreeClassifier)'
# cross validation with 50 iterations to have a smoother curve
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
model = etc
plot_learning_curve(model,title,X_train, y_train, n_jobs=-1,ylim=None,cv=cv)
plt.show()

