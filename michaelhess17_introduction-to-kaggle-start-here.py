#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
Image("../input/images/welcome.png")


# In[2]:


# Import the necessary packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Magic function that elimates the necessity of using plt.show() after plotting with matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Importing our data using pandas
df = pd.read_csv('../input/qclkaggle/train.csv') # df is now our Pandas DataFrame!
df.head() # print the first 5 rows of df


# In[4]:


print(df.shape, '\n')
print(df.columns.to_list()) # returns a list of the column names


# In[5]:


print(df.describe()) # gives a summary of statistics for each column


# In[6]:


print(df.info()) # Returns information about data types and missing entries


# In[7]:


(df.Survived == df['Survived']).head() # Both of these are methods for selecting one column


# In[8]:


# To select more than one column, we must pass a list of the columns we want!
cols = ['PassengerId', 'Sex', 'Parch']
df[cols].head()


# In[9]:


# `pairplot()` may become very slow with the SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")


# In[10]:


# First step exploratory data analysis. Shows interactions of all features.
sns.pairplot(df)


# In[11]:


# Countplot will give a barchart of the frequencies of nominal data
sns.countplot('Pclass', data=df)
# Try changing Pclass to PassengerId (a non-nominal feature) to see what happens!


# In[12]:


# Using the "hue" parameter allows us to visualize interactions between two features
sns.countplot('Sex', hue='Survived', data=df)


# In[13]:


# We can grab a subset of the data by using logical indexing, loc, or iloc
df[df.Survived == 0].head()


# In[14]:


# loc and iloc methods use square brackets, not parentheses!
# loc searches for a specified row in the index, then returns the value of the given column, or returns all values for that row if no column given
print(df.loc[0])
print('\n')
print(df.loc[0, 'Name'])


# In[15]:


df.index = [f'Passenger_{a}' for a in range(len(df))]
print(df.loc['Passenger_0'])


# In[16]:


df.head()


# In[17]:


df.index = list(range(len(df)))


# In[18]:


df.iloc[:50, :].head()


# In[19]:


print(df.iloc[:50, :].shape)


# In[20]:


print(df[df.Survived == 1].Age.mean())
print(df[df.Survived == 0].Age.mean())
df[df.Survived == 1].Age.plot.hist()


# In[21]:


print(df[df.Survived == 1]['Age'].mean())
print(df[df.Survived == 0]['Age'].mean())
df[df.Survived == 1]['Age'].plot.hist()


# In[22]:


# using the .plot.PLOT_TYPE() method works on pandas DataFrames and allows for quick, easy plotting
df[df.Survived == 0].Age.plot.hist(bins=30, color='orange')
df[df.Survived == 1].Age.plot.hist(bins=30, color='blue', alpha=0.6)


# In[23]:


# Whereas a countplot will tell you HOW MANY of each feature interactions there are, a barplot will tell you what the AVERAGE interaction is
sns.barplot('Sex','Survived',hue='Pclass',data=df)


# In[24]:


# using the .corr() method on pandas DataFrames returns on array of correlations between variables
print(df.corr())
sns.heatmap(df.corr())


# In[25]:


# as the .astype() method to change data storage types in dataframes for more efficient storage
# we will drop the target variable, and features that are not too useful for prediction
target = df['Survived'].astype('int16')
df = df.drop(['Ticket','Survived','PassengerId','Cabin'], axis=1)
target.head()


# In[26]:


# Hardly any machine learning algorithm can handle missing values (Go RandomForest!!) so we have to do something about these entries
df.isnull().sum()


# In[27]:


# Figure out which Embarked value is most common
sns.countplot('Embarked', data=df)


# In[28]:


df['Age'] = df['Age'].fillna(np.nanmedian(df['Age'])) # Impute missing values with average age
df['Embarked'] = df['Embarked'].fillna('S') # Impute missing values with most common value for Embarked column


# In[29]:


df['Name'] = df['Name'].str.len() # Replace the actual name with the length of the name. Turns out to be a good predictor!
df['FamilySize'] = df['SibSp'] + df['Parch'] # SibSp = Number of siblings/spouses and Parch = number of parents/children
df['isAlone'] = [1 if p == 0 else 0 for p in df['FamilySize']]
df['Embarked'] = df['Embarked'].astype('str')
df.head()


# In[30]:


from sklearn import preprocessing
# We can use question marks to get information about modules in jupyter! Double question marks takes us right to the source code.
get_ipython().run_line_magic('pinfo2', 'preprocessing.label')


# In[31]:


# Using LabelEncoder on numerical data
encoder = preprocessing.LabelEncoder()
Pclass = df['Pclass'].astype('str')
print(Pclass.head(10))
print(encoder.fit_transform(Pclass)[:10])


# In[32]:


# Using LabelEncoder on alphabetical data
encoder = preprocessing.LabelEncoder()
Embarked = df['Embarked'].astype('str')
print(Embarked.head(10))
print(encoder.fit_transform(Embarked)[:10])


# In[33]:


# Convert categorical variables to numerical
cat_names = ['Sex','Embarked']
col_names = df.columns.to_list()
col_names = [i for i in col_names if i not in cat_names]
encoders = []
scalers = []
for i in cat_names:
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(df[i].astype('str'))
    df[i] = encoder.transform(df[i].astype('str'))
    encoders.append(encoder)
for i in col_names:
    scaler = preprocessing.StandardScaler()
    scaler = scaler.fit(np.array(df[i]).reshape(-1,1))
    df[i] = scaler.transform(np.array(df[i]).reshape(-1,1))
    scalers.append(scaler)
df.head()


# In[34]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


# In[35]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('PCT', Perceptron()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier(n_estimators=300, n_jobs=-1)))
models.append(('RF', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)))


# In[36]:


import time
results = []
names = []
times = []
scoring = 'roc_auc'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    time0 = time.time()
    cv_results = model_selection.cross_val_score(model, df, target, cv=kfold, scoring=scoring)
    time1 = time.time()
    results.append(cv_results)
    names.append(name)
    times.append(time1-time0)
    msg = f"{name} / {scoring}: {round(cv_results.mean(),3)} / StDev: {round(cv_results.std(),4)} / Time: {round(time1-time0,4)}"
    print(msg)


# In[37]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(121)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.xlabel('Model')
ax1 = fig.add_subplot(122)
plt.bar(names,times)
plt.ylabel('Training time')
plt.xlabel('Model')
fig = plt.gcf()
fig.set_size_inches(16, 9)
plt.show()


# In[38]:



# max_depth = [6, 8, 10, 12]
# n_estimators = [100, 250]
# subsample = np.linspace(0,1,5)
# colsample_bytree = np.linspace(0,1,5)
# models = []
# for depth in max_depth:
#     for sub in subsample:
#         for col in colsample_bytree:
#             for n in n_estimators:
#                 models.append(XGBClassifier(n_estimators=n, max_depth=depth, subsample=sub, colsample_bytree=col))
# results = []
# scoring = 'accuracy'
# for model in models:
#     kfold = model_selection.KFold(n_splits=5, random_state=42)
#     cv_results = model_selection.cross_val_score(model, df, target, cv=kfold, scoring=scoring)
#     results.append(cv_results.mean())
# best = models[np.argmax(results)]
# print(best)


# In[39]:


# Now we have to load our test data in and apply the same transformations to it as we did our training data
test_df = pd.read_csv('../input/qclkaggle/test.csv')
test_df.drop(['PassengerId', 'Cabin', 'Ticket'],axis=1, inplace=True)
test_df['Name'] = test_df['Name'].str.len() # Replace the actual name with the length of the name. Turns out to be a good predictor!
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] # SibSp = Number of siblings/spouses and Parch = number of parents/children
test_df['isAlone'] = [1 if p == 0 else 0 for p in test_df['FamilySize']]
test_df['Age'] = test_df['Age'].fillna(np.nanmedian(df['Age'])) # Impute missing values with average age
test_df['Embarked'] = test_df['Embarked'].fillna('S') # Impute missing values with most common value for Embarked column
for k, i in enumerate(cat_names):
    test_df[i] = encoders[k].transform(test_df[i].astype('str'))
for k, i in enumerate(col_names):
    test_df[i] = scalers[k].transform(np.array(test_df[i]).reshape(-1,1))
test_df.head()


# In[40]:


# define a new model based on the best performance from our grid search
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=12,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.5, verbosity=1)
model.fit(df, target)
predictions = model.predict(test_df)
print(model.score(df, target))


# In[41]:


sub = pd.read_csv('../input/qclkaggle/sample_submission.csv')
sub.columns = ['Id', 'Category']
sub.Category = predictions
sub.to_csv('first_predictions.csv', index=False)


# In[ ]:





# In[ ]:




