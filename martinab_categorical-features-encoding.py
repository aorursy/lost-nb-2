#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing numpy (linear algebra) and pandas (data processing): 
import numpy as np 
import pandas as pd 

# Imports for plotting:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import matplotlib.ticker as ticker


# In[2]:


# Explore what's in the cat-in-the-dat folder:
print(os.listdir("../input/cat-in-the-dat"))


# In[3]:


# Read train, test and sample_submission data:
train_df = pd.read_csv("../input/cat-in-the-dat/train.csv")
test_df = pd.read_csv("../input/cat-in-the-dat/test.csv")
submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")


# In[4]:


submission.head()


# In[5]:


# Shape of the train and testdataset:
print(train_df.shape)


# In[6]:


# To display first 5 rows of the train_df:
train_df.head()


# In[7]:


# Print the names of all columns in train DataFrame:
print(train_df.columns.values)


# In[8]:


# Are there any missing values in train_df?
# train_df.apply(axis=0, func=lambda x : any(pd.isnull(x)))


# In[9]:


# Function to describe variables
def desc(df):
    summ = pd.DataFrame(df.dtypes,columns=['Data_Types'])
    summ = summ.reset_index()
    summ['Columns'] = summ['index']
    summ = summ[['Columns','Data_Types']]
    summ['Missing'] = df.isnull().sum().values    
    summ['Uniques'] = df.nunique().values
    return summ

# Function to analyse missing values
def nulls_report(df):
    nulls = df.isnull().sum()
    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)
    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])
    return nulls_report


# In[10]:


# Use desc function to describe test data:
desc(train_df)


# In[11]:


# Bar chart of frequency of digit occurance in our train dataset:
total = float(len(train_df))

plt.figure(figsize=(16,4))
ax = sns.countplot(x = 'target', data=train_df,  palette = 'rocket_r')

# Make twin axis
ax2=ax.twinx()
ax2.set_ylabel('Frequency [%]')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height*100/total),
           # '{0:.0%}'.format(height/total),
            ha="center") 


# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the Frequency [%] range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,300000)

# And use a MultipleLocator to ensure a tick spacing of 10
ax.yaxis.set_major_locator(ticker.MultipleLocator(25000))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Turn the grid on ax2 off, otherwise the gridlines will cut through percentages %:
ax.grid(False)
ax2.grid(False)   
    
plt.title('Target Distribution')
plt.show()


# In[12]:


print(train_df['target'].value_counts())


# In[13]:


# Define bin list:
bin = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']


# In[14]:


# Bar charts for binary features, split according to the target:
for i in bin:
    plt.figure(figsize=(16,4))
    ax = sns.countplot(x=i, 
                       hue="target", 
                       palette= 'ocean_r',
                       data=train_df
                       )
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}'.format(height*100/total),
                #'{0:.0%}'.format(height/total),
                ha="center") 
       
        ax.set_ylim(0,200000)
        ax.grid(False)

        plt.title('Target Distribution')
plt.show()


# In[15]:


# Define nom as:
nom = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']


# In[16]:


# Bar charts for nominal features, split according to the target:
for i in nom[0:5]:
    plt.figure(figsize=(16,4))
    ax = sns.countplot(x=i, 
                       hue="target", 
                       palette= 'gist_heat_r',
                       data=train_df
                       )
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height*100/total),
                #'{0:.0%}'.format(height/total),
                ha="center") 
       
        ax.set_ylim(0,100000)
        ax.grid(False)

        plt.title('Target Distribution')
plt.show()  


# In[17]:


# Create a crosstab with nom_1 and target:
print('Crosstab for numerical target distribution in nom_1:')

pd.crosstab([train_df.target], 
            [train_df.nom_1],
             margins=True).style.background_gradient(cmap='autumn_r')


# In[18]:


# Create a crosstab with nom_2 and target:
print('Crosstab for numerical target distribution in nom_2:')

pd.crosstab([train_df.target], 
            [train_df.nom_2],
             margins=True).style.background_gradient(cmap='autumn_r')


# In[19]:


# Create a crosstab with nom_3 and target:
print('Crosstab for numerical target distribution in nom_3:')

pd.crosstab([train_df.target], 
            [train_df.nom_3],
             margins=True).style.background_gradient(cmap='autumn_r')


# In[20]:


ord = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']


# In[21]:


# Bar charts for ordinal features, split according to the target:

for i in ord[0:3]:
    plt.figure(figsize=(16,4))
    ax = sns.countplot(x=i, 
                       hue="target", 
                       palette= 'winter_r',
                       data=train_df
                       )
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.1f}%'.format(height*100/total),
                #'{0:.0%}'.format(height/total),
                ha="center") 
       
        ax.set_ylim(0,150000)
        ax.grid(False)

        plt.title('Target Distribution')
plt.show()


# In[22]:


for i in ord[3:5]:
    plt.figure(figsize=(16,4))
    ax = sns.countplot(x=i, 
                       hue="target", 
                       palette= 'winter_r',
                       data=train_df
                       )
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                #'{:1.1f}%'.format(height*100/total),
                '{0:.0%}'.format(height/total),
                ha="center") 
       
        ax.set_ylim(0,35000)
        ax.grid(False)

        plt.title('Target Distribution')
plt.show()


# In[23]:


# Number of unique values in ord_5:
print('Number of unique values for ord_5: ' + str(train_df['ord_5'].nunique()))


# In[24]:


print('Unique values of day:',train_df.day.unique())
print('Unique values of month:',train_df.month.unique())


# In[25]:


cyc = ['day', 'month']


for i in cyc:
    plt.figure(figsize=(16,4))
    ax = sns.countplot(x=i, 
                       hue="target", 
                       palette= 'cool_r',
                       data=train_df
                       )
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                #'{:1.1f}%'.format(height*100/total),
                '{0:.0%}'.format(height/total),
                ha="center") 
       
        ax.set_ylim(0,60000)
        ax.grid(False)

        plt.title('Target Distribution')
plt.show()      


# In[26]:


# Assign output target to the following variable:
target = train_df['target']


# In[27]:


# Merge train and test data into tetra_df and drop target and id column:
tetra_df = train_df.append(test_df, ignore_index = True, sort = 'True')
tetra_df = tetra_df.drop(['target', 'id'], axis = 1)


# In[28]:


# Check if merge worked (must have 500,000 entries):
tetra_df.shape


# In[29]:


# Create indexes to separate data later:
train_df_idx = len(train_df)
test_df_idx = len(tetra_df) - len(test_df)


# In[30]:


# Convert T, F in bin_3 to binary values (0,1):
tetra_df['bin_3'] = tetra_df['bin_3'].map({'T':1, 'F':0})

# Similarly convert Y, N in bin_4 to binary values:
tetra_df['bin_4'] = tetra_df['bin_4'].map({'Y':1, 'N':0})


# In[31]:


# Check the outcome:
tetra_df[bin].head()


# In[32]:


# One hot encoding for column : nom_0 to nom_4
tetra_df = pd.get_dummies(tetra_df, columns = nom[0:5],
                        prefix = nom[0:5], 
                        drop_first = True)


# In[33]:


# Encoding hex features
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features_hex = nom[5:]

for col in features_hex:
    labelencoder.fit(tetra_df[col])
    tetra_df[col] = labelencoder.transform(tetra_df[col])


# In[34]:


#tetra_df[ord].head()


# In[35]:


# Convert ord_1 by dictionary mapping as follows:
tetra_df['ord_1'] = tetra_df['ord_1'].map({
    'Novice': 0,
    'Contributor': 1,
    'Master': 2,
    'Expert' : 3,
    'Grandmaster': 4
})

# Similarly convert ord_2:
tetra_df['ord_2'] = tetra_df['ord_2'].map({
    'Freezing': 0,
    'Cold': 1,
    'Warm': 2,
    'Hot' : 3,
    'Boiling Hot': 4,
    'Lava Hot' : 5
})


# In[36]:


# Change type of ord_3 to category, create a dictionary alph that orders letters alphabetically:
tetra_df['ord_3'] = tetra_df['ord_3'].astype('category')
alph = dict(zip(tetra_df['ord_3'],tetra_df['ord_3'].cat.codes))
# Map alphord to ord_3 and change type of ord_3 to integer:
tetra_df['ord_3'] = tetra_df['ord_3'].map(alph)
tetra_df['ord_3'] = tetra_df['ord_3'].astype(int)

# Similarly change ord_4:
tetra_df['ord_4'] = tetra_df['ord_4'].astype('category')
alph1 = dict(zip(tetra_df['ord_4'],tetra_df['ord_4'].cat.codes))
tetra_df['ord_4'] = tetra_df['ord_4'].map(alph1)
tetra_df['ord_4'] = tetra_df['ord_4'].astype(int)


# In[37]:


# Create sorted list of ord_5 values (ordered alphabetically):
ordli = sorted(list(set(tetra_df['ord_5'].values)))

# Create mapping dictionary alph2 for ord_5
alph2 = dict(zip(ordli, range(len(ordli))))  

# Map alph2 dictionary to ord_5
tetra_df['ord_5'] = tetra_df['ord_5'].map(alph2)


# In[38]:


# Cyclical encoding for day:
tetra_df['day_sin'] = np.sin(2 * np.pi * tetra_df['day']/7.0)
tetra_df['day_cos'] = np.cos(2 * np.pi * tetra_df['day']/7.0)

# Cyclical encoding for month:
tetra_df['month_sin'] = np.sin(2 * np.pi * tetra_df['month']/12.0)
tetra_df['month_cos'] = np.cos(2 * np.pi * tetra_df['month']/12.0)


# In[39]:


# Show that Encoded values are now placed on the circle with radius 1 and origing at [0,0]:
x = tetra_df.day_sin
y = tetra_df.day_cos

tetra_df.sample(5000).plot.scatter('day_sin','day_cos').set_aspect('equal')
tetra_df.sample(5000).plot.scatter('month_sin','month_cos').set_aspect('equal')


# In[40]:


tetra_df = tetra_df.drop(['day', 'month'], axis = 1)


# In[41]:


# Print the names of all columns in tetra_df DataFrame:
 print(tetra_df.columns.values)


# In[42]:


#from sklearn.preprocessing import MinMaxScaler
#min_max_scaler = MinMaxScaler()

# x returns a numpy array
#x = tetra_df.values 


#x_scaled = min_max_scaler.fit_transform(x)
#tetra_df = pd.DataFrame(x_scaled)


# In[43]:


#tetra_df.describe()


# In[44]:


# Creating training and testing data:
training = tetra_df[ : train_df_idx]
testing = tetra_df[test_df_idx :]


# In[45]:


# For splitting data we will be using train_test_split from sklearn:
from sklearn.model_selection import train_test_split


# In[46]:


X = training
y = target


# In[47]:


# Splitting the training data into test and train, we are testing on 0.20 = 20% of dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=13)


# In[48]:


from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# In[49]:


xgb = XGBClassifier(objective= 'binary:logistic'
                    , learning_rate=0.7
                    , max_depth=3
                    , n_estimators=250
                    , scale_pos_weight=2
                    , random_state=42
                    , colsample_bytree=0.5
                    )
    
xgb.fit(X_train, y_train)   


# In[50]:


y_predict = xgb.predict(X_test)
print(classification_report(y_test,y_predict))


# In[51]:


# Confusion matrix cm:
cm = confusion_matrix(y_test,y_predict)
cm


# In[52]:


# Quick overview of our confusion matrix:
sns.heatmap(cm, annot = True, square = True, fmt='g')


# In[53]:


prediction = xgb.predict(testing)


# In[54]:


# Combine ImageID and Label into one DataFrame:
final_result = pd.DataFrame({'target': prediction, 'id': submission.id})
final_result = final_result[['id', 'target']]

# Downloading final_result dataset as digit_output.csv:
final_result.to_csv('cat_output.csv', index = False)

