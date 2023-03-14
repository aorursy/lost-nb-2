#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pyplot as plt
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[2]:


path='/kaggle/input/petfinder-sentiment/'
path1='/kaggle/input/petfinder-adoption-prediction/'
# t=pd.read_json(path+'train_sentiment/25a834a2e.json', orient='split')
train_meta=pd.read_csv(path+'train_meta.csv')
train_meta
test_meta=pd.read_csv(path+'test_meta.csv')
test_meta
state=pd.read_csv(path1+'StateLabels.csv')
state
stateNames=[]
for x in train_meta.State:
    for j in range(len(state.StateID)):
        if x==state.StateID[j]:
            h= state.StateName[j]
            stateNames.append(h)
            break
train_meta['StateName']=stateNames

stateNames=[]
for x in test_meta.State:
    for j in range(len(state.StateID)):
        if x==state.StateID[j]:
            h= state.StateName[j]
            stateNames.append(h)
            break
test_meta['StateName']=stateNames
test_meta


# In[3]:


test_meta.info()


# In[4]:


stateAdoption=train_meta.groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption.columns=['StateName','countPetID']
stateAdoption
plt.figure(figsize=(20,11))
# plt.title("Distribution of PetID by States")
plt.xlabel('States of Malaysia')
plt.ylabel('Counted by PetID')
plt.plot(stateAdoption.StateName,stateAdoption.countPetID)
plt.legend()

plt.show()


# In[5]:


train_meta.State.isnull().count()


# In[6]:


stateAdoption0=train_meta[train_meta.AdoptionSpeed==0].groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption0.columns=['State','AdoptionSpeed0']
stateAdoption0=stateAdoption0.sort_values(by=['State'])

stateAdoption1=train_meta[train_meta.AdoptionSpeed==1].groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption1.columns=['State','AdoptionSpeed1']
stateAdoption1=stateAdoption1.sort_values(by=['State'])

stateAdoption2=train_meta[train_meta.AdoptionSpeed==2].groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption2.columns=['State','AdoptionSpeed2']
stateAdoption2=stateAdoption2.sort_values(by=['State'])

stateAdoption3=train_meta[train_meta.AdoptionSpeed==3].groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption3.columns=['State','AdoptionSpeed3']
stateAdoption3=stateAdoption3.sort_values(by=['State'])

stateAdoption4=train_meta[train_meta.AdoptionSpeed==4].groupby('StateName').agg({'PetID': ['count']}).reset_index()
stateAdoption4.columns=['State','AdoptionSpeed4']
stateAdoption4=stateAdoption4.sort_values(by=['State'])

# print('AdoptionSpeed==0')
# stateAdoption
fig = plt.figure(figsize=(15,7))
# plt.set_title('Distribution of PetID by States')
plt.xlabel('States of Malaysia')
plt.ylabel('Counted by PetID')
plt.scatter(stateAdoption0.State,stateAdoption0.AdoptionSpeed0,c='g',label='AdoptionSpeed=0',s=200)
plt.scatter(stateAdoption1.State,stateAdoption1.AdoptionSpeed1,c='b',label='AdoptionSpeed=1',s=150)
plt.scatter(stateAdoption2.State,stateAdoption2.AdoptionSpeed2,c='yellow',label='AdoptionSpeed=2',s=100)
plt.scatter(stateAdoption3.State,stateAdoption3.AdoptionSpeed3,c='pink',label='AdoptionSpeed=3',s=50)
plt.scatter(stateAdoption4.State,stateAdoption4.AdoptionSpeed4,c='black',label='AdoptionSpeed=4',s=25)
plt.legend()
plt.show()


# In[7]:


train_meta.info()


# In[8]:


# count the number of duplicate values
from collections import Counter
c = Counter(list(zip(train_meta.columns)))
c


# In[9]:


train_meta.corr()


# In[10]:


f = plt.figure(figsize=(19, 15))
plt.matshow(train_meta.corr(), fignum=f.number)
plt.xticks(range(train_meta.shape[1]), train_meta.columns, fontsize=14, rotation=45)
plt.yticks(range(train_meta.shape[1]), train_meta.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# In[11]:


train_meta.hist(figsize=(15,15))


# In[12]:


#Linear Regression?

# PCA to see the data in 2 components
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
features=["AdoptionSpeed",'Name','RescuerID','Description','PetID','StateName','RescuerID','Color1Name','Breed1Name']
x = train_meta.drop(features, axis=1).values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

pca = PCA(n_components=2)
principalComponents=pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, train_meta[['AdoptionSpeed']]], axis = 1)
finalDf


# In[13]:


plt.scatter(finalDf['AdoptionSpeed'],finalDf['principal component 1'], c = 'red',s=50)
plt.show()


# In[14]:


plt.scatter(finalDf['principal component 2']
               , finalDf['AdoptionSpeed']
               , c = 'red',s=50)
plt.show()


# In[15]:


pca.explained_variance_ratio_


# In[16]:


pca = PCA(n_components=1)
principalComponents=pca.fit_transform(train_meta.drop(features, axis=1))
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1'])
finalDf = pd.concat([principalDf, train_meta[['AdoptionSpeed']]], axis = 1)
plt.scatter(finalDf['principal component 1']
               , finalDf['AdoptionSpeed']
               , c = 'r'
               , s = 50)
plt.show()


# In[17]:


train_meta.shape


# In[18]:



features.append('Unnamed: 0')
features.append('Unnamed: 0.1')
feat=train_meta.drop(features, axis=1).columns
feat


# In[19]:


# This examples shows the use of forests of trees to evaluate the importance of features on an artificial classification task. 
# The red bars are the feature importances of the forest, along with their inter-trees variability.
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

X=train_meta.drop(features, axis=1)
Y=train_meta["AdoptionSpeed"]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], feat[indices[f]],importances[indices[f]]))

xvalues=[feat[x] for x in indices]
plt.rcParams["figure.figsize"] = (20,15)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), xvalues)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[20]:


from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

X=train_meta[['Age', 'PhotoAmt']]
Y=train_meta["AdoptionSpeed"]
# valid_size: what proportion of original data is used for valid set
train, valid, train_lbl, valid_lbl = train_test_split(
    X, Y, test_size=0.15, random_state=122)
#Normalization
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train)
# Apply transform to both the training set and the test set.
train1 = scaler.transform(train)
valid1 = scaler.transform(valid)
test1=scaler.transform(test_meta[['Age', 'PhotoAmt']])

#fit the model
model = LogisticRegression(solver = 'lbfgs')
model.fit(train1, train_lbl)

#Validating the fit
# use the model to make predictions with the test data
y_pred = model.predict(valid1)
# how did our model perform?
count_misclassified = (valid_lbl != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(valid_lbl, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
plt.rcParams["figure.figsize"] = (7,5)
# Plot the feature importances of the forest
plt.figure()
plt.scatter(valid_lbl, y_pred)
print(y_pred)


# In[ ]:





# In[21]:


features


# In[22]:


test_meta[features.remove('AdoptionSpeed')].columns


# In[23]:


from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
features=['Unnamed: 0','Unnamed: 0.1','Name','RescuerID','Description','PetID','StateName','Color1Name','Breed1Name','AdoptionSpeed']
X=train_meta.drop(features, axis=1)
Y=train_meta["AdoptionSpeed"]
# valid_size: what proportion of original data is used for valid set
train, valid, train_lbl, valid_lbl = train_test_split(
    X, Y, test_size=0.15, random_state=122)
#Normalization
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train)
# Apply transform to both the training set and the test set.
train1 = scaler.transform(train)
valid1 = scaler.transform(valid)

test_X=test_meta.drop(['Unnamed: 0','Unnamed: 0.1','Name','RescuerID','Description','PetID','StateName','Color1Name','Breed1Name'], axis=1)
# print(test_X)
scaler.fit(test_X)
test1=scaler.transform(test_X)
#fit the model
model = LogisticRegression(solver = 'lbfgs')
model.fit(train1, train_lbl)

#Validating the fit
# use the model to make predictions with the test data
y_pred = model.predict(valid1)
# how did our model perform?
count_misclassified = (valid_lbl != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(valid_lbl, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

plt.scatter(valid_lbl, y_pred)
# print(y_pred)

#submission
y_pred = model.predict(test1)
# test_X.columns
submission=pd.DataFrame(columns=['PetID','AdoptionSpeed'])
submission['PetID']=test_meta['PetID']
submission['AdoptionSpeed']=y_pred
submission.head()
submission['AdoptionSpeed'].hist()
submission.to_csv('submission.csv', index=False)


# In[24]:


submission=pd.read_csv('submission.csv')
submission


# In[25]:


#submit
submission=model.predict(test1)
submission = pd.DataFrame(data = submission
             , columns = ['AdoptionSpeed'])
submission = pd.concat([test_meta['PetID'], submission], axis = 1)
submission.to_csv('samplesubmission.csv', index=False)
submission


# In[26]:


from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

X=finalDf.drop(["AdoptionSpeed"], axis=1)
Y=finalDf["AdoptionSpeed"]
# valid_size: what proportion of original data is used for valid set
train, valid, train_lbl, valid_lbl = train_test_split(
    X, Y, test_size=0.15, random_state=122)
#fit the model
model = LogisticRegression(solver = 'lbfgs')
model.fit(train, train_lbl)

#Validating the fit
# use the model to make predictions with the test data
y_pred = model.predict(valid)
# how did our model perform?
count_misclassified = (valid_lbl != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(valid_lbl, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

plt.scatter(valid_lbl, y_pred)
print(y_pred)


# In[27]:


# Importing the required packages 
import numpy as np 
import pandas as pd 
import math
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
    
    
X=train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID','StateName'], axis=1)
Y=train_meta["AdoptionSpeed"]
X_train, X_valid, y_train, y_valid = train_test_split( 
          X, Y, test_size = 0.2, random_state = 100)

clf_gini = train_using_gini(X_train, X_valid, y_train) 
clf_entropy = train_using_entropy(X_train, X_valid, y_train) 
      
# Operational Phase 
print("Results Using Gini Index:") 
      
# Prediction using gini 
y_pred_gini = prediction(X_valid, clf_gini) 
cal_accuracy(y_valid, y_pred_gini) 
      
print("Results Using Entropy:") 
# Prediction using entropy 
y_pred_entropy = prediction(X_valid, clf_entropy) 
cal_accuracy(y_valid, y_pred_entropy) 

#Predict for the test_meta based on GINI:
# Operational Phase 
print("Results Using Gini Index for test_meta:") 
# Prediction using gini 
X_test=test_meta.drop(['Name','RescuerID','Description','PetID'], axis=1)
y_pred_gini = prediction(X_test, clf_gini) 
y_valid_gini=  prediction(X_valid, clf_gini) 
print("Results Using Entropy:") 
# Prediction using entropy 
y_pred_entropy = prediction(X_test, clf_entropy) 

print('diff between Decison Tree with gini and entropy, check their histogram')
print('GINI prediction for test_meta histogram:')
_=plt.hist(y_pred_gini, bins='auto')
plt.title("Histogram with 'auto' bins for GINI")
plt.show()

print('ENTROPY prediction for test histogram:')

#submit
submission_gini = pd.DataFrame(data = y_pred_gini
             , columns = ['AdoptionSpeed'])
submission_gini = pd.concat([test_meta['PetID'], submission_gini], axis = 1)
submission_gini.to_csv('samplesubmissionTreeGini.csv', index=False)

submission_entropy = pd.DataFrame(data = y_pred_gini
             , columns = ['AdoptionSpeed'])
submission_entropy = pd.concat([test_meta['PetID'], submission_entropy], axis = 1)
submission_entropy.to_csv('samplesubmissionTreeEntropy.csv', index=False)


# In[28]:


_=plt.hist(y_pred_entropy, bins='auto')
plt.title("Histogram with 'auto' bins for Entropy")
plt.show()


# In[29]:


print('ENTROPY prediction for test_meta histogram:')
_=plt.hist(y_pred_entropy, bins='auto')
plt.title("Histogram with 'auto' bins for ENTROPY")
plt.show()


# In[30]:


# Visualize the decision tree: gini
import graphviz
from sklearn import tree
data = tree.export_graphviz(clf_gini,out_file=None,feature_names=X_test.columns,class_names=[str(x) for x in range(5)],   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph


# In[31]:


# Decision tree classifier – Decision tree classifier is a systematic approach for multiclass classification. 
# It poses a set of questions to the dataset (related to its attributes/features). 
# The decision tree classification algorithm can be visualized on a binary tree. 
# On the root and each of the internal nodes, a question is posed and the data on 
# that node is further split into separate records that have different characteristics. 
# The leaves of the tree refer to the classes in which the dataset is split. 
# In the following code snippet, we train a decision tree classifier in scikit-learn.
# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

X=train_meta.drop(["AdoptionSpeed",'Name','RescuerID','Description','PetID'], axis=1)
Y=train_meta["AdoptionSpeed"]
X_train, X_valid, y_train, y_valid = train_test_split( 
          X, Y, test_size = 0.2, random_state = 100)
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_valid) 

# creating a confusion matrix 
cm = confusion_matrix(y_valid, dtree_predictions) 
plot_confusion_matrix(cm, classes = ['0', '1','2','3','4'],
                      title = 'Decision Tree Confusion Matrix (n=4)')
plt.savefig('cmDecisionTree.png')


# In[32]:


# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
# accuracy on X_test 
accuracy = knn.score(X_valid, y_valid) 
print(accuracy)
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
plot_confusion_matrix(cm, classes = ['0', '1','2','3','4'],
                      title = 'Decision Tree Confusion Matrix (n=4)')
plt.savefig('cmDecisionTree.png')
# X_test=test_meta.drop(['Name','RescuerID','Description','PetID'], axis=1)


# In[33]:


#Name
# Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# train_meta.Name list of unique words ratio ..> 40% are duplication in names
print("Ratio of unique Name to the sample size: ",len(set(test_meta.Name.unique()))/len(list(test_meta.Name)))

# train_meta.Name list of unique words ratio ..> 40% are duplication in names
print("Ratio of unique Description to the sample size: ",len(set(test_meta.Description.unique()))/len(list(test_meta.Description)))


# In[34]:


train_meta.Name.head(20)


# In[35]:


#Name: Maximum and minimum font size
t=''
for x in test_meta.Name:
    t=t+str(x)

text_file = open("Names.txt", "wt")
n = text_file.write(t)
text_file.close()

wordcloud = WordCloud(width=480, height=480, max_font_size=70, min_font_size=12).generate(t)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[36]:


t=''
for x in test_meta.Name:
    t=t+str(x)
wordcloud = WordCloud(width=480, height=480, max_words=20).generate(t)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Top 20 names")
plt.show()


# In[37]:


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

path='/kaggle/input/petfinder/cat.png'
catdog_mask = np.array(Image.open(path))
# catdog_mask


# In[38]:


# train_meta.AdoptionSpeed
test_meta.Type.hist()


# In[39]:


#Dog
dog=test_meta.Name[test_meta.Type==1]
ratioD=len(dog)/len(test_meta.Name)
print(ratioD)

path='/kaggle/input/petfinder/dog.png'
catdog_mask = np.array(Image.open(path))
catdog_mask

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=catdog_mask, contour_width=3, contour_color='firebrick')

text = " ".join(str(review) for review in dog)
text=set(text)
text = " ".join(str(review) for review in dog)
# nan_dog=text.find('nan')
# text_dog=text.split(' ')

print(nan_dog)
# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("dog.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[40]:


#Dog
dog=train_meta.Name[train_meta.Type==1]
ratioD=len(dog)/len(train_meta.Name)
print(ratioD)

path='/kaggle/input/petfinder/dog.png'
catdog_mask = np.array(Image.open(path))
catdog_mask

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=catdog_mask, contour_width=3, 
               contour_color='firebrick',stopwords=['nan','No Name','NaN','And','For','The'])

text = " ".join(str(review) for review in dog)
nan_dog=text.find('nan')
text_dog=text.split(' ')
print(nan_dog)
# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("dogSW.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[41]:


#Cat
cat=test_meta.Name[test_meta.Type==2]
ratioC=len(cat)/len(test_meta.Name)
print(ratioC)

path='/kaggle/input/petfinder/cat.png'
catdog_mask = np.array(Image.open(path))
catdog_mask

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=catdog_mask, contour_width=3, contour_color='firebrick')

text = " ".join(str(review) for review in cat)
nan_cat=text.find('nan')
text_cat=text.split(' ')
# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("cat.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[42]:


#Cat
cat=train_meta.Name[train_meta.Type==2]
ratioC=len(cat)/len(train_meta.Name)
print(ratioC)

path='/kaggle/input/petfinder/cat.png'
catdog_mask = np.array(Image.open(path))
catdog_mask

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=catdog_mask, 
               contour_width=3, contour_color='firebrick',stopwords=['nan','No Name','NaN','And','For','The'])

text = " ".join(str(review) for review in cat)
nan_cat=text.find('nan')
text_cat=text.split(' ')
# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("catSW.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[43]:


train_meta.head()


# In[44]:


obj_df = train_meta.select_dtypes(include=['object']).copy()
obj_df.head()


# In[45]:


print(obj_df.shape)
for i in obj_df.columns:
    t=set(obj_df[i].unique())
    print('================================')
    print(i,' has unique values= ', obj_df[i].unique(), ' : ',len(t) )


# In[46]:


obj_df.info()


# In[47]:


t=obj_df["StateName"].value_counts()
print(t)
t.plot(figsize =(10,10))
#use it for label encoding
print(t.index)


# In[48]:


obj_df[obj_df.isnull().any(axis=1)]


# In[49]:


#Hot One encoding for StateName
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
obj_df["StateNameLabel"] = lb_make.fit(t)

obj_df[["make", "make_code"]].head(11)


# In[ ]:





# In[50]:


path='/kaggle/input/petfinder-adoption-prediction/'
# t=pd.read_json(path+'train_sentiment/25a834a2e.json', orient='split')
train_meta=pd.read_csv(path+'train/train.csv')
train_meta
test_meta=pd.read_csv(path+'test/test.csv')
test_meta
state=pd.read_csv(path+'StateLabels.csv')
state
color=pd.read_csv(path+'ColorLabels.csv')
color
breed=pd.read_csv(path+'BreedLabels.csv')
breed
# -----------STATE NAME--------------------
stateNames=[]
for x in train_meta.State:
    for j in range(len(state.StateID)):
        if x==state.StateID[j]:
            h= state.StateName[j]
            stateNames.append(h)
            break
train_meta['StateName']=stateNames

stateNames=[]
for x in test_meta.State:
    for j in range(len(state.StateID)):
        if x==state.StateID[j]:
            h= state.StateName[j]
            stateNames.append(h)
            break
test_meta['StateName']=stateNames

# -----------COLOR NAME--------------------
# ------------------TRAIN for ColorName ---------------
colorNames=[]
for x in train_meta.Color1:
    for j in range(len(color.ColorID)):
        if x==color.ColorID[j]:
            h= color.ColorName[j]
            colorNames.append(h)
            break
train_meta['Color1Name']=colorNames
# ------------------TEST for ColorName ---------------
colorNames=[]
for x in test_meta.Color1:
    for j in range(len(color.ColorID)):
        if x==color.ColorID[j]:
            h= color.ColorName[j]
            colorNames.append(h)
            break
test_meta['Color1Name']=colorNames

# -----------BREED NAME--------------------
# ------------------TRAIN for BreedName ---------------
breedNames=[]
for x in train_meta.Breed1:
    h=''
    for j in range(len(breed.BreedID)):
        if x==breed.BreedID[j]:
            h= breed.BreedName[j]
            breedNames.append(h)
            break
    if h=='':
        breedNames.append('nan')
#         if x.Type==1:
#             breedNames.append('dog')
#         else:
#             breedNames.append('cat')
print('breedNames=',len(breedNames))
train_meta['Breed1Name']=breedNames
# ------------------TEST for BreedName ---------------
breedNames=[]
for i in test_meta.Breed1:
    x=test_meta.Breed1[i]
    h=''
    for j in range(len(breed.BreedID)):
        if x==breed.BreedID[j]:
            h= breed.BreedName[j]
            breedNames.append(h)
            break
    if h=='':
        breedNames.append('nan')
#         if x.Type==1:
#             breedNames.append('dog')
#         else:
#             breedNames.append('cat')
test_meta['Breed1Name']=breedNames


print('**************************************')
print('Train data')
print('**************************************')
print('---------TRAIN dataset-------------')
print(train_meta.info())
print(train_meta.head())
print('---------TEST dataset-------------')
print(test_meta.info())
print(test_meta.head())


# In[51]:


train_meta['Breed1Name'][(train_meta.Type == 1) & (train_meta.Breed1Name == 'nan')] = "dog"
train_meta['Breed1Name'][(train_meta.Type == 2) & (train_meta.Breed1Name == 'nan')] = "cat"
train_meta[train_meta.Breed1Name=='nan']


# In[52]:


test_meta['Breed1Name'][(test_meta.Type == 1) & (test_meta.Breed1Name == 'nan')] = "dog"
test_meta['Breed1Name'][(test_meta.Type == 2) & (test_meta.Breed1Name == 'nan')] = "cat"
test_meta[test_meta.Breed1Name=='nan']


# In[53]:


train_meta.info()


# In[54]:


#24/12/2019
train.to_csv('train_meta.csv')#: including name of states, color1, breed1
test.to_csv('test_meta.csv')
# # read from train_meta.csv and test_meta.csv
# # train=pd.read_csv('train_meta.csv')
# # train
# test=pd.read_csv('test_meta.csv')
# test


# In[55]:


breed.groupby(['Type'])['BreedID'].count()


# In[56]:


# # FOR TRAIN dataset
# # fillna for Description column
# train['Description'][(train.Type == 1) & (train.Description.isnull())] = "dog"
# train['Description'][(train.Type == 2) & (train.Description.isnull())] = "cat"
# train[train.Description.isnull()]
# # fillna for Name column
# train['Name'][(train.Type == 1) & (train.Name.isnull())] = "dog"
# train['Name'][(train.Type == 2) & (train.Name.isnull())] = "cat"
# train[train.Name.isnull()]

# # FOR TEST dataset
# # fillna for Description column
# test['Description'][(test.Type == 1) & (test.Description.isnull())] = "dog"
# test['Description'][(test.Type == 2) & (test.Description.isnull())] = "cat"
# test[test.Description.isnull()]
# # fillna for Name column
# test['Name'][(test.Type == 1) & (test.Name.isnull())] = "dog"
# test['Name'][(test.Type == 2) & (test.Name.isnull())] = "cat"
# test[test.Name.isnull()]


# In[57]:


# print('---------TRAIN dataset-------------')
# print(train.info())
# print(train.head())
# print('---------TEST dataset-------------')
# print(test.info())
# print(test.head())


# In[58]:


# print('train meta dataset len PetID= ',len(train.PetID.unique()))
# print('**************************')
# print(train.shape)
# print(test.shape)
# print('**************************')
# print('train_dfs_metadata len PetID= ',len(train_dfs_metadata.PetID.unique()))
# print(train_dfs_metadata.shape)
# print(train_dfs_metadata.columns)
# print(test_dfs_metadata.shape)
# print('**************************')
# print('train_dsf_sentiment len PetID= ',len(train_dsf_sentiment.PetID.unique()))
# print(train_dsf_sentiment.shape)
# print(train_dsf_sentiment.columns)
# print(test_dfs_sentiment.shape)


# In[59]:


# %%timeit
# t=pd.Series(list(set(train.PetID).intersection(set(train_dsf_sentiment.PetID))))
# t=train.PetID[train.PetID.isin(train_dsf_sentiment.PetID)]


# In[60]:


# train=train_meta[train_meta.PhotoAmt!=0].set_index('PetID')
# print(train)
# imagesAtt=imagesAtt.set_index('PetID')
# print(imagesAtt.columns)
# print(imagesAtt.head())
train=train.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
train=train.drop(['Unnamed: 0.1'],axis=1)
train['metadata_annots_score']=imagesAtt['metadata_annots_score']
train['metadata_color_score']=imagesAtt['metadata_color_score']
train['metadata_color_pixelfrac']=imagesAtt['metadata_color_pixelfrac']
train['metadata_crop_conf']=imagesAtt['metadata_crop_conf']
print(train)
train.to_csv('trainWithImageDatameda.csv')


# In[61]:


train_meta.shape[0]-imagesAtt.shape[0]


# In[62]:


# t=train.PetID[train.PetID.isin(train_dsf_sentiment.PetID)]
# columns in train.PetID that are not in train_dsf_sentiment.PetID
# import numpy as np
# c1 = np.setdiff1d(train.PetID, train_dsf_sentiment.PetID)
# print(len(c1))
# #save c1 to file: PetID not have sentiment data
# np.savetxt("PetID_no_sentiment.csv", c1, delimiter=",", fmt='%s')

# t=train.PetID[train.PetID.isin(train_sentiment.PetID)]
# columns in train.PetID that are not in train_sentiment.PetID
# import numpy as np
# c1 = np.setdiff1d(train.PetID, train_sentiment.PetID)
# print(len(c1))
# #save c1 to file: PetID not have sentiment data
# np.savetxt("PetID_no_sentiment.csv", c1, delimiter=",", fmt='%s')


# In[63]:


import pandas as pd
PetID_no_sentiment = pd.read_csv("../input/petfinder-sentiment/PetID_no_sentiment.csv")
test_dfs_metadata = pd.read_csv("../input/petfinder-sentiment/test_dfs_metadata.csv")
test_dfs_sentiment = pd.read_csv("../input/petfinder-sentiment/test_dfs_sentiment.csv")
test_meta = pd.read_csv("../input/petfinder-sentiment/test_meta.csv")
train_dfs_metadata = pd.read_csv("../input/petfinder-sentiment/train_dfs_metadata.csv")
train_dsf_sentiment = pd.read_csv("../input/petfinder-sentiment/train_dsf_sentiment.csv")
train_meta = pd.read_csv("../input/petfinder-sentiment/train_meta.csv")


# In[64]:


k=train_meta.groupby(['RescuerID'])['AdoptionSpeed'].count().reset_index(name='meanAdoptionSpeed')
print(k)
k.meanAdoptionSpeed.hist()
print(k.meanAdoptionSpeed.min())
print(k.meanAdoptionSpeed.max())
print(k.meanAdoptionSpeed.mean())
maxRescuer=k[k.meanAdoptionSpeed==k.meanAdoptionSpeed.max()].RescuerID
print(maxRescuer)
maxR=train_meta[train_meta.RescuerID=='fa90fa5b1ee11c86938398b60abc32cb']
pd.set_option('display.max_rows', maxR.shape[0]+1)
print(maxR.groupby(['Type','Breed1Name','Age'])['PetID'].count())


# In[65]:


k=train_meta.groupby(['RescuerID'])['AdoptionSpeed'].mean().reset_index(name='meanAdoptionSpeed')
print(k)
k.meanAdoptionSpeed.hist()


# In[66]:


#There are duplication of Names...try to label encoding
print('There are duplication of Names...try to label encoding')
print(train_meta.shape[0])
print(len(train_meta.Name.unique()))
print(1-len(train_meta.Name.unique())/train_meta.shape[0])


#There are duplication of StateNames...try to label encoding
print('There are duplication of StateNames...try to label encoding')
print(train_meta.shape[0])
print(len(train_meta.StateName.unique()))
print(1-len(train_meta.StateName.unique())/train_meta.shape[0])

#There are duplication of ColorNames...try to label encoding
print('There are duplication of ColorNames...try to label encoding')
print(train_meta.shape[0])
print(len(train_meta.Color1Name.unique()))
print(1-len(train_meta.Color1Name.unique())/train_meta.shape[0])

#There are duplication of Breed1Names...try to label encoding
print('There are duplication of Breed1Names...try to label encoding')
print(train_meta.shape[0])
print(len(train_meta.Breed1Name.unique()))
print(1-len(train_meta.Breed1Name.unique())/train_meta.shape[0])

#There are duplication of RescuerID...try to label encoding
print('There are duplication of RescuerID...try to label encoding')
print(train_meta.shape[0])
print(len(train_meta.RescuerID.unique()))
print(1-len(train_meta.RescuerID.unique())/train_meta.shape[0])


# In[67]:


#Label encoding the Name, StateName, Color1Name, Breed1Name, RescuerID
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'Name'. 
train_meta['NameL']= label_encoder.fit_transform(train_meta['Name']) 
# Encode labels in column 'StateName'. 
train_meta['StateNameL']= label_encoder.fit_transform(train_meta['StateName']) 
# Encode labels in column 'Color1Name'. 
train_meta['Color1NameL']= label_encoder.fit_transform(train_meta['Color1Name']) 
# Encode labels in column 'Breed1Name'. 
train_meta['Breed1NameL']= label_encoder.fit_transform(train_meta['Breed1Name']) 
# Encode labels in column 'RescuerID'. 
train_meta['RescuerIDL']= label_encoder.fit_transform(train_meta['RescuerID']) 

print(train_meta.head())
print(train_meta.columns)


# In[68]:


train_meta=train_meta.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
test_meta=test_meta.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
print(train_meta.columns)
print(test_meta.columns)


# In[69]:


features=['Type', 'Age','Gender', 'MaturitySize', 'FurLength',
       'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt', 'NameL', 'StateNameL', 'Color1NameL', 'Breed1NameL', 'RescuerIDL']
print(train_meta[features].info())
train_meta.drop(['AdoptionSpeed','Unnamed: 0', 'Unnamed: 0.1'],axis=1).corrwith(train_meta['AdoptionSpeed']).plot.bar(
        figsize = (20, 10), title = "Correlation with AdoptionSpeed", fontsize = 15,
        rot = 45, grid = True)


# In[70]:


train_meta[features].corrwith(train_meta['AdoptionSpeed']).plot.bar(
        figsize = (20, 10), title = "Correlation with AdoptionSpeed", fontsize = 15,
        rot = 45, grid = True)
train_meta[features].hist(figsize=(15,15))


# In[71]:


# Basic Logistic Regression
#LOgistic regression for features
# features=['Type', 'Age','Gender', 'MaturitySize', 'FurLength',
#        'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
#        'VideoAmt', 'PhotoAmt', 'NameL', 'StateNameL', 'Color1NameL', 'Breed1NameL', 'RescuerIDL']
#split train into training and testing/validating
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(train_meta[features], train_meta.AdoptionSpeed, test_size=0.2)

#Scaling training and validating
scaler = StandardScaler()
# Apply transform to both the training set and the test set.
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

#Fit the model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_valid)
# Use score method to get accuracy of model
# accuracy is defined as: 
# (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(X_valid, y_valid)
print(score)

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_valid, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[72]:



print('**************************************')
print('Meta data')
print('**************************************')
print('---------TRAIN dataset-------------')
print(train_meta.info())
print(train_meta.head())
print('---------TEST dataset-------------')
print(test_meta.info())
print(test_meta.head())

path='../input/petfinder-sentiment/'
test_image = pd.read_csv(path+"test_dfs_metadata.csv")
test_sentiment = pd.read_csv(path+"test_dfs_sentiment.csv")
train_image = pd.read_csv(path+"train_dfs_metadata.csv")
train_sentiment = pd.read_csv(path+"train_dsf_sentiment.csv")
# train_meta, test_meta
print('**************************************')
print('Meta data of images')
print('**************************************')
print('---------Meta data of images TRAIN dataset-------------')
print(train_image.info())
print(train_image.head())
print('---------Meta data of images TEST dataset-------------')
print(test_image.info())
print(test_image.head())
print('**************************************')
print('Sentiment data')
print('**************************************')
print('---------Sentiment data TRAIN dataset-------------')
print(train_sentiment.info())
print(train_sentiment.head())
print('---------Sentiment data TEST dataset-------------')
print(test_sentiment.info())
print(test_sentiment.head())


# In[73]:


train_image.info()


# In[74]:


imagesAtt.hist(figsize=(15,15))


# In[75]:


t=train_meta.PetID[train_meta.PetID.isin(train_image.PetID)]
# columns in train.PetID that are not in train_dsf_sentiment.PetID
import numpy as np
c1 = np.setdiff1d(train_meta.PetID, train_sentiment.PetID)
print(len(c1)/train_meta.shape[0])
# #save c1 to file: PetID not have sentiment data
# np.savetxt("PetID_no_sentiment.csv", c1, delimiter=",", fmt='%s')


# In[ ]:





# In[76]:


print(train_sentiment.columns)
print(train_sentiment.head())


# In[77]:


imagesAtt=test_image.groupby(['PetID'])['metadata_annots_score','metadata_color_score',
                                         'metadata_color_pixelfrac','metadata_crop_conf',
                                         'metadata_crop_importance'].mean().reset_index()
sentimentAtt=test_sentiment.groupby(['PetID'])['sentiment_magnitude_sum',
       'sentiment_score_sum', 'sentiment_magnitude_mean',
       'sentiment_score_mean', 'sentiment_magnitude_var',
       'sentiment_score_var', 'sentiment_magnitude_std', 
       'sentiment_score_std'].mean().reset_index()

import numpy as np
# ratio of missing image metadata
c1 = np.setdiff1d(test_meta.PetID, sentimentAtt.PetID)
print(len(c1)/test_meta.shape[0])
# ratio of missing sentiment data
c2 = np.setdiff1d(test_meta.PetID, imagesAtt.PetID)
print(len(c2)/test_meta.shape[0])


# In[78]:


train.columns


# In[79]:


# 'trainWithImageDatameda.csv': contain images metadata
# Index(['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
#        'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
#        'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
#        'VideoAmt', 'Description', 'PhotoAmt', 'AdoptionSpeed', 'StateName',
#        'Color1Name', 'Breed1Name', 'NameL', 'StateNameL', 'Color1NameL',
#        'Breed1NameL', 'RescuerIDL', 'metadata_annots_score',
#        'metadata_color_score', 'metadata_color_pixelfrac',
#        'metadata_crop_conf'],
#       dtype='object')
featuresImagesMeta=['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt', 'NameL', 'StateNameL', 'Color1NameL',
       'Breed1NameL', 'RescuerIDL', 'metadata_annots_score',
       'metadata_color_score', 'metadata_color_pixelfrac',
       'metadata_crop_conf']


# In[80]:


train[featuresImagesMeta].corrwith(train['AdoptionSpeed']).plot.bar(
        figsize = (20, 10), title = "Images attributes is correlated with AdoptionSpeed", fontsize = 15,
        rot = 45, grid = True)
train[featuresImagesMeta].hist(figsize=(15,15))


# In[81]:


#Logistics with imagesAtt
# Basic Logistic Regression
#LOgistic regression for features
#split train into training and testing/validating
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(train[featuresImagesMeta],train.AdoptionSpeed, test_size=0.2)

#Scaling training and validating
scaler = StandardScaler()
# Apply transform to both the training set and the test set.
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

#Fit the model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_valid)
# Use score method to get accuracy of model
# accuracy is defined as: 
# (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(X_valid, y_valid)
print(score)

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_valid, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[82]:


# Basic Logistic Regression for sentiment and image metadata
print(train_sentiment.columns)
print(train_sentiment.info())
print(train_sentiment.head())


# In[83]:


featuresSent=train_sentiment.groupby(['PetID'])['sentiment_magnitude_sum',
       'sentiment_score_sum', 'sentiment_magnitude_mean', 'sentiment_score_mean', 'sentiment_magnitude_var',
       'sentiment_score_var', 'sentiment_magnitude_std', 'sentiment_score_std'].mean().reset_index()
print(len(train))
print(len(featuresSent))
print(len(imagesAtt))
print(len(train)-len(featuresSent))


# In[84]:


train=train.reset_index()
train


# In[85]:


import numpy as np
c1 = np.setdiff1d(test.PetID, featuresSent.PetID)
print(len(c1))
#save c1 to file: PetID not have sentiment data
np.savetxt("PetID_no_sentiment.csv", c1, delimiter=",", fmt='%s')


# In[86]:


print(len(featuresSent.PetID))
print(featuresSent)


# In[87]:


print(len(have_sent.PetID))
print(have_sent)


# In[88]:


have_sent=train[~train.PetID.isin(c1)]
have_sent.to_csv('train_images_PetID_forSent.csv')


# In[89]:


have_sent


# In[90]:


have_sent.set_index('PetID')
have_sent


# In[91]:


featuresSent


# In[92]:


from functools import reducefeaturesSent.set_index('PetID')
featuresSent.to_csv('featureSent.csv')


# In[93]:


common_ID=np.intersect1d(featuresSent.PetID,have_sent.PetID)
common_ID
print(len(common_ID))


# In[94]:


train_st=train[train.PetID.isin(common_ID)]
print(train_st.shape[0])


# In[95]:


featuresSent.columns
featuresSent=featuresSent.set_index('PetID')
featuresSent


# In[96]:


train_st.columns
train_st=train_st.set_index('PetID')
train_st


# In[97]:


# Add columns for train_st
# 'train_images_PetID_forSent.csv'
# 'featureSent'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/kaggle/input/petfinder-sentiment/'

train_st=pd.read_csv(path+'train_images_PetID_forSent.csv')
train_st.info()


# In[98]:


sent=pd.read_csv(path+'featureSent.csv')
sent.info()


# In[99]:


common_ID=np.intersect1d(sent.PetID,train_st.PetID)
common_ID
print(len(common_ID))


# In[100]:


train_st=train_st[train_st.PetID.isin(common_ID)]
print(train_st.shape[0])


# In[101]:


sent.info()


# In[102]:


sent=sent.fillna(0)
sent


# In[103]:


sent.columns


# In[104]:


test_st=train_st.sort_values(by=['PetID'])
sent=sent.sort_values(by=['PetID'])
# 'sentiment_magnitude_sum', 'sentiment_score_sum',
#        'sentiment_magnitude_mean', 'sentiment_score_mean',
#        'sentiment_magnitude_var', 'sentiment_score_var',
#        'sentiment_magnitude_std', 'sentiment_score_std'
train_st['sentiment_magnitude_sum']=sent['sentiment_magnitude_sum']
train_st['sentiment_score_sum']=sent['sentiment_score_sum']
train_st['sentiment_magnitude_mean']=sent['sentiment_magnitude_mean']
train_st['sentiment_score_mean']=sent['sentiment_score_mean']
train_st['sentiment_magnitude_var']=sent['sentiment_magnitude_var']
train_st['sentiment_score_var']=sent['sentiment_score_var']
train_st['sentiment_magnitude_std']=sent['sentiment_magnitude_std']
train_st['sentiment_score_std']=sent['sentiment_score_std']


# In[105]:


train_st.to_csv('train_final.csv')


# In[106]:


print(train_st)


# In[107]:


train_st.columns


# In[108]:


sent_features=['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength',
       'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt','NameL','StateNameL', 'Color1NameL', 'Breed1NameL', 
        'RescuerIDL','metadata_annots_score', 'metadata_color_score',
       'metadata_color_pixelfrac', 'metadata_crop_conf', 'sentiment_magnitude_sum', 
        'sentiment_score_sum', 'sentiment_magnitude_mean', 'sentiment_score_mean',
       'sentiment_magnitude_var', 'sentiment_score_var', 'sentiment_magnitude_std', 
        'sentiment_score_std']
target=['AdoptionSpeed']
X=train_st[sent_features]
y=train_st['AdoptionSpeed']
print(X.columns)
print(y)


# In[109]:


# Correlation betwen target and other features
train_st[sent_features].corrwith(y).plot.bar(
        figsize = (20, 10), title = "Correlation with AdoptionSpeed, including metadata from images and sentiments", fontsize = 15,
        rot = 45, grid = True)
train_st[sent_features].hist(figsize=(15,15))


# In[110]:


print(X.info())


# In[111]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

#Scaling training and validating
scaler = StandardScaler()
# Apply transform to both the training set and the test set.
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

#Fit the model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_valid)
# Use score method to get accuracy of model
# accuracy is defined as: 
# (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(X_valid, y_valid)
print(score)

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_valid, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[112]:


short_features=['metadata_annots_score', 'Type', 'Age', 'FurLength','Sterilized']
X_short=train_st[short_features]

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(X_short,y, test_size=0.2)

#Scaling training and validating
scaler = StandardScaler()
# Apply transform to both the training set and the test set.
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

#Fit the model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_valid)
# Use score method to get accuracy of model
# accuracy is defined as: 
# (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(X_valid, y_valid)
print(score)

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_valid, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[113]:


# Importing the required packages 
import numpy as np 
import pandas as pd 
import math
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

clf_gini = train_using_gini(X_train, X_valid, y_train) 
clf_entropy = train_using_entropy(X_train, X_valid, y_train) 
      
# Operational Phase 
print("Results Using Gini Index:") 
      
# Prediction using gini 
y_pred_gini = prediction(X_valid, clf_gini) 
cal_accuracy(y_valid, y_pred_gini) 
      
print("Results Using Entropy:") 
# Prediction using entropy 
y_pred_entropy = prediction(X_valid, clf_entropy) 
cal_accuracy(y_valid, y_pred_entropy) 

#Predict for the test_meta based on GINI:
# Operational Phase 
print("Results Using Gini Index for test_meta:") 
# Prediction using gini 
# X_test=test_meta.drop(['Name','RescuerID','Description','PetID'], axis=1)
# y_pred_gini = prediction(X_valid, clf_gini) 
y_valid_gini=  prediction(X_valid, clf_gini) 
print("Results Using Entropy:") 
# Prediction using entropy 
# y_pred_entropy = prediction(X_test, clf_entropy) 

print('diff between Decison Tree with gini and entropy, check their histogram')
print('GINI prediction for test_meta histogram:')
_=plt.hist(y_valid_gini, bins='auto')
plt.title("Histogram with 'auto' bins for GINI")
plt.show()

# print('ENTROPY prediction for test histogram:')

# #submit
# submission_gini = pd.DataFrame(data = y_pred_gini
#              , columns = ['AdoptionSpeed'])
# submission_gini = pd.concat([test_meta['PetID'], submission_gini], axis = 1)
# submission_gini.to_csv('samplesubmissionTreeGini.csv', index=False)

# submission_entropy = pd.DataFrame(data = y_pred_gini
#              , columns = ['AdoptionSpeed'])
# submission_entropy = pd.concat([test_meta['PetID'], submission_entropy], axis = 1)
# submission_entropy.to_csv('samplesubmissionTreeEntropy.csv', index=False)


# In[114]:


train_st.shape[1]


# In[115]:


from sklearn.preprocessing import StandardScaler
train_st=pd.read_csv('train_final.csv')

sent_features=['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength',
       'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt','NameL','StateNameL', 'Color1NameL', 'Breed1NameL', 
        'RescuerIDL','metadata_annots_score', 'metadata_color_score',
       'metadata_color_pixelfrac', 'metadata_crop_conf', 'sentiment_magnitude_sum', 
        'sentiment_score_sum', 'sentiment_magnitude_mean', 'sentiment_score_mean',
       'sentiment_magnitude_var', 'sentiment_score_var', 'sentiment_magnitude_std', 
        'sentiment_score_std']
target=['AdoptionSpeed']
X=train_st[sent_features]
y=train_st[target]

# create training and testing vars
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_valid = pca.transform(X_valid)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
t=0
for i in range(len(explained_variance)):
        t=t+explained_variance[i]
        print(i,":",t)
# print(t)


# In[116]:


pca = PCA(n_components=22)
X_train = pca.fit_transform(X_train)
X_valid = pca.transform(X_valid)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


# In[117]:


# In this case we'll use random forest classification 
# with PCA about 95% (22 components)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_valid)
plt.scatter(y_pred,y_valid)
plt.show()


# In[118]:


# META CODE
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer

train_st=pd.read_csv('train_final.csv')

sent_features=['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength',
       'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt','NameL','StateNameL', 'Color1NameL', 'Breed1NameL', 
        'RescuerIDL','metadata_annots_score', 'metadata_color_score',
       'metadata_color_pixelfrac', 'metadata_crop_conf', 'sentiment_magnitude_sum', 
        'sentiment_score_sum', 'sentiment_magnitude_mean', 'sentiment_score_mean',
       'sentiment_magnitude_var', 'sentiment_score_var', 'sentiment_magnitude_std', 
        'sentiment_score_std']
target=['AdoptionSpeed']
X=train_st[sent_features]
y=train_st[target]


clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

# You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] 
# to a multilabel format before training.
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
clf.fit(X, y)

# https://www.freecodecamp.org/news/multi-class-classification-with-sci-kit-learn-xgboost-a-case-study-using-brainwave-data-363d7fca5f69/



# In[119]:


import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import matplotlib.pylab as pl
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# https://evgenypogorelov.com/multiclass-xgb-shap.html


# In[120]:


train_meta=pd.read_csv(path+'train_meta.csv')
test_meta=pd.read_csv(path+'test_meta.csv')
test_image = pd.read_csv(path+"test_dfs_metadata.csv")
test_sentiment = pd.read_csv(path+"test_dfs_sentiment.csv")
train_image = pd.read_csv(path+"train_dfs_metadata.csv")
train_sentiment = pd.read_csv(path+"train_dsf_sentiment.csv")

sent_features=['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength',
       'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee',
       'VideoAmt', 'PhotoAmt','NameL','StateNameL', 'Color1NameL', 'Breed1NameL', 
        'RescuerIDL','metadata_annots_score', 'metadata_color_score',
       'metadata_color_pixelfrac', 'metadata_crop_conf', 'sentiment_magnitude_sum', 
        'sentiment_score_sum', 'sentiment_magnitude_mean', 'sentiment_score_mean',
       'sentiment_magnitude_var', 'sentiment_score_var', 'sentiment_magnitude_std', 
        'sentiment_score_std']
target=['AdoptionSpeed']
# X=train_st[sent_features]
# y=train_st[target]


# In[121]:


#Check for missing values
print(test_meta.isna().sum()[test_meta.isna().sum()>0])


# In[122]:


# train_meta=pd.read_csv(path+'train_meta.csv')
# test_meta=pd.read_csv(path+'test_meta.csv')

# test_image = pd.read_csv(path+"test_dfs_metadata.csv")
# test_sentiment = pd.read_csv(path+"test_dfs_sentiment.csv")

# train_image = pd.read_csv(path+"train_dfs_metadata.csv")
# train_sentiment = pd.read_csv(path+"train_dsf_sentiment.csv")

#reset index=PetID
# train_meta=train_meta.set_index('PetID')
# train_meta=train_meta.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
# train_meta
# test_meta=test_meta.set_index('PetID')
# test_meta=test_meta.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
# test_meta
test_image=test_image.set_index('PetID')
test_sentiment=test_sentiment.set_index('PetID')
train_image=train_image.set_index('PetID')
train_sentiment=train_sentiment.set_index('PetID')


# In[123]:


print('train_meta.columns=',train_meta.columns)
print('shape(train_meta.columns)=',train_meta.shape)
print('test_meta.columns=',test_meta.columns)
print('shape(test_meta.columns)=',test_meta.shape)
print('train_image.columns=',train_image.columns)
print('shape(train_image.columns)=',train_image.shape)
print('test_image.columns=',test_image.columns)
print('shape(test_image.columns)=',test_image.shape)
print('train_sentiment.columns=',train_sentiment.columns)
print('shape(train_sentiment.columns)=',train_sentiment.shape)
print('test_sentiment.columns=',test_sentiment.columns)
print('shape(test_sentiment.columns)=',test_sentiment.shape)


# In[124]:


#TRAIN: check PetID common for all sentiment and image
from functools import reduce
train=reduce(np.intersect1d, (train_meta.index, train_image.index, train_sentiment.index))
print('len(train)=',len(train),': ', len(train)/train_meta.shape[0])
test=reduce(np.intersect1d, (test_meta.index, test_image.index, test_sentiment.index))
print('len(train)=',len(test),': ', len(test)/test_meta.shape[0])


# In[125]:


#Reduce the train and test dataset to get image metadata and sentiment data
#reset the index for train_meta and test_meta
train_meta=train_meta.reset_index()
test_meta=test_meta.reset_index()
train_image=train_image.reset_index()
test_image=test_image.reset_index()
train_sentiment=train_sentiment.reset_index()
test_sentiment=test_sentiment.reset_index()
#Update the datasets
train_meta=train_meta[train_meta.PetID.isin(train)]
test_meta=test_meta[test_meta.PetID.isin(test)]
train_image=train_image[train_image.PetID.isin(train)]
train_sentiment=train_sentiment[train_sentiment.PetID.isin(train)]
test_image=test_image[test_image.PetID.isin(test)]
test_sentiment=test_sentiment[test_sentiment.PetID.isin(test)]
print(train_meta.info())
print(test_meta.info())


# In[126]:


#**********************FOR IMAGE**********************
#---TRAIN------
# 'metadata_annots_score', 'metadata_color_score',
#        'metadata_color_pixelfrac', 'metadata_crop_conf',
#        'metadata_crop_importance', 'metadata_annots_top_desc'
train_meta['metadata_annots_score']=train_image['metadata_annots_score']
train_meta['metadata_color_score']=train_image['metadata_color_score']
train_meta['metadata_color_pixelfrac']=train_image['metadata_color_pixelfrac']
train_meta['metadata_crop_conf']=train_image['metadata_crop_conf']
train_meta['metadata_crop_importance']=train_image['metadata_crop_importance']
train_meta['metadata_annots_top_desc']=train_image['metadata_annots_top_desc']
#---TEST-------
test_meta['metadata_annots_score']=test_image['metadata_annots_score']
test_meta['metadata_color_score']=test_image['metadata_color_score']
test_meta['metadata_color_pixelfrac']=test_image['metadata_color_pixelfrac']
test_meta['metadata_crop_conf']=test_image['metadata_crop_conf']
test_meta['metadata_crop_importance']=test_image['metadata_crop_importance']
test_meta['metadata_annots_top_desc']=test_image['metadata_annots_top_desc']

#**********************FOR SENTIMENT**********************
#---TRAIN------
# 'sentiment_magnitude', 'sentiment_score', 'sentiment_magnitude_sum',
#        'sentiment_score_sum', 'sentiment_magnitude_mean',
#        'sentiment_score_mean', 'sentiment_magnitude_var',
#        'sentiment_score_var', 'sentiment_magnitude_std', 'sentiment_score_std'
train_meta['sentiment_magnitude']=train_sentiment['sentiment_magnitude']
train_meta['sentiment_score']=train_sentiment['sentiment_score']
train_meta['sentiment_magnitude_sum']=train_sentiment['sentiment_magnitude_sum']
train_meta['sentiment_score_sum']=train_sentiment['sentiment_score_sum']
train_meta['sentiment_magnitude_mean']=train_sentiment['sentiment_magnitude_mean']
train_meta['sentiment_score_mean']=train_sentiment['sentiment_score_mean']
train_meta['sentiment_magnitude_var']=train_sentiment['sentiment_magnitude_var']
train_meta['sentiment_score_var']=train_sentiment['sentiment_score_var']
train_meta['sentiment_magnitude_std']=train_sentiment['sentiment_magnitude_std']
train_meta['sentiment_score_std']=train_sentiment['sentiment_score_std']
#---TEST-------
test_meta['sentiment_magnitude']=test_sentiment['sentiment_magnitude']
test_meta['sentiment_score']=test_sentiment['sentiment_score']
test_meta['sentiment_magnitude_sum']=test_sentiment['sentiment_magnitude_sum']
test_meta['sentiment_score_sum']=test_sentiment['sentiment_score_sum']
test_meta['sentiment_magnitude_mean']=test_sentiment['sentiment_magnitude_mean']
test_meta['sentiment_score_mean']=test_sentiment['sentiment_score_mean']
test_meta['sentiment_magnitude_var']=test_sentiment['sentiment_magnitude_var']
test_meta['sentiment_score_var']=test_sentiment['sentiment_score_var']
test_meta['sentiment_magnitude_std']=test_sentiment['sentiment_magnitude_std']
test_meta['sentiment_score_std']=test_sentiment['sentiment_score_std']


# In[127]:


print(train_meta.info())
print(test_meta.info())


# In[128]:


train_meta.to_csv('final_train.csv',index=False)
test_meta.to_csv('final_test.csv',index=False)

