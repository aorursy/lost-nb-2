#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


'''
Import train and test data
'''

train_data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')

#Making copies of original dataset for future use
test_data_copy = test_data
train_data_copy = train_data
train_data


# In[ ]:


#Exporting train and test datasets as dataframes
train_data.to_csv("train.csv",index=False)
test_data.to_csv("test.csv",index=False)


# In[ ]:


'''
Exploratory Data Analysis
'''
import nltk
import matplotlib.pyplot as plt

#Total entries
print ("Total number of recipes: ",len(train_data))
#Total number of cuisines
print ("Total number of cuisines: ",len(nltk.FreqDist(train_data["cuisine"]).keys()))

#Visualizing frequency of each cuisine
plt.figure(figsize=(12,8))
plt.bar(nltk.FreqDist(train_data["cuisine"]).keys(),nltk.FreqDist(train_data["cuisine"]).values(),alpha=0.8,width=0.8,align="center")
plt.xlabel("Cuisine",size=15)
plt.ylabel("Frequency",size=15)
plt.title("Train data cuisine distribution",size=20)
plt.xticks(rotation=45,size=12)
plt.show()

'''
Determining the highest and lowest occurring words
'''

#Collecting all the words and creating a frequency distribution 
total_words = []
for i in range(len(train_data)):
    total_words.extend(train_data.iloc[i,2])
words_freq = nltk.FreqDist(total_words)

#Total number of words 
print (len(total_words))
print ('')
words_freq_sorted = sorted(words_freq.items(),key=lambda x: x[1])

#Highest occurring words
print ('Highest occurring ingredients in the dataset')
print (words_freq_sorted[-20:])
print ('')

#Lowest occurring words
print ('20 Lowest occurring ingredients in the dataset')
print (words_freq_sorted[:20])
print ('')

#Visualizing frequency of highest occurring words
plt.figure(figsize=(12,8))
#plt.bar(list(list(zip(*words_freq_sorted[-20:]))[0]),list(list(zip(*words_freq_sorted[-20:]))[1]),alpha=0.8,width=0.8,align="center")
plt.barh(list(list(zip(*words_freq_sorted[-20:]))[0]),list(list(zip(*words_freq_sorted[-20:]))[1]),alpha=0.8,align="center",color='red')
plt.ylabel("Words",size=15)
plt.xlabel("Frequency",size=15)
plt.title("Frequency of highest occurring ingredients",size=20)
plt.xticks(rotation=90,size=12)
plt.show()


# In[ ]:


'''
Adding new feature that contains the number of ingredients in each recipe
'''
num_ingredients_train = train_data["ingredients"].apply(len)
num_ingredients_test = test_data["ingredients"].apply(len)
print (len(num_ingredients_train))

#Will add this feature to the modeling dataset later following vectorizing 


# In[ ]:


'''
Pre-processing train data
'''
from nltk.stem.porter import PorterStemmer

#Function to combine a list of strings into a sentence for text modeling
def join_strings_to_sentence(x):
    result = ' '.join(x)
    return result

#Similar to the 'join_strings_to_sentence' function except this function joins ingredients with
#with a space in between into a single word
def combine_and_stem(x):
    x = pd.Series(x)
    stemmer = PorterStemmer().stem
    x = x.apply(stemmer)
    x = x.apply(lambda k: ''.join(k.split(' ')))
    x = ' '.join(list(x))
    return x

#Separating the words in each document to form sentences for vectorization
train_data["ingredients"] = train_data["ingredients"].apply(combine_and_stem)
test_data["ingredients"] = test_data["ingredients"].apply(combine_and_stem)


#train_data["ingredients"] = train_data["ingredients"].apply(join_strings_to_sentence)
train_data = train_data.drop("id",axis=1)
test_data_id = test_data["id"]
test_data = test_data.drop("id",axis=1)

#Printing both datasets to make sure that the ingredients have been processed appropriately
print (train_data.head(10))
print ("")
print (test_data.head(10))


# In[ ]:


'''
Function to add a new feature to the document-term matrix following vectorization
'''
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[ ]:


'''
Count Vectorizer Fit - using every single word as separate features
'''
'''
from sklearn.feature_extraction.text import CountVectorizer

#Creating a document-term matrix using Count Vectorizer modeling and fitting train and test data
vect = CountVectorizer().fit(train_data["ingredients"])
train_data_vect = vect.transform(train_data["ingredients"])

#Adding the number of ingredients variable to the dataset
train_data_vect_new = add_feature(train_data_vect,num_ingredients_train)

#Total number of features
print ("Number of unique features: ",len(vect.get_feature_names()))

print (train_data_vect_new.shape)
'''


# In[ ]:


'''
Splitting train data into further train and test for modeling
'''
'''
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, find

X_train,X_test,y_train,y_test = train_test_split(train_data_vect_new,train_data["cuisine"],random_state=0,train_size=0.9)

#Unique features
print (vect.get_feature_names())

#Document-term matrix
print (X_train)
'''


# In[ ]:


'''
Logistic Regression Modeling
'''
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score

#Creating model and fitting
logreg = LogisticRegression(C=1)
model = logreg.fit(X_train,y_train)

#Prediction and score
y_predict = model.predict(X_test)
score = accuracy_score(y_test,y_predict)

print (score)
print (y_predict)
print (y_test)
'''

'''
Scores using CountVectorizer
'''
'''
Scores with features represented as individual words
'''
#0.7938 with C=1 - best score with individual words as features
#0.7921 with C=3

'''
Scores with features represented as entire terms as opposed to single words
'''
#0.7905 with C=1
#0.7931 with C=2
#0.7941 with C=3 - the best score with considering multiple word ingredients as a single term
#0.7926 with C=4
#0.7913 with C=5
#0.7863 with C=10


# In[ ]:


'''
TFIDF Vectorizer
'''
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#vect = TfidfVectorizer(min_df=2).fit(train_data["ingredients"].append(test_data["ingredients"]))
vect = TfidfVectorizer(min_df=2).fit(train_data["ingredients"])
train_data_vect = vect.transform(train_data["ingredients"])
test_data_vect = vect.transform(test_data["ingredients"])

#Adding the number of ingredients variable to the dataset
train_data_vect_new = add_feature(train_data_vect,num_ingredients_train)
test_data_vect_new = add_feature(test_data_vect,num_ingredients_test)

'''
Train Test Split
'''
X_train,X_test,y_train,y_test = train_test_split(train_data_vect_new,train_data["cuisine"],random_state=0,train_size=0.9)


# In[ ]:


'''
Logistic Regression with TFIDF
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score

#Creating model and fitting
logreg = LogisticRegression(C=6)
model = logreg.fit(X_train,y_train)

#Prediction and score
y_predict = model.predict(X_test)
score = accuracy_score(y_test,y_predict)

print (score)

results = pd.DataFrame()
results["y_test"] = y_test
results["y_predict"] = y_predict
print (results)

'''
Scores with features represented as entire terms as opposed to single words
'''
#0.78 with C=1 and min_df=1
#0.7797 with C=1 and min_df=5
#0.77 with C=1 and min_df=10
#0.7911 with C=10 and min_df=10
#0.7948 with C=4 and min_df=5
#0.7966 with C=5 and min_df=5 - best score with TFIDF and using ingredient terms as features
#0.7963 with C=6 and min_df=5
#0.7966 with C=10 and min_df=5
#0.7941 with C=15 and min_df=5

'''
Scores with stemming
'''
#0.795 with C=5 and min_df=5
#0.7963 with C=6 and min_df=5
#0.7948 with C=7 and min_df=5
#0.7981 with C=6 and min_df=1
#0.7986 with C=7 and min_df=1
#0.7983 with C=8 and min_df=1
#0.7986 with C=8 and min_df=2
#0.7993 with C=7 and min_df=2
#0.8006 with C=6 and min_df=2 - best score with ingredient terms as features and stemming
#0.7986 with C=5 and min_df=2
#0.7998 with C=5 and min_df=3
#0.7961 with C=4 and min_df=4
#0.7998 with C=5 and min_df=4


# In[ ]:


'''
Analyzing the results
'''

#The best results appears to be 0.7938 with the default CountVectorizer
#parameters such as min_fd, max_df, analyzer, max_features, ngram_range did not improve the result
#Adding the extra feature with the number of tokens in each document did not affect the result

results = pd.DataFrame()
results["test"] = y_test
results["predict"] = y_predict
results_incorrect = results[results["test"]!=results["predict"]]
results_correct = results[results["test"]==results["predict"]]

#Cuisine pairs that were predicted correctly and incorrectly
print ("Number of cuisines not predicted correctly :",len(results_incorrect))
print ("Number of cuisines predicted correctly :",len(results_correct))
print ("")
print ("Pairs of incorrect predictions")
print (results_incorrect)

#Visualizing how many times each cuisine was predict correctly
plt.figure(figsize=(12,8))
plt.bar(nltk.FreqDist(results_correct["test"]).keys(),nltk.FreqDist(results_correct["test"]).values(),alpha=0.8,width=0.8,align="center")
plt.xlabel("Cuisine",size=15)
plt.ylabel("Frequency",size=15)
plt.title("Frequency of cuisines predicted correctly",size=20)
plt.xticks(rotation=45,size=12)
plt.show()

#Visualizing how many times each cuisine was predicted incorrectly
plt.figure(figsize=(12,8))
plt.bar(nltk.FreqDist(results_incorrect["test"]).keys(),nltk.FreqDist(results_incorrect["test"]).values(),alpha=0.8,width=0.8,align="center")
plt.xlabel("Cuisine",size=15)
plt.ylabel("Frequency",size=15)
plt.title("Frequency of cuisines predicted incorrectly",size=20)
plt.xticks(rotation=45,size=12)
plt.show()

#Determining which cuisines were predicted incorrectly the most
print ("Number of times each cuisine was incorrectly predicted")
print ((results_incorrect.groupby("test").apply(len)))

#Looking at the results, French, Italian, Southern US and Mexican cuisines were predicted incorrectly
#the most. Although, it is to be noted that these are 4 of the highest occurring cuisines 
#occurring in the original data set. Therefore, a percentage value may be more helpful

#The case is similar for correctly predicted cuisines with French being the exception


# In[ ]:


'''
Cross-Validation
'''
'''
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(logreg,train_data_vect_new,train_data["cuisine"],cv=5,scoring='accuracy')

cv_score
'''


# In[ ]:


'''
Determine important features for Logistic Regression classifier
'''
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print ("Most important features")
print (feature_names[sorted_coef_index[:10]])
print ('')
print ("Least important features")
print (feature_names[sorted_coef_index[-10:]])

#Although good to know, these feature importances do not provide too much insight as this is a
#multi-class model. Feature importance would be more useful for sentiment analysis or spam 
#detection, hence binary classification. 


# In[ ]:


'''
Multinomial Naive Bayes Classifier with TFIDF
'''
'''
from sklearn.naive_bayes import MultinomialNB

#Modeling and fitting
mnb = MultinomialNB(alpha=0.1)
model = mnb.fit(X_train,y_train)

#Prediction and score
y_predict = model.predict(X_test)
score = accuracy_score(y_test,y_predict)

print (score)

model.get_params()
'''
'''
Representing features represented as entire terms as opposed to single words
'''
#0.7584 with min_df=5 and alpha=0.1
#0.7536 with min_df=5
'''


# In[ ]:


'''
Bernoulli Naive Bayes Classifier
'''
'''
from sklearn.naive_bayes import BernoulliNB

#Modeling and fitting
mnb = BernoulliNB(alpha=0.1)
model = mnb.fit(X_train,y_train)

#Prediction and score
y_predict = model.predict(X_test)
score = accuracy_score(y_test,y_predict)

print (score)

model.get_params()

#0.7644 with alpha=0.1 - better than MNB
'''


# In[ ]:


'''
Support Vector Machines
'''
'''
from sklearn.svm import SVC

#Creating model and fitting
svc = SVC(kernel='linear',C=6)
model = svc.fit(X_train,y_train)

#Prediction and score
y_predict = model.predict(X_test)
score = accuracy_score(y_test,y_predict)
print (score)
print (y_predict)
print (y_test)

'''
Score using CountVectorizer
'''
#SVC is taking a long time to execute
#SVC with kernel as 'linear' is returning a score of 0.7772
'''


# In[ ]:


'''
Test Data Prediction - Logistic Regression 
'''

#Prediction and score
y_predict = model.predict(test_data_vect_new)

print (y_predict)


# In[ ]:


'''
Presenting prediction results
'''
prediction_results = pd.DataFrame()
prediction_results["id"] = test_data_copy["id"]
prediction_results["cuisine"] = y_predict

prediction_results.to_csv("cuisine_prediction.csv",index=False)
prediction_results_svm = prediction_results

