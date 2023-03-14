#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


# In[2]:


#Do not display warnings in notebook 
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.options.display.max_seq_items = 2000

import sys

np.set_printoptions(threshold=sys.maxsize)


# In[3]:


# Importing the json file
J_tweets = pd.read_json ("../input/financial-data/train_data_JSON.json")
# df = pd.read_json(filename)
# df.head()


# In[4]:


# Importing the json file
JSON_tweets_test = pd.read_json("../input/financial-data/test_data.json")


# In[5]:


# Print the shape of the tweeets data
J_tweets.records.shape


# In[6]:


JSON_tweets_test.records.shape


# In[7]:


# Print the first json record
J_tweets.records[0]


# In[8]:


JSON_tweets_test.records[0]


# In[9]:


# Convert the json data into dataframe
from pandas.io.json import json_normalize


# In[10]:


tweets_data = json_normalize(J_tweets.records)


# In[11]:


tweets_data1 = json_normalize(JSON_tweets_test.records)


# In[12]:


# Save the dataframe into a csv file
tweets_data.to_csv("tweets.csv",index=False)
tweets_data1.to_csv("tweets1.csv",index=False)


# In[13]:


# Import the tweets csv file
tweets = pd.read_csv("tweets.csv")
tweets1 = pd.read_csv("tweets.csv")


# In[14]:


tweets.skew(), tweets.kurt()


# In[15]:


tweets.sentiment_score.value_counts()


# In[16]:


Sentiment_count = tweets['sentiment_score'].value_counts()
plt.figure(figsize=(10,4))
sns.barplot(Sentiment_count.index, Sentiment_count.values, alpha=0.8,)
plt.ylabel("COUNT")
plt.xlabel("sentiment_score")
plt.title('sentiment_score counts across the text data', loc='Center', fontsize=19)
plt.show()


# In[17]:


tweets.head(5)


# In[18]:


tweets1.head(5)


# In[19]:


#Number of words in train data
tweets['word_count'] = tweets['stocktwit_tweet'].apply(lambda x: len(str(x).split(" ")))
tweets[['stocktwit_tweet','word_count']].head()


# In[20]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[21]:


#Number of words in testdata
tweets1['word_count'] = tweets1['stocktwit_tweet'].apply(lambda x: len(str(x).split(" ")))
tweets1[['stocktwit_tweet','word_count']].head()


# In[22]:


tweets['word_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Review Text Word Count Distribution')
plt.show()


# In[23]:


# Avg word length
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

tweets['avg_word'] = tweets['stocktwit_tweet'].apply(lambda x: avg_word(x))
tweets[['stocktwit_tweet','avg_word']].head()


# In[24]:


tweets['avg_word'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Review Text Word Count Distribution')
plt.show()


# In[25]:



# Avg word length in test data
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

tweets1['avg_word'] = tweets1['stocktwit_tweet'].apply(lambda x: avg_word(x))
tweets1[['stocktwit_tweet','avg_word']].head()


# In[26]:


tweets1['avg_word'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Review Text Word Count Distribution')
plt.show()


# In[27]:


# Number of special characters in train data
tweets['hastags'] = tweets['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
tweets[['stocktwit_tweet','hastags']].head()


# In[28]:


# Number of special characters in test data
tweets1['hastags'] = tweets1['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
tweets1[['stocktwit_tweet','hastags']].head()


# In[29]:


# Number of numerics in train data
tweets['numerics'] = tweets['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
tweets[['stocktwit_tweet','numerics']].head()


# In[30]:


# Number of numerics in test data
tweets1['numerics'] = tweets1['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
tweets1[['stocktwit_tweet','numerics']].head()


# In[31]:


# Number of uppercase words in train data
tweets['upper'] = tweets['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
tweets[['stocktwit_tweet','upper']].head()


# In[32]:


# Number of uppercase words in test data
tweets1['upper'] = tweets1['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
tweets1[['stocktwit_tweet','upper']].head()


# In[33]:


# Conversion in train_data
tweets['Dates'] = pd.to_datetime(tweets['timestamp']).dt.date
tweets['Time'] = pd.to_datetime(tweets['timestamp']).dt.time


# In[34]:


# Conversion in test_data
tweets1['Dates'] = pd.to_datetime(tweets1['timestamp']).dt.date
tweets1['Time'] = pd.to_datetime(tweets1['timestamp']).dt.time


# In[35]:


days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}

tweets["Dateoftweet"]=pd.to_datetime(tweets["Dates"])
tweets["Day"]=tweets["Dateoftweet"].dt.day
tweets["month"]=tweets["Dateoftweet"].dt.month
tweets["year"]=tweets["Dateoftweet"].dt.month
tweets["dayOftheweek"]=tweets["Dateoftweet"].dt.dayofweek

tweets['dayOftheweek'] = tweets['dayOftheweek'].apply(lambda x: days[x])


# In[36]:


tweets.drop(['timestamp'],axis=1,inplace=True)


# In[37]:


tweets.head()


# In[38]:


tweets.dtypes


# In[39]:


Tweet_Day_Count = tweets['Day'].value_counts()
plt.figure(figsize=(10,4))
sns.barplot(Tweet_Day_Count.index, Tweet_Day_Count.values, alpha=0.8)
plt.ylabel("Number Of Tweet")
plt.xlabel("Tweets By Days")
plt.title('Total tweets count by Day', loc='Center', fontsize=14)
plt.show()


# In[40]:


# Number of stop words in train_data
from nltk.corpus import stopwords
stop = stopwords.words('english')

tweets['stopwords'] = tweets['stocktwit_tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
tweets[['stocktwit_tweet','stopwords']].head()


# In[41]:


# Number of characters
tweets1['char_count'] = tweets1['stocktwit_tweet'].str.len() ## this also includes spaces
tweets1[['stocktwit_tweet','char_count']].head()


# In[42]:


# Print the shape of the dataframe(train_data)
tweets.shape


# In[43]:


# Print the shape of the dataframe(test_data)
tweets1.shape


# In[44]:


tweets.head(5)


# In[45]:


Sentiment_count = tweets['sentiment_score'].value_counts()
plt.figure(figsize=(10,4))
sns.barplot(Sentiment_count.index, Sentiment_count.values, alpha=0.8)
plt.ylabel("Count")
plt.xlabel("sentiment Count")
plt.title('sentiment counts across the tweet data', loc='Center', fontsize=14)
plt.show()


# In[46]:


tweets1.head(5)


# In[47]:


## Checking the null values in the data


# In[48]:


tweets.info()


# In[49]:


tweets1.info()


# In[50]:


### find the sentiment score for train_data
tweets.sentiment_score.value_counts()


# In[51]:


# Preprocess the tweets data of ticker feature (train_data)
tweets['ticker']=tweets['ticker'].apply(lambda x:x.lower().replace('$',''))
tweets['ticker'] = tweets['ticker'].apply(lambda x: '$'+x)
len(tweets['ticker'].str.lower().unique())


# In[52]:


# Preprocess the tweets data of ticker feature(test_data)
tweets1['ticker']=tweets1['ticker'].apply(lambda x:x.lower().replace('$',''))
tweets1['ticker'] = tweets1['ticker'].apply(lambda x: '$'+x)
len(tweets1['ticker'].str.lower().unique())


# In[53]:


top_ticker = tweets.ticker.value_counts()[:10]
top_ticker


# In[54]:


plt.figure(figsize=(15,12))
sns.barplot(top_ticker.index, top_ticker.values, alpha=0.8)
plt.ylabel("Count")
plt.xlabel("Frequent words")
plt.title('frequency of Ticker', loc='Center', fontsize=14)
plt.show()


# In[55]:


# Plot the graph between the sentiment_score and the count(train_data)
Sentiment_count = tweets['sentiment_score'].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(Sentiment_count.index, Sentiment_count.values, alpha=0.8)
plt.ylabel("COUNT")
plt.xlabel("sentiment_score")
plt.title('sentiment_score counts across the text data', loc='Center', fontsize=19)
plt.show()


# In[56]:


# Import the libraries
import nltk
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[57]:


tweets.head(5)


# In[58]:


tweets1.head(5)


# In[59]:


# Removing non_letters(train_data)
tweets["stocktwit_tweet"]=tweets["stocktwit_tweet"].apply(lambda x:re.sub("[^A-Za-z]", " ", x.strip()))


# In[60]:


# Removing non_letters-(test_data) 
tweets1["stocktwit_tweet"]=tweets1["stocktwit_tweet"].apply(lambda x:re.sub("[^A-Za-z]", " ", x.strip()))


# In[61]:


# Converting into Lower Cases- (train_data)
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[62]:


# Converting into Lower Cases- (test_data)
tweets1['stocktwit_tweet'] = tweets1['stocktwit_tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[63]:


# Removing Numbers-train_data
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].str.replace('[\d]', '')


# In[64]:


# Removing Numbers- (test_data)
tweets1['stocktwit_tweet'] = tweets1['stocktwit_tweet'].str.replace('[\d]', '')


# In[65]:


# Removing Punctuational marks- (train_data)
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].str.replace('[^\w\s]','')


# In[66]:


# Removing Punctuational marks- (test_data)
tweets1['stocktwit_tweet'] = tweets1['stocktwit_tweet'].str.replace('[^\w\s]','')


# In[67]:


# Removing Stop words- (train_data)
stop = stopwords.words('english')
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[68]:


# Common word removal (train_data)
freq = pd.Series(' '.join(tweets['stocktwit_tweet']).split()).value_counts()[:10]
print(freq)

# Remove these words as their presence will not of any use in classification of our text data.
freq = list()
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
tweets['stocktwit_tweet'].head()


# In[69]:


# Common word removal(test_data)
freq = pd.Series(' '.join(tweets1['stocktwit_tweet']).split()).value_counts()[:10]
print(freq)


# In[70]:


# Remove these words as their presence will not of any use in classification of our text data.
freq = list()
tweets1['stocktwit_tweet'] = tweets1['stocktwit_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
tweets1['stocktwit_tweet'].head()


# In[71]:


# Rare words removal (train_data)
freq = pd.Series(' '.join(tweets['stocktwit_tweet']).split()).value_counts()[-10:]
print(freq)

freq = list(freq.index)
tweets['stocktwit_tweet'] = tweets['stocktwit_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
tweets['stocktwit_tweet'].head()


# In[72]:


# Rare words removal (test_data)
freq = pd.Series(' '.join(tweets1['stocktwit_tweet']).split()).value_counts()[-10:]
print(freq)

freq = list(freq.index)
tweets1['stocktwit_tweet'] = tweets1['stocktwit_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
tweets1['stocktwit_tweet'].head()


# In[73]:


# Word Frequency- train_data
Word_freq = pd.Series(' '.join(tweets['stocktwit_tweet']).split()).value_counts()[:10]
Word_freq


# In[74]:


# Word Frequency- test_data
Word_freq = pd.Series(' '.join(tweets1['stocktwit_tweet']).split()).value_counts()[:10]
Word_freq


# In[75]:


# Word Frequency Plot - 
plt.figure(figsize=(10,4))
sns.barplot(Word_freq.index, Word_freq.values, alpha=0.8)
plt.ylabel("Count")
plt.xlabel("Frequent words")
plt.title('Frequent words in review_data', loc='Center', fontsize=14)
plt.show()


# In[76]:


# Word Frequency Plot - (test_data)
plt.figure(figsize=(10,4))
sns.barplot(Word_freq.index, Word_freq.values, alpha=0.8)
plt.ylabel("Count")
plt.xlabel("Frequent words")
plt.title('Frequent words in review_data', loc='Center', fontsize=14)
plt.show()


# In[77]:


# Stemming (train_data)
st = PorterStemmer()
tweets["stocktwit_tweet"] = tweets["stocktwit_tweet"].apply(lambda x: " ".join([st.stem(word)
                                                                   for word in x.split()]))


# In[78]:


# Stemming (test_data)
st = PorterStemmer()
tweets1["stocktwit_tweet"] = tweets1["stocktwit_tweet"].apply(lambda x: " ".join([st.stem(word)
                                                                   for word in x.split()]))


# In[79]:


nltk.download('wordnet')


# In[80]:


# # Lemmatization (train_data)
Lem = WordNetLemmatizer()
tweets["stocktwit_tweet"] = tweets["stocktwit_tweet"].apply(lambda x: " ".join([Lem.lemmatize(word)
                                                          for word in x.split()]))


# In[81]:


# # Lemmatization (test_data)
Lem = WordNetLemmatizer()
tweets1["stocktwit_tweet"] = tweets1["stocktwit_tweet"].apply(lambda x: " ".join([Lem.lemmatize(word)
                                                           for word in x.split()]))


# In[82]:


tweets.head(5)


# In[83]:


tweets1.head(5)


# In[84]:


# Splitting into train and test
X_train,X_test,Y_train,Y_test = train_test_split(tweets['stocktwit_tweet'],
                                                 tweets['sentiment_score'],
                                                 test_size=0.25,
                                                 random_state=7)


# In[85]:



# Print the shape of train and test
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[86]:


# Function for cleaning the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)   
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)   
    text = re.sub(r'www.[^ ]+', '', text)  
    text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)  
    text = re.sub(r'[^a-zA-Z]', ' ', text)   
    text = [token for token in text.split() if len(token) > 2]
    text = ' '.join(text)
    #text = emoji.demojize(text)
    
    return text

X_train = X_train.apply(clean_text)
X_test =X_test.apply(clean_text)


# In[87]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, 
                             min_df=3, max_features=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)


# In[88]:


# Fit the model
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)


# In[89]:


# Print the shape of the train and test
print(X_train_tfidf.shape)
print(X_test_tfidf.shape)


# In[90]:


# Build a naive_bayes model
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB().fit(X_train_tfidf, Y_train)
pred_test = nb_clf.predict(X_test_tfidf)

# Print the mertics
print('f1_score       :', f1_score(Y_test, pred_test, average='macro'))
print('accuracy score :', accuracy_score(Y_test, pred_test))


# In[91]:


# Print the classification reports
print(classification_report(Y_test, pred_test))


# In[92]:


# Import the libraries
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

# Build the model
sgd = SGDClassifier(loss='log', max_iter=200, random_state=0, class_weight='balanced')
ovr = OneVsRestClassifier(sgd)
ovr.fit(X_train_tfidf, Y_train)
y_pred_class = ovr.predict(X_test_tfidf)

# Print the metrics score
print('f1_score       :', f1_score(Y_test, y_pred_class, average='macro'))
print('accuracy score :', accuracy_score(Y_test, y_pred_class))


# In[93]:


# Print the classification report
print(classification_report(Y_test, y_pred_class)) 


# In[94]:


# Print the confusion matrix
confusion_matrix(Y_test,y_pred_class)


# In[95]:


# Validation DataFrame
validation = pd.DataFrame({'stocktwit_tweet':X_test,
                           'predicted_sentiment':''})


# In[96]:


validation['predicted_sentiment']=y_pred_class
validation['actual_sentiment']=Y_test


# In[97]:


validation.head(10)


# In[98]:


validation.shape


# In[99]:


validation['predicted_sentiment'].value_counts()


# In[100]:



validation['actual_sentiment'].value_counts()


# In[102]:


tweets1.head()


# In[104]:


# Preprocessing the test text
X_test_data = tweets1['stocktwit_tweet'].apply(clean_text)
X_test_data_tfidf = tfidf_vect.transform(X_test_data )
print(X_test_data_tfidf.shape)

y_pred_class_data = ovr.predict(X_test_data_tfidf)
tweets1['sentiment_score'] = y_pred_class_data


# In[105]:


tweets1.head()


# In[ ]:




