#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#read the data in
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


#encode labels to integer classes
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder().fit(train['author'])

#Original labels are stored in a class property
#and binarized labels correspond to indexes of this array - 0,1,2 in our case of three classes
lb.classes_


# In[4]:


#after transformation the label will look like an array of integer taking values 0,1,2
lb.transform(train['author'])


# In[5]:


from sklearn.model_selection import train_test_split

X_train_part, X_valid, y_train_part, y_valid =    train_test_split(train['text'], 
                     lb.transform(train['author']), 
                test_size=0.3,random_state=17, stratify=train['author'])


# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

pipe1 = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('logit', LogisticRegression()),
])


# In[7]:


pipe1.fit(X_train_part, y_train_part)


# In[8]:


pipe1.steps


# In[9]:


pipe1.named_steps['logit'].coef_


# In[10]:


from sklearn.metrics import log_loss

pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# In[11]:


pipe1.named_steps['logit'].get_params()


# In[12]:


pipe1.get_params()


# In[13]:


#set_params(cv__lowercase=True)
pipe1.set_params(cv__min_df=6, 
                 cv__lowercase=False).fit(X_train_part, y_train_part)
pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# In[14]:


from sklearn.naive_bayes import MultinomialNB, BernoulliNB

pipe1 = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('logit', LogisticRegression()),
    ('bnb', BernoulliNB()),
   
])


# In[15]:


pipe1.fit(X_train_part, y_train_part)
pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# In[16]:


import nltk

text = "And now we are up for 'something' completely different;"
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
tagged


# In[17]:


from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

class PosTagMatrix(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, tokenizer=lambda x: x.split(), normalize=True):
        self.tokenizer=tokenizer
        self.normalize=normalize

    #helper function to tokenize and count parts of speech
    def pos_func(self, sentence):
        return Counter(tag for word,tag in nltk.pos_tag(self.tokenizer(sentence)))

    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        X_tagged = X.apply(self.pos_func).apply(pd.Series).fillna(0)
        X_tagged['n_tokens'] = X_tagged.apply(sum, axis=1)
        if self.normalize:
            X_tagged = X_tagged.divide(X_tagged['n_tokens'], axis=0)

        return X_tagged


# In[18]:


from sklearn.pipeline import FeatureUnion

pipe2 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
        ])),
    ])),
    ('logit', LogisticRegression()),

])


# In[19]:


pipe2.fit(X_train_part, y_train_part)
pred = pipe2.predict_proba(X_valid)
log_loss(y_valid, pred)


# In[20]:


class CountVectorizerPlus(CountVectorizer):
    def __init__(self, *args, fit_add=None, **kwargs):
        #this will store a reference to an extra data to include for fitting only
        self.fit_add = fit_add
        super().__init__(*args, **kwargs)
    
    def transform(self, X):
        U = super().transform(X)
        return U
    
    def fit_transform(self, X, y=None):
        if self.fit_add is not None:
            X_new = pd.concat([X, self.fit_add])
        else:
            X_new = X
        #calling CountVectorizer.fit_transform()
        super().fit_transform(X_new, y)

        U = self.transform(X)
        return U
    


# In[21]:


pipe1a = Pipeline([
    ('cv', CountVectorizerPlus(fit_add=test['text'])),
    #('cv', CountVectorizerPlus()),
    ('tfidf', TfidfTransformer()),
    #('logit', LogisticRegression()),
    ('bnb', BernoulliNB()),
   
])


# In[22]:


pipe1a.fit(X_train_part, y_train_part)
pred = pipe1a.predict_proba(X_valid)
print(log_loss(y_valid, pred))


# In[23]:


#stacking trick
from sklearn.metrics import get_scorer
class ClassifierWrapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator, verbose=None, fit_params=None, use_proba=True, scoring=None):
        self.estimator = estimator
        self.verbose = verbose #True = 1, False = 0, 1 - moderately verbose, 2- extra verbose    
        if verbose is None:
            self.verbose=0
        else:
            self.verbose=verbose
        self.fit_params= fit_params
        self.use_proba = use_proba #whether to use predict_proba in transform
        self.scoring = scoring # calculate validation score, takes score function name
        #TODO check if scorer imported?
        self.score = None #variable to keep the score if scoring is set.

    def fit(self,X,y):
        fp=self.fit_params
        if self.verbose==2: print("X: ", X.shape, "\nFit params:", self.fit_params)
        
        if fp is not None:
            self.estimator.fit(X,y, **fp)
        else:
            self.estimator.fit(X,y)
        
        return self
    
    def transform(self, X):
        if self.use_proba:
            return self.estimator.predict_proba(X) #[:, 1].reshape(-1,1)
        else:
            return self.estimator.predict(X)
    
    def fit_transform(self,X,y,**kwargs):
        self.fit(X,y)
        p = self.transform(X)
        if self.scoring is not None:
            self.score = eval(self.scoring+"(y,p)")
            #TODO print own instance name?
            if self.verbose >0: print("score: ", self.score) 
        return p
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)


# In[24]:


from xgboost import XGBClassifier
#params are from the above mentioned tutorial
xgb_params={
    'objective': 'multi:softprob',
    'eta': 0.1,
    'max_depth': 3,
    'silent' :1,
    'num_class' : 3,
    'eval_metric' : "mlogloss",
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'seed':17,
    'num_rounds':2000,
}


# In[25]:


pipe3 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('tfidf_logit', ClassifierWrapper(LogisticRegression())),
        ])),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
            ('pos_logit', ClassifierWrapper(LogisticRegression())),
        ])),
    ])),
    ('xgb', XGBClassifier(**xgb_params)),
])


# In[26]:


get_ipython().run_cell_magic('time', '', 'pipe3.fit(X_train_part, y_train_part)\npred = pipe3.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# In[27]:


pipe4 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('tfidf_logit', ClassifierWrapper(LogisticRegression())),
        ], memory="/tmp")),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
            ('pos_logit', ClassifierWrapper(LogisticRegression())),
        ], memory="/tmp")),
    ])),
    ('xgb', XGBClassifier(**xgb_params)),
])


# In[28]:


get_ipython().run_cell_magic('time', '', 'pipe4.fit(X_train_part, y_train_part)\npred = pipe4.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# In[29]:


get_ipython().run_cell_magic('time', '', 'pipe4.fit(X_train_part, y_train_part)\npred = pipe4.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# In[30]:


#refit on the full train dataset
pipe4.fit(train['text'], lb.transform(train['author']))

# obtain predictions
pred = pipe4.predict_proba(test['text'])

#id,EAP,HPL,MWS
#id07943,0.33,0.33,0.33
#...
pd.DataFrame(dict(zip(lb.inverse_transform(range(pred.shape[1])),
                      pred.T
                     )
                 ),index=test.id).to_csv("submission.csv", index_label='id')


# In[31]:




