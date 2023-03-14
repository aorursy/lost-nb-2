#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import seaborn as sns
import copy

TOTAL_RECORDS = 10000
PERCENT_TRAINING = 0.8
TRAINING_RECORDS = int(TOTAL_RECORDS * PERCENT_TRAINING)


# In[2]:


df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

print("number of training samples : %i" %len(df_train) )
print("number of testing samples : %i" %len(df_test) )

# Merge  training and testing
### concatenate both train and test data set.
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
### add all product info to the above dataframe
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')


df_all = df_all.iloc[:TOTAL_RECORDS] # TO-DO: remove hardcodings
print("total number of samples : %i" %len(df_all))


# In[ ]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

def str_stemmer(s):
    ''' To stem and lamatize the sentences so that we can avoid the difference between computing , computed , computs'''
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
    '''Get count of words common in two input strings. Basic word matching'''
    return sum(int(str2.find(word)>=0) for word in str1.split())


# In[ ]:


def runFeatureEngineeringSet1(df_all):    
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

    # calculating the length of search term
    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

    # combine search_term , product_title and product_description
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

    # get common words in search_term and product_title
    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

    # get count of common words in search_term and product_description
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

    # display first rows in dataframe
    print(df_all.head())
    
    return df_all

df_all = runFeatureEngineeringSet1(df_all)


# In[ ]:


from sklearn import metrics

class TreeCapsule:
    def __init__(self, max_depth=50, n_estimators=100, min_child_weight=2, learning_rate=0.01):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.sklearnHook = None
        self.sklearnHookReady = False
        self.y_hat = None
        
        
    def getTrainingTestingData(self, df, numTrain, columnsToTrain=None):
        df_train = df.iloc[:numTrain]
        df_test = df.iloc[numTrain:]

        if not columnsToTrain: # include all columns except "relevance"
            print("using all columns except 'relevance'")
            columnsToTrain = list(df.columns)
            columnsToTrain.remove("relevance")
            
        self.x_train = df_train[columnsToTrain]
        self.y_train = df_train["relevance"]
        
        
        self.x_test = df_test[columnsToTrain]
        self.y_test = list(df_test["relevance"])
        
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    
    def fit(self):
        assert self.sklearnHook != None, "cannot init base class object"
        
        if self.sklearnHookReady:
            # assign default params
            self.sklearnHook.max_depth = self.max_depth
            self.sklearnHook.n_estimators = self.n_estimators
            self.sklearnHook.min_child_weight = self.min_child_weight
            self.sklearnHook.learning_rate = self.learning_rate
        
            self.sklearnHook.fit(self.x_train, self.y_train)
        
        
    def predict(self):
        assert self.sklearnHookReady, "cannot run predictions on unfitted algo"
        
        self.y_hat = list(self.sklearnHook.predict(self.x_test))
        
    
    def calculateMetrics(self):
        assert self.y_hat , "no predictions to run metrics on"
        
        self.rmse = metrics.mean_squared_error(self.y_test, self.y_hat)
        # self.accuracyScore = metrics.accuracy_score(self.y_test, self.y_hat)
        
        
    def reset(self):
        self.sklearnHookReady = False
        
    
    def run(self):
        self.fit()
        self.predict()
        self.calculateMetrics()
        
        return {"rmse" : self.rmse}
        # return json.dumps({"accuracyScore" : "self.accuracyScore", "rmse" : self.rmse}, indent=4)
        


# In[ ]:


import xgboost as xgb
from xgboost import plot_importance

class XGBoostCapsule(TreeCapsule):
    def __init__(self, max_depth=50, n_estimators=100, min_child_weight=2, learning_rate=0.01):
        super().__init__(max_depth, n_estimators, min_child_weight, learning_rate)
        self.sklearnHook = xgb.XGBRegressor()
        self.sklearnHookReady = True
        
    def run(self):
        val = super().run()
        plot_importance(self.sklearnHook)
        return val
        
from sklearn.ensemble import RandomForestRegressor
class RandomForestCapsule(TreeCapsule):
    def __init__(self, max_depth=50, n_estimators=100, min_child_weight=2, learning_rate=0.01):
        super().__init__(max_depth, n_estimators, min_child_weight, learning_rate)
        self.sklearnHook = RandomForestRegressor()
        self.sklearnHookReady = True
        
from sklearn.ensemble import BaggingRegressor
class BaggingRegressorCapsule(TreeCapsule):
    def __init__(self, max_depth=50, n_estimators=100, min_child_weight=2, learning_rate=0.01):
        super().__init__(max_depth, n_estimators, min_child_weight, learning_rate)
        self.sklearnHook = BaggingRegressor()
        self.sklearnHookReady = True
        
        
from sklearn.ensemble import GradientBoostingRegressor
class GradBoostingRegressorCapsule(TreeCapsule):
     def __init__(self, max_depth=50, n_estimators=100, min_child_weight=2, learning_rate=0.01):
        super().__init__(max_depth, n_estimators, min_child_weight, learning_rate)
        self.sklearnHook = GradientBoostingRegressor()
        self.sklearnHookReady = True


# In[ ]:


columnsToTrain = ['len_of_query' , 'word_in_title' , 'word_in_description']


# In[ ]:


from fuzzywuzzy import fuzz
def fuzzy_partial_ratio(string_1 , string_2):
    return fuzz.partial_ratio(string_1, string_2)

def fuzzy_token_sort_ratio(string_1,string_2):
    return fuzz.token_sort_ratio(string_1,string_2)


# In[ ]:


def runFeatureEngineeringSet2(df_all):
    df_all['fuzzy_ratio_in_title'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0],x.split('\t')[1]))
    df_all['fuzzy_ratio_in_description'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0],x.split('\t')[2]))
    df_all['fuzzy_ratio_in_title_description'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0]," ".join(x.split('\t')[1:])))
    df_all['fuzzy_token_sort_ratio_in_title_description'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0]," ".join(x.split('\t')[1:])))
    df_all['fuzzy_token_sort_ratio_in_title'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0],x.split('\t')[1]))
    df_all['fuzzy_token_sort_ratio_in_description'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0],x.split('\t')[2]))
    
    return df_all

df_all = runFeatureEngineeringSet2(df_all)


# In[ ]:





# In[ ]:


def rnd(columnsToTrain):
   print("Random Forests")
   rndForest = RandomForestCapsule()
   _ = rndForest.getTrainingTestingData(df_all, TRAINING_RECORDS, columnsToTrain)

   print(rndForest.run())


# In[ ]:


def xboost(columnsToTrain):
    print("XG Boost")
    xgBoost = XGBoostCapsule()
    _ = xgBoost.getTrainingTestingData(df_all, TRAINING_RECORDS, columnsToTrain)

    print(xgBoost.run())


# In[ ]:


def bag(columnsToTrain):   
    print("Bagging Regressor")
    bagging = BaggingRegressorCapsule()
    _ = bagging.getTrainingTestingData(df_all, TRAINING_RECORDS, columnsToTrain)

    print(bagging.run())


# In[ ]:


def grad(columnsToTrain):
    print("Gradient Boosted Regressor")
    gradRegressor = GradBoostingRegressorCapsule()
    _ = gradRegressor.getTrainingTestingData(df_all, TRAINING_RECORDS, columnsToTrain)

    print(gradRegressor.run())


# In[ ]:


columnsToTrain = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description' , 'fuzzy_ratio_in_title_description' , 'fuzzy_token_sort_ratio_in_title_description']
columnsToTrainMaster = copy(columnsToTrain)
rnd(columnsToTrain)
xboost(columnsToTrain)
bag(columnsToTrain)
grad(columnsToTrain)


# In[ ]:


corr_matrix = df_all.corr()
print(corr_matrix["relevance"].map(lambda x : abs(x)).sort_values(ascending=False))
sns.heatmap(corr_matrix, vmax=.8, square=True);


# In[ ]:


# use only the highest correlation values - we do not care if the co-relation is positive or negative
columnsToTrain = ["fuzzy_ratio_in_title", "fuzzy_ratio_in_title_description", "fuzzy_ratio_in_description", "word_in_title", "word_in_description", "fuzzy_token_sort_ratio_in_title"]

rnd(columnsToTrain)
xboost(columnsToTrain)
bag(columnsToTrain)
grad(columnsToTrain)


# In[ ]:


# use only the highest correlation values - we do not care if the co-relation is positive or negative
columnsToTrain = ["fuzzy_token_sort_ratio_in_title", "fuzzy_ratio_in_title", "fuzzy_ratio_in_description", "fuzzy_ratio_in_title_description", "word_in_title", "fuzzy_token_sort_ratio_in_description"]

rnd(columnsToTrain)
xboost(columnsToTrain)
bag(columnsToTrain)
grad(columnsToTrain)


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
columnsToTrain = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description' , 'fuzzy_ratio_in_title_description' , 'fuzzy_token_sort_ratio_in_title_description']
capObj = TreeCapsule()
x_train, y_train, _, _ = capObj.getTrainingTestingData(df_all, 600, columnsToTrain)


# In[ ]:


skb = SelectKBest(f_regression, k=5)
skb.fit(x_train, y_train)

x_train_transformed = skb.transform(x_train)
sorted(zip(map(lambda x: round(x, 4), skb.scores_), columnsToTrain), 
             reverse=True)


# In[ ]:


columnsToTrain =  ['fuzzy_ratio_in_title', 'fuzzy_ratio_in_title_description','word_in_title', 'fuzzy_ratio_in_description', 'word_in_description']
rnd(columnsToTrain)
xboost(columnsToTrain)
bag(columnsToTrain)
grad(columnsToTrain)


# In[ ]:


from random import shuffle

for column in columnsToTrain:
    print("shuffling column : %s" %column)
    df_all_copy = copy.deepcopy(df_all)
    allValues = list(df_all_copy[column])
    shuffle(allValues)
    df_all_copy[column] = allValues
    
    rnd(columnsToTrain)
    xboost(columnsToTrain)
    bag(columnsToTrain)
    grad(columnsToTrain)
    
    print("\n\n")
    
    


# In[ ]:




