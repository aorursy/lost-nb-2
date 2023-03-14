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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


'''from sklearn.model_selection import train_test_split

def read_data():
    df = pd.read_csv("../input/train.csv", nrows=20000)
    print ("Shape of base training File = ", df.shape)
    # Remove missing values and duplicates from training data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("Shape of base training data after cleaning = ", df.shape)
    return df

df = read_data()
df_train, df_test = train_test_split(df, test_size = 0.02)
print (df_train.head(2))
print (df_test.shape)'''


# In[3]:


''''''from collections import Counter
import matplotlib.pyplot as plt
import operator

def eda(df):
    print ("Duplicate Count = %s , Non Duplicate Count = %s" 
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
    
    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()
    
    print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))
    
    question_ids_counter = Counter(question_ids_combined)
    sorted_question_ids_counter = sorted(question_ids_counter.items(), key=operator.itemgetter(1))
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
    print ("Count of Quesitons appearing more than once = %s" %(len(question_appearing_more_than_once)))
    
    
eda(df_train)'''


# In[4]:


'''import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.porter import *

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')
stemmer = PorterStemmer()

def tokenize_questions(df):
    question_1_tokenized = []
    question_2_tokenized = []

    for q in df.question1.tolist():
        question_1_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) if i not in stopword])

    for q in df.question2.tolist():
        question_2_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) if i not in stopword])

    df["Question_1_tok"] = question_1_tokenized
    df["Question_2_tok"] = question_2_tokenized
    
    return df

def train_dictionary(df):
    
    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()
    
    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.8)
    dictionary.compactify()
    
    return dictionary
    
df_train = tokenize_questions(df_train)
dictionary = train_dictionary(df_train)
print ("No of words in the dictionary = %s" %len(dictionary.token2id))

df_test = tokenize_questions(df_test)'''


# In[5]:


'''def get_vectors(df, dictionary):
    
    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
    return question1_csc.transpose(),question2_csc.transpose()


q1_csc, q2_csc = get_vectors(df_train, dictionary)

print (q1_csc.shape)
print (q2_csc.shape)'''


# In[6]:


'''from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

minkowski_dis = DistanceMetric.get_metric('minkowski')
mms_scale_man = MinMaxScaler()
mms_scale_euc = MinMaxScaler()
mms_scale_mink = MinMaxScaler()

def get_similarity_values(q1_csc, q2_csc):
    cosine_sim = []
    manhattan_dis = []
    eucledian_dis = []
    jaccard_dis = []
    minkowsk_dis = []
    
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
        sim = md(i,j)
        manhattan_dis.append(sim[0][0])
        sim = ed(i,j)
        eucledian_dis.append(sim[0][0])
        i_ = i.toarray()
        j_ = j.toarray()
        try:
            sim = jsc(i_,j_)
            jaccard_dis.append(sim)
        except:
            jaccard_dis.append(0)
            
        sim = minkowski_dis.pairwise(i_,j_)
        minkowsk_dis.append(sim[0][0])
    
    return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis    


# cosine_sim = get_cosine_similarity(q1_csc, q2_csc)
cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc, q2_csc)
print ("cosine_sim sample= \n", cosine_sim[0:2])
print ("manhattan_dis sample = \n", manhattan_dis[0:2])
print ("eucledian_dis sample = \n", eucledian_dis[0:2])
print ("jaccard_dis sample = \n", jaccard_dis[0:2])
print ("minkowsk_dis sample = \n", minkowsk_dis[0:2])

eucledian_dis_array = np.array(eucledian_dis).reshape(-1,1)
manhattan_dis_array = np.array(manhattan_dis).reshape(-1,1)
minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1,1)
    
manhattan_dis_array = mms_scale_man.fit_transform(manhattan_dis_array)
eucledian_dis_array = mms_scale_euc.fit_transform(eucledian_dis_array)
minkowsk_dis_array = mms_scale_mink.fit_transform(minkowsk_dis_array)

eucledian_dis = eucledian_dis_array.flatten()
manhattan_dis = manhattan_dis_array.flatten()
minkowsk_dis = minkowsk_dis_array.flatten()'''


# In[7]:


''''''from sklearn.metrics import log_loss

def calculate_logloss(y_true, y_pred):
    loss_cal = log_loss(y_true, y_pred)
    return loss_cal

q1_csc_test, q2_csc_test = get_vectors(df_test, dictionary)
y_pred_cos, y_pred_man, y_pred_euc, y_pred_jac, y_pred_mink = get_similarity_values(q1_csc_test, q2_csc_test)
y_true = df_test.is_duplicate.tolist()

y_pred_man_array = mms_scale_man.transform(np.array(y_pred_man).reshape(-1,1))
y_pred_man = y_pred_man_array.flatten()

y_pred_euc_array = mms_scale_euc.transform(np.array(y_pred_euc).reshape(-1,1))
y_pred_euc = y_pred_euc_array.flatten()

y_pred_mink_array = mms_scale_mink.transform(np.array(y_pred_mink).reshape(-1,1))
y_pred_mink = y_pred_mink_array.flatten()

logloss = calculate_logloss(y_true, y_pred_cos)
print ("The calculated log loss value on the test set for cosine sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_man)
print ("The calculated log loss value on the test set for manhattan sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_euc)
print ("The calculated log loss value on the test set for euclidean sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_jac)
print ("The calculated log loss value on the test set for jaccard sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_mink)
print ("The calculated log loss value on the test set for minkowski sim is = %f" %logloss)'''


# In[8]:


'''from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

X_train = pd.DataFrame({"cos" : cosine_sim, "man" : manhattan_dis, "euc" : eucledian_dis, "jac" : jaccard_dis, "min" : minkowsk_dis})
y_train = df_train.is_duplicate

X_test = pd.DataFrame({"cos" : y_pred_cos, "man" : y_pred_man, "euc" : y_pred_euc, "jac" : y_pred_jac, "min" : y_pred_mink})
y_test = y_true

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

svr = SVR()
svr.fit(X_train,y_train)'''


# In[9]:


''''y_rfr_predicted = rfr.predict(X_test)
y_svr_predicted = svr.predict(X_test)

logloss_rfr = calculate_logloss(y_test, y_rfr_predicted)
logloss_svr = calculate_logloss(y_test, y_svr_predicted)

print ("The calculated log loss value on the test set using RFR is = %f" %logloss_rfr)
print ("The calculated log loss value on the test set using SVR is = %f" %logloss_svr)'''


# In[10]:


'''
Predict Duplicate using basic ML + NLP techniques
I am trying to predict the duplicate sentences using vector 
similarity blnCreateSubmitFile and NLP technique in this module and its other forked versions.
'''
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler


from __future__ import unicode_literals
from sklearn.metrics import log_loss
import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import SnowballStemmer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

coding:'utf-8'
FILE_PATH='C:\\Kaggle\\Quora\\QuoraComp\\'
FILE_PATH="../input/"
global df_train 
global X_train 
global y_train
#global cosine_sim, manhattan_dis,eucledian_dis,jaccard_dis,minkowsk_dis
global y_pred_cos,y_pred_euc,y_pred_jac,y_pred_mink,y_pred_man
global y_true

global blnCreateSubmitFile


def calculate_logloss(y_true, y_pred):
    loss_cal = log_loss(y_true, y_pred)
    return loss_cal

def mainPredict():

    global df_train
    global y_true
    #global cosine_sim, manhattan_dis,eucledian_dis,jaccard_dis,minkowsk_dis
    global y_pred_cos,y_pred_euc,y_pred_jac,y_pred_mink,y_pred_man
    global X_train, y_train

    def read_data():
        global blnCreateSubmitFile
        lstrFile='train.csv'

        if blnCreateSubmitFile:
            lstrFile='test.csv'

        df = pd.read_csv(FILE_PATH+lstrFile) #, nrows=250000)
        df.reindex(np.random.permutation(df.index))
        df=df[0:70000]
        print ("Shape of base training File = ", df.shape)

        if not blnCreateSubmitFile:
            # Remove missing values and duplicates from training data
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)

        print("Shape of base training data after cleaning = ", df.shape)
        return df

    df = read_data()

    if blnCreateSubmitFile:
        #create submission file, use df_train variable but it holds test data file.
        df_train =df
        df_test =df[0:6]
        del df
    else:
        df_train, df_test = train_test_split(df, test_size = 0.14)

        print (df_train.head(2))
        print (df_test.shape)
        print (str(len(df_train))+' rows in df_train')


    '''Train Dictionary¶
    First we will tokenize the sentences to extract words from the question. 
    Lets also apply porter stemmer to break down words into their basic form. 
    This should help us increase the accuracy of the system.
    Then we use gensims to train a dictionary of words available in the corpus.
    We are training the dictionary based on the Bag Of Words concept. 
    Gensims dictionary will assign a id to each word which we can use 
    later to convert documents into vectors.
    Also, filter extremes to remove words appearing less than 5 
    times in the corpus or in more than 80% of the questions.
    '''


    words = re.compile(r"\w+",re.I)
    stopword = stopwords.words('english')
    #stemmer = PorterStemmer()
    stemmer = SnowballStemmer('english')
    def tokenize_questions(df):
        question_1_tokenized = []
        question_2_tokenized = []

        try:
            for q in df.question1.tolist():
                question_1_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) if i not in stopword ])

            for q in df.question2.tolist():
                question_2_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) if i not in stopword ])

            df["Question_1_tok"] = question_1_tokenized
            df["Question_2_tok"] = question_2_tokenized

        except Exception as e:
            #print(len(question_1_tokenized))
            print ('hit exception')
            #print (df[78217:78218]['question1'])
        return df

    def train_dictionary(df):
    
        questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()
    
        dictionary = corpora.Dictionary(questions_tokenized)
        dictionary.filter_extremes(no_below=5, no_above=0.8)
        dictionary.compactify()
    
        return dictionary
    
    df_train = tokenize_questions(df_train)
    dictionary = train_dictionary(df_train)
    print ("No of words in the dictionary = %s" %len(dictionary.token2id))

    df_test = tokenize_questions(df_test)


    '''
    Create Vector
    Here we are using the simple method of Bag Of Words Technique
    to convert sentences into vectors.
    There are two vector matrices thus created where each of
    the matrix is a sparse matrix to save memory in the system.
    '''
    def get_vectors(df, dictionary):
    
        question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
        question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    
        question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
        question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
        return question1_csc.transpose(),question2_csc.transpose()


    q1_csc, q2_csc = get_vectors(df_train, dictionary)

    print (q1_csc.shape)
    print (q2_csc.shape)

    '''
    Define Similarity Calculation Functions:
    Here we have defined various Distance calculation functions for
    Cosine Distance
    Euclidean Distance
    Manhattan Distance
    Jaccard Distance
    Minkowski Distance
    As Eucledian, Manhattan and Minkowski Distance 
    may go beyond 1 we must scale them down between0 - 1 ,
    for that we are using MinMaxScaler and training them on training data.
    '''
    minkowski_dis = DistanceMetric.get_metric('minkowski')
    mms_scale_man = MinMaxScaler()
    mms_scale_euc = MinMaxScaler()
    mms_scale_mink = MinMaxScaler()

    def get_similarity_values(q1_csc, q2_csc):
        #global cosine_sim, manhattan_dis,eucledian_dis,jaccard_dis,minkowsk_dis
        cosine_sim = []
        manhattan_dis = []
        eucledian_dis = []
        jaccard_dis = []
        minkowsk_dis = []
    
        for i,j in zip(q1_csc, q2_csc):
            sim = cs(i,j)
            cosine_sim.append(sim[0][0])
            sim = md(i,j)
            manhattan_dis.append(sim[0][0])
            sim = ed(i,j)
            eucledian_dis.append(sim[0][0])
            i_ = i.toarray()
            j_ = j.toarray()
            try:
                sim = jsc(i_,j_)
                jaccard_dis.append(sim)
            except:
                jaccard_dis.append(0)
            
            sim = minkowski_dis.pairwise(i_,j_)
            minkowsk_dis.append(sim[0][0])
    
        return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis    


    # cosine_sim = get_cosine_similarity(q1_csc, q2_csc)
    cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc, q2_csc)
    print ("cosine_sim sample= \n", cosine_sim[0:2])
    print ("manhattan_dis sample = \n", manhattan_dis[0:2])
    print ("eucledian_dis sample = \n", eucledian_dis[0:2])
    print ("jaccard_dis sample = \n", jaccard_dis[0:2])
    print ("minkowsk_dis sample = \n", minkowsk_dis[0:2])

    eucledian_dis_array = np.array(eucledian_dis).reshape(-1,1)
    manhattan_dis_array = np.array(manhattan_dis).reshape(-1,1)
    minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1,1)
    
    manhattan_dis_array = mms_scale_man.fit_transform(manhattan_dis_array)
    eucledian_dis_array = mms_scale_euc.fit_transform(eucledian_dis_array)
    minkowsk_dis_array = mms_scale_mink.fit_transform(minkowsk_dis_array)

    eucledian_dis = eucledian_dis_array.flatten()
    manhattan_dis = manhattan_dis_array.flatten()
    minkowsk_dis = minkowsk_dis_array.flatten()

    '''
    Calculate Log Loss¶
    Here we will use log loss formula to set a base criteria as
    to what accuracy our algorithm is able to achieve in terms 
    of log loss which is the competition calucation score.
    We will also use Eucledian, Manhattan ,
    Minkowski and Jaccard to calculate the similarity and then
    have a look at the log loss from each one of them. These are
    the five most widely used similarity classes used in Data Science
    so Lets use each one of them to see which performs best.
    '''

    q1_csc_test, q2_csc_test = get_vectors(df_test, dictionary)
    y_pred_cos, y_pred_man, y_pred_euc, y_pred_jac, y_pred_mink = get_similarity_values(q1_csc_test, q2_csc_test)
    
    if not blnCreateSubmitFile:
        y_true = df_test.is_duplicate.tolist()

    y_pred_man_array = mms_scale_man.transform(np.array(y_pred_man).reshape(-1,1))
    y_pred_man = y_pred_man_array.flatten()

    y_pred_euc_array = mms_scale_euc.transform(np.array(y_pred_euc).reshape(-1,1))
    y_pred_euc = y_pred_euc_array.flatten()

    y_pred_mink_array = mms_scale_mink.transform(np.array(y_pred_mink).reshape(-1,1))
    y_pred_mink = y_pred_mink_array.flatten()

    if not blnCreateSubmitFile:
        logloss = calculate_logloss(y_true, y_pred_cos)
        print ("The calculated log loss value on the test set for cosine sim is = %f" %logloss)

        logloss = calculate_logloss(y_true, y_pred_man)
        print ("The calculated log loss value on the test set for manhattan sim is = %f" %logloss)

        logloss = calculate_logloss(y_true, y_pred_euc)
        print ("The calculated log loss value on the test set for euclidean sim is = %f" %logloss)

        logloss = calculate_logloss(y_true, y_pred_jac)
        print ("The calculated log loss value on the test set for jaccard sim is = %f" %logloss)

        logloss = calculate_logloss(y_true, y_pred_mink)
        print ("The calculated log loss value on the test set for minkowski sim is = %f" %logloss)

    X_train = pd.DataFrame({"cos" : cosine_sim, "man" : manhattan_dis, "euc" : eucledian_dis, "jac" : jaccard_dis, "min" : minkowsk_dis})
    
    if not blnCreateSubmitFile:
        y_train = df_train.is_duplicate

def fnRunSVR(*pArgs):
    '''
    Adding Machine Learning Models to improve logloss accuracy
    Now in order to improve on the accuracy let us feed the
    results from these similarity coefficients to a Random Forest Regressor
    and Support Vector Regressor and check if we can improve on the log loss values.
    Not concentrating on the hyper parameters of RF and SVM we 
    are just allowing the algorithms to run as it is.
    '''
    print (pArgs)
    pC=int(pArgs[0][0])
    pGamma=pArgs[0][1]
    global blnCreateSubmitFile
    global df_train
    #global cosine_sim, manhattan_dis,eucledian_dis,jaccard_dis,minkowsk_dis
    global y_pred_cos,y_pred_euc,y_pred_jac,y_pred_mink,y_pred_man
    global y_true
    global X_train 
    global y_train
    
    X_test = pd.DataFrame({"cos" : y_pred_cos, "man" : y_pred_man, "euc" : y_pred_euc, "jac" : y_pred_jac, "min" : y_pred_mink})
    
    if not blnCreateSubmitFile:
        y_test = y_true

    #rfr = RandomForestRegressor()
    #rfr.fit(X_train, y_train)

    if not blnCreateSubmitFile:
        svr = SVR(C=pC, gamma=pGamma)
        print ('training SVR model')
        svr.fit(X_train,y_train)
        joblib.dump(svr, 'Quora_SVR.pkl') 
        print ('saved SVR to pickle')
    else:
        #load from pickle saved file
        svr = joblib.load('Quora_SVR.pkl') 

    '''
    Now that we have trained the model . 
    Lets predict duplicate from models and calculate logloss
    from them to check if their is any improvement in the logloss values.
    '''
    #y_rfr_predicted = rfr.predict(X_test)
    if not blnCreateSubmitFile:
        y_svr_predicted = svr.predict(X_test)

        #logloss_rfr = calculate_logloss(y_test, y_rfr_predicted)
        logloss_svr = calculate_logloss(y_test, y_svr_predicted)
        print ("The calculated log loss value on the test set using SVR is = %f" %logloss_svr)
    else:
        y_svr_predicted =svr.predict(X_train)
    #print ("The calculated log loss value on the test set using RFR is = %f" %logloss_rfr)
    
    
    if blnCreateSubmitFile:
        df_train['is_duplicate']=y_svr_predicted
        df_train['is_duplicate'] =np.maximum(0,df_train['is_duplicate'] )
        #submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':y_svr_predicted})
        df_train[['test_id','is_duplicate']].to_csv('QuoraSubmission.csv', index=False)

    return logloss_svr

if __name__=='__main__':
    blnCreateSubmitFile=False
    mainPredict()
    lBounds=[(1,100),(.000000001,.25)]
    fnRunSVR([  5.42448063e+01,   2.37859253e-02])
    #result=differential_evolution(func=fnRunSVR,bounds=lBounds,disp=1)
    #print(result)

