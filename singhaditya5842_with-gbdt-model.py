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


import os, time, re

import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import pickle

from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from prettytable import PrettyTable

# https://stackoverflow.com/a/14463362/12005970
import warnings
warnings.filterwarnings("ignore")


# In[3]:


d = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/"
df = pd.read_csv(d+"train.csv")
print("Shape of dataset is: ", df.shape)
# df.head(2)

#keep only those columns which are required
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian',
                    'christian', 'jewish','muslim', 'black', 'white',
                    'psychiatric_or_mental_illness']

cols = ["id","comment_text"] + identity_columns + ["target"]
df = df[cols]
print("Now shape of the data: ", df.shape)
df.head(2)


# In[4]:


def convert_to_bool(data, cols):
    for col in cols:
        data[col] = np.where(data[col] >= 0.5, True, False)
    return data

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
# df = df.astype({"comment_text":"str"})
print(df.dtypes)

df = convert_to_bool(df, identity_columns)

# converting  target feature to 0 and 1
df["target"] = df["target"].apply(lambda x: 1 if x >= 0.5 else 0)
print("\n\n")
df.head()


# In[5]:


comments = df["comment_text"].values
print(comments[0])
print("="*100)
print(comments[50])
print("="*100)
print(comments[100])
print("="*100)
print(comments[1000])
print("="*100)


# In[6]:


import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

#removing "not" stop word from stop_words
stop_words = stop_words - {"not"}


# In[7]:


def text_process(row):
    try:
        text = row["comment_text"]
        text = str(text).lower()

        #expansion
        text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")        .replace("n't", " not").replace("what's", "what is").replace("it's", "itis")        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar")        .replace("€", " euro ").replace("'ll", " will")

        text = re.sub(r"<.*?>","", text) # removes the htmltags: https://stackoverflow.com/a/12982689

        #special character removal
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        #extra space removal
        text = re.sub('\s+',' ', text)

        # stopword removal
        text_to_words = []
        for word in text.split():
            if word not in stop_words:
                text_to_words.append(word)
            else:
                continue
        text = " ".join(text_to_words)

        return text
    except:
        print("There is no value in comment_text, so returnin 'nan'")
        return np.nan


# In[8]:


tic = time.time()
print("processing train data...")
df.loc[:,"comment_text"] = df.apply(text_process, axis = 1)
print("Time take to process the text data: {:.2f} seconds".format(time.time()-tic))


# In[9]:


X = df[[col for col in df.columns]]
Y = df[["target"]]

X_train, X_test, y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    train_size = 0.8,
                                                                    stratify = Y, 
                                                                    random_state = 42)
X_test, X_cv, y_test, y_cv = model_selection.train_test_split(X_test, Y_test,
                                                              train_size = 0.5,
                                                              stratify = Y_test,
                                                              random_state = 42)

print("Number of datapoints in train data: {:,}\nNumber of datapoints in CV data: {:,}\nNumber of datapoints in test data: {:,}".format(X_train.shape[0],
                                              X_cv.shape[0],
                                              X_test.shape[0]))


# In[10]:


a = y_train["target"].value_counts()
cl_wt = {0: a[1], 1: a[0]}
print(cl_wt)

subgroups    = identity_columns
actual_label = "target" 
pred_label   = "pred_target"


# In[11]:


from sklearn.metrics import roc_auc_score

########################################################################################
#######################     function to calculate the AUC        #######################
########################################################################################

def cal_auc(y_true, y_pred):
    "returns the auc value"
    return roc_auc_score(y_true, y_pred)

########################################################################################
#######################  function to calculate the Subgroup AUC  #######################
########################################################################################

def cal_subgroup_auc(data, subgroup, actual_label, pred_label):
    subgroup_examples = data[data[subgroup]]
    return cal_auc(subgroup_examples[actual_label], subgroup_examples[pred_label])

########################################################################################
#######################   function to calculate the BPSN AUC     #######################
########################################################################################

def cal_bpsn_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BPSN auc"""
    # subset where subgroup is True and target label is 0
    subgroup_negative_examples = data[data[subgroup] & ~data[actual_label]]

    # subset where subgroup is False and target label is 1
    background_positive_examples = data[~data[subgroup] & data[actual_label]]

    # combine above tow sets
    bpsn_examples = subgroup_negative_examples.append(background_positive_examples)

    return cal_auc(bpsn_examples[actual_label], bpsn_examples[pred_label])


########################################################################################
#######################   function to calculate the BNSP AUC     #######################
########################################################################################
def cal_bnsp_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BNSP auc"""
    # subset where subgroup is True and target label is 1
    subgroup_positive_examples = data[data[subgroup] & data[actual_label]]

    # subset where subgroup is False and target label is 0
    background_negative_examples = data[~data[subgroup] & ~data[actual_label]]

    # combine above tow sets
    bnsp_examples = subgroup_positive_examples.append(background_negative_examples)

    return cal_auc(bnsp_examples[actual_label], bnsp_examples[pred_label])

########################################################################################
#######################    function to calculate Bias metric     #######################
########################################################################################
def cal_bias_metric(data, subgroups, actual_label, pred_label):
    """Computes per-subgroup metrics for all subgroups and one model
    and returns the dataframe which will have all three Bias metrices
    and number of exmaples for each subgroup"""
    records = []
    for subgroup in subgroups:
        record = {"subgroup": subgroup, "subgroup_size": len(data[data[subgroup]])}
        record["subgroup_auc"] = cal_subgroup_auc(data, subgroup, actual_label, pred_label)
        record["bpsn_auc"]     = cal_bpsn_auc(data, subgroup, actual_label, pred_label)
        record["bnsp_auc"]     = cal_bnsp_auc(data, subgroup, actual_label, pred_label)

        records.append(record)
    submetric_df = pd.DataFrame(records).sort_values("subgroup_auc", ascending = True)

    return submetric_df

########################################################################################
#######################   function to calculate Overall metric   #######################
########################################################################################
def cal_overall_auc(data, actual_label, pred_label):
    return roc_auc_score(data[actual_label], data[pred_label])

########################################################################################
#######################    function to calculate final metric    #######################
########################################################################################
def power_mean(series, p):
    total_sum = np.sum(np.power(series, p))
    return np.power(total_sum/len(series), 1/p)

def final_metric(submetric_df, overall_auc, p = -5, w = 0.25):
    generalized_subgroup_auc = power_mean(submetric_df["subgroup_auc"], p)
    generalized_bpsn_auc = power_mean(submetric_df["bpsn_auc"], p)
    generalized_bnsp_auc = power_mean(submetric_df["bnsp_auc"], p)
    
    overall_metric = w*overall_auc + w*(generalized_subgroup_auc
                                        + generalized_bpsn_auc
                                        + generalized_bnsp_auc)
    return overall_metric


########################################################################################
#######################   function all above function into one   #######################
########################################################################################

def return_final_metric(data, subgroups,actual_label, pred_label, verbose = False):
    """Data is dataframe which include whole data 
    and it also has the predicted target column"""
    submetric_df = cal_bias_metric(data, subgroups, actual_label, pred_label)

    if verbose:
        print("printing the submetric table for each identity or subgroup")
        print(submetric_df)

    overall_auc =  cal_overall_auc(data, actual_label, pred_label)
    overall_metric = final_metric(submetric_df, overall_auc, p = -5, w = 0.25)

    return overall_metric, submetric_df


from sklearn.metrics import confusion_matrix

########################################################################################
#######################    function to plot Confusion matrix     #######################
########################################################################################
def plot_confusion_matrix(train, cv, test):
    tr_pred = np.where(train["pred_target"] >= 0.5, 1, 0)
    cv_pred = np.where(cv["pred_target"] >= 0.5, 1, 0)
    te_pred = np.where(test["pred_target"] >= 0.5, 1, 0)   

    tr_con_mat = confusion_matrix(train["target"], tr_pred)
    cv_con_mat = confusion_matrix(cv["target"], cv_pred)
    te_con_mat = confusion_matrix(test["target"], te_pred)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(19,4))
    sns.heatmap(tr_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax1)
    ax1.set_title("For Train data", fontsize = 15)
    ax1.set_xlabel("Pridicted target", fontsize = 12)
    ax1.set_ylabel("Actual target", fontsize = 12)

    sns.heatmap(cv_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax2)
    ax2.set_title("For CV data", fontsize = 15)
    ax2.set_xlabel("Pridicted target", fontsize = 12)
    ax2.set_ylabel("Actual target", fontsize = 12)
    
    sns.heatmap(te_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax3)
    ax3.set_title("For Test data", fontsize = 15)
    ax3.set_xlabel("Pridicted target", fontsize = 12)
    ax3.set_ylabel("Actual target", fontsize = 12)

    plt.show()

########################################################################################
##############    function to plot Confusion matrix for each identity   ################
########################################################################################
def plot_confusion_for_each_identity(train, cv, test, subgroups):
    for subgroup in subgroups:
        print("{}{} for '{}' identity {}".format(" "*5, "*"*10, subgroup, "*"*10))
        TR, CV, TE = train[train[subgroup]], cv[cv[subgroup]], test[test[subgroup]]
        plot_confusion_matrix(TR, CV, TE)
        print("\n\n")
        
def plot_auc(params, train_auc, cv_auc, hyp_name):
    plt.figure(figsize = (12,8))
    plt.plot(params, train_auc, "bo-", label = "Train")
    plt.plot(params, cv_auc, "ro-", label = "CV")
    plt.title("Final Metric (AUC) Plot", fontsize = 18)
    plt.xlabel("Hyperparameter '{}'".format(hyp_name), fontsize = 14)
    plt.ylabel("Modified AUC (Final Metric)", fontsize = 14)
    plt.legend(fontsize = 14)
    plt.grid(1)
    plt.show()

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm-lgbmclassifier

########################################################################################################
############################          function to tune the GBDT        #################################
########################################################################################################

def tune_GBDT(train_x, train_y):
    params = {"n_estimators":[50, 100, 200, 400, 500, 800, 1000],
            "max_depth": [2, 5, 8, 10, 15, 25, 50],
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "colsample_bytree": [0.4, 0.6, 0.8, 1.0]}
            
    gbdt_clf   = LGBMClassifier(boosting_type = "gbdt", n_jobs = -1,
                               class_weight = cl_wt, random_state = 42)
    gbdt_model = RandomizedSearchCV(gbdt_clf, params, cv = 3,
                                    scoring = "roc_auc", n_jobs = -1)
    start = time.time()
    print('Tunning the parameters...')
    gbdt_model.fit(train_x, train_y)
    print("Done!\nTime take to tune the hyper-parameters: {:.4f} seconds.\n".format(time.time()-start))

    print("Best parameters of tunned model is:\n", gbdt_model.best_params_)
    print("\nBest score of of tunned model is: {:.4f}\n".format(gbdt_model.best_score_))

    best_model = gbdt_model.best_estimator_
    print("Best model is:\n", best_model)
    return best_model


# In[12]:


########################################################################################################
############################      function to train the best model     #################################
########################################################################################################

def return_best_model(tr_df, cv_df, train_x, train_y, CV_x, 
                      subgroups, actual_label, pred_label, model_name, path, gram):
    """retruns trained model and save it given path"""
    if model_name == "log_reg":
        best_model = tune_log_reg(tr_df, cv_df, train_x, train_y, CV_x, subgroups, actual_label, pred_label)
    elif model_name == "DT":
        best_model = tune_DT(tr_df, cv_df, train_x, train_y, CV_x, subgroups, actual_label, pred_label)
    elif model_name == "RF":
        best_model = tune_RF(train_x, train_y)
    elif model_name == "GBDT":
        # it accepts training data as float32 or float64 only.
        #so don't give it integers value
        train_x = train_x.astype("float64")
        best_model = tune_GBDT(train_x, train_y)
    print("\n\nTraining the best model...")
    best_model.fit(train_x, train_y)

    file = path + model_name+str(gram) + ".pkl"
    print("Saving the model in path...")
    with open(file, 'wb') as f:
        pickle.dump(best_model, f)

    return best_model

########################################################################################################
############################     function to report the best model     #################################
########################################################################################################

def report_model(model, tr_df, cv_df, te_df, train_x, cv_x, test_x,
                 subgroups, actual_label, pred_label, CV_bool = True):
    tr_df["pred_target"] = model.predict(train_x)
    cv_df["pred_target"]  = model.predict(cv_x)
    te_df["pred_target"]  = model.predict(test_x)

    final_train_auc, _ = return_final_metric(tr_df, subgroups, actual_label, pred_label, verbose = False)
    final_cv_auc, _ = return_final_metric(cv_df, subgroups, actual_label, pred_label, verbose = False)
    final_test_auc, _ = return_final_metric(te_df, subgroups, actual_label, pred_label, verbose = False)

    print("Final metric for:\nTrain: {:.5f}\nCV: {:.5f}\nTest: {:.5f}".format(final_train_auc,
                                                                              final_cv_auc,
                                                                              final_test_auc))
    print("\n\nPloting Confusion matrix for whole data....\n")
    plot_confusion_matrix(tr_df, cv_df, te_df)
    
    print("\n\nPlotting confusion matrix indentity-wise ...\n")
    plot_confusion_for_each_identity(tr_df, cv_df, te_df, subgroups)

    return final_train_auc, final_cv_auc, final_test_auc


# In[13]:


uni_bow_vectorizer = CountVectorizer(min_df = 1, max_features = 10000)
uni_bow_train = uni_bow_vectorizer.fit_transform(X_train["comment_text"].values)
uni_bow_cv = uni_bow_vectorizer.transform(X_cv["comment_text"].values) 
uni_bow_test = uni_bow_vectorizer.transform(X_test["comment_text"].values)


# In[14]:


uni_bow_train = uni_bow_train.astype("float64")
uni_bow_cv    = uni_bow_cv.astype("float64")
uni_bow_test  = uni_bow_test.astype("float64")

if os.path.isfile("/kaggle/input/my-model/GBDT1.pkl"):
    with open("/kaggle/input/my-model/GBDT1.pkl", 'rb') as f:
        best_model = pickle.load(f)

    tr, cv, te = report_model(best_model, X_train, X_cv, X_test,
                            uni_bow_train, uni_bow_cv, uni_bow_test, 
                            subgroups, actual_label, pred_label)
else:
    print("Tunned model is not is not in disk. Run above cell.")


# In[15]:


# load the test data, preprocess it, vectorize it using train data then predict it
submission = pd.read_csv("/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

#preprocess
tic = time.time()
print("processing test data...")
submission.loc[:,"comment_text"] = submission.apply(text_process, axis = 1)
print("Time take to process the text data: {:.2f} seconds".format(time.time()-tic))
submission.head(2)


# In[16]:


test_x_to_submit = uni_bow_vectorizer.transform(submission["comment_text"].values)
print(test_x_to_submit.shape)


# In[ ]:





# In[17]:


test_x_to_submit = test_x_to_submit.astype("float64")
y_test_pred = best_model.predict(test_x_to_submit)
y_test_pred = y_test_pred.astype("float32")
submission["prediction"] = y_test_pred
submission.drop(["comment_text"], inplace = True, axis = 1)
submission.head()


# In[18]:


submission.to_csv("submission.csv", index = False)


# In[ ]:




