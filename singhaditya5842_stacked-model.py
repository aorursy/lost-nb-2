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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from prettytable import PrettyTable

# https://stackoverflow.com/a/14463362/12005970
import warnings
warnings.filterwarnings("ignore")

# import DL libraries
import tensorflow as tf
import keras
from keras.layers import Input, Conv1D, Dense, Activation, LSTM, Reshape
from keras.layers import BatchNormalization, Dropout, concatenate, Embedding, Flatten, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[3]:


d = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/"
df = pd.read_csv(d+"train.csv")
print("Shape of dataset is: ", df.shape)


#keep only those columns which are required
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian',
                    'christian', 'jewish','muslim', 'black', 'white',
                    'psychiatric_or_mental_illness']

cols = ["id","comment_text"] + identity_columns + ["target"]
df = df[cols]

def convert_to_bool(data, cols):
    for col in cols:
        data[col] = np.where(data[col] >= 0.5, True, False)
    return data

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
df = df.astype({"comment_text":"str"})
print(df.dtypes)

df = convert_to_bool(df, identity_columns)

# converting  target feature to 0 and 1
df["target"] = df["target"].apply(lambda x: 1 if x >= 0.5 else 0)
print("\n\n")
df.head()


# In[4]:


# list of punctuations
punc = [".", "?", "!", ",", ";", ":", "-", "--", "(", ")", "[", "]", "{", "}", "'", '"', "..."]
# symbols list
symbols = ["@", "#", "$", "%", "^", "&", "*", "~"]


def feature(data):
    print("Creating hand crafted features...")
    start = time.time()
    data_df = data.copy()
    # 1.
    print(" For 'word_count' feature...")
    data_df['word_count'] = data_df['comment_text'].apply(lambda x : len(x.split()))

    # 2.
    print(" For 'char_count' feature...")
    data_df['char_count'] = data_df['comment_text'].apply(lambda x : len(x.replace(" ","")))

    # 3.
    print(" For 'word_density' feature...")
    data_df['word_density'] = data_df['word_count'] / (data_df['char_count'] + 1)

    # 4.
    print(" For 'total_length' feature...")
    data_df['total_length'] = data_df['comment_text'].apply(len)

    # 5.
    print(" For 'capitals' feature...")
    data_df['capitals'] = data_df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    # 6.
    print(" For 'caps_vs_length' feature...") 
    data_df['caps_vs_length'] = data_df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

    # 7.
    print(" For 'punc_count' feature...")
    data_df['punc_count'] = data_df['comment_text'].apply(lambda x : len([a for a in x if a in punc]))

    # 8.
    print(" For 'num_exclamation_marks' feature...")     
    data_df['num_exclamation_marks'] =data_df['comment_text'].apply(lambda x: x.count('!'))

    # 9.
    print(" For 'exlamation_vs_punc_count' feature...")     
    data_df['exlamation_vs_punc_count'] = data_df['num_exclamation_marks']/data_df['punc_count']

    # 10.
    print(" For 'num_question_marks' feature...")
    data_df['num_question_marks'] = data_df['comment_text'].apply(lambda x: x.count('?'))

    # 11.
    print(" For 'question_vs_punc_count' feature...")     
    data_df['question_vs_punc_count'] = data_df['num_question_marks']/data_df['punc_count']

    # 12.
    print(" For 'num_symbols' feature...")
    data_df['num_symbols'] = data_df['comment_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))

    # 13.
    print(" For 'num_unique_words' feature...")  
    data_df['num_unique_words'] = data_df['comment_text'].apply(lambda x: len(set(w for w in x.split())))
    
    # 14.
    print(" For 'words_vs_unique' feature...") 
    data_df['words_vs_unique'] = data_df['num_unique_words'] / data_df['word_count']

    data_df.fillna(0, inplace = True)
    print("\nALL Done!\nTime take for this is {:.4f} seconds".format(time.time() - start))
    return data_df

df = feature(df)
print(df.shape)
df.head(2)


# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

#removing "not" stop word from stop_words
stop_words = stop_words - {"not"}

def text_process(row):
    try:
        text = row["comment_text"]
        text = str(text).lower()
        porter = PorterStemmer()

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
        
        # stemming the words
        text = porter.stem(text)

        return text

    except:
        print("There is no value in comment_text, so returnin 'nan'")
        
        return np.nan
    
tic = time.time()
print("processing train data...")
df.loc[:,"comment_text"] = df.apply(text_process, axis = 1)
print("Time take to process the text data: {:.2f} seconds".format(time.time()-tic))


# In[6]:


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


# In[7]:


print("Number of datapoints in:\nTrain: {:,}\nCV: {:,}\nTest: {:,}".format(len(y_train),
                                                                           len(y_cv),
                                                                           len(y_test)))

# creating class weight dictionary
a = y_train["target"].value_counts()
cl_wt = {0: a[1], 1: a[0]}
print("\n\n",cl_wt)

subgroups    = identity_columns
actual_label = "target" 
pred_label   = "pred_target"


# In[8]:


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
        print("{}{} for '{}' identity {}".format(" "*40, "*"*15, subgroup, "*"*15))
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
        CV_x = CV_x.astype("float64")
        best_model = tune_GBDT(tr_df, cv_df, train_x, train_y, CV_x, subgroups, actual_label, pred_label)

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

def report_model1(model, tr_df, cv_df, te_df, train_x, cv_x, test_x,
                 subgroups, actual_label, pred_label, CV_bool = True):
    tr_df["pred_target"] = model.predict(train_x, batch_size = 8192, verbose=1)
    cv_df["pred_target"]  = model.predict(cv_x, batch_size = 8192, verbose=1)
    te_df["pred_target"]  = model.predict(test_x, batch_size = 8192, verbose=1)

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

########################################################################################################
############################     function to report the best model     #################################
########################################################################################################

def report_model1(model, tr_df, cv_df, te_df, train_x, cv_x, test_x,
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


# In[9]:


uni_bow_vectorizer = CountVectorizer(min_df = 1, max_features = 10000)
uni_bow_train = uni_bow_vectorizer.fit_transform(X_train["comment_text"].values)
uni_bow_cv = uni_bow_vectorizer.transform(X_cv["comment_text"].values) 
uni_bow_test = uni_bow_vectorizer.transform(X_test["comment_text"].values)

print("Shape of featurized\nTrain data: {}\nCV data: {}\nTest data: {}\n".format(uni_bow_train.shape,
                                                                               uni_bow_cv.shape,
                                                                               uni_bow_test.shape))
hand_crafted_train = X_train[X_train.columns[-14:]].values
hand_crafted_cv    = X_cv[X_train.columns[-14:]].values
hand_crafted_test  = X_test[X_train.columns[-14:]].values

std = StandardScaler()
hand_crafted_train = std.fit_transform(hand_crafted_train)
hand_crafted_cv    = std.transform(hand_crafted_cv)
hand_crafted_test  = std.transform(hand_crafted_test)

from scipy.sparse import hstack

print("Stacking Uni-Gram...")
# uni-gram
train_bow_uni = hstack((uni_bow_train, hand_crafted_train)).tocsr()
cv_bow_uni = hstack((uni_bow_cv, hand_crafted_cv)).tocsr()
test_bow_uni = hstack((uni_bow_test, hand_crafted_test)).tocsr()
print("Shape of data:\n Train: {}\n CV: {}\n Test: {}".format(train_bow_uni.shape,
                                                              cv_bow_uni.shape,
                                                              test_bow_uni.shape))


# In[10]:


#laod the W2V vector
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
file_name = "/kaggle/input/glove6b/glove.6B.200d.txt"
print("Loading the W2V model...")
tic = time.time()
with open(file_name, 'r') as f:
    w2v_loaded_dict = {}
    for line in f:
        values = line.split()
        word = values[0]
        vector = [float(i) for i in values[1:]]
        w2v_loaded_dict[word] = vector

glove_words = w2v_loaded_dict.keys()
print("Done!\nTime taken to laod the mdoel: {:.4f} seconds".format(time.time() - tic))
print("\n{:,} words loaded from the model.".format(len(w2v_loaded_dict)))


# In[11]:


def text_to_seq(texts, keras_tokenizer, max_len):
    """this function  return sequence of text after padding/truncating"""
    x = pad_sequences(keras_tokenizer.texts_to_sequences(texts),
                      maxlen = max_len, padding = 'post',truncating = 'post')
    return x

tokens = Tokenizer()
tokens.fit_on_texts(X_train["comment_text"].values)

max_lenght = 400

# padding the encoded data to make each datapoint of same dimension
encoded_text_train = text_to_seq(X_train["comment_text"].values, tokens, max_lenght)
encoded_text_cv    = text_to_seq(X_cv["comment_text"].values, tokens, max_lenght)
encoded_text_test  = text_to_seq(X_test["comment_text"].values, tokens, max_lenght)

print("Shape of train, cv and test {} features are: {}, {}, {}".format('essay',
                                                                       encoded_text_train.shape,
                                                                       encoded_text_cv.shape,
                                                                       encoded_text_test.shape))

# gettting the length of unique words in train data, and adding (+1)
# becasue of zeros padding and words are encoded from 1 to n
vocab_size = len(tokens.word_index) + 1
# below array will be used in Embedding layer
embedding_matrix1 = np.zeros((vocab_size, 200), dtype = 'float32')
for word, index in tokens.word_index.items():
    embedding_vector = w2v_loaded_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix1[index] = embedding_vector


# In[12]:


#clearing the graph
keras.backend.clear_session()

seq_input  = Input(shape = (max_lenght, ), name = "input_layer")
embed_text = Embedding(input_dim = vocab_size, output_dim = 200,
                            weights=[embedding_matrix1], input_length = max_lenght,
                            trainable = False, name = 'text_embedding')(seq_input)
x   = Conv1D(128, 2, activation='relu', padding='same', name = "conv_1d_1")(embed_text)
x   = MaxPooling1D(5, padding='same', name = "maxpool_1")(x)
x   = Conv1D(128, 3, activation='relu', padding='same', name = "conv_1d_2")(x)
x   = MaxPooling1D(5, padding='same', name = "maxpool_2")(x)
x   = Conv1D(128, 4, activation='relu', padding='same', name = "conv_1d_3")(x)
x   = MaxPooling1D(40, padding='same', name = "maxpool_3")(x)
x   = Flatten(name = "flatten_1")(x)
x   = Dropout(0.3, name = "dropout_1")(x)
x   = Dense(128, activation='relu', name = "dense_layer_1")(x)
out = Dense(1, activation='sigmoid', name = "output_layer")(x)

model2 = Model(inputs = seq_input, outputs = out, name = "Model_2")
print(model2.summary())               


# In[13]:


n_epochs = 10
batch_size = 8192


# In[14]:


es = EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta = 0.001, patience = 10, verbose = 1)

model2.compile(loss='binary_crossentropy', optimizer = "rmsprop", metrics=['acc'])
model2.fit(encoded_text_train, y_train, validation_data = (encoded_text_cv, y_cv),
          epochs = n_epochs, batch_size = batch_size, callbacks = [es])


# In[15]:


# laoding the DL model
print("Loading the trained DL model...")
model_1 = load_model("/kaggle/input/my-models-2/Model_1.h5")

print("laoding the trained ML models...")
with open("/kaggle/input/my-dataset/GBDT1.pkl", "rb") as f:
    model3 = pickle.load(f)
    
with open("/kaggle/input/my-dataset/log_reg1.pkl", "rb") as f:
    model_4 = pickle.load(f)


# In[16]:


# mdoel_1 prediction
print("\npredicting for model_1...\n")
y_pred_tr_1 = model_1.predict(uni_bow_train, batch_size = 8192, verbose=1)
y_pred_cv_1 = model_1.predict(uni_bow_cv, batch_size = 8192, verbose=1)
y_pred_te_1 = model_1.predict(uni_bow_test, batch_size = 8192, verbose=1)

# mdoel_2 prediction
print("\npredicting for model_2...\n")
y_pred_tr_2 = model2.predict(encoded_text_train, batch_size = 8192, verbose=1)
y_pred_cv_2 = model2.predict(encoded_text_cv, batch_size = 8192, verbose=1)
y_pred_te_2 = model2.predict(encoded_text_test, batch_size = 8192, verbose=1)

# model_5 prediction
print("\npredicting for model_4...\n")
y_pred_tr_4 = model_4.predict(train_bow_uni)
y_pred_cv_4 = model_4.predict(cv_bow_uni)
y_pred_te_4 = model_4.predict(test_bow_uni)


train_bow_uni = train_bow_uni.astype("float64")
cv_bow_uni    = cv_bow_uni.astype("float64")
test_bow_uni  = test_bow_uni.astype("float64")

# model_4 prediction
print("\npredicting for model_3...\n")
y_pred_tr_3 = model3.predict(train_bow_uni)
y_pred_cv_3 = model3.predict(cv_bow_uni)
y_pred_te_3 = model3.predict(test_bow_uni)

tr = [y_pred_tr_1, y_pred_tr_2, y_pred_tr_3, y_pred_tr_4]
cv = [y_pred_cv_1, y_pred_cv_2, y_pred_cv_3, y_pred_cv_4]
te = [y_pred_te_1, y_pred_te_2, y_pred_te_3, y_pred_te_4]


# In[17]:


def report_stacked_mode():
    alphas = [2.7*0.88, 1.8*0.81, 0.78, 0.77]

    # define arrays of zeros to store above predicted values
    a = np.zeros((X_train.shape[0], 4))
    b = np.zeros((X_cv.shape[0], 4))
    c = np.zeros((X_test.shape[0], 4))

    # storing with wieghtage
    for i in range(4):
        a[:,i] = alphas[i] * tr[i].flatten()/sum(alphas)
        b[:,i] = alphas[i] * cv[i].flatten()/sum(alphas)
        c[:,i] = alphas[i] * te[i].flatten()/sum(alphas)

    # final prediction
    X_train["pred_target"] = np.sum(a, axis = 1)
    X_cv["pred_target"]    = np.sum(b, axis = 1)
    X_test["pred_target"]  = np.sum(c, axis = 1)

    final_train_auc, _ = return_final_metric(X_train, subgroups, actual_label, pred_label, verbose = False)
    final_cv_auc, _ = return_final_metric(X_cv, subgroups, actual_label, pred_label, verbose = False)
    final_test_auc, _ = return_final_metric(X_test, subgroups, actual_label, pred_label, verbose = False)

    print("Final metric for:\nTrain: {:.5f}\nCV: {:.5f}\nTest: {:.5f}".format(final_train_auc,
                                                                              final_cv_auc,
                                                                              final_test_auc))

    print("\n\nPloting Confusion matrix for whole data....\n")
    plot_confusion_matrix(X_train, X_cv, X_test)
    
    print("\n\nPlotting confusion matrix indentity-wise ...\n")
    plot_confusion_for_each_identity(X_train, X_cv, X_test, subgroups)


# In[18]:


report_stacked_mode()


# In[19]:


print("loading the test data...")
df_test = pd.read_csv("/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

print("\nhand crafted features...")
df_test = feature(df_test)

tic = time.time()
print("\nprocessing train data...")
df_test.loc[:,"comment_text"] = df_test.apply(text_process, axis = 1)
print("Time take to process the text data: {:.2f} seconds".format(time.time()-tic))


# In[20]:


test_hand_craft = df_test[df.columns[-14:]]


# In[21]:


test_stacked.shape


# In[22]:


print("BoW vectorizer...")
test_uni_bow = uni_bow_vectorizer.transform(df_test["comment_text"].values)

print("\nHand crafted vectorizer...")
test_hand_crafted = std.transform(test_hand_craft.values)

print("\nStacking bow and hand crafted...")
test_stacked = hstack((test_uni_bow, test_hand_crafted)).tocsr()

print("word to sequences...")
test_encoded_text = text_to_seq(df_test["comment_text"].values, tokens, max_lenght)


# In[23]:


# mdoel_1 prediction
print("\npredicting for model_1...\n")
y_pred_tr_1 = model_1.predict(test_uni_bow, batch_size = 8192, verbose=1)

# mdoel_2 prediction
print("\npredicting for model_2...\n")
y_pred_tr_2 = model2.predict(test_encoded_text, batch_size = 8192, verbose=1)

# model_5 prediction
print("\npredicting for model_4...\n")
y_pred_tr_4 = model_4.predict(test_stacked)

test_stacked = test_stacked.astype("float64")

# model_4 prediction
print("\npredicting for model_3...\n")
y_pred_tr_3 = model3.predict(test_stacked)

tr = [y_pred_tr_1, y_pred_tr_2, y_pred_tr_3, y_pred_tr_4]


# In[24]:


alphas = [2.7*0.88, 1.8*0.81, 0.78, 0.77]

# define arrays of zeros to store above predicted values
a = np.zeros((df_test.shape[0], 4))

# storing with wieghtage
for i in range(4):
    a[:,i] = tr[i].flatten() * alphas[i] / sum(alphas)


# In[25]:


# final prediction
df_test["prediction"] = np.sum(a, axis = 1)
required_cols = ["id", "prediction"]
df_test = df_test[required_cols]
df_test.head(20)


# In[26]:


# save it
df_test.to_csv('submission.csv', index = False)


# In[ ]:




