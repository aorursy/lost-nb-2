#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Common imports
import numpy as np
import pandas as pd
import re, string
from itertools import product

# Imports for models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# To format floats
from IPython.display import display
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[ ]:


#Load all the csv files into Pandas dataframes, properly parsing dates

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print("Train DF:\n\n{}\n".format(train.head()))


# In[ ]:


print("Test DF:\n\n{}\n".format(test.head()))


# In[ ]:


print("Train DF:\n")
print(train.info(),"\n\n")
print(train.drop('id',axis=1).describe(),"\n\n")
print("\n\nNulls:\n\n{}\n\n".format(train.isnull().sum()))


# In[ ]:


d = train
split = 0.7
d_train = d[:int(split*len(d))]
d_test = d[int((split)*len(d)):]

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(d_train.comment_text)
test_features = vectorizer.transform(d_test.comment_text)

i = 15000
j = 15000

#An example of what features have inside

words = vectorizer.get_feature_names()[i:i+10]
pd.DataFrame(features[j:j+7,i:i+10].todense(), columns=words)


# In[ ]:


print("Different words:",len(vectorizer.get_feature_names()))


# In[ ]:


def performance(y_true, pred, ann=True):
    acc = accuracy_score(y_true, pred[:,1]>0.5)
    auc = roc_auc_score(y_true, pred[:,1])
    fpr, tpr, thr = roc_curve(y_true, pred[:,1])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 7))
    plt.plot(fpr, tpr, color='royalblue', linewidth="3")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    if ann:
        ax.annotate("Acc: %0.2f" % acc, (0.2,0.7), size=14)
        ax.annotate("AUC: %0.2f" % auc, (0.2,0.6), size=14)


# In[ ]:


model1 = MultinomialNB()
model1.fit(features, d_train.toxic)
pred1 = model1.predict_proba(test_features)

performance(d_test.toxic, pred1)


# In[ ]:


comment = "What a stupid comment is the one you made you dumbass"
print(model1.predict(vectorizer.transform([comment]))[0])


# In[ ]:


comment = "Great comment, thanks for contributing"
print(model1.predict(vectorizer.transform([comment]))[0])


# In[ ]:


vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(d_train.comment_text)

model2 = MultinomialNB()
model2.fit(features, d_train.toxic)

pred2 = model2.predict_proba(vectorizer.transform(d_test.comment_text))
performance(d_test.toxic, pred2)


# In[ ]:


def build_model(Tfid = True, max_features=None, min_df=1, nb_alpha=1.0):
    if Tfid:
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    else:
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
    features = vectorizer.fit_transform(d_train.comment_text)
    model = MultinomialNB(alpha=nb_alpha)
    model.fit(features, d_train.toxic)
    pred = model.predict_proba(vectorizer.transform(d_test.comment_text))
    return {
        "Tfid": Tfid,
        "max_features": max_features,
        "min_df": min_df,
        "nb_alpha": nb_alpha,
        "auc": roc_auc_score(d_test.toxic, pred[:,1])
    }


# In[ ]:


param_values = {
  "Tfid": [True,False],
  "max_features": [10000, 30000, 50000, None],
  "min_df": [1,2,3],
  "nb_alpha": [0.01, 0.1, 1.0]
}

results = []
max_auc = 0

for p in product(*param_values.values()):
    res = build_model(**dict(zip(param_values.keys(), p)))
    results.append(res)
    if res.get('auc')>max_auc:
        max_auc = res.get('auc')
        Tfid_opt = res.get('Tfid')
        max_features_opt = res.get('max_features')
        min_df_opt = res.get('min_df')
        nb_alpha_opt = res.get('nb_alpha')
    print(res)


# In[ ]:


if Tfid_opt:
    vectorizer = TfidfVectorizer(max_features=max_features_opt, min_df=min_df_opt)
else:
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
    
features = vectorizer.fit_transform(d_train.comment_text)

model2_opt = MultinomialNB(alpha=nb_alpha_opt)
model2_opt.fit(features, d_train.toxic)

pred2_opt = model2_opt.predict_proba(vectorizer.transform(d_test.comment_text))
performance(d_test.toxic, pred2_opt)


# In[ ]:


roc_auc_score(d_test.toxic, pred2_opt[:,1])


# In[ ]:


stop_words = set(['all', "she'll", "don't", 'being', 'over', 'through', 
'yourselves', 'its', 'before', "he's", "when's", "we've", 'had', 'should',
"he'd", 'to', 'only', "there's", 'those', 'under', 'ours', 'has', 
"haven't", 'do', 'them', 'his', "they'll", 'very', "who's", "they'd", 
'cannot', "you've", 'they', 'not', 'during', 'yourself', 'him', 'nor', 
"we'll", 'did', "they've", 'this', 'she', 'each', "won't", 'where', 
"mustn't", "isn't", "i'll", "why's", 'because', "you'd", 'doing', 'some', 
'up', 'are', 'further', 'ourselves', 'out', 'what', 'for', 'while', 
"wasn't", 'does', "shouldn't", 'above', 'between', 'be', 'we', 'who', 
"you're", 'were', 'here', 'hers', "aren't", 'by', 'both', 'about', 'would', 
'of', 'could', 'against', "i'd", "weren't", "i'm", 'or', "can't", 'own', 
'into', 'whom', 'down', "hadn't", "couldn't", 'your', "doesn't", 'from', 
"how's", 'her', 'their', "it's", 'there', 'been', 'why', 'few', 'too', 
'themselves', 'was', 'until', 'more', 'himself', "where's", "i've", 'with', 
"didn't", "what's", 'but', 'herself', 'than', "here's", 'he', 'me', 
"they're", 'myself', 'these', "hasn't", 'below', 'ought', 'theirs', 'my', 
"wouldn't", "we'd", 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 
'at', 'have', 'in', 'any', 'if', 'again', 'no', 'that', 'when', 'same', 
'how', 'other', 'which', 'you', "shan't", 'our', 'after', "let's", 'most', 
'such', 'on', "he'll", 'a', 'off', 'i', "she'd", 'yours', "you'll", 'so', 
"we're", "she's", 'the', "that's", 'having', 'once'])

def tokenize(docs):
    pattern = re.compile('[\W_]+', re.UNICODE)
    sentences = []
    for d in docs:
        sentence = d.lower().split(" ")
        sentence = [pattern.sub('', w) for w in sentence]
        sentences.append( [w for w in sentence if w not in stop_words] )
    return sentences

def featurize_w2v(model, sentences):
    f = np.zeros((len(sentences), model.vector_size))
    for i,s in enumerate(sentences):
        for w in s:
            try:
                vec = model[w]
            except KeyError:
                continue
            f[i,:] = f[i,:] + vec
        f[i,:] = f[i,:] / len(s)
    return f

def delete_nans(features):
    rows_to_delete = []
    for i in range(len(features)):
        if np.isnan(features[i].sum()):
            rows_to_delete.append(i)
    return rows_to_delete


# In[ ]:


sentences = tokenize(d_train.comment_text)
model = Word2Vec(sentences, size=500, window=5, min_count=6, sample=1e-3, workers=2)
model.init_sims(replace=True)

features_w2v = featurize_w2v(model, sentences)

rows_to_delete = delete_nans(features_w2v)
features_w2v = np.delete(features_w2v, rows_to_delete, 0)

model3 = RandomForestClassifier(n_estimators=600, n_jobs=-1, max_features="log2")
model3.fit(features_w2v, d_train.toxic.drop(d_train.index[rows_to_delete]))

test_sentences = tokenize(d_test.comment_text)
test_features_w2v = featurize_w2v(model, test_sentences)


# In[ ]:


rows_to_delete = delete_nans(test_features_w2v)
test_features_w2v = np.delete(test_features_w2v, rows_to_delete, 0)


# In[ ]:


pred3 = model3.predict_proba(test_features_w2v)
performance(d_test.toxic.drop(d_test.index[rows_to_delete]), pred3)


# In[ ]:



roc_auc_score(d_test.toxic.drop(d_test.index[rows_to_delete]), pred3[:,1])


# In[ ]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((len(test), len(labels)))


# In[ ]:


sentences = tokenize(train.comment_text)
model_w2v = Word2Vec(sentences, size=500, window=5, min_count=6, sample=1e-3, workers=2)
model_w2v.init_sims(replace=True)
features_w2v = featurize_w2v(model_w2v, sentences)
train_rows_to_delete = delete_nans(features_w2v)
features_w2v = np.delete(features_w2v, train_rows_to_delete, 0)

for i, label in enumerate(labels):
    model_rfc = RandomForestClassifier(n_estimators=600, n_jobs=-1, max_features="log2")
    model_rfc.fit(features_w2v, train[label].drop(train.index[train_rows_to_delete]))

    test_sentences = tokenize(test.comment_text.astype('str'))
    test_features_w2v = featurize_w2v(model_w2v, test_sentences)
    np.nan_to_num(test_features_w2v,copy=False)
    
    preds[:,i] = model_rfc.predict_proba(test_features_w2v)[:,1]
    
    print('{}/{} labels fitted'.format(i+1,len(labels)))


# In[ ]:


subm_id = pd.DataFrame({'id': sample_submission["id"]})
submission = pd.concat([subm_id, pd.DataFrame(preds, columns = labels)], axis=1)
submission.to_csv('submission.csv', index=False)

