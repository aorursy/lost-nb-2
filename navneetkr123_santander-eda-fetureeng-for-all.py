#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns            #For plots
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


train.shape, test.shape


# In[4]:


train.head(5)


# In[5]:


test.head(5)


# In[6]:


#target = train["target"]


# In[7]:


#train = train.drop(["target"], axis=1)


# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# In[10]:


train.describe()


# In[11]:


test.describe()


# In[12]:


train_vis = train.iloc[:, 1:13]


# In[13]:


train_vis.head(2)


# In[14]:


# PROVIDE CITATIONS TO YOUR CODE IF YOU TAKE IT FROM ANOTHER WEBSITE.
# https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py


y_value_counts = train['target'].value_counts()
print("Number of people transacted the money in future ", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
print("Number of people not transacted the money in future  ", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
#above codes will give the%age of approved and not approved project

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
recipe = ["transacted", "not transacted"]

data = [y_value_counts[1], y_value_counts[0]]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment, **kw)

ax.set_title("Number of people transacted money or not")

plt.show()


# In[15]:


#https://seaborn.pydata.org/generated/seaborn.pairplot.html
plt.close()  #Closing all open window
sns.set_style("whitegrid");
sns.pairplot(train_vis, hue="target", height=3);
plt.show()


# In[16]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[17]:


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]


# In[18]:


features = train.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[19]:


features = train.columns.values[102:202]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[20]:


train_5000 = train.head(5000)
y =train_5000["target"]
x = train_5000.iloc[:,2:202].values


# In[21]:


# https://github.com/pavlin-policar/fastTSNE you can try this also, this version is little faster than sklearn 
#reference: aaic tsne
import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)

X_embedding = tsne.fit_transform(x)
# if x is a sparse matrix you need to pass it as X_embedding = tsne.fit_transform(x.toarray()) , .toarray() will convert the sparse matrix into dense matrix

for_tsne = np.vstack((X_embedding.T, y)).T#y.reshape(-1,1)
for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dim_1','Dim_2','label'])
# Ploting the result of tsne
sns.FacetGrid(for_tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("Visualise tsne ")
plt.show()


# In[22]:


features_train = train.columns.values[2:202]
features_test = test.columns.values[1:201]
row_mean_train = train[features_train].mean(axis=1)
train["row_mean"] =row_mean_train
row_mean_test = test[features_test].mean(axis=1)
test["row_mean"] = row_mean_test


# In[23]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_mean")             .add_legend()
plt.title("Histogram of mean")
plt.ylabel("Density of mean")
plt.plot()


# In[24]:


#reference : aaic haberman


# In[25]:


row_median_train = train[features_train].median(axis=1)
train["row_median"] =row_median_train
row_median_test = test[features_test].median(axis=1)
test["row_median"] = row_median_test


# In[26]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_median")             .add_legend()
plt.title("Histogram of median")
plt.ylabel("Density of median")
plt.plot()


# In[27]:


row_std_train = train[features_train].std(axis=1)
train["row_std"] =row_std_train
row_std_test = test[features_test].std(axis=1)
test["row_std"] = row_std_test


# In[28]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_std")             .add_legend()
plt.title("Histogram of std")
plt.ylabel("Density of std")
plt.plot()


# In[29]:


row_min_train = train[features_train].min(axis=1)
train["row_min"] =row_min_train
row_min_test = test[features_test].min(axis=1)
test["row_min"] = row_min_test


# In[30]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_min")             .add_legend()
plt.title("Histogram of min")
plt.ylabel("Density of min")
plt.plot()


# In[31]:


row_max_train = train[features_train].max(axis=1)
train["row_max"] =row_max_train
row_max_test = test[features_test].max(axis=1)
test["row_max"] = row_max_test


# In[32]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_max")             .add_legend()
plt.title("Histogram of max")
plt.ylabel("Density of max")
plt.plot()


# In[33]:


row_skew_train = train[features_train].skew(axis=1)
train["row_skew"] =row_skew_train
row_skew_test = test[features_test].skew(axis=1)
test["row_skew"] = row_skew_test


# In[34]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_skew")             .add_legend()
plt.title("Histogram of skew")
plt.ylabel("Density of skew")
plt.plot()


# In[35]:


row_kurt_train = train[features_train].kurtosis(axis=1)
train["row_kurt"] =row_kurt_train
row_kurt_test = test[features_test].kurtosis(axis=1)
test["row_kurt"] = row_kurt_test


# In[36]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_kurt")             .add_legend()
plt.title("Histogram of kurt")
plt.ylabel("Density of kurt")
plt.plot()


# In[37]:


row_sum_train = train[features_train].sum(axis=1)
train["row_sum"] =row_sum_train
row_sum_test = test[features_test].sum(axis=1)
test["row_sum"] = row_sum_test


# In[38]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "row_sum")             .add_legend()
plt.title("Histogram of sum")
plt.ylabel("Density of sum")
plt.plot()


# In[39]:


#https://www.kaggle.com/hjd810/keras-lgbm-aug-feature-eng-sampling-prediction
row_ma_train = train[features_train].apply(lambda x: np.ma.average(x), axis=1)
train["ma"] = row_ma_train
row_ma_test = test[features_test].apply(lambda x: np.ma.average(x), axis=1)
test["ma"] = row_ma_test


# In[40]:


#https://seaborn.pydata.org/generated/seaborn.distplot.html
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.average.html
sns.FacetGrid(train, hue = "target", height = 5)             .map(sns.distplot, "ma")             .add_legend()
plt.title("Histogram of ma")
plt.ylabel("Density of ma")
plt.plot()


# In[41]:


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]


# In[42]:


#reference: aaic haberman
counts, bin_edges=np.histogram(t0["row_mean"], bins=10, density=True)
pdf=counts/(sum(counts))
print(pdf);    #this will return 10 values
print(bin_edges);  #this will return 11 values
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label="mean0pdf");
plt.plot(bin_edges[1:], cdf, label="mean0pdf");

counts, bin_edges=np.histogram(t1["row_mean"], bins=10, density=True)
pdf=counts/(sum(counts))
print(pdf);    #this will return 10 values
print(bin_edges);  #this will return 11 values
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label="mean1pdf");
plt.plot(bin_edges[1:], cdf, label="mean1pdf");
plt.legend()
plt.title("Pdf & Cdf of year")
plt.xlabel("Year")
plt.ylabel("percentage")


# In[43]:


#reference aaic haberman
sns.boxplot(x="target", y="row_sum", data=train)
plt.title("Boxplot for row_sum")
plt.plot()


# In[44]:


sns.boxplot(x="target", y="row_mean", data=train)
plt.title("Boxplot for mean")
plt.plot()


# In[45]:


#create a function which makes the plot:
#https://www.kaggle.com/sicongfang/eda-feature-engineering
from matplotlib.ticker import FormatStrFormatter
def visualize_numeric(ax1, ax2, ax3, df, col, target):
    #plot histogram:
    df.hist(column=col,ax=ax1,bins=200)
    ax1.set_xlabel('Histogram')
    
    #plot box-whiskers:
    df.boxplot(column=col,by=target,ax=ax2)
    ax2.set_xlabel('Transactions')
    
    #plot top 10 counts:
    cnt = df[col].value_counts().sort_values(ascending=False)
    cnt.head(10).plot(kind='barh',ax=ax3)
    ax3.invert_yaxis()  # labels read top-to-bottom
#     ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #somehow not working 
    ax3.set_xlabel('Count')


# In[46]:


##https://www.kaggle.com/sicongfang/eda-feature-engineering
for col in list(train.columns[10:20]):
    fig, axes = plt.subplots(1, 3,figsize=(10,3))
    ax11 = plt.subplot(1, 3, 1)
    ax21 = plt.subplot(1, 3, 2)
    ax31 = plt.subplot(1, 3, 3)
    fig.suptitle('Feature: %s'%col,fontsize=5)
    visualize_numeric(ax11,ax21,ax31,train,col,'target')
    plt.tight_layout()


# In[47]:


train.head(2)


# In[48]:


train.to_csv("../train_sant.csv")


# In[49]:


test.to_csv("../test_sant.csv")


# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


# In[51]:


train = pd.read_csv("../train_sant.csv")
#test = pd.read_csv("/content/drive/My Drive/test_santander.csv")


# In[52]:


train.head(2)


# In[53]:


#target values
target = train["target"].values


# In[54]:


#imp features from column 3 to 212
train = train.iloc[:,3:212]


# In[55]:


train.shape


# In[56]:


#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
train, test, y_train, y_test = train_test_split(train, target, test_size=0.4)


# In[57]:


train.head(2)


# In[58]:


test.head(2)


# In[59]:


#https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaler = scaler.fit(train) 
train = scaler.transform(train)
test = scaler.transform(test)


# In[60]:


def batch_predict(clf, data):
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    y_data_pred = []
    tr_loop = data.shape[0] - data.shape[0]%1000
    # consider you X_tr shape is 49041, then your cr_loop will be 49041 - 49041%1000 = 49000
    # in this for loop we will iterate unti the last 1000 multiplier
    for i in range(0, tr_loop, 1000):
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
    # we will be predicting for the last data points
    y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
    return y_data_pred


# In[61]:


#From facebook recommendation applied this code is taken and modified according to use
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    TN = C[0,0]       
    FP = C[0,1]  
    FN = C[1,0]
    TP = C[1,1]
    print("True Positive",TP)
    print("False Negative",FN)
    print("False Positive",FP)
    print("True Negative",TN)
    
    
    
    A =(((C.T)/(C.sum(axis=1))).T)
    
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(30,6))
    
    labels = [0,1]
    # representing A in heatmap format
    cmap=sns.light_palette("Navy", as_cmap=True)#https://stackoverflow.com/questions/37902459/seaborn-color-palette-as-matplotlib-colormap
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    
    plt.show()


# In[62]:


# we are writing our own function for predict, with defined thresould
# we will pick a threshold that will give the least fpr
def predict(proba, threshould, fpr, tpr):
    
    t = threshould[np.argmax(tpr*(1-fpr))]
    
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    predictions = []
    for i in proba:
        if i>=t:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[63]:


#As mentioned in logistic regression assignment I am changing alpha to log to plot a goog graph
import numpy as np
def log_alpha(al):
    alpha=[]
    for i in al:
        a=np.log(i)
        alpha.append(a)
    return alpha    


# In[64]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

lg = SGDClassifier(loss='log', class_weight='balanced', penalty="l2")
alpha=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
parameters = {'alpha':alpha}
clf = GridSearchCV(lg, parameters, cv=3, scoring='roc_auc', n_jobs=-1, return_train_score=True,)
clf.fit(train, y_train)

print("Model with best parameters :\n",clf.best_estimator_)

alpha = log_alpha(alpha)


best_alpha = clf.best_estimator_.alpha
#best_split = clf.best_estimator_.min_samples_split

print(best_alpha)
#print(best_split)

train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score'] 
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(alpha, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(alpha, train_auc, label='Train AUC points')
plt.scatter(alpha, cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha and l1")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[65]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

lg = SGDClassifier(loss='log', alpha=best_alpha, penalty="l2", class_weight="balanced")
#lg.fit(train_1, project_data_y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

sig_clf = CalibratedClassifierCV(lg, method="isotonic")
lg = sig_clf.fit(train, y_train)


y_train_pred = lg.predict_proba(train)[:,1]   
y_test_pred = lg.predict_proba(test)[:,1] 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[66]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,predict(y_train_pred, tr_thresholds, train_fpr, train_fpr))


# In[67]:


print('Test confusion_matrix')
plot_confusion_matrix(y_test,predict(y_test_pred, tr_thresholds, train_fpr, train_fpr))


# In[68]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

svm = SGDClassifier(loss='hinge', class_weight='balanced', penalty="l2")
alpha=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
parameters = {'alpha':alpha}
clf = GridSearchCV(svm, parameters, cv=3, scoring='roc_auc', n_jobs=-1, return_train_score=True,)
clf.fit(train, y_train)

print("Model with best parameters :\n",clf.best_estimator_)

alpha = log_alpha(alpha)


best_alpha = clf.best_estimator_.alpha
#best_split = clf.best_estimator_.min_samples_split

print(best_alpha)
#print(best_split)

train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score'] 
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(alpha, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(alpha, train_auc, label='Train AUC points')
plt.scatter(alpha, cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha and l1")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[69]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

svm = SGDClassifier(loss='hinge', alpha=best_alpha, penalty="l2", class_weight="balanced")
#svm.fit(train_1, project_data_y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

sig_clf = CalibratedClassifierCV(svm, method="isotonic")
svm = sig_clf.fit(train, y_train)


y_train_pred = svm.predict_proba(train)[:,1]   
y_test_pred = svm.predict_proba(test)[:,1] 

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[70]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,predict(y_train_pred, tr_thresholds, train_fpr, train_fpr))


# In[71]:


print('Test confusion_matrix')
plot_confusion_matrix(y_test,predict(y_test_pred, tr_thresholds, train_fpr, train_fpr))


# In[72]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
naive = MultinomialNB(fit_prior=False)
alpha=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
parameters = {'alpha':alpha}
clf = GridSearchCV(naive, parameters, cv=3, scoring='roc_auc', return_train_score=True)
clf.fit(train, y_train)

print("Model with best parameters :\n",clf.best_estimator_)

train_auc= list(clf.cv_results_['mean_train_score'])
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = list(clf.cv_results_['mean_test_score']) 
cv_auc_std= clf.cv_results_['std_test_score']

best_alpha=clf.best_estimator_.alpha

alpha = log_alpha(alpha)

plt.plot(alpha, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(alpha, train_auc, label='Train AUC points')
plt.scatter(alpha, cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha and l1")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[73]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc

naive = MultinomialNB(alpha=best_alpha, fit_prior=False)
naive.fit(train, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = naive.predict_proba(train)[:,1]    
y_test_pred = naive.predict_proba(test)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[74]:


print('Train confusion_matrix')
plot_confusion_matrix(y_train,predict(y_train_pred, tr_thresholds, train_fpr, train_fpr))


# In[75]:


print('Test confusion_matrix')
plot_confusion_matrix(y_test,predict(y_test_pred, tr_thresholds, train_fpr, train_fpr))


# In[76]:


train = pd.read_csv("../train_sant.csv")
test = pd.read_csv("../test_sant.csv")


# In[77]:


#from sklearn.model_selection import train_test_split
#train, test, y_train, y_test = train_test_split(train, target, test_size=0.4)


# In[78]:


#taking all the columns except idcode, target, unnamed0
features = [c for c in train.columns if c not in ['ID_code', 'target',"Unnamed:0"]]
target = train['target']


# In[79]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# In[80]:


#setting parameters
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}


# In[81]:


#making 10 folds
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


# In[82]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,28))
#plotting a bar plot where y represents features and x represents its importance.
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('FI.png')


# In[83]:


sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# In[84]:


#http://zetcode.com/python/prettytable/
from prettytable import PrettyTable

x = PrettyTable()
x.field_names =["Models","Test auc"]
x.add_row(["Logistic ",0.862])
x.add_row(["SVM ",0.863])
x.add_row(["Naive ",0.854])
x.add_row(["LightGbm",0.90])

print(x)

