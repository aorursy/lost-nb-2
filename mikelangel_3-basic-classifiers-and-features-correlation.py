#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Let's start importing the libraries that we will use
import csv as csv 
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from random import randint
from scipy import stats  

#Here are the sklearn libaries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.decomposition import PCA

#Here we define a function to calculate the Pearson's correlation coefficient
#which we will use in a later part of the notebook
def pearson(x,y):
	if len(x)!=len(y):
		print("I can't calculate Pearson's Coefficient, sets are not of the same length!")
		return
	else:
		sumxy = 0
		for i in range(len(x)):
			sumxy = sumxy + x[i]*y[i]
		return (sumxy - len(x)*np.mean(x)*np.mean(y))			/((len(x)-1)*np.std(x, ddof=1)*np.std(y, ddof=1))


# In[2]:


traindf = pd.read_csv('../input/train.csv', header=0) 
traindf.describe
traindf.head()

traindf.head()


# In[3]:


#Import the dataset and do some basic manipulation
traindf = pd.read_csv('../input/train.csv', header=0) 
#testdf = pd.read_csv('../input/test.csv', header=0) I won't use the test set in this notebook

#We can have a look at the data, shape and types, but I'll skip this step here
#traindf.dtypes
#traindf.info()
#traindf.describe
#The dataset is complete, so there's no need here to clean it from empty entries.
#traindf = traindf.dropna() 

#We separate the features from the classes, 
#we can either put them in ndarrays or leave them as pandas dataframes, since sklearn can handle both. 
#x_train = traindf.values[:, 2:] 
#y_train = traindf.values[:, 1]
x_train = traindf.drop(['id', 'species'], axis=1)
y_train = traindf.pop('species')
#x_test = traindf.drop(['id'], axis=1)

#Sometimes it may be useful to encode labels with numeric values, but is unnecessary in this case 
#le = LabelEncoder().fit(traindf['species']) 
#y_train = le.transform(train['species'])
#classes = list(le.classes_)

#However, it's a good idea to standardize the data (namely to rescale it around zero 
#and with unit variance) to avoid that certain unscaled features 
#may weight more on the classifier decision 
scaler = StandardScaler().fit(x_train) #find mean and std for the standardization
x_train = scaler.transform(x_train) #standardize the training values
#x_test = scaler.transform(x_test)


# In[4]:


y_train.shape


# In[5]:


#Initialise the K-fold with k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=4)


# In[6]:


a = 0
for treno, test in kfold.split(x_train):
    print(test.shape)
    print("OOOOO")
    a = a+1
    if a>=100:
        break


# In[7]:


#Initialise Naive Bayes
nb = GaussianNB()
#We can now run the K-fold validation on the dataset with Naive Bayes
#this will output an array of scores, so we can check the mean and standard deviation
nb_validation=[nb.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()            for train, test in kfold.split(x_train)]


# In[8]:


#Initialise Extra-Trees Random Forest
rf = ExtraTreesClassifier(n_estimators=500, random_state=0)
#Run K-fold validation with RF
#Again the classifier is trained on the k-1 sub-sets and then tested on the remaining k-th subset
#and scores are calcualted
rf_validation=[rf.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]


# In[9]:


#We extract the importances, their indices and standard deviations
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)

#And we plot the first and last 10 features out of curiosity
fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 
ax1.bar(range(10), importances[indices][:10],        color="#6480e5", yerr=imp_std[indices][:10], ecolor='#31427e', align="center")
ax2.bar(range(10), importances[indices][-10:],        color="#e56464", yerr=imp_std[indices][-10:], ecolor='#7e3131', align="center")
ax1.set_xticks(range(10)), ax2.set_xticks(range(10))
ax1.set_xticklabels(indices[:10]), ax2.set_xticklabels(indices[-10:])
ax1.set_xlim([-1, 10]), ax2.set_xlim([-1, 10])
ax1.set_ylim([0, 0.035]), ax2.set_ylim([0, 0.035])
ax1.set_xlabel('Feature #'), ax2.set_xlabel('Feature #')
ax1.set_ylabel('Random Forest Normalized Importance') 
ax2.set_ylabel('Random Forest Normalized Importance')
ax1.set_title('First 10 Important Features'), ax2.set_title('Last 10 Important Features')
gs1.tight_layout(fig)
#plt.show()


# In[10]:


#We first define the ranges for each parameter we are interested in searching 
#(while the others are left as default):
#C is the inverse of the regularization strength
#tol is the tolerance for stopping the criteria
params = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
#We initialise the Logistic Regression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#We initialise the Exhaustive Grid Search, we leave the scoring as the default function of 
#the classifier singe log loss gives an error when running with K-fold cross validation
#add n_jobs=-1 in a parallel computing calculation to use all CPUs available
#cv=3 increasing this parameter makes it too difficult for kaggle to run the script
gs = GridSearchCV(lr, params, scoring=None, refit='True', cv=3) 
gs_validation=[gs.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]


# In[11]:


print("Validation Results\n==========================================")
print("Naive Bayes: " + '{:1.3f}'.format(np.mean(nb_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(nb_validation)))
print("Random Forest: " + '{:1.3f}'.format(np.mean(rf_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(rf_validation)))
print("Logistic Regression: " + '{:1.3f}'.format(np.mean(gs_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(gs_validation)))


# In[12]:


#First we find the sets of margin, shape and texture columns 
margin_cols = [col for col in traindf.columns if 'margin' in col]
shape_cols = [col for col in traindf.columns if 'shape' in col] 
texture_cols = [col for col in traindf.columns if 'texture' in col] 
margin_pear, shape_pear, texture_pear = [],[],[]

#Then we calculate the correlation coefficients for each couple of columns: we can either do this
#between random columns of between consecutive columns, the difference won't matter much since we are
#just exploring the data
for i in range(len(margin_cols)-1):
    margin_pear.append(pearson(traindf[margin_cols[i]],traindf[margin_cols[i+1]]))
	#margin_pear.append(pearson(traindf[margin_cols[randint(0,len(margin_cols)-1)]],\
        #traindf[margin_cols[randint(0,len(margin_cols)-1)]]))
for i in range(len(shape_cols)-1):
	shape_pear.append(pearson(traindf[shape_cols[i]],traindf[shape_cols[i+1]]))
	#shape_pear.append(pearson(traindf[shape_cols[randint(0,len(shape_cols)-1)]],\
        #traindf[shape_cols[randint(0,len(shape_cols)-1)]]))
for i in range(len(texture_cols)-1):
	texture_pear.append(pearson(traindf[texture_cols[i]],traindf[texture_cols[i+1]]))
	#texture_pear.append(pearson(traindf[texture_cols[randint(0,len(texture_cols)-1)]],\
        #traindf[texture_cols[randint(0,len(texture_cols)-1)]]))

#We calculate average and standard deviation for each cathergory 
#and we give it a position on the X axis of the graph
margin_mean, margin_std = np.mean(margin_pear), np.std(margin_pear, ddof=1)
margin_x=[0]*len(margin_pear)
shape_mean, shape_std =	np.mean(shape_pear), np.std(shape_pear, ddof=1)
shape_x=[1]*len(shape_pear)	
texture_mean, texture_std =	np.mean(texture_pear), np.std(texture_pear, ddof=1)	
texture_x=[2]*len(texture_pear)

#We set up the graph
fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 

#We fill the first graph with a scatter plot on a single axis for each category and we add
#mean and standard deviation, which we can also print to screen as a reference
ax1.scatter(margin_x, margin_pear, color='blue', alpha=.3, s=100)
ax1.errorbar([0],margin_mean, yerr=margin_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.scatter(shape_x, shape_pear, color='red', alpha=.3, s=100)
ax1.errorbar([1],shape_mean, yerr=shape_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.scatter(texture_x, texture_pear, color='green', alpha=.3, s=100)
ax1.errorbar([2],texture_mean, yerr=texture_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.set_ylim(-1.25, 1.25), ax1.set_xlim(-0.25, 2.25)
ax1.set_xticks([0,1,2]), ax1.set_xticklabels(['margin','shape','texture'], rotation='vertical')
ax1.set_xlabel('Category'), ax1.set_ylabel('Pearson\'s Correlation')
ax1.set_title('Neighbours Correlation')
ax1.set_aspect(2.5)

print("Pearson's Correlation between neighbours\n==========================================")
print("Margin: " + '{:1.3f}'.format(margin_mean) + u' \u00B1 '        + '{:1.3f}'.format(margin_std))
print("Shape: " + '{:1.3f}'.format(shape_mean) + u' \u00B1 '        + '{:1.3f}'.format(shape_std))
print("Texture: " + '{:1.3f}'.format(texture_mean) + u' \u00B1 '        + '{:1.3f}'.format(texture_std))

#And now, we build a more detailed (and expensive!) correlation matrix, 
#but only for the shape category, which, as we will see, is highly correlated
shape_mat=[]

for i in range(traindf[shape_cols].shape[1]):
    shape_mat.append([])
    for j in range(traindf[shape_cols].shape[1]):
        shape_mat[i].append(pearson(traindf[shape_cols[i]],traindf[shape_cols[j]]))

cmap = cm.RdBu_r
MS= ax2.imshow(shape_mat, interpolation='none', cmap=cmap, vmin=-1, vmax=1)
ax2.set_xlabel('Shape Feature'), ax2.set_ylabel('Shape Feature')
cbar = plt.colorbar(MS, ticks=np.arange(-1.0,1.1,0.2))
cbar.set_label('Pearson\'s Correlation')
ax2.set_title('Shape Category Correlation Matrix')

#And we have a look at the resulting graphs
gs1.tight_layout(fig)
#plt.show()


# In[13]:


#We initialise pca choosing Minka’s MLE to guess the minimum number of output components necessary
#to maintain the same information coming from the input descriptors and we ask to solve SVD in full
pca = PCA(n_components = 'mle', svd_solver = 'full')
#Then we fit pca on our training set and we apply to the same entire set
x_train_pca=pca.fit_transform(x_train)

#Now we can compare the dimensions of the training set before and after applying PCA and see if we 
#managed to reduce the number of features. 
print("Number of descriptors before PCA: " + '{:1.0f}'.format(x_train.shape[1]))
print("Number of descriptors after PCA: " + '{:1.0f}'.format(x_train_pca.shape[1]))


# In[14]:


#Naive Bayes
nb_validation=[nb.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()            for train, test in kfold.split(x_train)]
#Random Forest
rf_validation=[rf.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]
#Logistic Regression
gs_validation=[gs.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]

#And we print the results
print("Validation Results After PCA\n==========================================")
print("Naive Bayes: " + '{:1.3f}'.format(np.mean(nb_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(nb_validation)))
print("Random Forest: " + '{:1.3f}'.format(np.mean(rf_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(rf_validation)))
print("Logistic Regression: " + '{:1.3f}'.format(np.mean(gs_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(gs_validation)))


# In[15]:


#Again, we can check if anything changed in the features importance of our Random Forest classifier
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)

fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 
ax1.bar(range(10), importances[indices][:10],        color="#6480e5", yerr=imp_std[indices][:10], ecolor='#31427e', align="center")
ax2.bar(range(10), importances[indices][-10:],        color="#e56464", yerr=imp_std[indices][-10:], ecolor='#7e3131', align="center")
ax1.set_xticks(range(10)), ax2.set_xticks(range(10))
ax1.set_xticklabels(indices[:10]) ,ax2.set_xticklabels(indices[-10:])
ax1.set_xlim([-1, 10]), ax2.set_xlim([-1, 10])
ax1.set_ylim([0, 0.035]), ax2.set_ylim([0, 0.035])
ax1.set_xlabel('Feature #'), ax2.set_xlabel('Feature #')
ax1.set_ylabel('Random Forest Normalized Importance'), ax2.set_ylabel('Random Forest Normalized Importance')
ax1.set_title('First 10 Important Features (after PCA)'), ax2.set_title('Last 10 Important Features (after PCA)')
gs1.tight_layout(fig)
#plt.show()

