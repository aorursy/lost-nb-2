#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()

# Press shift+enter to execute this cell


# In[2]:


# happy customers have TARGET==0, unhappy custormers have TARGET==1
# A little less then 4% are unhappy => unbalanced dataset
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df


# In[3]:


# Top-10 most common values
train.var3.value_counts()[:10]


# In[4]:


# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
train.loc[train.var3==-999999].shape


# In[5]:


# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape


# In[6]:


X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']


# In[7]:


# According to dmi3kno (see https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
# num_var4 is the number of products. Let's plot the distribution:
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()


# In[8]:


# Let's look at the density of the of happy/unhappy customers in function of the number of bank products
sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var4")    .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()


# In[9]:


train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');


# In[10]:


train.var38.describe()


# In[11]:


# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()


# In[12]:


# Histogram for var 38 is not normal distributed
train.var38.hist(bins=1000);


# In[13]:


train.var38.map(np.log).hist(bins=1000);


# In[14]:


# where is the spike between 11 and 12  in the log plot ?
train.var38.map(np.log).mode()


# In[15]:


# What are the most common values for var38 ?
train.var38.value_counts()


# In[16]:


# the most common value is very close to the mean of the other values
train.var38[train['var38'] != 117310.979016494].mean()


# In[17]:


# what if we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()


# In[18]:


# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);


# In[19]:


# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0


# In[20]:


#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())


# In[21]:


train['var15'].describe()


# In[22]:


#Looks more normal, plot the histogram
train['var15'].hist(bins=100);


# In[23]:


# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Unhappy customers are slightly older');


# In[24]:


train.saldo_var30.hist(bins=100)
plt.xlim(0, train.saldo_var30.max());


# In[25]:


# improve the plot by making the x axis logarithmic
train['log_saldo_var30'] = train.saldo_var30.map(np.log)


# In[26]:


# Let's look at the density of the age of happy/unhappy customers for saldo_var30
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "log_saldo_var30")    .add_legend();


# In[27]:


sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "var38", "var15")    .add_legend();


# In[28]:


sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]); # Age must be positive ;-)


# In[29]:


# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]);


# In[30]:


# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend();


# In[31]:


# What is density of n0 ?
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "n0")    .add_legend()
plt.title('Unhappy customers have a lot of features that are zero');


# In[32]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


# In[33]:


# Make a dataframe with the selected features and the target variable
X_sel = train[features+['TARGET']]


# In[34]:


X_sel['var36'].value_counts()


# In[35]:


# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36")    .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');


# In[36]:


# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend();


# In[37]:


sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()
plt.title('If var36==0, only happy customers');


# In[38]:


# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38")    .add_legend();


# In[39]:


train.num_var5.value_counts()


# In[40]:


train[train.TARGET==1].num_var5.value_counts()


# In[41]:


train[train.TARGET==0].num_var5.value_counts()


# In[42]:


sns.FacetGrid(train, hue="TARGET", size=6)    .map(plt.hist, "num_var5")    .add_legend();


# In[43]:


sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "num_var5")    .add_legend();


# In[44]:


sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde");


# In[45]:


train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6));


# In[46]:


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET");


# In[47]:


features


# In[48]:


radviz(train[features+['TARGET']], "TARGET");


# In[49]:


sns.pairplot(train[features+['TARGET']], hue="TARGET", size=2, diag_kind="kde");


# In[50]:


cor_mat = X.corr()


# In[51]:


f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);


# In[52]:


cor_mat = X_sel.corr()


# In[53]:


f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);


# In[54]:


# only important correlations and not auto-correlations
threshold = 0.7
important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0])     .unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]
unique_important_corrs


# In[55]:


# Recipe from https://github.com/mgalardini/python_plotting_snippets/blob/master/notebooks/clusters.ipynb
import matplotlib.patches as patches
from scipy.cluster import hierarchy
from scipy.stats.mstats import mquantiles
from scipy.cluster.hierarchy import dendrogram, linkage


# In[56]:


# Correlate the data
# also precompute the linkage
# so we can pick up the 
# hierarchical thresholds beforehand

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# scale to mean 0, variance 1
train_std = pd.DataFrame(scale(X_sel))
train_std.columns = X_sel.columns
m = train_std.corr()
l = linkage(m, 'ward')


# In[57]:


# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(14, 14),
               row_linkage=l,
               col_linkage=l)


# In[58]:


# Threshold 1: median of the
# distance thresholds computed by scipy
t = np.median(hierarchy.maxdists(l))


# In[59]:


# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(12, 12),
               row_linkage=l,
               col_linkage=l)

# Draw the threshold lines
mclust.ax_col_dendrogram.hlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)
mclust.ax_row_dendrogram.vlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)

# Extract the clusters
clusters = hierarchy.fcluster(l, t, 'distance')
for c in set(clusters):
    # Retrieve the position in the clustered matrix
    index = [x for x in range(m.shape[0])
             if mclust.data2d.columns[x] in m.index[clusters == c]]
    # No singletons, please
    if len(index) == 1:
        continue

    # Draw a rectangle around the cluster
    mclust.ax_heatmap.add_patch(
        patches.Rectangle(
            (min(index),
             m.shape[0] - max(index) - 1),
                len(index),
                len(index),
                facecolor='none',
                edgecolor='r',
                lw=3)
        )

plt.title('Cluster matrix')

pass


# In[60]:


train.var38.mode()


# In[61]:




