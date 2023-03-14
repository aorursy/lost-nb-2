#!/usr/bin/env python
# coding: utf-8

# In[1]:


# will need this later for dynamic time warping
get_ipython().system('pip install dtw-python')


# In[2]:


import numpy as np
import pandas as pd 
from random import sample
import random
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# dynamic time warping
from dtw import *


# In[3]:


# load data
sales = pd.read_csv("../input/m5-forecasting-uncertainty/sales_train_validation.csv")


# In[4]:


sales.loc[sales.item_id == 'HOBBIES_1_001']    .sort_values("id")    .head()


# In[5]:


# make time series columns integers for more meaningful plotting (can order numbers)
_cols = list(sales.columns)
sales.columns = pd.Index(_cols[:6] + [int(c.replace("d_","")) for c in _cols[6:]])
del _cols


# In[6]:


def series_from_id(_id:str) -> pd.DataFrame:
    """
    Get a daily time series for a single id
    """
    return sales.loc[sales.id == _id]    .iloc[:,6:]    .T

# Create a global lookup table for fast plotting by department
daily_sales_dept_lookup = sales[["dept_id"] + list(sales.columns[6:])]    .melt(id_vars = "dept_id")    .groupby("dept_id variable".split())    .agg({"value":"sum"})

def series_from_dept(dept:str) -> pd.DataFrame:
    return daily_sales_dept_lookup.loc[dept]

# create a global lookup table for fast plotting by item
daily_sales_item_lookup = sales[["item_id"] + list(sales.columns[6:])]    .melt(id_vars = "item_id")    .groupby("item_id variable".split())    .agg({"value":"sum"})

def series_from_item(item:str) -> pd.DataFrame:
    return daily_sales_item_lookup.loc[item]

"""
Time series for particular items are quite noisy on a daily level. 
Provide the ability to bin sales (for examply - to a weekly bin) for more stable plots
"""
def series_from_id_binned(_id:str, bin_every:int = 7) -> pd.DataFrame:
    """
    Get the sales for an id, grouped by a fixed interval (default 7 - weekly)
    """
    t = series_from_id(_id).reset_index()
    t["index"] = t.index.map(lambda x: x - (x % bin_every))
    t.columns = pd.Index(["day", "sales"])
    return t.groupby("day")        .agg({"sales":"sum"})

def series_from_dept_binned(dept:str, bin_every:int = 7) -> pd.DataFrame:
    """
    Get the sales for a department, grouped by a fixed interval (default 7 - weekly)
    """
    t = series_from_dept(dept).reset_index()
    t["variable"] = t.index.map(lambda x: x - (x % bin_every))
    return t.groupby("variable")        .agg({"value":"sum"})

def series_from_item_binned(item:str, bin_every:int = 7) -> pd.DataFrame:
    """
    Get the sales for an item (across stores), grouped by a fixed interval (default 7 - weekly)
    """
    t = series_from_item(item).reset_index()
    t["variable"] = t.index.map(lambda x: x - (x % bin_every))
    return t.groupby("variable")        .agg({"value":"sum"})


# In[7]:


fig, axes = plt.subplots(nrows = 5, figsize = (12,20))
_ids = sales["id"].sample(n = 5, random_state = 1)
for i in range(len(_ids)):
    series_from_id(_ids.iloc[i]).plot(ax = axes[i])
del _ids


# In[8]:


# bin the items by week and plot again
fig, axes = plt.subplots(nrows = 5, figsize = (12,20))
_ids = sales["id"].sample(n = 5, random_state = 1)
for i in range(len(_ids)):
    series_from_id_binned(_ids.iloc[i], bin_every = 7).plot(ax = axes[i])


# In[9]:


fig, axes = plt.subplots(nrows = 5, figsize = (12,20))
random.seed(2)
_ids = sample(list(sales["item_id"].unique()), 5)
for i in range(len(_ids)):
    series_from_item_binned(_ids[i], bin_every = 7).plot(ax = axes[i])
    axes[i].set_title("Item: %s" % _ids[i])


# In[10]:


fig, axes = plt.subplots(nrows = 5, figsize = (12,20))
random.seed(3)
_ids = sample(list(sales["dept_id"].unique()), 5)
for i in range(len(_ids)):
    series_from_dept_binned(_ids[i], bin_every = 7).plot(ax = axes[i])
    axes[i].set_title("Department: %s" % _ids[i])


# In[11]:


# plotting 10 series, for demonstration
daily_sales_item_lookup.pivot_table(index = "variable", columns = "item_id", values = "value")    .iloc[:,:5]    .plot(figsize = (12,6))


# In[12]:


# Create a lookup table for scaled series
daily_sales_item_lookup_scaled = daily_sales_item_lookup    .pivot_table(index = "variable", columns = "item_id", values = "value").copy()
daily_sales_item_lookup_scaled = daily_sales_item_lookup_scaled.div(daily_sales_item_lookup_scaled.mean(axis = 0), axis = 1)
# bin by week
daily_sales_item_lookup_scaled_weekly = daily_sales_item_lookup_scaled.copy().reset_index()
daily_sales_item_lookup_scaled_weekly["variable"] = daily_sales_item_lookup_scaled_weekly.variable.map(lambda x: x - (x%7))
daily_sales_item_lookup_scaled_weekly = daily_sales_item_lookup_scaled_weekly.groupby("variable").mean()


# In[13]:


# plot those same series, but this time normalized by the series' means. 
random.seed(1)
daily_sales_item_lookup_scaled_weekly.iloc[:,random.sample(range(1000),10)]    .plot(figsize = (12,6))


# In[14]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, ward, dendrogram


# In[15]:


# heirarchical clustering of scales weekly item sales. 
clf = AgglomerativeClustering(n_clusters=None, distance_threshold = 0).fit(daily_sales_item_lookup_scaled_weekly.T.values)


# In[16]:


# given a linkage model, plog dendogram, with the colors indicated by the a cutoff point at which we define clusters
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

plt.figure(figsize = (14,6))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
Z = plot_dendrogram(clf, p=5, color_threshold = 110)
plt.show()


# In[17]:


# extract clusters from dendogram
clusters = fcluster(Z, 100, criterion='distance')
# create a lookup table for series in a given cluster
daily_sales_item_lookup_scaled_clustered = daily_sales_item_lookup_scaled_weekly.T.reset_index()
daily_sales_item_lookup_scaled_clustered["cluster"] = clusters
daily_sales_item_lookup_scaled_clustered = daily_sales_item_lookup_scaled_clustered.set_index("cluster item_id".split())    .sort_index()


# In[18]:


# cluster 1
random.seed(1)
daily_sales_item_lookup_scaled_clustered.loc[1]    .T    .iloc[:, random.sample(range(daily_sales_item_lookup_scaled_clustered.loc[1].shape[0]), 10)]    .plot(figsize = (12,6))


# In[19]:


# series 2
random.seed(1)
daily_sales_item_lookup_scaled_clustered.loc[2]    .T    .iloc[:, random.sample(range(daily_sales_item_lookup_scaled_clustered.loc[2].shape[0]), 10)]    .plot(figsize = (12,6))


# In[20]:


# cluster 3
random.seed(1)
daily_sales_item_lookup_scaled_clustered.loc[3]    .T    .iloc[:, random.sample(range(daily_sales_item_lookup_scaled_clustered.loc[3].shape[0]), 10)]    .plot(figsize = (12,6))


# In[21]:


# cluster 7
random.seed(1)
daily_sales_item_lookup_scaled_clustered.loc[7]    .T    .iloc[:, random.sample(range(daily_sales_item_lookup_scaled_clustered.loc[7].shape[0]), 10)]    .plot(figsize = (12,6))


# In[22]:


# show two series that look similar but are misaligned, for demonstration purposes
fig, [ax1,ax2] = plt.subplots(nrows = 2, figsize = (12,6))
daily_sales_item_lookup_scaled_weekly["HOBBIES_1_062"].plot(ax = ax1, color = "C0")
daily_sales_item_lookup_scaled_weekly["HOUSEHOLD_2_040"].plot(ax = ax2, color = "C1")
ax1.set_title("HOBBIES_1_062", fontsize= 14)
ax2.set_title("HOUSEHOLD_2_040", fontsize= 14)
ax1.set_xlabel("")
ax2.set_xlabel("Days since start")


# In[23]:


## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(
     daily_sales_item_lookup_scaled_weekly["HOUSEHOLD_2_040"],\
    daily_sales_item_lookup_scaled_weekly["HOBBIES_1_062"],\
    keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(3, "c"))\
    .plot(type="twoway",offset=10)


# In[24]:


def get_dtw_diff_matrix(cols:list):
    """
    From a list of series, compute a distance matrix by computing the 
    DTW distance of all pairwise combinations of series.
    """
    diff_matrix = {}
    cross = itertools.product(cols, cols)
    for (col1, col2) in cross:
        series1 = daily_sales_item_lookup_scaled_weekly[col1]
        series2 = daily_sales_item_lookup_scaled_weekly[col2]
        diff = dtw(
            series1, 
            series2,
            keep_internals=True, 
            step_pattern=rabinerJuangStepPattern(2, "c")
            )\
            .normalizedDistance
        diff_matrix[(col1, col2)] = [diff]
    return diff_matrix


# In[25]:


# sample 50 series, and compute the DTW distance matrix
random.seed(1)
sample_cols = random.sample(list(daily_sales_item_lookup_scaled_weekly.columns), 50)
dtw_diff_dict = get_dtw_diff_matrix(sample_cols)
# make into a df
dtw_diff_df = pd.DataFrame(dtw_diff_dict).T.reset_index()    .rename(columns = {"level_0":"item1", "level_1":"item2", 0:"diff"})    .pivot_table(index = "item1", columns = "item2", values = "diff")


# In[26]:


# plot a similarity matrix, with a dendogram imposed
import seaborn as sns
sns.clustermap(1-dtw_diff_df)


# In[27]:


# ward clustering from difference matrix, where distance is Dynamic time warping distance instead of Euclidean
t = ward(dtw_diff_df)
# extract clusters
dtw_clusters = pd.DataFrame({"cluster":fcluster(t, 1.15)}, index = dtw_diff_df.index)


# In[28]:


dtw_clusters.cluster.value_counts().sort_index().plot.barh()
plt.title("Frequency of DTW clusters", fontsize = 14)


# In[29]:


# cluster 1
daily_sales_item_lookup_scaled_weekly.T.merge(
    dtw_clusters.loc[dtw_clusters.cluster == 1], 
    left_index = True,
    right_index = True
)\
    .T\
    .plot(figsize = (12,4))


# In[30]:


def plot_dtw(series1:str, series2:str) -> None:
    dtw(daily_sales_item_lookup_scaled_weekly[series1],            daily_sales_item_lookup_scaled_weekly[series2],        keep_internals=True, 
        step_pattern=rabinerJuangStepPattern(2, "c"))\
        .plot(type="twoway",offset=5)

plot_dtw("FOODS_1_119", "HOUSEHOLD_2_423")
plot_dtw("FOODS_2_043", "HOUSEHOLD_2_423")
plot_dtw("HOBBIES_1_300", "HOUSEHOLD_2_423")


# In[31]:


# cluster 5
daily_sales_item_lookup_scaled_weekly.T.merge(
    dtw_clusters.loc[dtw_clusters.cluster == 5], 
    left_index = True,
    right_index = True
)\
    .T\
    .plot(figsize = (12,4))


# In[32]:


# see which items are in cluster 5
plot_dtw("FOODS_3_247", "FOODS_3_284")
plot_dtw("FOODS_3_247", "HOBBIES_1_122")
plot_dtw("FOODS_3_247", "HOUSEHOLD_1_164")
plot_dtw("FOODS_3_247", "HOUSEHOLD_1_429")
plot_dtw("FOODS_3_247", "HOUSEHOLD_2_318")

