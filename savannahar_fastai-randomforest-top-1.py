#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install fastai==0.7.0')


# In[2]:


#It will automatically reload the latest module when you start again
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#Used to display plots and graphs inside Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#The following are fastAi imports
from fastai.imports import * 
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics


# In[4]:


get_ipython().run_line_magic('pinfo', 'display')


# In[5]:


get_ipython().run_line_magic('pinfo2', 'display')


# In[6]:


import os
print(os.listdir("../input"))


# In[7]:


PATH = "../input/train/"


# In[8]:


# ! says it is a bash command and not a jupyter command, and {} says it is a python variable
get_ipython().system('ls {PATH}')


# In[9]:


get_ipython().system('head -n 5 ../input/train/Train.csv')


# In[10]:


df_raw = pd.read_csv(PATH+ 'Train.csv', low_memory = False ,parse_dates=['saledate'])


# In[11]:


df_raw


# In[12]:


#function to display all the data of the dataframe at one go
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


# In[13]:


display_all(df_raw.transpose())


# In[14]:


df_raw.SalePrice  = np.log(df_raw.SalePrice)


# In[15]:


df_raw.saledate.head(5)


# In[16]:


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[17]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head(5)


# In[18]:


train_cats(df_raw) #behind scenes everything will be converted into numbers


# In[19]:


df_raw.UsageBand.cat.categories


# In[20]:


#the UsageBand is in wired order, so we'll convert into order High Med Low
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'],ordered = True ,inplace = True)


# In[21]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[22]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[23]:


nas #this is created by proc_df they are new columns along with there mean values


# In[24]:


#all columns
df.columns


# In[25]:


#now check everything is numeric
df.head()


# In[26]:


m = RandomForestRegressor(n_jobs=-1) #njobs = -1 use all the resouces for running our model
m.fit(df, y)
m.score(df, y)


# In[27]:


def split_vals(a,n) : return a[:n].copy(), a[n:].copy()


# In[28]:


n_valid = 12000 #same as kaggle's test size
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df, n_trn)
x_train, x_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)


# In[29]:


x_train.shape, y_train.shape, x_valid.shape


# In[30]:


#the following code will print the score
def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(x_train), y_train), rmse(m.predict(x_valid), y_valid), m.score(x_train, y_train), m.score(x_valid, y_valid)]
    if hasattr(m ,'obb_score_'): res.append(m.obb_score_)
    print(res)


# In[31]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(df,y)')
print_score(m)


# In[32]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000)
x_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)


# In[33]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[34]:


#RandomForest randomizes the things, so to stop it we set Bootstrap = False
#n_estimator depicts that we are using 1 tree with max_depth of that tree 3
m = RandomForestRegressor(n_jobs=-1, n_estimators=1, max_depth=3, bootstrap=False)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[35]:


draw_tree(m.estimators_[0], df_trn, precision=3)


# In[36]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_depth=35)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[37]:


#let's check how each tree is working differently and how's there predictions
preds = np.stack([t.predict(x_valid)for t in m.estimators_])
preds[:, 0], np.mean(preds[:, 0]), y_valid[0]


# In[38]:


preds.shape


# In[39]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis = 0)) for i in range(40)])


# In[40]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[41]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
x_train, _ = split_vals(df_trn, n_trn)
y_train, _ = split_vals(y_trn, n_trn)


# In[42]:


set_rf_samples(20000)


# In[43]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[44]:


reset_rf_samples()


# In[45]:


def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right
    
    def walk(node_id):
        if(children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            righ_max = 1 + walk(children_right)
            return max(left_max, right_max)
        else:
            return 1
        root_node_id = 0
        return walk(root_node_id)


# In[46]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[47]:


t = m.estimators_[0].tree_


# In[48]:


t.max_depth


# In[49]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40,oob_score=True, min_samples_leaf = 5)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[50]:


t = m.estimators_[0].tree_
dectree_max_depth(t)


# In[51]:


m = RandomForestRegressor(n_jobs=-1, n_estimators=40,oob_score=True, min_samples_leaf = 5, max_features=0.5)
get_ipython().run_line_magic('time', 'm.fit(x_train, y_train)')
print_score(m)


# In[52]:


set_rf_samples(50000)


# In[53]:


def get_preds(t): return t.predict(x_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:, 0]), np.std(preds[:,0])


# In[54]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis = 0)
x['pred'] = np.mean(preds, axis = 0)
x.Enclosure.value_counts().plot.barh()


# In[55]:


flds = ['Enclosure','pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index = False).mean()
enc_summ


# In[56]:


raw_valid.ProductSize.value_counts().plot.barh()


# In[57]:


fi = rf_feat_importance(m, df_trn)
fi[:10]


# In[58]:


fi.plot('cols', 'imp', figsize = (10,6), legend = False)


# In[59]:


def plot_fi(fi) : return fi.plot('cols', 'imp', 'barh', figsize = (12,7), legend=False)


# In[60]:


plot_fi(fi[:30])


# In[61]:


to_keep = fi[fi.imp > 0.005].cols
len(to_keep)


# In[62]:


df_keep = df_trn[to_keep].copy()
x_train, x_valid = split_vals(df_keep, n_trn)


# In[63]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)
print_score(m)


# In[64]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi)


# In[65]:


df_raw.YearMade.head(5)
df_raw.Coupler_System.head(5)


# In[66]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.scatter(df_raw['YearMade'], df_raw['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('YearMade')

plt.subplot(1,2,2)
sns.stripplot(df_raw['Coupler_System'], df_raw['SalePrice'])


# In[67]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.stripplot(df_raw['ProductSize'], df_raw['SalePrice'],order = ['Large', 'Medium', 'Small', 'Mini', 'Compact'])

plt.subplot(1,2,2)
sns.stripplot(df_raw['fiProductClassDesc'], df_raw['SalePrice'])


# In[68]:


plt.hist(df_raw['YearMade'], bins = 10)
plt.show()


# In[69]:


df_raw['Coupler_System'].value_counts().plot(kind='bar')


# In[70]:


df_raw['ProductSize'].value_counts().plot(kind='bar')


# In[71]:


reset_rf_samples()


# In[72]:


x_train, x_valid = split_vals(df_keep, n_trn)


# In[73]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)
print_score(m)


# In[74]:


set_rf_samples(50000)


# In[75]:


from scipy.cluster import hierarchy as hc


# In[76]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
plt.figure(figsize=(15,15))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size = 16)
plt.show()


# In[77]:


def get_obb(df):
    m = RandomForestRegressor(n_estimators=30,min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# In[78]:


get_obb(df_keep)


# In[79]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_obb(df_keep.drop(c, axis = 1)))


# In[80]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_obb(df_keep.drop(to_drop, axis = 1))


# In[81]:


df_keep.drop(to_drop, axis = 1, inplace=True)
x_train, x_valid = split_vals(df_keep, n_trn)


# In[82]:


reset_rf_samples()
m = RandomForestRegressor(n_estimators=60,min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)
print_score(m)


# In[83]:


df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear - df_raw.YearMade


# In[84]:


x_train, x_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(x_train, y_train)
plot_fi(rf_feat_importance(m, df_keep))


# In[85]:


m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
m.fit(x_train, y_train)
print_score(m)

