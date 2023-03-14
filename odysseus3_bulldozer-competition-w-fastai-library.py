#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics


# In[ ]:


PATH = "../input/"


# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, 
                     parse_dates=["saledate"])


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_raw.tail().T)


# In[ ]:


display_all(df_raw.describe(include='all').T)


# In[ ]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# In[ ]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()


# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# In[ ]:


df_raw.UsageBand = df_raw.UsageBand.cat.codes


# In[ ]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/bulldozers-raw')


# In[ ]:


import feather
feather.read_dataframe('tmp/bulldozers-raw')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'proc_df')


# In[ ]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)

