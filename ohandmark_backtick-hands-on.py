#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
from kaggle.competitions import twosigmanews
from sklearn.metrics import confusion_matrix, accuracy_score


# In[2]:


env = twosigmanews.make_env()

market_orig, news_orig = env.get_training_data()


# In[3]:


# We'll use data from 2013 onwards to speed things up a bit
market_orig = market_orig[market_orig['time'] >= "2013-01-01"]
market_orig.head()


# In[4]:


def unique_asset_names(df):
    return "TODO"

unique_asset_names(market_orig)


# In[5]:


def most_common_asset_name(df):
    return "TODO"

most_common_asset_name(market_orig)


# In[6]:


def max_next10(df):
    return "TODO"

max_next10(market_orig)


# In[7]:


# Matplotlib is available as "plt"
plt.figure(figsize=(12,7))

def plot_fb(df):
    return "TODO"

plot_fb(market_orig)


# In[8]:


def asset_presence(df):
    return "TODO"

asset_presence(market_orig)


# In[9]:


def remove_unknown(df):
    return df

market = remove_unknown(market_orig)


# In[10]:


def clip_next10(df):
    return df

market = clip_next10(market)


# In[11]:


def remove_short_lived(df):
    return df

market = remove_short_lived(market)


# In[12]:


# https://github.com/bukosabino/ta/blob/master/ta/momentum.py
def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()

def rsi(close, n=14, fillna=False):
    """Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.
    https://www.investopedia.com/terms/r/rsi.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n, fillna)
    emadn = ema(dn, n, fillna)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='rsi')


# In[13]:


# TODO: Add the RSI indicator to the market df


# In[14]:


def add_volume_avg(df):
    return df

add_volume_avg(market)


# In[15]:


def add_lag_features(df):
    return df
    
market = add_lag_features(market)


# In[16]:


# Returns:
#    X: numpy matrix with relevant features only
#    y: numpy array of class values, 0 if returnsOpenNextMktres10 is negative, else 1

TRAIN_END_DATE  = ""
TEST_START_DATE = ""

def get_data(df):
    pass

X_train, y_train = get_data()
X_test, y_test   = get_data()


# In[17]:


def train_clf(X_train, y_train):
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
        'objective': 'binary',
        'num_threads': 4
    }

    train_set = lgb.Dataset(X_train, y_train)

    # https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
    lgb_clf = lgb.train(params, train_set)
    
    return lgb_clf

clf = train_clf(X_train, y_train)


# In[18]:


def accuracy(clf, X_test, y_test):
    return 0

accuracy(clf, X_test, y_test)


# In[19]:


# You can use this helper, if you want.
def plot_feature_importances(clf, feature_columns):
    features_imp = pd.DataFrame()
    features_imp['features'] = list(feature_columns)[:]
    features_imp['importance'] = clf.feature_importance()
    features_imp = features_imp.sort_values(by='importance', ascending=False).reset_index()
    shape = features_imp.shape[0]
    
    y_plot = -np.arange(shape)
    plt.figure(figsize=(10,7))
    plt.barh(y_plot, features_imp.loc[:shape,'importance'].values)
    plt.yticks(y_plot,(features_imp.loc[:shape,'features']))
    plt.xlabel('Feature importance')
    plt.title('Features importance')
    plt.tight_layout()


# In[20]:


# Returns a series of confidence values
def get_confidence(clf, X_test):
    y_pred = clf.predict(X_test)
    
    return "TODO"

confidence = get_confidence(clf, X_test)


# In[21]:


# You can use this helper:
def get_scoring_data(market):
    test_df         = market[[market['time'] > TEST_START_DATE]
    test_df['date'] = df['time'].dt.date
                     
    actual_returns  = test_df['returnsOpenNextMktres10'].values.clip(-1, 1)
    universe        = test_df['universe']
    dates           = test_df['date']

    return actual_returns, universe, dates

actual_returns, universe, dates = get_scoring_data(market)


# In[22]:


def score(confidence, actual_returns, universe, dates):
    return 0

score(confidence, actual_returns, universe, dates)


# In[23]:


# Modify the "score" function above to plot your strategy's daily returns


# In[24]:


def cross_validate():
    # 1. Create folds based on date from the data
    # 2. Train a classifier for each fold
    # 3. Test against related test set
    # 4. Evaluate the results
    
    pass

cross_validate()


# In[25]:


# Implement a voting strategy, you can reuse a lot of the code from the cross_validation step


# In[26]:





# In[26]:




