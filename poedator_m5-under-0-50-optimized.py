#!/usr/bin/env python
# coding: utf-8

# In[1]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb


# In[2]:


CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


# In[3]:


pd.options.display.max_columns = 50


# In[4]:


h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday


# In[5]:


def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt


# In[6]:


def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


# In[7]:


FIRST_DAY = 350 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !


# In[8]:


get_ipython().run_cell_magic('time', '', '\ndf = create_dt(is_train=True, first_day= FIRST_DAY)\ndf.shape')


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


get_ipython().run_cell_magic('time', '', '\ncreate_fea(df)\ndf.shape')


# In[12]:


df.info()


# In[13]:


df.head()


# In[14]:


df.dropna(inplace = True)
df.shape


# In[15]:


cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]


# In[16]:


get_ipython().run_cell_magic('time', '', "\nnp.random.seed(777)\n\nfake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)\ntrain_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], \n                         categorical_feature=cat_feats, free_raw_data=False)\nfake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],\n                              categorical_feature=cat_feats,\n                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!")


# In[17]:


del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()


# In[18]:


params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 2**11-1,
    "min_data_in_leaf":  2**12-1,
}


# In[19]:


get_ipython().run_cell_magic('time', '', '\nm_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) ')


# In[20]:


fig, ax = plt.subplots(figsize=(12,6))
lgb.plot_importance(m_lgb, max_num_features=30, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15);


# In[21]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 4)
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(12,8))
lgb.plot_importance(m_lgb, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[22]:


m_lgb.save_model("model.lgb")


# In[23]:


def create_lag_features_for_test(dt, day):
    # create lag feaures just for single day (faster)
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] =             dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main

    windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] =                 df_window_grouped.sales.values     


# In[24]:


def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")


# In[25]:


get_ipython().run_cell_magic('time', '', '\nalphas = [1.028, 1.023, 1.018]\nweights = [1/len(alphas)]*len(alphas)  # equal weights\n\nte0 = create_dt(False)  # create master copy of `te`\ncreate_date_features_for_test (te0)\n\nfor icount, (alpha, weight) in enumerate(zip(alphas, weights)):\n    te = te0.copy()  # just copy\n    cols = [f"F{i}" for i in range(1, 29)]\n\n    for tdelta in range(0, 28):\n        day = fday + timedelta(days=tdelta)\n        print(tdelta, day.date())\n        tst = te[(te.date >= day - timedelta(days=max_lags))\n                 & (te.date <= day)].copy()\n#         create_fea(tst)  # correct, but takes much time\n        create_lag_features_for_test(tst, day)  # faster  \n        tst = tst.loc[tst.date == day, train_cols]\n        te.loc[te.date == day, "sales"] = \\\n            alpha * m_lgb.predict(tst)  # magic multiplier by kyakovlev\n\n    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()\n\n    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")[\n        "id"].cumcount()+1]\n    te_sub = te_sub.set_index(["id", "F"]).unstack()[\n        "sales"][cols].reset_index()\n    te_sub.fillna(0., inplace=True)\n    te_sub.sort_values("id", inplace=True)\n    te_sub.reset_index(drop=True, inplace=True)\n    te_sub.to_csv(f"submission_{icount}.csv", index=False)\n    if icount == 0:\n        sub = te_sub\n        sub[cols] *= weight\n    else:\n        sub[cols] += te_sub[cols]*weight\n    print(icount, alpha, weight)')


# In[26]:


sub.head(10)


# In[27]:


sub.id.nunique(), sub["id"].str.contains("validation$").sum()


# In[28]:


sub.shape


# In[29]:


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv",index=False)

