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


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gc
import lightgbm as lgb
import time
# import datetime
# import xgboost as xgb
# import time
# import itertools
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


INPUT_DIR = '/kaggle/input/m5-forecasting-accuracy'

calendar_df = pd.read_csv(f"{INPUT_DIR}/calendar.csv")
sell_prices_df = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv")
sales_train_validation_df = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv")
sample_submission_df = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")


# In[4]:


# Calendar data type cast -> Memory Usage Reduction
calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")
calendar_df[["wm_yr_wk", "year"]] = calendar_df[["wm_yr_wk", "year"]].astype("int16") 
calendar_df["date"] = calendar_df["date"].astype("datetime64")

nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in nan_features:
    calendar_df[feature].fillna('unknown', inplace = True)

calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")


# In[5]:


# Sales Training dataset cast -> Memory Usage Reduction
sales_train_validation_df.loc[:, "d_1":] = sales_train_validation_df.loc[:, "d_1":].astype("int16")


# In[6]:


# Make ID column to sell_price dataframe
sell_prices_df.loc[:, "id"] = sell_prices_df.loc[:, "item_id"] + "_" + sell_prices_df.loc[:, "store_id"] + "_validation"


# In[7]:


sell_prices_df = pd.concat([sell_prices_df, sell_prices_df["item_id"].str.split("_", expand=True)], axis=1)
sell_prices_df = sell_prices_df.rename(columns={0:"cat_id", 1:"dept_id"})
sell_prices_df[["store_id", "item_id", "cat_id", "dept_id"]] = sell_prices_df[["store_id","item_id", "cat_id", "dept_id"]].astype("category")
sell_prices_df = sell_prices_df.drop(columns=2)


# In[8]:


def make_dataframe():
    # Wide format dataset 
    df_wide_train = sales_train_validation_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_wide_train.index = calendar_df["date"][:1913]
    df_wide_train.columns = sales_train_validation_df["id"]
    
    # Making test label dataset
    df_wide_test = pd.DataFrame(np.zeros(shape=(56, len(df_wide_train.columns))), index=calendar_df.date[1913:], columns=df_wide_train.columns)
    df_wide = pd.concat([df_wide_train, df_wide_test])

    # Convert wide format to long format
    df_long = df_wide.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_wide_train, df_wide_test, df_wide
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), calendar_df, on="date"), sell_prices_df, on=["id", "wm_yr_wk"])
    df = df.drop(columns=["d"])
#     df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")
    df["sell_price"] = df["sell_price"].astype("float16")   
    df["value"] = df["value"].astype("int32")
    df["state_id"] = df["store_id"].str[:2].astype("category")


    del df_long
    gc.collect()

    return df

df = make_dataframe()


# In[9]:


def add_date_feature(df):
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["week"] = df["date"].dt.week.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["quarter"]  = df["date"].dt.quarter.astype("int8")
    return df


# In[10]:


df = add_date_feature(df)
df


# In[11]:


temp_series = df.groupby(["cat_id", "date"])["value"].sum()
temp_series


# In[12]:


plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Category")
plt.legend()


# In[13]:


temp_series = temp_series.loc[temp_series.index.get_level_values("date") >= "2015-01-01"]
plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year-Month")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Category from 2015")
plt.legend()


# In[14]:


# Plot only December, 2015
temp_series = temp_series.loc[(temp_series.index.get_level_values("date") >= "2015-12-01") & (temp_series.index.get_level_values("date") <= "2015-12-31")]
plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total sold item per day in December, 2015")
plt.legend()


# In[15]:


temp_series.loc[(temp_series.index.get_level_values("date") >= "2015-12-24") & (temp_series.index.get_level_values("date") <= "2015-12-26")]


# In[16]:


temp_series = df.groupby(["cat_id", "wday"])["value"].sum()
temp_series


# In[17]:


plt.figure(figsize=(6, 4))
left = np.arange(1,8) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, width=width, label="FOODS")
plt.bar(left + width, temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, width=width, label="HOUSEHOLD")
plt.bar(left + width + width, temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, width=width, label="HOBBIES")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xticks(left, weeklabel, rotation=60)
plt.xlabel("day of week")
plt.ylabel("# of sold items")
plt.title("Total sold item in each daytype")


# In[18]:


temp_series = df.groupby(["state_id", "date"])["value"].sum()
temp_series


# In[19]:


plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "CA"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "CA"].values, label="CA")
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "TX"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "TX"].values, label="TX")
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "WI"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "WI"].values, label="WI")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each State")
plt.legend()


# In[20]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].values, label="CA_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].values, label="CA_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].values, label="CA_3")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].values, label="CA_4")

plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Store in CA")
plt.legend()


# In[21]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].values, label="CA_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].values, label="CA_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].values, label="CA_3")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].values, label="CA_4")
plt.xlabel("Year")
plt.ylabel("# of item entries")
plt.title("Total item entries in each CA stores")
plt.legend()


# In[22]:


temp_series = df.groupby(["state_id", "store_id", "year", "month"])["value"].std()
temp_series


# In[23]:


fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

# We can use for loop, of course! And that'll be better, sorry, this is for my easy trial. 
sns.lineplot(x=temp_series[(temp_series.index.get_level_values("state_id") == "CA")].index.get_level_values("month"), 
             y=temp_series[(temp_series.index.get_level_values("state_id") == "CA")].values, 
             hue=temp_series[(temp_series.index.get_level_values("state_id") == "CA")].index.get_level_values("store_id"), 
             legend=False,
             ax=axs[0])
sns.lineplot(x=temp_series[temp_series.index.get_level_values("state_id") == "TX"].index.get_level_values("month"),
             y=temp_series[temp_series.index.get_level_values("state_id") == "TX"].values, 
             hue=temp_series[temp_series.index.get_level_values("state_id") == "TX"].index.get_level_values("store_id"), 
             legend=False,
             ax=axs[1])
sns.lineplot(x=temp_series[temp_series.index.get_level_values("state_id") == "WI"].index.get_level_values("month"),
             y=temp_series[temp_series.index.get_level_values("state_id") == "WI"].values, 
             hue=temp_series[temp_series.index.get_level_values("state_id") == "WI"].index.get_level_values("store_id"),
             ax=axs[2])



plt.legend(bbox_to_anchor=(1.01, 1.01))
axs[0].set_title("CA")
axs[0].set_xticks(range(1, 13))
axs[0].set_ylabel("Standard deviation of sold items in one month")
axs[1].set_title("TX")
axs[1].set_xticks(range(1, 13))
axs[2].set_title("WI")
axs[2].set_xticks(range(1, 13))

fig.suptitle("Standard deviation of sold items in one month in each store")


# In[24]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].values, label="WI_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].values, label="WI_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].values, label="WI_3")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total item sold in each WI stores")
plt.legend()


# In[25]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].values, label="WI_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].values, label="WI_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].values, label="WI_3")
plt.xlabel("Year")
plt.ylabel("# of item entries")
plt.title("Total item entries in each WI stores")
plt.legend()


# In[26]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].values, label="TX_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].values, label="TX_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].values, label="TX_3")
plt.xlabel("Year")
plt.ylabel("Total sold item per day")
plt.title("Total item sold in each TX stores")
plt.legend()


# In[27]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].values, label="TX_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].values, label="TX_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].values, label="TX_3")
plt.xlabel("Year")
plt.ylabel("Total item entries")
plt.title("Total item entries in each TX stores")
plt.legend()


# In[28]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()
temp_series


# In[29]:


# Find the day when items are sold less than 1000 of each store
# Let's take a look at TX_2 for example
temp_series.loc[(temp_series.values < 1000) & (temp_series.index.get_level_values("date") <= "2016-04-22")].loc["TX_2"]


# In[30]:


# Find the day when items are sold most of each store
temp_series.groupby(["store_id"]).idxmax()


# In[31]:


temp_series = temp_series.reset_index()
temp_series


# In[32]:


plt.plot(temp_series[(temp_series["store_id"] == "CA_1") & ((temp_series["date"] >= "2013-07-15") & (temp_series["date"] <= "2013-10-15"))]["date"],
         temp_series[(temp_series["store_id"] == "CA_1") & ((temp_series["date"] >= "2013-07-15") & (temp_series["date"] <= "2013-10-15"))]["value"])
plt.xticks(rotation=60)
plt.ylabel("# of sold items")
plt.xlabel("date")
plt.title("Item sold transition around its most sold day in CA_1 store")


# In[33]:


import statsmodels.api as sm
import scipy


# In[34]:


# Analysis target item is the most sold one.
# In this Dynamic Factor Analysis Session, we'll try to find the hidden factor of this item sales transition among states.
item_id = "FOODS_3_090"
state_list = ["CA", "TX", "WI"]


# In[35]:


# First, we extract the target item sold sum in each state
temp_series = df[(df.date >= "2015-01-01") & (df.date <= "2016-01-01") & (df.item_id == item_id)].groupby(["date", "state_id"])["value"].sum()

# Then convert it to dataframe type.
temp_df = pd.concat([pd.DataFrame(temp_series[temp_series.index.get_level_values("state_id") == "CA"].values),
                     pd.DataFrame(temp_series[temp_series.index.get_level_values("state_id") == "TX"].values), 
                     pd.DataFrame(temp_series[temp_series.index.get_level_values("state_id") == "WI"].values)], axis=1) 
temp_df.columns = state_list
temp_df.index = temp_series[temp_series.index.get_level_values("state_id") == "CA"].index.get_level_values("date")
temp_df.head()


# In[36]:


# First, we plot this item transition in each state
temp_df.plot(figsize=(12, 4))
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.ylabel("# of sold item")
plt.title(f"{item_id} sales transition in each state")


# In[37]:


# Next Step, we create diff column, which shows the difference of today and tommorow's sold count.  
# For flattening, we apply log function and standardization

diff_cols = ["diff_" + state for state in state_list]      # diff columns (row data)
std_cols = ["std_diff_" + state for state in state_list]   # diff columns (after standardization)

for state in state_list:
    col = "diff_" + state
    temp_df[col] = np.log(temp_df[state] + 0.1).diff() * 100
    temp_df[col] = temp_df[col].fillna(method="bfill")
    
    # Standardization
    std_col = "std_" + col
    temp_df[std_col] = (temp_df[col] - temp_df[col].mean()) / temp_df[col].std()


# In[38]:


# Conveert it to Z-value
temp_df[std_cols] = temp_df[std_cols].apply(scipy.stats.zscore, axis=0)


# In[39]:


temp_df


# In[40]:


# Create the model
# This time, for simplicity, unobserved factor (k_factors) is 1, and factor_order is 1 (i.e. it follows an AR(1) process) , 
# error_order is 1 (i.e. error has order 1 autocorelated), 

endog = temp_df.loc[:, std_cols]
mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=1, error_order=1)
initial_res = mod.fit(method='powell', disp=False)
res = mod.fit(initial_res.params, disp=False)


# In[41]:


print(res.summary(separate_params=False))


# In[42]:


from pandas_datareader.data import DataReader

fig, ax = plt.subplots(figsize=(13,3))

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, res.factors.filtered[0], label='Factor')
ax.legend()

# Retrieve and also plot the NBER recession indicators
rec = DataReader('USREC', 'fred', start=temp_df.index.min(), end=temp_df.index.max())
ylim = ax.get_ylim()
plt.title("Fluctuations extracted by Factor 1")
plt.ylabel("Fluctuations")
plt.xlabel("Year-Month")


# In[43]:


res.coefficients_of_determination


# In[44]:


res.plot_coefficients_of_determination();


# In[45]:


temp_series = df.groupby(["store_id", "cat_id"])["value"].sum()


# In[46]:


store_id_list_by_state = [["CA_1", "CA_2", "CA_3", "CA_4"], ["TX_1", "TX_2", "TX_3"], ["WI_1", "WI_2", "WI_3"]] 


# In[47]:


fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        axs[row, col].bar(x=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].index.get_level_values("cat_id"),
                          height=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].values,
                         color=["orange", "green", "blue"], label=["FOODS", "HOBBIES", "HOUSEHOLD"])
        axs[row, col].set_title(store_id_list_by_state[row][col])
        axs[row, col].set_ylabel("# of items")

fig.suptitle("Each category item sold in each store")


# In[48]:


fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        axs[row, col].bar(x=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].index.get_level_values("cat_id"),
                          height=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].values / temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].sum(),
                         color=["orange", "green", "blue"], label=["FOODS", "HOBBIES", "HOUSEHOLD"])
        axs[row, col].set_title(store_id_list_by_state[row][col])
        axs[row, col].set_ylabel("% of each category")

fig.suptitle("Each category item sold percentage in each store")


# In[49]:


cat_id = "FOODS"

temp_series = df.groupby(["store_id", "cat_id", "wday"])["value"].sum()
temp_series = temp_series[temp_series.index.get_level_values("cat_id") == cat_id]
temp_series


# In[50]:


weekday = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]


# In[51]:


# Combine all these three figures.
cat_list = ["FOODS", "HOBBIES", "HOUSEHOLD"]
color_list = ["orange", "green", "blue"]
temp_series = df.groupby(["store_id", "cat_id", "wday"])["value"].sum()
width = 0.25

fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        for i, cat in enumerate(cat_list):
            height_numerator = temp_series[(temp_series.index.get_level_values("cat_id") == cat) & (temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col])].values
            height_denominater = height_numerator.sum()

            axs[row, col].bar(x=temp_series[(temp_series.index.get_level_values("cat_id") == cat) & (temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col])].index.get_level_values("wday") + width * (i-1),
                              height=height_numerator / height_denominater,
                             tick_label=weekday, color=color_list[i], width=width, label=cat)
            axs[row, col].set_title(store_id_list_by_state[row][col])
            axs[row, col].legend()
            
fig.suptitle("HOBBIES item sold in each store in each day")


# In[52]:


fig, axs = plt.subplots(1, 3, sharey=True)
fig.suptitle("Snap Purchase Enable Day Count of each store")

sns.countplot(x="snap_CA", data =calendar_df, ax=axs[0])
sns.countplot(x="snap_TX", data =calendar_df, ax=axs[1])
sns.countplot(x="snap_WI", data =calendar_df, ax=axs[2])


# In[53]:


temp_df = calendar_df.groupby(["year"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# In[54]:


# This cell is just visuallizing the above dataframe.
plt.bar(temp_df.index, temp_df.snap_CA)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Year")
plt.title("Snap Purchase allowed day yearly transition")


# In[55]:


temp_df = calendar_df[calendar_df["year"] == 2015].groupby(["month"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# In[56]:


# Just visualizing the above dataframe
plt.bar(temp_df.index, temp_df.snap_CA)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Month")
plt.title("Snap Purchase allowed day monthly trend")


# In[57]:


temp_df = calendar_df[calendar_df["year"] == 2015].groupby(["weekday"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# In[58]:


plt.bar(temp_df.index, temp_df.snap_CA)
plt.xticks(rotation=60)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Day type")
plt.title("Snap Purchase allowed day weekly trend")


# In[59]:


# Make temp dataframe with necessary information
temp_df = df.groupby(["date", "state_id"])[["value"]].sum()
temp_df = temp_df.reset_index()
temp_df = temp_df.merge(calendar_df[["date", "snap_CA", "snap_TX", "snap_WI"]], on="date")
temp_df


# In[60]:


np.argmax(temp_df.groupby(["date", "state_id"])["value"].sum())


# In[61]:


temp_df = temp_df[(temp_df.date >= "2016-02-15") & (temp_df.date <= "2016-03-25") & (temp_df.state_id == "CA")]
temp_df


# In[62]:


fig, ax1 = plt.subplots()
plt.xticks(rotation=60)
ax1.plot("date", "value", data=temp_df[temp_df.state_id == "CA"])
ax2 = ax1.twinx()  
ax2.scatter("date", "snap_CA", data=temp_df[temp_df.state_id == "CA"])


# In[63]:


plt.figure(figsize=(8, 6))
sns.countplot(x="event_type_1", data=calendar_df[calendar_df["event_name_1"] != "unknown"])
plt.xticks(rotation=90)
plt.title("Event Type Count in event name 1 column")


# In[64]:


# Let's check the distribution of snap purchase day and event day
# Accirding to the graph, Snap CA is allowed especially when sport event occurs.

plt.figure(figsize=(8, 6))
sns.countplot(x="event_type_1", data=calendar_df[calendar_df["event_name_1"] != "unknown"], hue="snap_CA")
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.title("Snap Purchse allowed day Count in each event category")


# In[65]:


temp_series = df.groupby(["cat_id", "event_type_1"])["value"].mean()
temp_series


# In[66]:


plt.bar(x=temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("event_type_1"), 
        height=temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values)
plt.title("HOBBIES Item Sold mean in each event type")
plt.ylabel("Item sold mean")
plt.xlabel("Event Type")


# In[67]:


# find out most sold item for example
df[df["value"] == df["value"].max()]


# In[68]:


target_id = "FOODS_3_090_CA_3_validation"
temp_df = df[df["id"] == target_id]
temp_df


# In[69]:


weekday = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

# Create one hot weekday column from wday column to calculate correlation later. 
for idx, val in enumerate(weekday):
    temp_df.loc[:, val] = (temp_df["wday"] == idx + 1).astype("int8")

temp_df
# sns.heatmap(temp_df[["value", "snap_CA", ]].corr(), annot=True)


# In[70]:


# Create Event Flag (Any events occur: 1, otherwise: 0)
# Create Each Event Type Flag
temp_df.loc[:, "is_event_day"] = (temp_df["event_name_1"] != "unknown").astype("int8")
temp_df.loc[:, "is_sport_event"] = (temp_df["event_type_1"] == "Sporting").astype("int8")
temp_df.loc[:, "is_cultural_event"] = (temp_df["event_type_1"] == "Cultural").astype("int8")
temp_df.loc[:, "is_national_event"] = (temp_df["event_type_1"] == "National").astype("int8")
temp_df.loc[:, "is_religious_event"] = (temp_df["event_type_1"] == "Religious").astype("int8")

temp_df.head()


# In[71]:


# Plot Heatmap with these columns made in previous cells
plt.figure(figsize=(14, 10))
sns.heatmap(temp_df[["value", "sell_price", "snap_CA", "is_event_day", "is_sport_event", "is_cultural_event", "is_national_event", "is_religious_event"] + weekday].corr(), annot=True)
plt.title("Heatmap with values, snap_CA,  event_flag and weekday columns")


# In[72]:


df.groupby("cat_id")["sell_price"].mean()


# In[73]:


df.groupby("cat_id")["sell_price"].describe()


# In[74]:


sns.boxplot(data=df, x="cat_id", y='sell_price')
plt.title("Boxplot of sell prices in each category")


# In[75]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="cat_id", y='sell_price', hue="store_id")
plt.title("Boxplot of sell prices in each store")


# In[76]:


# One Item Sell Price Transition
sns.lineplot(data=df[df["item_id"] == "FOODS_3_090"], x='date', y='sell_price', hue="store_id")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.title("Sell price change of 'FOODS_3_090' in each store")


# In[77]:


df["is_event_day"] = (df["event_name_1"] != "unknown").astype("int8")
df.head()


# In[78]:


sns.heatmap(df[df["item_id"] == "FOODS_3_090"][["value", "sell_price", "is_event_day"]].corr(), annot=True)
plt.title("Heatmap of value, sell_price and event flag")


# In[79]:


temp_df = df.groupby(["date", "cat_id"])["sell_price"].mean()
temp_df


# In[80]:


plt.figure(figsize=(8,4))
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Mean price")
plt.title("Mean price transition of each category")


# In[81]:


temp_df = df.groupby(["date", "cat_id"])["item_id"].count()


# In[82]:


plt.figure(figsize=(8,4))
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Registered Item Counts")
plt.title("Registered Item Counts Transition in each category")


# In[83]:


sns.jointplot(df["value"], df["sell_price"])


# In[84]:


df["sell_price_diff"] = df.groupby("id")["sell_price"].transform(lambda x: x - x.mean()).astype("float32")


# In[85]:


sns.lineplot(df[df["item_id"] == "FOODS_3_090"]["date"],df[df["item_id"] == "FOODS_3_090"]["sell_price_diff"], hue=df["store_id"]) 
plt.legend(bbox_to_anchor=(1.01, 1.01))


# In[86]:


temp_df = df[df["item_id"] == "FOODS_3_090"].groupby("date")["value"].sum()


# In[87]:


fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(temp_df)
ax1.set_ylabel("# of Sold items")
ax2 = ax1.twinx()  
ax2.plot(df[(df["item_id"] == "FOODS_3_090") & (df["store_id"] == "CA_3")]["date"],
         df[df["item_id"] == "FOODS_3_090"].groupby("date")["sell_price_diff"].mean(), color="red")
ax2.set_ylabel("price_diff [\$]") 
plt.title("FOODS_3_090 sold number and price difference from mean")


# In[88]:


# Find the most sold items in  HOBBIES section
np.argmax(df[df["cat_id"] == "HOBBIES"].groupby("item_id")["value"].sum())


# In[89]:


temp_df = df[df["item_id"] == "HOBBIES_1_371"].groupby("date")["value"].sum()


# In[90]:


fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(temp_df)
ax1.set_ylabel("# of Sold items")
ax2 = ax1.twinx()  
ax2.plot(df[(df["item_id"] == "HOBBIES_1_371") & (df["store_id"] == "CA_3")]["date"],
         df[df["item_id"] == "HOBBIES_1_371"].groupby("date")["sell_price_diff"].mean(), color="red")
ax2.set_ylabel("price_diff [\$]") 
plt.title("HOBBIES_1_371 sold number and price difference from mean")


# In[91]:


temp_series = df.groupby(["date", "item_id"])["value"].sum()
temp_series


# In[92]:


# Find Top 12 items that the mean of sold counts in all stores is high
high_sold_item_top12 = temp_series.groupby("item_id").mean().sort_values(ascending=False)[:12].index
high_sold_item_top12


# In[93]:


fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharey=True, sharex=True) 

for row in range(3):
    for col in range(4):
        target_item = high_sold_item_top12[row*4+col]
        
        axs[row, col].plot(temp_series[temp_series.index.get_level_values("item_id") == target_item].values)
        axs[row, col].set_title(target_item)
        axs[row, col].set_ylabel("# of sold items in all stores")
#         axs[idx, col].set_xticks(temp_series[temp_series.index.get_level_values("item_id") == target_item].index.get_level_values("date"))
        axs[row, col].legend()
            
fig.suptitle("Top 12 item sold in all stores in each day")


# In[94]:


df.head()


# In[95]:


# Take a closer look,
sns.violinplot(x = df.loc[df["year"] == 2015, "quarter"], y = df.loc[df["year"] == 2015, "value"])
plt.ylim(0, 10)


# In[96]:


temp_df = df.groupby(["date", "cat_id", "dept_id"])["value", "sell_price"].mean()


# In[97]:


fig, axs = plt.subplots(2, 3, figsize=(14, 10)) 


for col in range(3):
    target_cat = cat_list[col]
    sns.scatterplot("value", "sell_price", hue=temp_df[temp_df.index.get_level_values("cat_id") == target_cat].index.get_level_values("dept_id") ,
                data=temp_df[temp_df.index.get_level_values("cat_id") == target_cat], ax=axs[0, col])
#     axs[0, col].plot(temp_series[temp_series.index.get_level_values("item_id") == target_item].values)
    axs[0, col].set_title(f"{target_cat} ")
    axs[0, col].set_ylabel("sell_price")
    axs[0, col].set_xlabel("value")
#         axs[idx, col].set_xticks(temp_series[temp_series.index.get_level_values("item_id") == target_item].index.get_level_values("date"))
    axs[0, col].legend()
    
    temp_series = df[df["cat_id"] == target_cat].groupby(["date", "dept_id"])["item_id"].count()
    sns.lineplot(x=temp_series.index.get_level_values("date"), y=temp_series.values, hue=temp_series.index.get_level_values("dept_id"), ax=axs[1, col])
    axs[1, col].set_title(f"{target_cat}")
    axs[1, col].set_ylabel("count")
    axs[1, col].legend()
            
fig.suptitle("Daily Average value - price plot and item count transition in each dept.")


# In[98]:


df["num_of_next_week_event"] = df.groupby("id")["is_event_day"].transform(lambda x: x.shift(-7).rolling(7).sum().fillna(0)).astype("int8")


# In[99]:


df[df.id == "HOBBIES_1_008_CA_1_validation"][25:50]


# In[100]:


fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharey=True)
sns.scatterplot("num_of_next_week_event", "value", data=df[df["cat_id"] == "HOBBIES"], color="green",ax=axs[0])
sns.scatterplot("num_of_next_week_event", "value", data=df[df["cat_id"] == "HOUSEHOLD"], color="orange", ax=axs[1])
sns.scatterplot("num_of_next_week_event", "value", data=df[df["cat_id"] == "FOODS"], ax=axs[2])
axs[0].set_title("HOBBIES")
axs[1].set_title("HOUSEHOLD")
axs[2].set_title("FOODS")

fig.suptitle("Relationship between next week events count and sold item counts")


# In[101]:


df["lag_1"] = df.groupby("id")["value"].transform(lambda x: x.shift(1)).astype("float16")
df["lag_7"] = df.groupby("id")["value"].transform(lambda x: x.shift(7)).astype("float16")


# In[102]:


# plt.figure(figsize=(8, 8))
# sns.pairplot(df[["cat_id", "value", "lag_1"]], hue="cat_id")


# In[103]:


sns.pairplot(df[["cat_id", "value", "lag_1", "lag_7"]], hue="cat_id")


# In[104]:


get_ipython().system('pip install calmap')


# In[105]:


import calmap

temp_df = df.groupby(["state_id", "date"])["value"].sum()
temp_df = temp_df.reset_index()
temp_df = temp_df.set_index("date")
temp_df


# In[106]:


fig, axs = plt.subplots(3, 1, figsize=(10, 10))
calmap.yearplot(temp_df.loc[temp_df["state_id"] == "CA", "value"], year=2015, ax=axs[0])
axs[0].set_title("CA")
calmap.yearplot(temp_df.loc[temp_df["state_id"] == "TX", "value"], year=2015, ax=axs[1])
axs[1].set_title("TX")
calmap.yearplot(temp_df.loc[temp_df["state_id"] == "WI", "value"], year=2015, ax=axs[2])
axs[2].set_title("WI")


# In[107]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# In[108]:


# First, we try store clustering by using weekly total sales 
temp_df = df.groupby(["store_id", "wm_yr_wk"])["value"].sum()
temp_df = temp_df.reset_index()
temp_df


# In[109]:


# Convert it to wide format
temp_df_wide = temp_df.pivot(index="store_id", columns="wm_yr_wk", values="value")
temp_df_wide


# In[110]:


# By using PCA, we could decrease 10 rows * 282 dimentions to 10 rows * 2 dimentions
pca = PCA(n_components=2)


# In[111]:


pca.fit(temp_df_wide)


# In[112]:


pca.explained_variance_ratio_


# In[113]:


result = pca.transform(temp_df_wide)
result


# In[114]:


result_df = pd.DataFrame(result)
result_df.index = temp_df_wide.index
result_df.columns = ["PC1", "PC2"]
result_df


# In[115]:


ax = result_df.plot(kind='scatter', x='PC2', y='PC1', figsize=(12, 6))

for idx, store_id in enumerate(result_df.index):
    ax.annotate(  
        store_id,
       (result_df.iloc[idx].PC2, result_df.iloc[idx].PC1)
    )

ax.set_title("PCA result of all shops")


# In[116]:


temp_df = df.groupby(["item_id", "wm_yr_wk"])["value"].sum()
temp_df = temp_df.reset_index()
temp_df


# In[117]:


temp_df = temp_df.fillna(method="bfill")


# In[118]:


# Convert it to wide format
temp_df_wide = temp_df.pivot(index="item_id", columns="wm_yr_wk", values="value")
temp_df_wide


# In[119]:


# By using PCA, we could decrease 10 rows * 282 dimentions to 10 rows * 2 dimentions
pca = PCA(n_components=2)


# In[120]:


pca.fit(temp_df_wide)


# In[121]:


pca.explained_variance_ratio_


# In[122]:


result = pca.transform(temp_df_wide)
result


# In[123]:


result_df = pd.DataFrame(result)
result_df.index = temp_df_wide.index
result_df.columns = ["PC1", "PC2"]
result_df


# In[124]:


result_df.index.str[:5]


# In[125]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC2", y="PC1", data=result_df, hue=result_df.index.str[:5])
plt.title("PCA result for item_id column")


# In[ ]:




