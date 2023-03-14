#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')

# display all the outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


# utility functions
def get_col_stats(df):
    temp_nulls = df.isnull().sum()
    temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls*100/df.shape[0]})
    
    uniqs = []
    for col in df.columns:
        uniqs.append(df[col].unique().shape[0])
    temp["uniqs"] = uniqs
    return temp

def get_categorical_stats(df, col):
    counts = df[col].value_counts(dropna=False)
    temp = pd.DataFrame({
        "counts": counts,
        "count_per": counts * 100 / df.shape[0],
    })
    return temp

def numeric_to_cat(df, cols):
    for col in cols:
        df[col] = df[col].astype("category")
    return df


# In[3]:


DATA_ROOT = Path("../input/")

card_f = DATA_ROOT / "train.csv"
merchant_f = DATA_ROOT / "merchants.csv"
hist_f = DATA_ROOT / "historical_transactions.csv"
new_hist_f = DATA_ROOT / "new_merchant_transactions.csv"

test_f = DATA_ROOT / "test.csv"


# In[4]:


card_df = pd.read_csv(card_f)
card_df.shape
card_df.head()


# In[5]:


test_df = pd.read_csv(test_f)
test_df.shape
test_df.head()


# In[6]:


card_df.card_id.unique().shape
card_df.card_id.unique().shape[0] == card_df.shape[0]


# In[7]:


test_df.card_id.unique().shape
test_df.card_id.unique().shape[0] == test_df.shape[0]


# In[8]:


len(set(test_df.card_id).intersection(set(card_df.card_id)))


# In[9]:


print("Training data:")
card_df.isna().sum()
print("Testing data:")
test_df.isna().sum()


# In[10]:


test_df[test_df.first_active_month.isnull()]


# In[11]:


ax = card_df.target.plot.hist(bins=20, figsize=(10, 5))
_ = ax.set_title("target histogram")
plt.show()

fig, axs = plt.subplots(1,2, figsize=(20, 5))
_ = card_df.target[card_df.target > 10].plot.hist(ax=axs[0])
_ = axs[0].set_title("target histogram for values greater than 10")
_ = card_df.target[card_df.target < -10].plot.hist(ax=axs[1])
_ = axs[1].set_title("target histogram for values less than -10")
plt.show()

card_df.target.describe()


# In[12]:


card_df["target_sign"] = card_df.target.apply(lambda x: 0 if x <= 0 else 1)
card_df.target_sign.value_counts()


# In[13]:


print("feature_1")
pd.DataFrame({"counts": card_df.feature_1.value_counts(), "counts_per": card_df.feature_1.value_counts()*100/card_df.shape[0]})
print("feature_2")
pd.DataFrame({"counts": card_df.feature_2.value_counts(), "counts_per": card_df.feature_2.value_counts()*100/card_df.shape[0]})
print("feature_3")
pd.DataFrame({"counts": card_df.feature_3.value_counts(), "counts_per": card_df.feature_3.value_counts()*100/card_df.shape[0]})


# In[14]:


temp = card_df.first_active_month.value_counts().sort_index()
ax = temp.plot(figsize=(10, 5))
_ = ax.set_xticklabels(range(2010, 2019))
_ = ax.set_title("Distribution across years")


# In[15]:


card_df["yr"] = card_df.first_active_month.str.split("-").str[0]
card_df["month"] = card_df.first_active_month.str.split("-").str[1]
card_df.head()


# In[16]:


temp = get_categorical_stats(card_df, "yr")
temp

ax = temp.counts.sort_index().plot()
_ = ax.set_xticklabels(range(2010, 2019))


# In[17]:


temp = get_categorical_stats(card_df, "month")
temp

ax = temp.counts.sort_index().plot()
_ = ax.set_xticklabels(range(-1, 13, 2))


# In[18]:


card_df["card_id_dec"] = card_df.card_id.str.split("_").str[2].apply(lambda x: int(x, 16))

card_df.card_id.str.split("_").str[0].unique()
card_df.card_id.str.split("_").str[1].unique()
card_df.card_id_dec.describe()


# In[19]:


card_df[["card_id_dec", "first_active_month"]].sort_values("card_id_dec")


# In[20]:


_ = card_df[["feature_1", "target"]].plot.scatter(x="feature_1", y="target")
_ = card_df[["feature_2", "target"]].plot.scatter(x="feature_2", y="target")
_ = card_df[["feature_3", "target"]].plot.scatter(x="feature_3", y="target")


# In[21]:


card_df.groupby(["yr", "feature_1"])["month"].count()


# In[22]:


card_df.groupby(["yr", "feature_2"])["month"].count()


# In[23]:


merc_df = pd.read_csv(merchant_f)

minus_1_to_nan_cols = ["city_id", "state_id", "merchant_group_id",
                      "merchant_category_id", "subsector_id"]
for col in minus_1_to_nan_cols:
    merc_df[col] = merc_df[col].replace(-1, pd.np.nan)

num_to_cat_cols = ["category_2", "city_id", "state_id",
                  "merchant_group_id", "merchant_category_id",
                   "subsector_id"]
merc_df = numeric_to_cat(merc_df, num_to_cat_cols)

merc_df.shape
merc_df.head()


# In[24]:


merc_df.merchant_id.unique().shape
merc_df.merchant_id.unique().shape[0] == merc_df.shape[0]


# In[25]:


temp = merc_df.merchant_id.value_counts()
temp[temp > 1]


# In[26]:


merc_df[merc_df.merchant_id == "M_ID_d123532c72"]


# In[27]:


temp_nulls = merc_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls*100/merc_df.shape[0]})
temp


# In[28]:


merc_df[["numerical_1", "numerical_2"]].describe()


# In[29]:


fig, ax = plt.subplots(1, 2)
_ = merc_df[["numerical_1"]].boxplot(ax=ax[0])
_ = merc_df[["numerical_2"]].boxplot(ax=ax[1])

fig, ax = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.numerical_1 < 0.1, ["numerical_1"]].boxplot(ax=ax[0])
_ = merc_df.loc[merc_df.numerical_2 < 0.01, ["numerical_2"]].boxplot(ax=ax[1])
plt.tight_layout()


# In[30]:


_ = merc_df[["numerical_1", "numerical_2"]].plot.scatter(x="numerical_1", y="numerical_2")


# In[31]:


print("category_1")
get_categorical_stats(merc_df, "category_1")
print("category_2")
get_categorical_stats(merc_df, "category_2")
print("category_4")
get_categorical_stats(merc_df, "category_4")


# In[32]:


merc_df.groupby(["category_4", "category_2"])["category_1"].count()


# In[33]:


print("Ratio of category_4 with value N to value Y based on category_2 freqeuncy counts")
temp1 = merc_df.loc[merc_df.category_4 == "N", "category_2"].value_counts(dropna=False)
temp2 = merc_df.loc[merc_df.category_4 == "Y", "category_2"].value_counts(dropna=False)
temp1/temp2


# In[34]:


merc_df.groupby(["category_1", "category_2"])["category_4"].count()


# In[35]:


merc_df.loc[merc_df.category_1 == "Y", "category_2"].value_counts(dropna=False)


# In[36]:


merc_df.groupby(["category_1", "category_4"])["category_2"].count()


# In[37]:


print("city_id")
get_categorical_stats(merc_df, "city_id")


# In[38]:


print("state_id")
get_categorical_stats(merc_df, "state_id")


# In[39]:


merc_df[merc_df.state_id.isna()].category_2.value_counts(dropna=False)


# In[40]:


get_col_stats(merc_df[["merchant_id", "merchant_group_id", "merchant_category_id", "subsector_id"]])


# In[41]:


(
    merc_df.merchant_id.astype(str)\
    + merc_df.merchant_group_id.astype(str)
).unique().shape
merc_df.shape


# In[42]:


print("most_recent_sales_range")
get_categorical_stats(merc_df, "most_recent_sales_range")
print("most_recent_purchases_range")
get_categorical_stats(merc_df, "most_recent_purchases_range")


# In[43]:


merc_df.groupby(["most_recent_sales_range", "most_recent_purchases_range"])    .merchant_id    .count()


# In[44]:


fig, axs = plt.subplots(2, 3, figsize=(10, 5))

i = 0
j = 0
for val in merc_df.most_recent_purchases_range.unique():
    ax = axs[i, j]
    _ = merc_df        .loc[merc_df.most_recent_purchases_range == val, "most_recent_sales_range"]        .value_counts()        .plot.bar(ax=ax)
    _ = ax.set_title(f"most_recent_purchases_range = {val}")
    if i==0 and j==2:
        i = 1
        j = 0
    else:
        j += 1

plt.tight_layout()


# In[45]:


merc_df[["avg_sales_lag3", "avg_purchases_lag3", "active_months_lag3"]].describe()


# In[46]:


get_categorical_stats(merc_df, "active_months_lag3")


# In[47]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag3"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag3"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[48]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag3 < 10, ["avg_sales_lag3"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag3 < 10, ["avg_purchases_lag3"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[49]:


temp = merc_df.loc[(merc_df.avg_sales_lag3 < 10) & (merc_df.avg_sales_lag3 > -10), ["avg_sales_lag3", "avg_purchases_lag3"]]
temp.plot.scatter(x="avg_sales_lag3", y="avg_purchases_lag3")


# In[50]:


merc_df[["avg_sales_lag6", "avg_purchases_lag6", "active_months_lag6"]].describe()


# In[51]:


get_categorical_stats(merc_df, "active_months_lag6")


# In[52]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag6"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag6"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[53]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag6 < 10, ["avg_sales_lag6"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag6 < 10, ["avg_purchases_lag6"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[54]:


temp = merc_df.loc[(merc_df.avg_sales_lag6 < 10) & (merc_df.avg_sales_lag6 > -10), ["avg_sales_lag6", "avg_purchases_lag6"]]
temp.plot.scatter(x="avg_sales_lag6", y="avg_purchases_lag6")


# In[55]:


merc_df[["avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12"]].describe()


# In[56]:


get_categorical_stats(merc_df, "active_months_lag12")


# In[57]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag12"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag12"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[58]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag12 < 10, ["avg_sales_lag12"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag12 < 10, ["avg_purchases_lag12"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[59]:


temp = merc_df.loc[(merc_df.avg_sales_lag12 < 10) & (merc_df.avg_sales_lag12 > -10), ["avg_sales_lag12", "avg_purchases_lag12"]]
temp.plot.scatter(x="avg_sales_lag12", y="avg_purchases_lag12")


# In[60]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

filt = (merc_df.avg_sales_lag3 < 10) & (merc_df.avg_sales_lag3 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag3"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag3", ax=axs[0])

filt = (merc_df.avg_sales_lag6 < 10) & (merc_df.avg_sales_lag6 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag6"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag6", ax=axs[1])

filt = (merc_df.avg_sales_lag12 < 10) & (merc_df.avg_sales_lag12 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag12"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag12", ax=axs[2])


# In[61]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

filt = (merc_df.avg_purchases_lag3 < 10) & (merc_df.avg_purchases_lag3 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag3"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag3", ax=axs[0])

filt = (merc_df.avg_purchases_lag6 < 10) & (merc_df.avg_purchases_lag6 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag6"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag6", ax=axs[1])

filt = (merc_df.avg_purchases_lag12 < 10) & (merc_df.avg_purchases_lag12 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag12"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag12", ax=axs[2])


# In[62]:


merc_df.head()


# In[63]:


lag_cols = [
    "avg_sales_lag3", "avg_purchases_lag3", "active_months_lag3",
    "avg_sales_lag6", "avg_purchases_lag6", "active_months_lag6",
    "avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12"
]
corr = merc_df[lag_cols].corr()
corr.style.background_gradient()


# In[64]:


filt = (merc_df.state_id.isna()) & (merc_df.category_2.isna())
temp = merc_df.loc[filt]
temp.head()


# In[65]:





# In[65]:





# In[65]:





# In[65]:





# In[65]:





# In[65]:





# In[65]:


hist_df = pd.read_csv(hist_f)
hist_df.shape
hist_df.head()


# In[66]:


len(set(hist_df.card_id) - set(card_df.card_id) - set(test_df.card_id))


# In[67]:


set(hist_df.merchant_id) - set(merc_df.merchant_id)


# In[68]:


temp_nulls = hist_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls/hist_df.shape[0]})
temp


# In[69]:


new_hist_df = pd.read_csv(new_hist_f)
new_hist_df.shape
new_hist_df.head()


# In[70]:


len(set(new_hist_df.card_id) - set(card_df.card_id) - set(test_df.card_id))


# In[71]:


set(new_hist_df.merchant_id) - set(merc_df.merchant_id)


# In[72]:


temp_nulls = new_hist_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls/new_hist_df.shape[0]})
temp


# In[73]:





# In[73]:





# In[73]:





# In[73]:





# In[73]:





# In[73]:





# In[73]:





# In[73]:





# In[73]:




