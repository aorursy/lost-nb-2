#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Changed the data-types in order to minimize the file-size
types = {
    'id': 'uint32',
    'item_nbr': 'uint32',
    'store_nbr': 'uint16',
    'unit_sales': 'float32',
    'onpromotion': bool,
}
df_train = pd.read_csv("../input/train.csv", parse_dates=['date'], dtype=types, infer_datetime_format=True)


# In[3]:


df_train.head()


# In[4]:


print("{} datapoints and {} features".format(*df_train.shape))


# In[5]:


df_train.loc[:, "year"] = df_train["date"].dt.year.astype("uint16")
df_train.loc[:, "week"] = df_train["date"].dt.week.astype("uint16")
df_train.loc[:, "weekday"] = df_train["date"].dt.weekday.astype("uint16")
df_train.loc[:, "month"] = df_train["date"].dt.month.astype("uint16")


# In[6]:


_ = df_train.groupby("year").aggregate({"id": "count"}).plot(kind="bar", alpha=.5, figsize=(12, 5))
_ = plt.title("Distribution of data pr. year")
_ = plt.ylabel("Count")


# In[7]:


data = df_train.groupby(["year", "month"], as_index=False).aggregate({"id": "count"})
_ = sns.factorplot(x="month", y="id", col="year", col_wrap=3, data=data)


# In[8]:


_ = df_train.groupby("store_nbr").aggregate({"id": "count"}).plot(kind="bar", figsize=(12, 6), alpha=.5, width=1)
_ = plt.title("Distribution of datapoints pr. store")
_ = plt.ylabel("Count")


# In[9]:


ax = df_train.set_index("date")["unit_sales"].resample("M").sum().plot(figsize=(12, 6))
_ = ax.set_ylabel("unit_sales")
_ = ax.set_title("Monthly Sales Volum")


# In[10]:


ax = df_train.set_index("date")["unit_sales"].resample("W").sum().plot(figsize=(12, 6))
_ = ax.set_ylabel("unit_sales")
_ = ax.set_title("Weekly Sales Volum")


# In[11]:


ax = df_train.set_index("date")["unit_sales"].resample("D").sum().plot(figsize=(12, 6))
_ = ax.set_ylabel("unit_sales")
_ = ax.set_title("Daily Sales Volum")


# In[12]:


ax = df_train.set_index("date")["unit_sales"].resample("W").mean().plot(figsize=(12, 6), label="Mean")
ax = df_train.set_index("date")["unit_sales"].resample("W").median().plot(figsize=(12, 6), label="Median")
ax = df_train.set_index("date")["unit_sales"].resample("W").std().plot(figsize=(12, 6), label="Std")

_ = ax.set_ylabel("unit_sales")
_ = plt.legend(loc="best")


# In[13]:


def detect_outliars(col, df):
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    step = 1.5 * (Q3 - Q1)
    outliar_mask = ~((df[col] >= Q1 - step) & (df[col] <= Q3 + step))
    
    return outliar_mask


outliar_mask = detect_outliars("unit_sales", df_train)

print("Percentage of training-data that is classified as outliars: {}%".format(round(len(df_train[outliar_mask]) / float(len(df_train)) * 100, 2)))


# In[14]:


df_train[outliar_mask]["store_nbr"].value_counts().plot(kind="bar", figsize=(12, 6), width=1, alpha=.5)
_ = plt.title("Distribution of outliars accross the different stores")


# In[15]:


_ = df_train["unit_sales"].apply(np.log).hist(bins=25, range=(-2, 6), figsize=(12, 6), alpha=.5)
_ = plt.xlabel("unit_sales log-transformed")
_ = plt.ylabel("Count")
_ = plt.title("Distribution of sales-volume")


# In[16]:


data = pd.crosstab(df_train["year"], df_train["week"], df_train["unit_sales"], aggfunc="sum", normalize=True)

_ = plt.figure(figsize=(14, 3))
_ = sns.heatmap(data, cmap="viridis")
_ = plt.title("Heatmap of Sales Volume Year vs. Week")


# In[17]:


data = pd.crosstab(df_train["year"], df_train["weekday"], df_train["unit_sales"], aggfunc="sum", normalize=True)
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_ = plt.figure(figsize=(14, 3))
_ = sns.heatmap(data, cmap="viridis")
_ = plt.xticks(range(7), weekday_names)
_ = plt.title("Heatmap of Sales Volume Year vs. Weekday")


# In[18]:


data = pd.crosstab(df_train["year"], df_train["month"], df_train["unit_sales"], aggfunc="sum", normalize=True)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_ = plt.figure(figsize=(14, 3))
_ = sns.heatmap(data, cmap="viridis")
_ = plt.xticks(range(12), month_names)
_ = plt.title("Heatmap of Sales Volume Year vs. Month")


# In[19]:


data = pd.crosstab(df_train["weekday"], df_train["month"], df_train["unit_sales"], aggfunc="sum", normalize=True)

_ = plt.figure(figsize=(14, 3))
_ = sns.heatmap(data, cmap="viridis")
_ = plt.xticks(range(12), month_names)
_ = plt.yticks(range(7), weekday_names, rotation=0)
_ = plt.title("Heatmap of Sales Volume Weekday vs. Month")


# In[20]:


def on_promotion(x):
    if pd.isnull(x):
        return -1
    elif x == True:
        return 1
    else:
        return 0

df_train.loc[:, "onpromotion"] = df_train["onpromotion"].apply(on_promotion)
df_train.loc[:, "onpromotion"] = df_train["onpromotion"].astype("int8")


# In[21]:


df_train["onpromotion"].value_counts()


# In[22]:


promotion = len(df_train[df_train["onpromotion"] == 1])
no_promotion = len(df_train[df_train["onpromotion"] == 0])
unknown = len(df_train[df_train["onpromotion"] == -1])

print("{}% of traning_set is on promotion".format(round(promotion / float(len(df_train)) * 100, 2)))
print("{}% of traning_set is not on promotion".format(round(no_promotion / float(len(df_train)) * 100, 2)))
print("{}% of traning_set is unkown with regards to promotion".format(round(unknown / float(len(df_train)) * 100, 2)))


# In[23]:


data = df_train     .groupby(["store_nbr", "onpromotion"], as_index=False)     .aggregate({"id": "count"})     .pivot(index="store_nbr", columns="onpromotion",values="id")
    
_ = data     .apply(lambda x: x / data.sum(axis=1) * 100)     .fillna(0)     .plot     .bar(stacked=True, width=1, figsize=(14, 6), alpha=.5)   
    
_ = plt.legend(["Unkown", "No Promotion", "Promotion"], bbox_to_anchor=(1.2, 1.0))
_ = plt.title("Which store has the most products on promotion")
_ = plt.ylim(0, 100)


# In[24]:


df_items = pd.read_csv("../input/items.csv")
df_items.head()


# In[25]:


_ = plt.figure(figsize=(6,10))
ax = sns.countplot(y=df_items["family"])
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


# In[26]:


plt.figure(figsize=(10, 3))
sns.kdeplot(df_items[df_items["perishable"] == 1]["class"], shade=True, color="g")
sns.kdeplot(df_items[df_items["perishable"] == 0]["class"], shade=True, color="r")
plt.legend(["Perishable", "Non perishable"])
_ = plt.xlabel("family [id]")


# In[27]:


plt.figure(figsize=(10, 3))
sns.kdeplot(df_items[df_items["perishable"] == 1]["item_nbr"], shade=True, color="g")
sns.kdeplot(df_items[df_items["perishable"] == 0]["item_nbr"], shade=True, color="r")
plt.legend(["Perishable", "Non perishable"])
_ = plt.xlabel("item_nbr [id]")


# In[28]:


df_transactions = pd.read_csv("../input/transactions.csv", parse_dates=["date"], infer_datetime_format=True)
df_transactions.head()


# In[29]:


_ = plt.figure(figsize=(12, 5))
_ = plt.hist(df_transactions["transactions"], bins=50, alpha=.5)
_ = plt.xlabel("Number of Transactions")
_ = plt.ylabel("Count")


# In[30]:


data = df_transactions.groupby("date").aggregate({"transactions": "sum"})
_ = data.resample("W").mean().plot(figsize=(12, 6))
_ = plt.ylabel("Transactions")
_ = plt.title("Mean Transactions Weekly")


# In[31]:


outliar_mask = detect_outliars("transactions", df_transactions)
print("Percentage of outliars in transactions: {}%".format(round(len(df_transactions[outliar_mask]) / float(len(df_transactions)) * 100, 2) ))


# In[32]:


df_transactions[outliar_mask]["store_nbr"].value_counts().plot(kind="bar", figsize=(12, 6), width=1, alpha=.5)
_ = plt.ylabel("Number of Transactions")


# In[33]:


df_stores = pd.read_csv("../input/stores.csv")
df_stores.head()


# In[34]:


f, ax = plt.subplots(1, 2, figsize=(12, 4))
_ = sns.countplot(df_stores["type"].sort_values(), ax=ax[0])
_ = sns.countplot(df_stores["cluster"], ax=ax[1])


# In[35]:


data = df_stores     .groupby(["cluster", "type"])     .aggregate({"type": "count"})     .rename(columns={'type': 'type_count'})     .unstack(level=0)     .fillna(0)


ax = data.plot(kind="bar", stacked=True, figsize=(12, 4))
_ = plt.legend(["Cluster %d" % i for i in range(1, 18)], bbox_to_anchor=(1.3, 1.2))
_ = plt.ylabel("Count")
_ = plt.title("Distribution of Clusters pr. Type")


# In[36]:


data = df_stores     .groupby(["city", "type"])     .aggregate({"type": "count"})     .unstack(level=1)     .fillna(0)
    
_ = data.plot(kind="bar", stacked=True, figsize=(12, 4))
_ = plt.legend(["Type A", "Type B", "Type C", "Type D", "Type E"], loc="best")
_ = plt.ylabel("Count")
_ = plt.title("Distribution of Types pr. City")


# In[37]:


_ = plt.figure(figsize=(14, 4))
_ = sns.heatmap(pd.crosstab(df_stores["cluster"], df_stores["store_nbr"]), cmap="viridis")
_ = plt.yticks(rotation=0)
_ = plt.title("Which store belongs to which cluster")


# In[38]:




