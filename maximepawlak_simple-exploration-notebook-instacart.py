#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


order_products_train_df = pd.read_csv("../input/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")
aisles_df = pd.read_csv("../input/aisles.csv")
departments_df = pd.read_csv("../input/departments.csv")


# In[4]:


orders_df.head()


# In[5]:


order_products_train_df.head()


# In[6]:


order_products_prior_df.head()


# In[7]:


cnt_srs = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel("# Occurences")
plt.xlabel("eval_set type")
plt.title("Count of rows in each dataset")
plt.xticks(rotation=45)
plt.show()


# In[8]:


def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
print(cnt_srs)


# In[9]:


cnt_srs = orders_df.groupby("user_id")["order_number"].aggregate(np.max)
cnt_srs = cnt_srs.reset_index()
cnt_srs = cnt_srs.order_number.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel("# occurences")
plt.xlabel("Maximum order number")
plt.title("# Max order number")
plt.xticks(rotation=45)
plt.show()


# In[10]:


plt.figure(figsize=(12,8))
sns.countplot(x="order_dow", data=orders_df, color=color[0])
plt.ylabel("Count")
plt.xlabel("Day of week")
plt.title("Frequency of order by week day")
plt.xticks(rotation=45)
plt.show()


# In[11]:


plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders_df, color=color[1])
plt.ylabel("# occurences")
plt.xlabel("Hour of day")
plt.title("Frequencies of order by hour of day")
plt.xticks(rotation=45)
plt.show()


# In[12]:


grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"]    .aggregate("count").reset_index()
grouped_df = grouped_df.pivot("order_dow", "order_hour_of_day", "order_number")

plt.figure(figsize=(12,8))
sns.heatmap(grouped_df)
plt.title("Frequency of day_of_week VS hour_of_day")
plt.show()


# In[13]:


plt.figure(figsize=(12,8))
sns.countplot(x="days_since_prior_order", data=orders_df, color=color[3])
plt.ylabel("#")
plt.xlabel("days_since_prior_order")
plt.title("Frequency by days_since_prior_order")
plt.show()


# In[14]:


order_products_prior_df["reordered"].sum() / order_products_prior_df.shape[0]


# In[15]:


order_products_train_df["reordered"].sum() / order_products_train_df.shape[0]


# In[16]:


def no_re_ordered_products_rate(df):
    grouped_df = df.groupby("order_id")["reordered"].aggregate("sum").reset_index()
    grouped_df["reordered"].loc[grouped_df["reordered"]>1] = 1
    return grouped_df.reordered.value_counts() / grouped_df.shape[0]

no_re_ordered_products_rate(order_products_prior_df)


# In[17]:


no_re_ordered_products_rate(order_products_train_df)


# In[18]:


grouped_df = order_products_train_df.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_df["add_to_cart_order"].value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel("# Count")
plt.xlabel("# of products in the given order")
plt.show()


# In[19]:


products_df.head()


# In[20]:


aisles_df.head()


# In[21]:


departments_df.head()


# In[22]:


order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')
order_products_prior_df.head()


# In[23]:


cnt_srs = order_products_prior_df["product_name"].value_counts().reset_index()
cnt_srs = cnt_srs.head(20)
cnt_srs.columns = ["product_name", "frequency_count"]
cnt_srs


# In[24]:


cnt_srs = order_products_prior_df["aisle"].value_counts().head(20)
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel("# count")
plt.xlabel("Aisle")
plt.title("Distribution of products by aisle")
plt.xticks(rotation=45)
plt.show()


# In[25]:


temp_series = order_products_prior_df["department"].value_counts()
plt.figure(figsize=(8,8))
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series/temp_series.sum())*100))

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=200)
plt.title("Distribution of products by department")
plt.show()


# In[26]:


grouped_df = order_products_prior_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df["department"].values, grouped_df["reordered"].values, alpha=0.8, color=color[2])
plt.ylabel("Reordered %")
plt.xlabel("Department")
plt.title("Reordered ratio by department")
plt.xticks(rotation=45)
plt.show()


# In[27]:


grouped_df = order_products_prior_df.groupby(["department_id", "aisle"])["reordered"]    .aggregate("mean").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_df.reordered.values, grouped_df.department_id.values)
for i, txt in enumerate(grouped_df.aisle.values):
    ax.annotate(txt, (grouped_df.reordered.values[i], grouped_df.department_id.values[i]),                rotation=45, ha="center", va="center", color="green")


plt.xlabel("Reorder ratio")
plt.ylabel("department_id")
plt.title("Reorder ratio by aisle and department")
plt.show()


# In[28]:


order_products_prior_df["add_to_cart_order_mod"] = order_products_prior_df["add_to_cart_order"]    .copy()
order_products_prior_df["add_to_cart_order_mod"].loc[    order_products_prior_df["add_to_cart_order_mod"] >70] = 70
grouped_df = order_products_prior_df.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.add_to_cart_order_mod.values, grouped_df.reordered.values, alpha=0.8, color=color[3])   
plt.ylabel("Reorder ratio")
plt.xlabel("add_to_cart")
plt.title("Reorder ratio by add_to_cart")
plt.show()


# In[29]:





# In[29]:





# In[29]:




