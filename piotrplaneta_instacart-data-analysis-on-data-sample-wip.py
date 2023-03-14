#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


aisles_df = pd.read_csv("../input/aisles.csv")
departments_df = pd.read_csv("../input/departments.csv")
order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
order_products_train_df = pd.read_csv("../input/order_products__train.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")


# In[3]:


orders_df[orders_df["eval_set"] == "train"]["user_id"].values


# In[4]:


orders_df[orders_df["eval_set"] == "train"].shape


# In[5]:


orders_df[orders_df["eval_set"] == "test"]["user_id"].values


# In[6]:


orders_df[orders_df["eval_set"] == "test"].shape


# In[7]:


TRAIN_ORDER_COUNT = 1000
VALIDATION_ORDER_COUNT = 500
TEST_ORDER_COUNT = 1000


# In[8]:


train_orders_df = orders_df[orders_df["eval_set"] == "train"].iloc[0:TRAIN_ORDER_COUNT, :]
train_orders_df.shape


# In[9]:


validation_orders_df = orders_df[orders_df["eval_set"] == "train"].iloc[TRAIN_ORDER_COUNT:TRAIN_ORDER_COUNT+VALIDATION_ORDER_COUNT, :]
validation_orders_df.shape


# In[10]:


test_orders_df = orders_df[orders_df["eval_set"] == "test"].iloc[0:TEST_ORDER_COUNT, :]
test_orders_df.shape


# In[11]:


order_products_prior_with_orders_df = pd.merge(order_products_prior_df, orders_df, on="order_id")


# In[12]:


def generate_product_sets_per_user(orders_df, prior_product_orders_df, check_if_in_train=True):
    all_order_products = pd.DataFrame(columns=["user_id", "product_id", "order_id", "reordered_in_train"])

    for index, order in tqdm(orders_df.iterrows()):
        user_id = order["user_id"]
        order_id = order["order_id"]
        user_order_products = prior_product_orders_df[prior_product_orders_df["user_id"] == user_id]
        product_ids = user_order_products["product_id"].unique()
        for product_id in product_ids:
            if check_if_in_train:
                reordered = len(
                    order_products_train_df[
                        (order_products_train_df["order_id"] == order_id) &\
                        (order_products_train_df["product_id"] == product_id)
                    ]
                ) > 0
            else:
                reordered = np.NaN
                
            all_order_products = all_order_products.append(
                {"user_id": user_id, "product_id": product_id, "order_id": order_id, "reordered_in_train": reordered},\
                ignore_index=True
            )
    
    return all_order_products


# In[13]:


train_set_prior_oder_products_with_orders = order_products_prior_with_orders_df[order_products_prior_with_orders_df["user_id"].isin(train_orders_df["user_id"])]
all_train_order_products = generate_product_sets_per_user(train_orders_df, train_set_prior_oder_products_with_orders)


# In[14]:


train_df = all_train_order_products


# In[15]:


train_df = train_df.astype(int)
train_df.describe()


# In[16]:


train_df = pd.merge(train_df, products_df, on="product_id")
train_df = pd.merge(train_df, departments_df, on="department_id")
train_df.head()


# In[17]:


by_department = train_df.groupby("department")["reordered_in_train"].mean()
by_department.sort_values()


# In[18]:


train_prior_oder_products_df = train_set_prior_oder_products_with_orders
grouped_by_user_and_product = train_prior_oder_products_df.groupby(["user_id", "product_id"]).count()
grouped_by_user_and_product = grouped_by_user_and_product.reset_index()[["user_id", "product_id", "order_id"]]
grouped_by_user_and_product.columns = [["user_id", "product_id", "product_reordered_count"]]
grouped_by_user_and_product.head()


# In[19]:


grouped_by_user_and_order = train_prior_oder_products_df.groupby(["user_id"]).count()
amount_of_orders_per_user = grouped_by_user_and_order.reset_index()[["user_id", "order_id"]]
amount_of_orders_per_user.columns = [["user_id", "amount_of_orders"]]
amount_of_orders_per_user.head()


# In[20]:


products_by_user_with_reordered_and_order_count = pd.merge(grouped_by_user_and_product, amount_of_orders_per_user, on="user_id")
products_by_user_with_reordered_and_order_count["reorder_ratio"] =    products_by_user_with_reordered_and_order_count["product_reordered_count"] /    products_by_user_with_reordered_and_order_count["amount_of_orders"]
    
products_by_user_with_reordered_and_order_count = products_by_user_with_reordered_and_order_count[["user_id", "product_id", "reorder_ratio"]]
products_by_user_with_reordered_and_order_count.head()


# In[21]:


train_df = pd.merge(train_df, products_by_user_with_reordered_and_order_count, on=["user_id", "product_id"])


# In[22]:


plt.figure(figsize=(20, 3))
plt.xticks(rotation='vertical')

reorder_ratio_by_department = train_df.groupby(["user_id", "department"]).mean()
reorder_ratio_by_department = reorder_ratio_by_department.reset_index()
reorder_ratio_by_department = reorder_ratio_by_department[reorder_ratio_by_department["reorder_ratio"] < 0.5]
sns.stripplot(x="department", y="reorder_ratio", data=reorder_ratio_by_department, jitter=0.2)

