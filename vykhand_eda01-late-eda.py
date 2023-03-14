#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn2, venn3
import squarify
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


pd.read_excel("../input/Data_Dictionary.xlsx", sheet = 0, header = 2)


# In[3]:


train  = pd.read_csv("../input/train.csv")
print(train.shape)

train.head()


# In[4]:


sns.distplot(train.target)


# In[5]:


test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# In[6]:


venn2([set(train.card_id), set(test.card_id)])


# In[7]:


sns.jointplot(train.index.values, train.target)


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


sns.jointplot(LabelEncoder().fit_transform(train.card_id), train.target)


# In[10]:


train["is_test"] = np.int8(0)
test["is_test"] = np.int8(1)
test["target"] = np.NaN
train_test = pd.concat([train,test], ignore_index=True, sort=True, axis = 0)
train_test["first_active_month"] = pd.to_datetime(train_test.first_active_month)
train_test["mon"] = train_test.first_active_month.dt.month
train_test["year"] = train_test.first_active_month.dt.year

print(train_test.shape)
train_test.head()


# In[11]:


del train
del test


# In[12]:


pd.merge(train_test.groupby(["is_test", "feature_1"])["card_id"].count().reset_index(),
                     train_test.groupby(["is_test"]).count().reset_index(), on = "is_test" )


# In[13]:


def compare_cat_pct_counts(df, col_name, compare_col_name = "is_test", ind_col ="card_id"):
    cnt_df = pd.merge(df.groupby([compare_col_name, col_name])[ind_col].count().reset_index(),
                      df.groupby([compare_col_name])[ind_col].count().reset_index(), on = "is_test" )
    cnt_df["cnt"] = cnt_df[ind_col + "_x"] / cnt_df[ind_col + "_y"] 
    sns.barplot(col_name, "cnt", hue = compare_col_name, data = cnt_df)
     


# In[14]:


compare_cat_pct_counts(train_test, "feature_1")


# In[15]:


compare_cat_pct_counts(train_test, "feature_2")


# In[16]:


compare_cat_pct_counts(train_test, "feature_3")


# In[17]:


plt.subplots(figsize=(15,6))
plt.subplot(131)
sns.boxplot(train_test["feature_1"], train_test.target)
plt.subplot(132)
sns.boxplot(train_test["feature_2"], train_test.target)
plt.subplot(133)
sns.boxplot(train_test["feature_3"], train_test.target)


# In[18]:


compare_cat_pct_counts(train_test, "mon")


# In[19]:


compare_cat_pct_counts(train_test, "year")


# In[20]:


plt.subplots(figsize=(15,6))
plt.subplot(121)
sns.boxplot(train_test.mon, train_test.target)
plt.subplot(122)
sns.boxplot(train_test.year, train_test.target)


# In[21]:


pd.read_excel("../input/Data_Dictionary.xlsx", 1)


# In[22]:


hist_trans = pd.read_csv("../input/historical_transactions.csv")
print(hist_trans.shape)
hist_trans.head()


# In[23]:


plt.figure(figsize=(20,10))
print(hist_trans.card_id.nunique())
venn3([set(hist_trans.card_id.unique()), set(train_test.query("is_test == 0").card_id), 
                                             set(train_test.query("is_test == 1").card_id)])


# In[24]:


hist_trans.card_id.value_counts().head(10)


# In[25]:


tmp_df = pd.merge(hist_trans.card_id.value_counts().reset_index(), train_test, left_on="index", right_on = "card_id")
plt.subplots(figsize = (14, 6))
plt.subplot(131)
sns.boxplot("is_test", "card_id_x", data=tmp_df )
plt.subplot(132)
sns.distplot(tmp_df.query("is_test == 0")["card_id_x"],  color = "blue" )
sns.distplot(tmp_df.query("is_test == 1")["card_id_x"],  color = "green")
plt.subplot(133)
sns.distplot(np.log(tmp_df.query("is_test == 0")["card_id_x"]),  color = "blue" )
sns.distplot(np.log(tmp_df.query("is_test == 1")["card_id_x"]),  color = "green")
del tmp_df


# In[26]:


# a helper function
def plot_cat_treemap(df, col_name, title = None):
    cnts = df[col_name].value_counts()
    cmap = matplotlib.cm.Spectral
    mini=min(cnts)
    maxi=max(cnts)
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in cnts]

    
    squarify.plot(sizes = cnts, label = cnts.index.values, value = cnts, color = colors)
    plt.axis("off")
    plt.title(title)


# In[27]:


card_auth_agg = pd.merge(train_test, 
                         hist_trans.groupby(["card_id", "authorized_flag"])["city_id"].count().unstack(level=-1),
                         on = "card_id").rename(columns = {"N":"cnt_unauthorized", "Y":"cnt_authorized"}).fillna(0)

card_auth_agg["cnt_trans"] = card_auth_agg["cnt_unauthorized"] + card_auth_agg["cnt_authorized"] 
#plt.subplot(311)
sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_trans)


# In[28]:


sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_unauthorized)


# In[29]:


sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_authorized)


# In[30]:


del card_auth_agg


# In[31]:


plt.figure(figsize=(20,14))
plot_cat_treemap(hist_trans, "city_id", "City ID")


# In[32]:


city_agg = hist_trans.groupby("city_id")["purchase_amount"].agg(["mean", "min", "max", "median"])


# In[33]:


city_agg.sort_values("mean", ascending=False).head()


# In[34]:


city_agg.sort_values("max", ascending=False).head()


# In[35]:


del city_agg


# In[36]:


city_card_agg  = pd.merge(hist_trans.groupby("card_id")["city_id"].agg(["nunique"]).reset_index(), train_test, on="card_id").rename(columns={"nunique":"city_count"})


# In[37]:


plt.subplot(111)
sns.distplot(city_card_agg.loc[~city_card_agg.target.isnull(), "city_count"], color = "blue")
sns.distplot(city_card_agg.loc[city_card_agg.target.isnull(), "city_count"], color = "green")


# In[38]:


del city_card_agg


# In[39]:


gc.collect()


# In[40]:


plt.figure(figsize=(20,14))
plot_cat_treemap(hist_trans, "state_id", "State ID")


# In[41]:


stat_city_df = hist_trans.groupby(["state_id", "city_id"])["card_id"].count().reset_index().sort_values("card_id", ascending=False)


# In[42]:


stat_city_df.query("state_id == -1")


# In[43]:


stat_city_df.query("city_id == -1")


# In[44]:


stat_city_df.query("city_id == 179")


# In[45]:


stat_city_df.query("city_id == 75")


# In[46]:


del stat_city_df


# In[47]:




