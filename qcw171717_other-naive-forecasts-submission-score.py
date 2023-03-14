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
from matplotlib.pyplot import figure
from tqdm import tqdm


# In[3]:


df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')


# In[4]:


price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")


# In[5]:


df.head()


# In[6]:


price_df.head()


# In[7]:


cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")


# In[8]:


cal_df.head()


# In[9]:


cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"


# In[10]:


cal_df[cal_df["d"]==1858]


# In[11]:


cal_df[cal_df["d"]==1886]


# In[12]:


for day in tqdm(range(1858, 1886)):
    wk_id = list(cal_df[cal_df["d"]==day]["wm_yr_wk"])[0]
    wk_price_df = price_df[price_df["wm_yr_wk"]==wk_id]
    df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
    df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]
    df.drop(columns=["sell_price"], inplace=True)


# In[13]:


df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales")==0]].sum(axis=1)


# In[14]:


df.drop(columns=[c for c in df.columns if c.find("unit_sales")==0], inplace=True)


# In[15]:


df["weight"] = df["dollar_sales"] / df["dollar_sales"].sum()


# In[16]:


df.drop(columns=["dollar_sales"], inplace=True)


# In[17]:


display(cal_df[cal_df["d"]==1886])
display(cal_df[cal_df["d"]==1858])


# In[18]:


df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) <= 1885] +       ["id"]].set_index("id").transpose()


# In[19]:


complete_historical_mean_df =    df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) <= 1885] +       ["id"]].set_index("id").transpose().mean().reset_index()


# In[20]:


complete_historical_mean_df.head()


# In[21]:


# Nothing is always 0
df[[c for c in df.columns if c.find("d_")==0]].sum(axis=1).min()


# In[22]:


def find_first_non_0(s):
    assert type(s) == np.ndarray
    return (s!=0).argmax(axis=0)


# In[23]:


non_0_strt_arr = []
hist_arr = np.array(df[[c for c in df.columns if c.find("d_")==0]])
for i in tqdm(range(len(df))):
    non_0_strt_arr.append(find_first_non_0(hist_arr[i, :]))


# In[24]:


df.head(1)


# In[25]:


test = list(df[[c for c in df.columns if c.find("d_")==0] +                ["id"]].set_index("id").transpose()["HOBBIES_1_001_CA_1_validation"])


# In[26]:


print("Supposedly first non-zero value equals:", test[non_0_strt_arr[0]], 
      "on the", non_0_strt_arr[0], "day",
     "\nSum of all values before the supposedly first non-zero value is:", 
     sum(test[: non_0_strt_arr[0]]),
     "\nSum of all values after the supposedly first non-zero value is:", 
     sum(test[non_0_strt_arr[0]:]))


# In[27]:


num_non_zero = 1885 - np.array(non_0_strt_arr)


# In[28]:


non_zero_historical_mean_arr = np.array(df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) <= 1885] +   ["id"]].set_index("id").transpose().sum().reset_index()[0]) / num_non_zero


# In[29]:


# days 1886 to 1913 are local test weeks
for d in range(1, 29):
    df["F_1_" + str(1885+d)] = list(complete_historical_mean_df[0])
    df["F_2_" + str(1885+d)] = non_zero_historical_mean_arr


# In[30]:


method_dict = {1: "complete historical mean", 2: "historical mean after first non-zero"}


# In[31]:


num_non_zero.min()


# In[32]:


historical_mean_df10 =    df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1876, 1886)] +       ["id"]].set_index("id").transpose().mean().reset_index()

historical_mean_df20 =    df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1866, 1886)] +       ["id"]].set_index("id").transpose().mean().reset_index()

historical_mean_df30 =    df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1856, 1886)] +       ["id"]].set_index("id").transpose().mean().reset_index()

historical_mean_df40 =    df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1846, 1886)] +       ["id"]].set_index("id").transpose().mean().reset_index()


# In[33]:


# days 1886 to 1913 are local test weeks
for d in range(1, 29):
    df["F_3_" + str(1885+d)] = list(historical_mean_df10[0])
    df["F_4_" + str(1885+d)] = list(historical_mean_df20[0])
    df["F_5_" + str(1885+d)] = list(historical_mean_df30[0])
    df["F_6_" + str(1885+d)] = list(historical_mean_df40[0])


# In[34]:


method_dict[3] = "historical mean of recent 10 days"
method_dict[4] = "historical mean of recent 20 days"
method_dict[5] = "historical mean of recent 30 days"
method_dict[6] = "historical mean of recent 40 days"


# In[35]:


for d in range(1, 29):
    df["F_7_" + str(1885 + d)] = df["d_" + str(1885 + d - 28)]


# In[36]:


method_dict[7] = "same as last 28 days"


# In[37]:


display(cal_df[cal_df["d"]==1886])
display(cal_df[cal_df["d"]==1886 - 364])


# In[38]:


denominator = [(num // 364) if (num // 364) > 0 else 1 for num in num_non_zero]
for d in range(1, 29): 
    df["F_8_" + str(1885 + d)] = (df["d_" + str(1885 + d - 364*1)]+                                  df["d_" + str(1885 + d - 364*2)]+                                  df["d_" + str(1885 + d - 364*3)]+                                  df["d_" + str(1885 + d - 364*4)]+                     df["d_" + str(1885 + d - 364*5)]) / denominator


# In[39]:


method_dict[8] = "average of same day in historical years"


# In[40]:


agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0 or c.find("F_") == 0]].sum()).transpose()
agg_df["level"] = 1
agg_df["weight"] = 1/12
column_order = agg_df.columns


# In[41]:


agg_df


# In[42]:


level_groupings = {2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"], 
              6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
              10: ["item_id"], 11: ["item_id", "state_id"]}


# In[43]:


for level in tqdm(level_groupings):
    temp_df = df.groupby(by=level_groupings[level]).sum().reset_index(drop=True)
    temp_df["level"] = level
    temp_df["weight"] /= 12
    agg_df = agg_df.append(temp_df[column_order])

del temp_df


# In[44]:


df["weight"] /= 12


# In[45]:


print(df.shape[0], agg_df.shape[0], df.shape[0] + agg_df.shape[0])


# In[46]:


agg_df["weight"].sum() + df["weight"].sum()


# In[47]:


h = 28
n = 1885
def rmsse(ground_truth, forecast, train_series, axis=1):
    # assuming input are numpy array or matrices
    assert axis == 0 or axis == 1
    assert type(ground_truth) == np.ndarray and type(forecast) == np.ndarray and type(train_series) == np.ndarray
    
    if axis == 1:
        # using axis == 1 we must guarantee these are matrices and not arrays
        assert ground_truth.shape[1] > 1 and forecast.shape[1] > 1 and train_series.shape[1] > 1
    
    numerator = ((ground_truth - forecast)**2).sum(axis=axis)
    if axis == 1:
        denominator = 1/(n-1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)
    else:
        denominator = 1/(n-1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)
    return (1/h * numerator/denominator) ** 0.5


# In[48]:


train_series_cols = [c for c in df.columns if c.find("d_") == 0][:-28]
ground_truth_cols = [c for c in df.columns if c.find("d_") == 0][-28:]

forecast_cols_dict = {}
for i in range(1, 9):
    forecast_cols_dict[i] = [c for c in df.columns if c.find("F_"+str(i)+"_") == 0]


# In[49]:


for i in range(1, 9):
    df["rmsse_" + str(i)] = rmsse(np.array(df[ground_truth_cols]), 
        np.array(df[forecast_cols_dict[i]]), np.array(df[train_series_cols]))
    agg_df["rmsse_" + str(i)] = rmsse(np.array(agg_df[ground_truth_cols]), 
        np.array(agg_df[forecast_cols_dict[i]]), np.array(agg_df[train_series_cols]))


# In[50]:


for i in range(1, 9):
    df["wrmsse_" + str(i)] = df["weight"] * df["rmsse_" + str(i)]
    agg_df["wrmsse_" + str(i)] = agg_df["weight"] * agg_df["rmsse_" + str(i)]


# In[51]:


for i in range(1, 9):
    print("method:", method_dict[i])
    print(df["wrmsse_" + str(i)].sum() + agg_df["wrmsse_" + str(i)].sum())
    print()


# In[52]:


sample_sub = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")


# In[53]:


sample_sub.head()


# In[54]:


sample_sub.tail()


# In[55]:


(sample_sub["id"][:len(df)] == df["id"]).all()


# In[56]:


submit_df = df[["id"]]
for i in range(1, 29):
    submit_df["F" + str(i)] = df["F_7_" + str(1885 + i)]


# In[57]:


submit_df2 = submit_df.copy()
submit_df2["id"] = submit_df2["id"].apply(lambda x: x.replace('validation',
                                                              'evaluation'))


# In[58]:


submit_df = submit_df.append(submit_df2).reset_index(drop=True)


# In[59]:


submit_df.to_csv("submission.csv", index=False)


# In[ ]:




