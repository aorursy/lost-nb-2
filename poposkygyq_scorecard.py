#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/GiveMeSomeCredit"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#载入数据
df = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')
#数据集确实和分布情况
df.describe()
df.info()


# In[ ]:


df=df[df.NumberOfDependents.notnull()]#删除比较少的缺失值
df.describe()


# In[ ]:


df.info()# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.ix[:,[5,0,1,2,3,4,6,7,8,9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()]
    unknown = process_df[process_df.MonthlyIncome.isnull()]
    var=known.columns.tolist()
    # X为特征属性值
    X = known.loc[:, var ].drop("MonthlyIncome",axis = 1)
    X.describe()
    y=known['MonthlyIncome']
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, 
    n_estimators=200,max_depth=3,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown.drop("MonthlyIncome",axis = 1)).round(0)
    print(predicted)
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


# In[ ]:


df=set_missing(df)#用随机森林填补比较多的缺失值
df.info()


# In[ ]:


df=df.dropna()#删除比较少的缺失值
df = df.drop_duplicates()#删除重复项
df.to_csv('MissingData.csv',index=False)


# In[ ]:


df['default.payment.next.month'].value_counts()


# In[ ]:





# In[ ]:


process_df = df.ix[:,[0,1,2,3,4,5,6,7,8,9]]
known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
X = known[:, 1:]
X

