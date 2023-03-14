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


get_ipython().system('pip install pyspark')


# In[3]:


import pyspark
import pandas as pd
import numpy as np

spark = pyspark.sql.SparkSession.builder.appName("MyApp")             .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1")             .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")             .getOrCreate()


# In[4]:


train = spark.read.csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv',header = True, inferSchema = True)


# In[5]:


train.printSchema()


# In[6]:


from pyspark.sql import functions as F
from pyspark.sql.functions import mean, exp ,lit , col, pow
from pyspark.sql import SQLContext
from pyspark import SparkContext



name = ["ps_car_06_cat","ps_car_01_cat","ps_car_11_cat" ]
for i in range(len(name)):
    train = train.withColumnRenamed('target','label')
    trn_series = train.select(name[i],"label")
    tr1 , tr2, tr3 , tr4 , tr5 = trn_series.randomSplit([.2,.2,.2,.2,.2])

    tr = [tr1, tr2,tr3, tr4, tr5]

    min_samples_leaf = 1
    smoothing = 1
    averages = [[],[],[],[],[]]

    for k in range(5):
    # Compute target mean 
        n = tr[k]
        a = n.groupBy(name[i]).agg(F.count(n.label),F.avg(n.label))
        prior = n.agg(mean(F.col("label").alias("mean"))).collect()[0]["avg(label AS `mean`)"]
        print(prior)
    # The bigger the count the less full_avg is taken into account
        a = a.withColumn("averages",prior * (1 - smoothing) + a["avg(label)"] * smoothing)
        a.drop("avg(label)", "count(label)")
        averages[k] = a
        print(a)
    total = averages[0].union(averages[1])
    total = total.union(averages[2])
    total = total.union(averages[3])
    total = total.union(averages[4])
    targetkfoldmean = total.groupBy(name[i]).agg(F.mean("averages"))
    targetkfoldmean = targetkfoldmean.withColumnRenamed(name[i],name[i]+'encod')
    train = train.withColumnRenamed(name[i],name[i]+'encod')
    targetkfoldmean = targetkfoldmean.withColumnRenamed( "avg(averages)",name[i]+'replaced')

    train = targetkfoldmean.join(train, on = name[i]+'encod' , how = 'full')

