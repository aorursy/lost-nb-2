#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark')


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyspark.sql.functions as func
import sys
import os
from datetime import  datetime, timedelta
from pyspark.sql.types import ArrayType,DateType,StructType
from pyspark.sql.functions import when,col,lit
from pyspark.sql.window import Window
from pyspark.sql import Window
from pyspark.sql import functions as F

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Rossmann').getOrCreate()


# In[4]:


path      = "../input"
file_name = path + '/train.csv' 


# In[5]:


from pyspark.sql.types import (StructField, StringType, StructType, IntegerType,
                               FloatType, DateType) 

data_schema = [StructField('Store', StringType(), True),
               StructField('DayOfWeek', StringType(), True),
               StructField('Date', DateType(), True),
               StructField('Sales',FloatType(), True),
               StructField('Customers',StringType(), True),
               StructField('Open',StringType(), True),
               StructField('Promo',StringType(), True),
               StructField('StateHoliday',StringType(), True)]


# In[6]:


final_struc = StructType(fields = data_schema)
df = spark.read.csv(file_name,schema=final_struc)


# In[7]:


#Consider only active stores
df = df.where(df['Open'] == 1)
df.count()


# In[8]:


df.printSchema()


# In[9]:


# function for getting date range
def generate_date_series(start, stop):
    date_range = []
    no_of_days = (stop-start).days + 1
    for x in range(0, no_of_days):
        date_list = [start + timedelta(days=x)]
        k = date_list[0]
        date_range.append([k])
    return date_range


# In[10]:


from pyspark.sql.functions import count, isnan


# In[11]:


df = df.where(df['Store'] == 1)


# In[12]:


df.createOrReplaceTempView('storedata')


# In[13]:


result = spark.sql("SELECT MIN(Date) as mindate,MAX(Date) as maxdate from storedata ")
minim = result.head(5)[0][0] 
maxim = result.head(5)[0][1]


# In[14]:


minim = minim - timedelta(days=1)
date_list = generate_date_series(minim,maxim)


# In[15]:



df_datelist = spark.createDataFrame(date_list,['Date'])
df_datelist.show(5)


# In[16]:


df = df_datelist.join(df,["Date"],"leftouter").orderBy("Date")
df.head(5)


# In[17]:



pd_df = df.toPandas()


# In[18]:


pd_df = pd_df.fillna(method = 'bfill').fillna(method = 'ffill')


# In[19]:


#placing back to spark dataframe
df = spark.createDataFrame(pd_df)


# In[20]:


df.head(5)


# In[21]:


df.filter((df["Customers"] == "") | df["Customers"].isNull() | isnan(df["Customers"])).count()


# In[22]:


df.filter((df["Store"] == "") | df["Store"].isNull() | isnan(df["Store"])).count()


# In[23]:


from pyspark.sql.functions import weekofyear,year
                                        


# In[24]:


df = df.withColumn("Week",weekofyear("Date"))


# In[25]:


df= df.withColumn('Year', year('Date'))


# In[26]:


df.printSchema()


# In[27]:


#Change the Year data type to string
df = df.withColumn("Year", df["Year"].cast(StringType()))
df = df.withColumn("Store", df["Store"].cast(IntegerType()))


# In[28]:


df.printSchema()

df.createOrReplaceTempView('storedata')


# In[29]:


result = spark.sql("select  Year, Week, sum(Sales) as Tot_Sales from storedata                    group by Year, Week order by Year desc,Week desc ")


# In[30]:


result.head(5)


# In[31]:




windowval = (Window.partitionBy('Year').orderBy(('Week'))
             .rangeBetween(Window.unboundedPreceding, 0))
df_w_cumsum = result.withColumn('cum_sum', F.sum('Tot_Sales').over(windowval))
df_w_cumsum.orderBy((df_w_cumsum['Year']).desc(),(df_w_cumsum['Week']).desc() ).show(4)
#df_w_cumsum.head(5)


# In[32]:





# In[32]:




