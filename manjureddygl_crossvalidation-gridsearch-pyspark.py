#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install pyspark')


# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# In[4]:


sc = SparkContext(appName = "Forest_Cover")


# In[5]:


spark = SparkSession.Builder().getOrCreate()


# In[6]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


# In[7]:


train = spark.read.csv('../input/train.csv',header = True,inferSchema=True)
test = spark.read.csv('../input/test.csv',header = True,inferSchema=True)


# In[8]:


train.limit(5).toPandas()


# In[9]:


test.limit(5).toPandas()


# In[10]:


train.count()


# In[11]:


test.count()


# In[12]:


train_mod = train.withColumn("HF1", train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points) .withColumn("HF2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)) .withColumn("HR1", abs(train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)) .withColumn("HR2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)) .withColumn("FR1", abs(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)) .withColumn("FR2", abs(train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)) .withColumn("ele_vert", train.Elevation - train.Vertical_Distance_To_Hydrology) .withColumn("slope_hyd", pow((pow(train.Horizontal_Distance_To_Hydrology,2) + pow(train.Vertical_Distance_To_Hydrology,2)),0.5)) .withColumn("Mean_Amenities", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)/3) .withColumn("Mean_Fire_Hyd", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology)/2)

test_mod = test.withColumn("HF1", test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Fire_Points) .withColumn("HF2", abs(test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Fire_Points)) .withColumn("HR1", abs(test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)) .withColumn("HR2", abs(test.Horizontal_Distance_To_Hydrology - test.Horizontal_Distance_To_Roadways)) .withColumn("FR1", abs(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Roadways)) .withColumn("FR2", abs(test.Horizontal_Distance_To_Fire_Points - test.Horizontal_Distance_To_Roadways)) .withColumn("ele_vert", test.Elevation - test.Vertical_Distance_To_Hydrology) .withColumn("slope_hyd", pow((pow(test.Horizontal_Distance_To_Hydrology,2) + pow(test.Vertical_Distance_To_Hydrology,2)),0.5)) .withColumn("Mean_Amenities", (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways)/3) .withColumn("Mean_Fire_Hyd", (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology)/2)


# In[13]:


train_mod.limit(2).toPandas()


# In[14]:


test_mod.limit(2).toPandas()


# In[15]:


train_columns = test_mod.columns[1:]


# In[16]:


train_mod.printSchema


# In[17]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(train_columns).setOutputCol("features")
train_mod01 = assembler.transform(train_mod)


# In[18]:


train_mod01.limit(2).toPandas()


# In[19]:


train_mod02 = train_mod01.select("features","Cover_Type")


# In[20]:


test_mod01 = assembler.transform(test_mod)
test_mod02 = test_mod01.select("Id","features")


# In[21]:


from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
rfClassifer = RandomForestClassifier(labelCol = "Cover_Type", numTrees = 100)


# In[22]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = [rfClassifer])


# In[23]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[24]:


paramGrid = ParamGridBuilder()   .addGrid(rfClassifer.maxDepth, [1, 2, 4, 5, 6, 7, 8])   .addGrid(rfClassifer.minInstancesPerNode, [1, 2, 4, 5, 6, 7, 8])   .build()


# In[25]:


evaluator = MulticlassClassificationEvaluator(labelCol = "Cover_Type", predictionCol = "prediction", metricName = "accuracy") 

crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 10)


# In[26]:


cvModel = crossval.fit(train_mod02)


# In[27]:


cvModel.avgMetrics


# In[28]:


cvModel.bestModel.stages


# In[29]:


prediction = cvModel.transform(test_mod02)


# In[30]:


selected = prediction.select("Id","features", "probability", "prediction")


# In[31]:


selected.limit(5).toPandas()


# In[32]:


sub_final = selected.select(col("Id"),col("prediction").cast(IntegerType()).alias("Cover_Type"))


# In[33]:


sub_final.limit(2).toPandas()


# In[34]:




