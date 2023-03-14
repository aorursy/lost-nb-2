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
get_ipython().system('unzip ../input/sf-crime/train.csv.zip ')
#ls /kaggle/input/sf-crime
#!ls


# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark


# In[4]:


df = spark.read.csv("train.csv",header=True, inferSchema=True)
df.printSchema()


# In[5]:


from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
df = df.dropDuplicates()

df = df.withColumn('Year',year("Dates"))
df = df.withColumn('Month',month("Dates"))
df = df.withColumn('Day',dayofmonth("Dates"))
df = df.withColumn('Hour',hour("Dates"))

df = df.withColumn('X',df["X"].cast(DoubleType()))
df = df.withColumn('Y',df["Y"].cast(DoubleType()))
df = df.withColumn('X',df["X"].cast(DoubleType()))
df = df.withColumn('Y',df["Y"].cast(DoubleType()))

AvgX1 = df.groupBy("PdDistrict").agg({"X": "avg"}).withColumnRenamed("avg(X)", "X_avg")

AvgY1 = df.groupBy("PdDistrict").agg({"Y": "avg"}).withColumnRenamed("avg(Y)", "Y_avg")


df_avg_x = df.join(AvgX1,on ="PdDistrict")
df = df_avg_x.join(AvgY1,on ="PdDistrict")

df = df.withColumn("Y11", when(col("Y") > 50, col("Y_avg")).otherwise(col("Y")))
df = df.withColumn("X11", when(col("Y") > 50, col("X_avg")).otherwise(col("X")))

df = df.withColumn('SPOT', when(df.Address.like("%Block%") , lit(0)).otherwise(lit(1)))

df = df.drop("Dates","Address","Descript","Resolution")
df.printSchema()


# In[6]:


from pyspark.sql.functions import *
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, VectorSlicer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StringIndexer,OneHotEncoder, OneHotEncoderEstimator, VectorAssembler, VectorSlicer
#gathering the string fields in one object execluding the target column
encoding_var = [i[0] for i in df.dtypes if (i[1]=='string') & (i[0]!='Category')]
encoding_var

#gathering the string integer in one object execluding the target column
num_var = [i[0] for i in df.dtypes if ((i[1]=='int') | (i[1]=='double')) & (i[0]!='Category')]
num_var

string_indexes = [StringIndexer(inputCol = c, outputCol = 'IDX_' + c, handleInvalid = 'skip') for c in encoding_var]
string_indexes

onehot_indexes = [OneHotEncoderEstimator(inputCols = ['IDX_' + c], outputCols = ['OHE_' + c]) for c in encoding_var]

label_indexes = StringIndexer(inputCol = 'Category', outputCol = 'label', handleInvalid = 'skip')

label_hotcodes =  OneHotEncoder().setInputCol("label").setOutputCol("categoryOHE")

assembler = VectorAssembler(inputCols = num_var + ['OHE_' + c for c in encoding_var], outputCol = "features")

## Defining two pipelines so as to be able to transform the test dataset using pipe1 only
pipe1 = Pipeline(stages = string_indexes + onehot_indexes + [assembler])
df_transformer1 = pipe1.fit(df)
df = df_transformer1.transform(df)

## pip2 will be used for indexing labels only in training phase
pipe2 = Pipeline(stages = [label_indexes] + [label_hotcodes])
df_transformer2 = pipe2.fit(df)   ### This transformer will be used to access the indexed labels when submitting to kaggle
df_train = df_transformer2.transform(df)
df_train.printSchema()


# In[7]:


from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

df_train = df_train.select('features','label')         #creating final data with only 2 columns
#train,test=final_data.randomSplit([0.5,0.5])  

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed = 8464,                            numTrees=8, cacheNodeIds = False, subsamplingRate = 0.7,maxDepth=10, maxBins=30 )
rfModel = rf.fit(df_train)


# In[8]:


get_ipython().system('unzip ../input/sf-crime/test.csv.zip ')
# Import test data and do same steps of adding feature columns
test_df = spark.read.csv("test.csv",header=True, inferSchema=True)

#df_train_processed = pipe1.fit(df).transform(df)

test_df = test_df.withColumn('Year',year("Dates"))
test_df = test_df.withColumn('Month',month("Dates"))
test_df = test_df.withColumn('Day',dayofmonth("Dates"))
test_df = test_df.withColumn('Hour',hour("Dates"))

test_df = test_df.withColumn('X',test_df["X"].cast(DoubleType()))
test_df = test_df.withColumn('Y',test_df["Y"].cast(DoubleType()))

df_avg_x = test_df.join(AvgX1,on ="PdDistrict")
test_df = df_avg_x.join(AvgY1,on ="PdDistrict")


test_df = test_df.withColumn("Y11", when(col("Y") > 50, col("Y_avg")).otherwise(col("Y")))
test_df = test_df.withColumn("X11", when(col("Y") > 50, col("X_avg")).otherwise(col("X")))


test_df = test_df.withColumn('SPOT', when(test_df.Address.like("%Block%") , lit(0)).otherwise(lit(1)))
#test_df = test_df.drop('X','Y','X_avg','Y_avg')
test_df = test_df.drop("Dates","Address")

test_df.printSchema()


# In[9]:


df_test_processed = df_transformer1.transform(test_df)
df_test_processed.printSchema()


# In[10]:


final_predictions = rfModel.transform(df_test_processed).select("id", "probability")


# In[11]:


label_list = df_transformer2.stages[0].labels

from pyspark.sql import types as T

#Build a function to convert predictions from DenseVector to Array
def dense_to_array(dv):
    dvArray = list([float(i) for i in dv])
    return dvArray
#Create corresponding UDF
dense_to_array_udf = udf(dense_to_array, T.ArrayType(T.FloatType()))

#Use the UDF
results = final_predictions.withColumn('probability', dense_to_array_udf('probability'))

#Build the columns with target category names and corresponding probabilities
for i in range(39):
    results = results.withColumn(label_list[i], results.probability[i])

results.printSchema()


# In[12]:


results.drop("probability").toPandas().to_csv('submission.csv',index=False,header=True)


# In[13]:


submit_df = spark.read.csv("submission.csv",header=True, inferSchema=True)
submit_df.count()


# In[14]:


submit_df.printSchema()


# In[15]:


#kaggle competitions submit -c sf-crime -f ../output/submission.csv -m "Spark_Submission_Kaggle_Kernel"


# In[ ]:




