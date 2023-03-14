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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install pyspark')


# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# In[4]:


#sc = SparkContext(appName = "forest_cover")


# In[5]:


spark = SparkSession.Builder().getOrCreate()


# In[6]:


import zipfile
datasets = ["train.csv"]
for data in datasets :
    # Will unzip the files so that you can see them..
    with zipfile.ZipFile("../input/forest-cover-type-kernels-only/"+data+".zip","r") as z:
        z.extractall(".")


# In[7]:


train = spark.read.csv('train.csv',header = True,inferSchema=True)


# In[8]:


train.limit(5).toPandas()


# In[9]:


train_mod = train.withColumn("HF1", train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points) .withColumn("HF2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)) .withColumn("HR1", abs(train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)) .withColumn("HR2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)) .withColumn("FR1", abs(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)) .withColumn("FR2", abs(train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)) .withColumn("ele_vert", train.Elevation - train.Vertical_Distance_To_Hydrology) .withColumn("slope_hyd", pow((pow(train.Horizontal_Distance_To_Hydrology,2) + pow(train.Vertical_Distance_To_Hydrology,2)),0.5)) .withColumn("Mean_Amenities", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)/3) .withColumn("Mean_Fire_Hyd", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology)/2)


# In[10]:


train_mod.limit(2).toPandas()


# In[11]:


train_columns = [col for col in train_mod.columns if col not in ['Cover_Type','Id']]


# In[12]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(train_columns).setOutputCol("features")
train_mod01 = assembler.transform(train_mod)


# In[13]:


train_mod01.limit(2).toPandas()


# In[14]:


train_mod02 = train_mod01.select("features","Cover_Type")


# In[15]:


from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
rfClassifer = RandomForestClassifier(featuresCol = 'features',labelCol = "Cover_Type")


# In[16]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 


# In[17]:


paramGrid = ParamGridBuilder()   .addGrid(rfClassifer.numTrees, [100, 200, 300])   .addGrid(rfClassifer.maxDepth, [1, 2, 3, 4, 5, 6, 7, 8])   .addGrid(rfClassifer.maxBins, [25, 28, 31])   .addGrid(rfClassifer.impurity, ["entropy", "gini"])   .build()


# In[18]:


evaluator = MulticlassClassificationEvaluator(labelCol = "Cover_Type", predictionCol = "prediction", metricName = "accuracy") 

crossval = CrossValidator(estimator = rfClassifer,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 5)


# In[19]:


train_rf, test_rf = train_mod02.randomSplit([0.8, 0.2], seed=12345)


# In[20]:


cvModel = crossval.fit(train_rf)


# In[21]:


predictions = cvModel.transform(test_rf)


# In[22]:


predictions.select("features", "probability", "prediction","Cover_Type").limit(5).toPandas()


# In[23]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="accuracy")

evaluator.evaluate(predictions)


# In[24]:


from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

regression_columns = [col for col in train_mod.columns if col not in ["Cover_Type",'Elevation','Id']]

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(regression_columns).setOutputCol("features")

train_mod011 = assembler.transform(train_mod)

train_mod022 = train_mod011.select("features","Elevation")

# Split the data into training and test sets (30% held out for testing)
(trainingData1, testData1) = train_mod022.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="features", labelCol="Elevation")

# Train model.  This also runs the indexer.
model1 = dt.fit(trainingData1)

# Make predictions.
predictions1 = model1.transform(testData1)

# Select example rows to display.
predictions1.select("prediction", "Elevation", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="Elevation", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions1)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[25]:


paramGrid1 = ParamGridBuilder()   .addGrid(dt.maxDepth, [1, 2, 3, 4, 5, 6, 7, 8])   .addGrid(dt.maxBins, [25, 28, 31])   .addGrid(dt.impurity, ["variance"])   .build()

evaluator1 = RegressionEvaluator(labelCol="Elevation", predictionCol="prediction", metricName="rmse")

crossval1 = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid1,
                          evaluator = evaluator1,
                          numFolds = 5)


# In[26]:


cvModel1 = crossval1.fit(trainingData1)


# In[27]:


predictions10 = cvModel1.transform(testData1)


# In[28]:


# Select example rows to display.
predictions1.select("prediction", "Elevation", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="Elevation", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions1)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[29]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

clus_columns = [col for col in train_mod.columns if col not in ["Cover_Type",'Id']]

from pyspark.ml.feature import VectorAssembler
assembler2 = VectorAssembler().setInputCols(clus_columns).setOutputCol("features")

train_mod0111 = assembler2.transform(train_mod)


# In[30]:


import numpy as np
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans()            .setK(k)            .setSeed(1)             .setFeaturesCol("features")            .setPredictionCol("cluster")

    model_k = kmeans.fit(train_mod0111)
    cost[k] = model_k.computeCost(train_mod0111)


# In[31]:


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


# In[32]:


# Trains a k-means model.
kmeans1 = KMeans().setK(3).setSeed(1).setFeaturesCol("features")
model2 = kmeans1.fit(train_mod0111)

# Make predictions
predictions3 = model2.transform(train_mod0111)

# Evaluate clustering by computing Silhouette score
evaluator3 = ClusteringEvaluator()

silhouette = evaluator3.evaluate(predictions3)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model2.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[33]:


predictions3.select("features","prediction").limit(5).toPandas()

