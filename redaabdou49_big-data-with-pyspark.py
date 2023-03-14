#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

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
from pyspark.ml.feature import VectorAssembler


# In[4]:


spark = SparkSession.Builder().getOrCreate()


# In[5]:


import zipfile
datasets = ["train.csv"]
for data in datasets :
    # Will unzip the files so that you can see them..
    with zipfile.ZipFile("../input/forest-cover-type-kernels-only/"+data+".zip","r") as z:
        z.extractall(".")


# In[6]:


train = spark.read.csv('train.csv',header = True,inferSchema=True)


# In[7]:


train.limit(5).toPandas()


# In[8]:


train.printSchema()


# In[9]:


train.count() 


# In[10]:


train.distinct()


# In[11]:


train.describe().toPandas()


# In[12]:


plt.figure(figsize=(12,5))
plt.title("Distribution of forest categories(Target Variable)")
ax = sns.distplot(train.toPandas()["Cover_Type"])


# In[13]:


for i in train.columns:
    sns.FacetGrid(train.toPandas(),hue="Cover_Type",height=8)        .map(sns.distplot ,i)
plt.legend()  


# In[14]:


train_mod = train.withColumn("HF1", train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Fire_Points) .withColumn("HF2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Fire_Points)) .withColumn("HR1", abs(train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)) .withColumn("HR2", abs(train.Horizontal_Distance_To_Hydrology - train.Horizontal_Distance_To_Roadways)) .withColumn("FR1", abs(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Roadways)) .withColumn("FR2", abs(train.Horizontal_Distance_To_Fire_Points - train.Horizontal_Distance_To_Roadways)) .withColumn("ele_vert", train.Elevation - train.Vertical_Distance_To_Hydrology) .withColumn("slope_hyd", pow((pow(train.Horizontal_Distance_To_Hydrology,2) + pow(train.Vertical_Distance_To_Hydrology,2)),0.5)) .withColumn("Mean_Amenities", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways)/3) .withColumn("Mean_Fire_Hyd", (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology)/2)


# In[15]:


train_mod.limit(5).toPandas()


# In[16]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(train_mod.toPandas())
for var in train_mod.columns:
    cT.TestIndependence(colX=var,colY="Cover_Type" ) 


# In[17]:


train_columns = [col for col in train_mod.columns if col not in ['Cover_Type','Id','Soil_Type7','Soil_Type8','Soil_Type15','Soil_Type25']]
assembler = VectorAssembler().setInputCols(train_columns).setOutputCol("features")
train_mod01 = assembler.transform(train_mod)


# In[18]:


train_mod02 = train_mod01.select("features","Cover_Type")
train_mod02.limit(5).toPandas()


# In[19]:


train, test= train_mod02.randomSplit([0.8, 0.2], seed=12345)


# In[20]:


from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier,GBTClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rfClassifer = RandomForestClassifier(featuresCol = 'features',labelCol = "Cover_Type",numTrees=100)
dt = DecisionTreeClassifier(labelCol="Cover_Type", featuresCol="features")

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="accuracy")


models = [rfClassifer,dt]

for model in models:
    Model = model.fit(train)
    predictions = Model.transform(test)
    print(evaluator.evaluate(predictions))


# In[21]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 


# In[22]:


paramGrid = ParamGridBuilder()   .addGrid(rfClassifer.maxDepth, [3, 4, 5, 6, 7, 8])   .addGrid(rfClassifer.maxBins, [25, 28, 31])   .addGrid(rfClassifer.impurity, ["entropy", "gini"])   .build()


# In[23]:


crossval = CrossValidator(estimator = rfClassifer,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 3)


# In[24]:


cvModel = crossval.fit(train)


# In[25]:


predictions = cvModel.transform(test)


# In[26]:


predictions.select("features", "probability", "prediction","Cover_Type").limit(5).toPandas()

evaluator.evaluate(predictions)


# In[27]:


# Compute the correlation matrix
corr = train_mod.toPandas().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[28]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(train_mod.toPandas())
for var in train_mod.columns:
    cT.TestIndependence(colX=var,colY="Elevation" )


# In[29]:


columns = [col for col in train_mod.columns if col not in ['Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type15','Soil_Type16','Soil_Type20',"Aspect",'Elevation','Id','FR2','HF2','Soil_Type34','Soil_Type28','Soil_Type26','Soil_Type25','Soil_Type21','Hillshade_3pm']]

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(columns).setOutputCol("features")

train_mod011 = assembler.transform(train_mod)

train_mod022 = train_mod011.select("features","Elevation")

train, test = train_mod022.randomSplit([0.8, 0.2])


# In[30]:


from pyspark.ml.regression import DecisionTreeRegressor,RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


dt = DecisionTreeRegressor(featuresCol="features", labelCol="Elevation")
rf = RandomForestRegressor(featuresCol="features", labelCol="Elevation")
gbt = GBTRegressor(featuresCol="features", labelCol="Elevation", maxIter=10)

evaluator = RegressionEvaluator(labelCol="Elevation", predictionCol="prediction", metricName="rmse")

models = [rf,dt,gbt]

for model in models:
    Model = model.fit(train)
    predictions = Model.transform(test)
    print(evaluator.evaluate(predictions))


# In[31]:


paramGrid = ParamGridBuilder()   .addGrid(gbt.maxDepth, [4, 5, 6, 7, 8])   .addGrid(gbt.maxBins, [25, 28, 31])   .addGrid(gbt.impurity, ["variance"])   .build()

crossval = CrossValidator(estimator = gbt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 3)


# In[32]:


cvModel = crossval.fit(train)


# In[33]:


predictions = cvModel.transform(test)


# In[34]:


predictions.select("prediction", "Elevation", "features").show(5)
evaluator.evaluate(predictions)


# In[35]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

columns = [col for col in train_mod.columns if col not in ['Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type15','Soil_Type16','Soil_Type20',"Aspect",'Elevation','Id','FR2','HF2','Soil_Type34','Soil_Type28','Soil_Type26','Soil_Type25','Soil_Type21','Hillshade_3pm']]

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(columns).setOutputCol("features")

train_mod01 = assembler.transform(train_mod)


# In[36]:


import numpy as np
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans()            .setK(k)            .setSeed(1)             .setFeaturesCol("features")            .setPredictionCol("cluster")

    model_k = kmeans.fit(train_mod01)
    cost[k] = model_k.computeCost(train_mod01)


# In[37]:


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


# In[38]:


# Trains a k-means model.
kmeans = KMeans().setK(3).setSeed(1).setFeaturesCol("features")
model= kmeans.fit(train_mod01)

# Make predictions
predictions = model.transform(train_mod01)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[39]:


predictions.select("features","prediction").limit(5).toPandas()

