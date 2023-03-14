#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark')


# In[2]:


import os

import pandas as pd
import sklearn as sk
import math
import psutil
from time import time
import calendar
import json

import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')


from pyspark.sql import SparkSession 
from pyspark.sql.functions import col,unix_timestamp,to_date,min,max,isnull,count,when
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType,TimestampType
import pyarrow.parquet as pq
get_ipython().run_line_magic('pylab', 'inline')



# In[3]:


#Initialise the Spark context
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

NumCores=4 #Kaggle offers 4 CPU cores/threads.  Change for local machine


Spark = SparkSession.builder.master(f'local[{int(NumCores)}]').appName("PBS_Kids_Spark").config("spark.executor.memory", "4g") .config("spark.driver.memory", "14g") .config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","10g").config("spark.driver.maxResultSize",0).config("spark.sql.execution.arrow.enabled",True).getOrCreate()


# In[4]:


get_ipython().run_cell_magic('time', '', '#Load data to DataFrames\n\nTrainDf=Spark.read.csv(\'../input/data-science-bowl-2019/train.csv\',header=True,quote=\'"\',escape=\'"\') #quote and escape options required to parse double quotes\nTrainlabelsDf=Spark.read.csv(\'../input/data-science-bowl-2019/train_labels.csv\',header=True,quote=\'"\',escape=\'"\')\nTestDf=Spark.read.csv(\'../input/data-science-bowl-2019/test.csv\',quote=\'"\',header=True,escape=\'"\')\n\n#Load smaller files as panda Dfs\nSpecsDf=pd.read_csv(\'../input/data-science-bowl-2019/specs.csv\')\nsample_submissionDf=pd.read_csv(\'../input/data-science-bowl-2019/sample_submission.csv\')')


# In[5]:


get_ipython().run_cell_magic('time', '', "#What is the shape of the data?\nprint(f'rows :{TrainDf.count()}, columns: {len(TrainDf.columns)}')\n#I considered using countApprox to speed up, but the required conversion to rdd slowed things down")


# In[6]:


get_ipython().run_cell_magic('time', '', 'TrainDf.createOrReplaceTempView("Train")\nkeepidDf=Spark.sql(f\'SELECT installation_id from Train WHERE type="Assessment"\').dropDuplicates()\nkeepidDf.createOrReplaceTempView("keepid")\nColumns=\',\'.join([\'Train.\'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f\'SELECT {Columns} from Train INNER JOIN keepid ON Train.installation_id=keepid.installation_id\')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter')


# In[7]:


get_ipython().run_cell_magic('time', '', '#convert timeestamp field to datetime.  \nTrainDf=TrainDf.withColumn(\'timestamp\',unix_timestamp(col(\'timestamp\'), "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'").cast("timestamp"))')


# In[8]:


get_ipython().run_cell_magic('time', '', 'NullDf=TrainDf.agg(*[count(when(isnull(c),c)).alias(c) for c in TrainDf.columns])\nNullDf.show()')


# In[9]:


TrainDf=TrainDf.na.drop()


# In[10]:


get_ipython().run_cell_magic('time', '', "print(f'rows :{TrainDf.count()}, columns: {len(TrainDf.columns)}')")


# In[11]:


print(f'rows :{keepidDf.count()}, columns: {len(keepidDf.columns)}')


# In[12]:


get_ipython().run_cell_magic('time', '', '\'\'\'\'We want to put the data in a pandas dataframe in order to do graphs etc.  \nThe most memory and time efficient method to convert from Spark to Pandas is via a parquet file save\nand read via PyArrow.  But Kaggle machines doen\'t have sufficient memory for this, so I\'ve commented that code out and used a normal Pandas dataframe load of the .csv source\n\'\'\'\n\n# TrainDf.write.mode("overwrite").save(\'trainDf.parquet\')  #Uncomment if using local machine\n# TrainPdDf=pq.read_table(\'trainDf.parquet\').to_pandas()\n\nTrainPdDf=pd.read_csv(\'../input/data-science-bowl-2019/train.csv\', parse_dates= [\'timestamp\']) #comment out if using local machine')


# In[13]:


plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
ax1 = sns.countplot(y="type", data=TrainPdDf, color="blue", order = TrainPdDf.type.value_counts().index)
plt.title("number of events by type")

ax2 = fig.add_subplot(212)
ax2 = sns.countplot(y="world", data=TrainPdDf, color="blue", order = TrainPdDf.world.value_counts().index)
plt.title("number of events by world")

plt.tight_layout(pad=0)
plt.show()


# In[14]:


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,10))
se = TrainPdDf.title.value_counts().sort_values(ascending=True)
se.plot.barh()
plt.title("Event counts by title")
plt.xticks(rotation=0)
plt.show()


# In[15]:


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,10))
se = TrainPdDf.installation_id.value_counts().sort_values(ascending=False).head(200)
se.plot.bar()
plt.title("Event counts by installation id (top 200)")
plt.show()


# In[16]:


get_ipython().run_cell_magic('time', '', 'Counts=TrainDf.groupBy(\'installation_id\').agg(count(\'installation_id\').alias(\'NumEvents\'))\nCounts.select("NumEvents").describe().show()\nprint(f\'50% and 90% quartile :{Counts.approxQuantile(["NumEvents"],[0.5,0.9],0.05)}\')  #Use approxQuantile rather than Quantile for speed')


# In[17]:


get_ipython().run_cell_magic('time', '', 'Counts.createOrReplaceTempView("Counts")\nwebbotsDf=Spark.sql(f\'SELECT * from Counts WHERE NumEvents>15000\').select(\'installation_id\')\nprint(f\'Number of suspected webbots: {webbotsDf.count()}\')\nNotWebbotsDf=Spark.sql(f\'SELECT * from Counts WHERE NumEvents<=15000\').select(\'installation_id\')\nNotWebbotsDf.registerTempTable("NotWebbots")  \nTrainDf.createOrReplaceTempView("Train")  \n\nColumns=\',\'.join([\'Train.\'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f\'SELECT {Columns} from Train \\\nINNER JOIN NotWebbots ON Train.installation_id=NotWebbots.installation_id\')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter')


# In[18]:


def get_time(df):
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df
train = get_time(TrainPdDf)


# In[19]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('date')['date'].count()
se.plot()
plt.title("Event counts by date")
plt.xticks(rotation=90)
plt.show()


# In[20]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('dayofweek')['dayofweek'].count()
se.index = list(calendar.day_abbr)
se.plot.bar()
plt.title("Event counts by day of week")
plt.xticks(rotation=0)
plt.show()


# In[21]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('hour')['hour'].count()
se.plot.bar()
plt.title("Event counts by hour of day")
plt.xticks(rotation=0)
plt.show()


# In[22]:


get_ipython().run_cell_magic('time', '', "#What is the shape of the data?\nprint(f'rows :{TestDf.count()}, columns: {len(TestDf.columns)}')")


# In[23]:


TestDf.select('installation_id').dropDuplicates().count()


# In[24]:


sample_submissionDf.shape[0]


# In[25]:


get_ipython().run_cell_magic('time', '', 'TestDf.createOrReplaceTempView("Test")\nSpark.sql(f\'SELECT Train.title from Train INNER JOIN Test ON Train.installation_id=Test.installation_id\').count()')


# In[26]:


get_ipython().run_cell_magic('time', '', '#convert timeestamp field to datetime.  \nTestDf=TestDf.withColumn(\'timestamp\',unix_timestamp(col(\'timestamp\'), "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'").\\\n                         cast(TimestampType()))')


# In[27]:


get_ipython().run_cell_magic('time', '', 'TestDates=TestDf.select(to_date(TestDf[\'timestamp\']).alias(\'date\'))\nTrainDates=TrainDf.select(to_date(TrainDf[\'timestamp\']).alias(\'date\'))\nTest_min_date, Test_max_date = TestDates.select(min("date"), max("date")).first()\nTrain_min_date, Train_max_date = TrainDates.select(min("date"), max("date")).first()\nprint(f\'The date range in train is: {Train_min_date} to {Train_max_date}\')\nprint(f\'The date range in test is: {Test_min_date} to {Test_max_date}\')')


# In[28]:


get_ipython().run_cell_magic('time', '', '\'\'\'\'We want to put the data in a pandas dataframe in order to do graphs etc.  \nThe most memory and time efficient method to convert from Spark to Pandas is via a parquet file save\n\'\'\'\n\n# TrainlabelsDf.write.mode("overwrite").save(\'TrainlabelsDf.parquet\')  #uncomment if using local machine\n# TrainlabelsPdDf=pq.read_table(\'TrainlabelsDf.parquet\').to_pandas()\n\nTrainlabelsPdDf=pd.read_csv(\'../input/data-science-bowl-2019/train_labels.csv\') #comment out if using local machine')


# In[29]:


plt.rcParams.update({'font.size': 22})

plt.figure(figsize=(12,6))
sns.countplot(y="title", data=TrainlabelsPdDf, color="blue", order = TrainlabelsPdDf.title.value_counts().index)
plt.title("Counts of titles")
plt.show()


# In[30]:


plt.rcParams.update({'font.size': 16})

se = TrainlabelsPdDf.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')
se.plot.bar(stacked=True, rot=0, figsize=(12,10))
plt.title("Counts of accuracy group")
plt.show()


# In[31]:


TrainlabelsPdDf[TrainlabelsPdDf.installation_id == "0006a69f"]


# In[32]:


get_ipython().run_cell_magic('time', '', 'Spark.sql(f\'SELECT {Columns} from Train WHERE event_code = 4100 AND installation_id = "0006a69f"\\\nAND title == "Bird Measurer (Assessment)"\').toPandas()')


# In[33]:


get_ipython().run_cell_magic('time', '', 'TrainlabelsDf.createOrReplaceTempView("Trainlabel")\nUniqueTrainlabelsDf=Spark.sql(f\'SELECT installation_id from Trainlabel\').dropDuplicates()\nUniqueTrainlabelsDf.createOrReplaceTempView("UnqTrainlabel")\nkeepidDf.createOrReplaceTempView("keepid")  #we created a list of unique train ids earlier\nSpark.sql(f\'SELECT keepid.installation_id as one, UnqTrainlabel.installation_id as two FROM keepid \\\nLEFT JOIN UnqTrainlabel ON keepid.installation_id=UnqTrainlabel.installation_id \\\nWHERE UnqTrainlabel.installation_id IS NULL\').count()')


# In[34]:


get_ipython().run_cell_magic('time', '', "Columns=','.join(['Train.'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f'SELECT {Columns} from Train \\\nINNER JOIN UnqTrainlabel ON Train.installation_id=UnqTrainlabel.installation_id')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter")


# In[35]:


Count1=TrainlabelsDf.count()
Count2=TrainlabelsDf.select('game_session').dropDuplicates().count()
print(f'Number of rows in train_labels: {Count1}')
print(f'Number of unique game_sessions in train_labels: {Count2}')


# In[36]:


#Uncomment these if running on local machine.  Don't need to save in Kaggle, will load data into subsequent notebooks
# TrainDf.write.mode("overwrite").save('TrainDf.parquet')
# TestDf.write.mode("overwrite").save('TestDf.parquet')
# TrainlabelsDf.write.mode("overwrite").save('TrainlabelsDf.parquet')


# In[ ]:




