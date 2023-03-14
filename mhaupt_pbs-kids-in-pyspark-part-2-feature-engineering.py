#!/usr/bin/env python
# coding: utf-8

# In[1]:


#you will need to install pyspark as it isn't part of the standard kaggle environment.  Make sure you set internet on for this workbook
get_ipython().system('pip install pyspark')
get_ipython().system('pip install spark_sklearn')


# In[2]:


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
import psutil
        
# Any results you write to the current directory are saved as output.
from scipy.stats import skew,norm
from scipy import stats


from pyspark.sql import SparkSession, Window 
from pyspark.sql.functions import col,when,unix_timestamp,to_date,min,max,isnull,count,concat_ws,lit,sum,instr,datediff
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType,TimestampType,BooleanType,LongType,FloatType,DoubleType,ArrayType
from pyspark.ml.feature import OneHotEncoder, StringIndexer,OneHotEncoderEstimator,VectorAssembler,MinMaxScaler,PCA
from pyspark.sql.functions import udf,pandas_udf, PandasUDFType,to_date

from pyspark.ml import Pipeline
import json
import pyarrow as pa
import pyarrow.parquet as pq
import spark_sklearn
from collections import Counter
from scipy import stats

import pyspark.sql.functions as F



pd.set_option('display.max_columns', 1000)
pd.option_context('mode.use_inf_as_na', True)

get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


#Initialise the Spark context
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

#NumCores=psutil.cpu_count(logical=False) #Not necessary to manually set number of workers, just for clarity

Spark = SparkSession.builder.master('local[4]').appName("PBS_Kids_Spark").config("spark.executor.memory", "4g") .config("spark.driver.memory", "14g") .config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","10g").config("spark.driver.maxResultSize",0).config("spark.sql.execution.arrow.enabled",True).getOrCreate()


# In[4]:


get_ipython().run_cell_magic('time', '', '#Load data to DataFrames\n\nTrainDf=Spark.read.csv(\'../input/data-science-bowl-2019/train.csv\',header=True,quote=\'"\',escape=\'"\') #quote and escape options required to parse double quotes\nTrainlabelsDf=Spark.read.csv(\'../input/data-science-bowl-2019/train_labels.csv\',header=True,quote=\'"\',escape=\'"\')\nTestDf=Spark.read.csv(\'../input/data-science-bowl-2019/test.csv\',quote=\'"\',header=True,escape=\'"\')\n\n#Load smaller files as panda Dfs\nSpecsDf=pd.read_csv(\'../input/data-science-bowl-2019/specs.csv\')\nsample_submissionDf=pd.read_csv(\'../input/data-science-bowl-2019/sample_submission.csv\')')


# In[5]:


get_ipython().run_cell_magic('time', '', '#getting rid of the installation_ids that never took an assessment.  We saw these in Part 1\nTrainDf.createOrReplaceTempView("Train")\nkeepidDf=Spark.sql(f\'SELECT installation_id from Train WHERE type="Assessment"\').dropDuplicates()\nkeepidDf.createOrReplaceTempView("keepid")\nColumns=\',\'.join([\'Train.\'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f\'SELECT {Columns} from Train INNER JOIN keepid ON Train.installation_id=keepid.installation_id\')')


# In[6]:


#drop rows wih na
TrainDf=TrainDf.na.drop()


# In[7]:


''' I've limited the size of the dataframe as kaggle machines don't really have enough HDD storage (<5Gb) to support Spark as a head node.  
Especially when I use Pyarrow to save the feature dataframe to parquet.  Hopefully you have access to a more powerful PC and can remove this '''
TrainDf=TrainDf.limit(1000000)


# In[8]:


get_ipython().run_cell_magic('time', '', '#Add indentifying column to Test and Train dfs\nTestDf=TestDf.withColumn(\'TestOrTrain\',lit("Test"))\nTrainDf=TrainDf.withColumn(\'TestOrTrain\',lit("Train"))')


# In[9]:


get_ipython().run_cell_magic('time', '', '#identify test records\nTestRecordsDf=TestDf.groupBy(\'installation_id\').agg(F.last(\'timestamp\').alias(\'timestamp\'))\nTestRecordsDf=TestRecordsDf.withColumn(\'TestFlag\',lit(1))\nTrainDf=TrainDf.withColumn(\'TestFlag\',lit(0))\n\nTestDf.createOrReplaceTempView("Test")\nTestRecordsDf.createOrReplaceTempView("TestRecords")\n\nTestDf=Spark.sql(f\'SELECT *, \\\nTest.installation_id as id1,Test.timestamp as ts1,\\\nTestRecords.installation_id,TestRecords.timestamp \\\nfrom Test LEFT JOIN TestRecords \\\nON Test.installation_id=TestRecords.installation_id \\\nAND Test.timestamp=TestRecords.timestamp\')\\\n.drop(\'installation_id\',\'timestamp\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'ts1\',\'timestamp\')')


# In[10]:


get_ipython().run_cell_magic('time', '', '#stack the test and train dataframes for combined operations\nTestDf=TestDf.select(TrainDf.columns) #ensur esame column order\nCombinedDf=TrainDf.union(TestDf).repartition(3) \n\n#concatenate the events and codes\nCombinedDf = CombinedDf.withColumn(\'title_event_code\',concat_ws(\'_\',CombinedDf.title,CombinedDf.event_code))\n\n#concatenate the world and event type\nCombinedDf = CombinedDf.withColumn(\'world_type\',concat_ws(\'_\',CombinedDf.world,CombinedDf.type))\n\n#String encode a number of fields via a pipeline\n# ColumnsToEncode=[\'title\',\'world\',\'type\',\'event_code\',\'event_id\']\n# indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid=\'skip\')\\\n#              for column in ColumnsToEncode ]\n# EncodePipeline = Pipeline(stages=indexers)\n# CombinedDf = EncodePipeline.fit(CombinedDf).transform(CombinedDf)\n\n\n#Flag the assessment tasks\nCombinedDf=CombinedDf.withColumn(\'win_code\',when(\\\n                     ((col(\'event_code\')==\'4100\')\\\n                      & (F.instr(CombinedDf[\'title\'],\'(Assessment)\')>0)\\\n                      &(col(\'title\')!=\'Bird Measurer (Assessment)\')   )\\\n                      |\n                     ((col(\'event_code\')==\'4110\') & (col(\'title\')==\'Bird Measurer (Assessment)\'))\\\n                      |\n                      (col(\'TestFlag\')==1)                           \n                    ,1).otherwise(0))\n\n#For assessment tasks, indicate if pass or fail\nCombinedDf=CombinedDf.withColumn(\'true_attempts\',when(\\\n                     (col(\'win_code\')==1)&(F.instr(CombinedDf[\'event_data\'],\'true\')>0)\\\n                    ,1).otherwise(0))\nCombinedDf=CombinedDf.withColumn(\'false_attempts\',when(\\\n                     (col(\'win_code\')==1)&(F.instr(CombinedDf[\'event_data\'],\'false\')>0)\\\n                    ,1).otherwise(0))\n\n#convert timestamp from string\nCombinedDf=CombinedDf.withColumn(\'timestamp\',unix_timestamp(col(\'timestamp\')\\\n                                                             , "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'").cast("timestamp"))\n\n#Flag if session type changes for each installation\nwindowval = Window.partitionBy(\'installation_id\').orderBy(\'timestamp\')\nCombinedDf=CombinedDf.withColumn("ChangeSession", when(\\\n                                                       (col(\'type\')!=F.lag(col(\'type\'), 1, 0).over(windowval)),True\n                                                      )\\\n                                                    .otherwise(False))\n\n#Show chaged session type\nCombinedDf=CombinedDf.withColumn(\'typeChange\',when(\\\n                     (col(\'ChangeSession\')==True),col(\'type\')).otherwise(\'Unchanged\'))\n\n#Ensure key variables are the proper type\nCombinedDf=CombinedDf.withColumn(\'game_time\',col(\'game_time\').cast(LongType()))\n#Cache as we\'ll be using CombinedDf a lot\nCombinedDf.cache() \n\n\nCombinedDf.createOrReplaceTempView("Combined")\n\n\n#create frame of the 4 assessment titles\nlist_of_assess_titlesDf=Spark.sql(f\'SELECT title from Combined WHERE type="Assessment"\').dropDuplicates()')


# In[11]:


get_ipython().run_cell_magic('time', '', '#get game data\n\n@udf(\'int\')\ndef json_attribute(data,attribute=\'misses\'):\n    try:\n        result =json.loads(data)[attribute]\n    except:\n        result =-1\n    \n    return result\n\n@udf(\'int\')\ndef json_conditional_attribute(data,attribute=\'round\'):\n    try:\n        result =json.loads(data)[attribute]\n    except:\n        result =-1\n    \n    return result\n\n\ngameDf=CombinedDf.where(col(\'type\')==\'Game\').where(col(\'event_code\')==\'2030\')\\\n    .select(\'installation_id\',\'timestamp\',\'event_data\')\ngameDf=gameDf.withColumn(\'misses_cnt\',json_attribute(col(\'event_data\'),lit(\'misses\')))\ngameDf=gameDf.withColumn(\'game_round\',json_conditional_attribute(col(\'event_data\'),lit(\'round\')))\ngameDf=gameDf.withColumn(\'game_level\',json_conditional_attribute(col(\'event_data\'),lit(\'level\')))\ngameDf=gameDf.drop(\'event_data\')\n\nCombinedDf.createOrReplaceTempView("Combined")\ngameDf.createOrReplaceTempView("game")\n\nCombinedDf=Spark.sql(f\'SELECT *, \\\nCombined.installation_id as id1,Combined.timestamp as ts1,\\\ngame.installation_id,game.timestamp \\\nfrom Combined LEFT JOIN game \\\nON Combined.installation_id=game.installation_id \\\nAND Combined.timestamp=game.timestamp\')\\\n.drop(\'installation_id\',\'timestamp\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'ts1\',\'timestamp\')\n\n\nCombinedDf=CombinedDf.fillna(0, subset=[\'misses_cnt\'])\nCombinedDf=CombinedDf.fillna(-1, subset=[\'game_round\',\'game_level\'])')


# In[12]:


get_ipython().run_cell_magic('time', '', '#Record number of Title-event code pair (essentially one-hot encoding)\nAllSessionTimingTEDf=CombinedDf.groupBy("installation_id","game_session").pivot("title_event_code").count().na.fill(0)\n\n#Record number of type-world pair\nAllSessionTimingTWDf=CombinedDf.groupBy("installation_id","game_session").pivot("world_type").count().na.fill(0)\n\n# #Record  number of  event codes\nAllSessionTimingECDf=CombinedDf.groupBy("installation_id","game_session").pivot("event_code").count().na.fill(0)\n\n# #Record  number of  titles\nAllSessionTimingTtlDf=CombinedDf.groupBy("installation_id","game_session").pivot("title").count().na.fill(0)\n\n#Record  number of  event id\nAllSessionTimingEIDf=CombinedDf.groupBy("installation_id","game_session").pivot("event_id").count().na.fill(0)\n\n#Record  number of  world\nAllSessionTimingWDf=CombinedDf.groupBy("installation_id","game_session").pivot("world").count().na.fill(0)\n\n#Record  number of  types\nAllSessionTimingTypDf=CombinedDf.groupBy("installation_id","game_session").pivot("type").count().na.fill(0)\n\n#Record  number of  type changes\nAllSessionTimingTypCDf=CombinedDf.groupBy("installation_id","game_session").pivot("typeChange").count().na.fill(0)\n\nColumnNames= [\'Activity\', \'Assessment\', \'Clip\', \'Game\']  #rename to avoid confustion with AllSessionTimingTypDf\nNewColumnNames={\'Activity\':\'ActivityC\', \'Assessment\':\'AssessmentC\', \'Clip\':\'ClipC\', \'Game\':\'GameC\'}\nfor Column in ColumnNames:\n    AllSessionTimingTypCDf=AllSessionTimingTypCDf.withColumnRenamed(Column,NewColumnNames[Column])')


# In[13]:


get_ipython().run_cell_magic('time', '', '#Record all assessment attempts and results by world\nAssess_Titles=list_of_assess_titlesDf.toPandas()[[\'title\']].values.tolist()\nAssess_Titles=[x[0] for x in Assess_Titles]\n\nAllAssessmentDf=CombinedDf.select("installation_id","game_session","title",\\\n                                "true_attempts","false_attempts","timestamp","win_code")\nfor Assess_Title in Assess_Titles:\n    AllAssessmentDf=AllAssessmentDf.withColumn(Assess_Title+\'True\',\\\n                                    when(col(\'title\')==Assess_Title,col(\'true_attempts\'))\\\n                                                 .otherwise(0).cast(IntegerType()))\n    AllAssessmentDf=AllAssessmentDf.withColumn(Assess_Title+\'False\',\\\n                                    when(col(\'title\')==Assess_Title,col(\'false_attempts\'))\\\n                                                 .otherwise(0).cast(IntegerType()))\n    AllAssessmentDf=AllAssessmentDf.withColumn(Assess_Title+\'AllAttemps\',\\\n                                    when(col(\'title\')==Assess_Title,col(\'true_attempts\')+col(\'false_attempts\'))\\\n                                                 .otherwise(0).cast(IntegerType()))')


# In[14]:


get_ipython().run_cell_magic('time', '', '#First get some miscellaeous session informantion\n\nMiscSessionInfoDf=CombinedDf.groupBy("installation_id","game_session")\\\n.agg(count(\'event_id\').alias(\'NumEvents\')\\\n    ,F.sum(\'win_code\').alias(\'NumAssessmentAttempts\')\n    ,min(\'timestamp\').alias(\'StartTime\')\\\n    ,max(\'timestamp\').alias(\'EndTime\')\\\n#     ,F.first(col("title_index")).alias(\'session_title\')\\\n    ,F.first(col("title")).alias(\'r_session_title\')\\\n    ,F.first(col("TestFlag")).alias(\'TestFlag\')\\\n    ,F.first(col("TestOrTrain")).alias(\'TestOrTrain\')\\\n     ,F.first(col("game_round")).alias(\'game_round\')\\\n     ,F.first(col("game_level")).alias(\'game_level\')\\\n     ,F.first(col("misses_cnt")).alias(\'misses_cnt\')\\\n     ,F.first(col("type")).alias(\'Type\')\\\n     ,F.first(col("World")).alias(\'World\')\\\n    ,(F.unix_timestamp(F.max(\'timestamp\'))-F.unix_timestamp(F.min(\'timestamp\'))).alias(\'SessionDuration\'))\nMiscSessionInfoDf=MiscSessionInfoDf.withColumn(\'hour\',F.hour(col(\'StartTime\')))')


# In[15]:


get_ipython().run_cell_magic('time', '', "#insert clip durations\nclip_time = {'Welcome to Lost Lagoon!':19,'Tree Top City - Level 1':17,'Ordering Spheres':61, 'Costume Box':61,\n        '12 Monkeys':109,'Tree Top City - Level 2':25, 'Pirate\\'s Tale':80, 'Treasure Map':156,'Tree Top City - Level 3':26,\n        'Rulers':126, 'Magma Peak - Level 1':20, 'Slop Problem':60, 'Magma Peak - Level 2':22, 'Crystal Caves - Level 1':18,\n        'Balancing Act':72, 'Lifting Heavy Things':118,'Crystal Caves - Level 2':24, 'Honey Cake':142, 'Crystal Caves - Level 3':19,\n        'Heavy, Heavier, Heaviest':61}\n\n@udf('int')\ndef insert_clip_duration(Type,session_title,SessionDuration):\n    if Type=='Clip':\n        return clip_time[session_title]\n    else:\n        return SessionDuration\n    \n\nMiscSessionInfoDf=MiscSessionInfoDf.withColumn('SessionDuration',\\\n                                    insert_clip_duration(col('Type'),col('r_session_title'),col('SessionDuration'))\\\n                                               .cast(IntegerType()))")


# In[16]:


get_ipython().run_cell_magic('time', '', '#sum occurences for each title-event code pair across each session \nSessionTimingTEDf=AllSessionTimingTEDf.groupby("installation_id","game_session").sum()\n#sum occurences for each world-type pair across each session \nSessionTimingTWDf=AllSessionTimingTWDf.groupby("installation_id","game_session").sum()\n#Same for title\nSessionTimingTtlDf=AllSessionTimingTtlDf.groupby("installation_id","game_session").sum()\n#Same for event id\nSessionTimingEIDf=AllSessionTimingEIDf.groupby("installation_id","game_session").sum()\n#Same for world\nSessionTimingWDf=AllSessionTimingWDf.groupby("installation_id","game_session").sum()\n#Same for type\nSessionTimingTypDf=AllSessionTimingTypDf.groupby("installation_id","game_session").sum()\n#Same for type change\nSessionTimingTypCDf=AllSessionTimingTypCDf.groupby("installation_id","game_session").sum()\n#Same for event code\nSessionTimingECDf=AllSessionTimingECDf.groupby("installation_id","game_session").sum()\n\n#Join the 7 occurence dataframes\nSessionTimingTEDf.createOrReplaceTempView("SessionTimingTE")\nSessionTimingTWDf.createOrReplaceTempView("SessionTimingTW")\nSessionTimingTtlDf.createOrReplaceTempView("SessionTimingTtl")\nSessionTimingEIDf.createOrReplaceTempView("SessionTimingEI")\nSessionTimingWDf.createOrReplaceTempView("SessionTimingW")\nSessionTimingTypDf.createOrReplaceTempView("SessionTimingTyp")\nSessionTimingTypCDf.createOrReplaceTempView("SessionTimingTypC")\nSessionTimingECDf.createOrReplaceTempView("SessionTimingEC")\nMiscSessionInfoDf.createOrReplaceTempView("MiscSessionInfo")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTimingTE.installation_id as id1,SessionTimingTE.game_session as gs1,\\\nSessionTimingTtl.installation_id,SessionTimingTtl.game_session \\\nfrom SessionTimingTE INNER JOIN SessionTimingTtl \\\nON SessionTimingTE.installation_id=SessionTimingTtl.installation_id \\\nAND SessionTimingTE.game_session=SessionTimingTtl.game_session\')\\\n.drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingTW.installation_id,SessionTimingTW.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingTW \\\nON SessionTiming.installation_id=SessionTimingTW.installation_id \\\nAND SessionTiming.game_session=SessionTimingTW.game_session\')\\\n.drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingEI.installation_id,SessionTimingEI.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingEI \\\nON SessionTiming.installation_id=SessionTimingEI.installation_id \\\nAND SessionTiming.game_session=SessionTimingEI.game_session\')\\\n.drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingW.installation_id,SessionTimingW.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingW \\\nON SessionTiming.installation_id=SessionTimingW.installation_id \\\nAND SessionTiming.game_session=SessionTimingW.game_session\')\\\n.drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingTyp.installation_id,SessionTimingTyp.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingTyp \\\nON SessionTiming.installation_id=SessionTimingTyp.installation_id \\\nAND SessionTiming.game_session=SessionTimingTyp.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingTypC.installation_id,SessionTimingTypC.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingTypC \\\nON SessionTiming.installation_id=SessionTimingTypC.installation_id \\\nAND SessionTiming.game_session=SessionTimingTypC.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nSessionTimingEC.installation_id,SessionTimingEC.game_session \\\nfrom SessionTiming INNER JOIN SessionTimingEC \\\nON SessionTiming.installation_id=SessionTimingEC.installation_id \\\nAND SessionTiming.game_session=SessionTimingEC.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")\n\nSessionTimingDf=Spark.sql(f\'SELECT *, \\\nSessionTiming.installation_id as id1,SessionTiming.game_session as gs1,\\\nMiscSessionInfo.installation_id,MiscSessionInfo.game_session \\\nfrom SessionTiming INNER JOIN MiscSessionInfo \\\nON SessionTiming.installation_id=MiscSessionInfo.installation_id \\\nAND SessionTiming.game_session=MiscSessionInfo.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionTimingDf.createOrReplaceTempView("SessionTiming")')


# In[17]:


get_ipython().run_cell_magic('time', '', 'windowval = Window.partitionBy(\'installation_id\').orderBy(\'StartTime\')\n\n# get cumulative lag timings\nColumnsToSum=[\'SessionDuration\']+SessionTimingDf.columns[:-14]+[\'misses_cnt\']#Don\'t include key columns in summations\nlag_summed_cols = [F.sum(F.lag(col(Column), 1).over(windowval)).over(windowval).alias(Column+\'LagCum\') \\\n                     for Column in ColumnsToSum]#lag one session to avoid double counting cumulative and current session\n\nsession_title_cols=list(SessionTimingTtlDf.columns)[2:]\nmissing_cols = [x for x in SessionTimingDf.columns if x not in ColumnsToSum+session_title_cols]\\\n                        +[\'SessionDuration\',\'misses_cnt\']\n\n\nSessionTimingCumDf=SessionTimingDf.select(missing_cols+lag_summed_cols+session_title_cols).na.fill(0)\n\n#Get the start time of previous session:\nSessionTimingCumDf=SessionTimingCumDf\\\n.withColumn(\'PreviousSessStart\',F.lag(col(\'StartTime\'), 1).over(windowval))\n\n# get time since last session\nSessionTimingCumDf=SessionTimingCumDf\\\n.withColumn(\'TimeSinceLastSess\'\\\n           ,F.unix_timestamp(col(\'StartTime\'))-F.unix_timestamp(col(\'PreviousSessStart\'))).na.fill(100000)\n\n# get cumulative count of sessions since last session\nSessionTimingCumDf=SessionTimingCumDf\\\n.withColumn("NumSessionsLagCum", F.count(col(\'game_session\')).over(windowval))\n\n\n\n#get rolling average duration\nSessionTimingCumDf=SessionTimingCumDf.withColumn(\'duration_lag_mean\',\\\n                                    (col(\'SessionDurationLagCum\')/col(\'NumSessionsLagCum\')))\n\n#get rolling average #events\nSessionTimingCumDf=SessionTimingCumDf.withColumn(\'numevents_lag_mean\',\\\n                                    (col(\'NumEventsLagCum\')/col(\'NumSessionsLagCum\')))')


# In[18]:


get_ipython().run_cell_magic('time', '', '#get some data on assessment sessions only\nAssessmentsDf=SessionTimingCumDf.filter((col(\'NumAssessmentAttempts\')>0))\nAssessmentsDf=AssessmentsDf.withColumn(\'AssessmentDurationLag\',F.lag(col(\'SessionDuration\'), 1).over(windowval))\\\n            .na.fill(-1)\n\nAssessmentsDf=AssessmentsDf.select(\'installation_id\',\'game_session\',\'AssessmentDurationLag\')\n\n#join the dataframes\nSessionTimingCumDf.createOrReplaceTempView("SessionTimingCum")\nAssessmentsDf.createOrReplaceTempView("Assessments")\n\nSessionTimingCumDf=Spark.sql(f\'SELECT *, \\\nSessionTimingCum.installation_id as id1,SessionTimingCum.game_session as gs1,\\\nAssessments.installation_id,Assessments.game_session \\\nfrom SessionTimingCum LEFT JOIN Assessments \\\nON SessionTimingCum.installation_id=Assessments.installation_id \\\nAND SessionTimingCum.game_session=Assessments.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')')


# In[19]:


get_ipython().run_cell_magic('time', '', "#sum duration by type\nTypes=['Activity','Assessment','Clip','Game']\nfor Type in Types:\n    SessionTimingCumDf=SessionTimingCumDf.withColumn(Type+'Dur',\\\n                                when(col('Type')==Type,col('SessionDuration'))\\\n                                             .otherwise(0).cast(IntegerType()))\n#Get cum lag duration    \nColumnsToSum=[Type+'Dur' for Type in Types]#Don't include key columns in summations\nlag_summed_cols = [F.sum(F.lag(col(Column), 1).over(windowval)).over(windowval).alias(Column+'LagCum') \\\n                     for Column in ColumnsToSum]\n\n#get lag cum mean\nlag_mean_cols = [F.avg(F.lag(col(Column), 1).over(windowval)).over(windowval).alias(Column+'LagMean') \\\n                     for Column in ColumnsToSum]\n\n#get lag stddev\nlag_std_cols = [F.stddev(F.lag(col(Column), 1).over(windowval)).over(windowval).alias(Column+'LagStd') \\\n                     for Column in ColumnsToSum]\n\n#get lag max\nlag_max_cols = [F.max(F.lag(col(Column), 1).over(windowval)).over(windowval).alias(Column+'Lagmax') \\\n                     for Column in ColumnsToSum]\n\nSessionTimingCumDf=SessionTimingCumDf.select(SessionTimingCumDf.columns\\\n            +lag_summed_cols+lag_mean_cols+lag_std_cols+lag_max_cols).na.fill(0)    ")


# In[20]:


get_ipython().run_cell_magic('time', '', "#get game data\nSessionTimingCumDf=SessionTimingCumDf.withColumn('game_missMeanLag',F.avg(F.lag(col('misses_cnt'), 1)\\\n                                    .over(windowval)).over(windowval)).na.fill(0)   \nSessionTimingCumDf=SessionTimingCumDf.withColumn('game_missStdLag',F.stddev(F.lag(col('misses_cnt'), 1)\\\n                                    .over(windowval)).over(windowval)).na.fill(0)    \n\nColumnsToLag=['game_round', 'game_level','misses_cnt']\nlag_cols = [F.lag(col(Column), 1).over(windowval).alias(Column+'Lag') \\\n                     for Column in ColumnsToLag]\nSessionTimingCumDf=SessionTimingCumDf.select(SessionTimingCumDf.columns+lag_cols)\nLaggedCols=[i+'Lag' for i in ColumnsToLag]\nSessionTimingCumDf=SessionTimingCumDf.fillna(-1, subset=LaggedCols)")


# In[21]:


get_ipython().run_cell_magic('time', '', '#do the same for assessment data\nAssessmentDf=AllAssessmentDf.groupby("installation_id","game_session").sum()\nMiscAssessmentInfoDf=MiscSessionInfoDf.select("installation_id","game_session"\\\n                                              ,\'StartTime\',\'NumAssessmentAttempts\')\n\nAssessmentDf.createOrReplaceTempView("Assessment")\nMiscAssessmentInfoDf.createOrReplaceTempView("MiscAssessmentInfo")\nSessionAssessmentDf=Spark.sql(f\'SELECT *, \\\nAssessment.installation_id as id1,Assessment.game_session as gs1,\\\nMiscAssessmentInfo.installation_id as aid,MiscAssessmentInfo.game_session as agame_session \\\nfrom Assessment INNER JOIN MiscAssessmentInfo \\\nON Assessment.installation_id=MiscAssessmentInfo.installation_id \\\nAND Assessment.game_session=MiscAssessmentInfo.game_session\')\\\n.drop(\'installation_id\',\'game_session\',\'agame_session\',\'aid\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\nSessionAssessmentDf.createOrReplaceTempView("SessionAssessment")')


# In[22]:


get_ipython().run_cell_magic('time', '', "#get lagged assessments\nColumnsToAdd=list(SessionAssessmentDf.columns)[:-4]+['NumAssessmentAttempts']#Don't include key columns in summation\nlag_summed_cols = [F.sum(F.lag(col(Column), 1, 0).over(windowval)).over(windowval).alias(Column+'LagCum') \\\n                     for Column in ColumnsToAdd]#lag one session to avoid double counting cumulative and current session\nmissing_cols = [i for i in SessionAssessmentDf.columns if i not in ColumnsToAdd]\n\n\nSessionAccuracyDf=SessionAssessmentDf.select(missing_cols+ColumnsToAdd+lag_summed_cols).na.fill(-1)\n\n@pandas_udf('int', PandasUDFType.SCALAR)\ndef Count_Trues(Unq,World):\n    return Unq+World")


# In[23]:


get_ipython().run_cell_magic('time', '', "#Get time since last assessment\n#Get the start time of previous session:\nSessionAccuracyDf=SessionAccuracyDf\\\n.withColumn('PreviousSessStart',F.lag(col('StartTime'), 1).over(windowval))\n\n# get time since last session\nSessionAccuracyDf=SessionAccuracyDf\\\n.withColumn('TimeSinceLastAssess'\\\n           ,F.unix_timestamp(col('StartTime'))-F.unix_timestamp(col('PreviousSessStart'))).na.fill(100000)")


# In[24]:


get_ipython().run_cell_magic('time', '', '#Get current accuracy\nfor Assess_Title in Assess_Titles:\n    SessionAccuracyDf=SessionAccuracyDf.withColumn(Assess_Title+"Accy",\\\n                                   (col(f\'sum({Assess_Title}True)\')\\\n                                     /col(f\'sum({Assess_Title}AllAttemps)\'))).na.fill(-1)\n    \nSessionAccuracyDf=SessionAccuracyDf.withColumn(\'AllAssessmentAccy\',\\\n                                   (col(f\'sum(true_attempts)\')\\\n                                     /(col(f\'sum(true_attempts)\')+col(f\'sum(false_attempts)\')))).na.fill(-1)\n\n# Get cumulative lagged accuracy\nfor Assess_Title in Assess_Titles:\n    SessionAccuracyDf=SessionAccuracyDf.withColumn(Assess_Title+\'AccyLagCum\',\\\n                                   col(f\'sum({Assess_Title}True)LagCum\')\\\n                                     /col(f\'sum({Assess_Title}AllAttemps)LagCum\')).na.fill(-1)\n    \nSessionAccuracyDf=SessionAccuracyDf.withColumn(\'AllAssessmentAccyLagCum\',\\\n                    col(\'sum(true_attempts)LagCum\')\\\n                    /(col(\'sum(true_attempts)LagCum\')+col(\'sum(false_attempts)LagCum\'))\\\n                    ).na.fill(-1)')


# In[25]:


get_ipython().run_cell_magic('time', '', "#Get lagged features\nColumnsToLag=['Cart Balancer (Assessment)Accy', 'Cauldron Filler (Assessment)Accy', 'Bird Measurer (Assessment)Accy',\n 'Mushroom Sorter (Assessment)Accy', 'Chest Sorter (Assessment)Accy', 'AllAssessmentAccy']\nlag_cols = [F.lag(col(Column), 1).over(windowval).alias(Column+'Lag') \\\n                     for Column in ColumnsToLag]\nSessionAccuracyDf=SessionAccuracyDf.select(SessionAccuracyDf.columns+lag_cols)\nLaggedCols=[i+'Lag' for i in ColumnsToLag]\nSessionAccuracyDf=SessionAccuracyDf.fillna(-1, subset=LaggedCols)")


# In[26]:


get_ipython().run_cell_magic('time', '', '#Add accuracy_group for each world across each session\nAssess_Titles=list_of_assess_titlesDf.toPandas()[[\'title\']].values.tolist()\nfor Assess_Title in Assess_Titles:\n    #Create accuracy group by world\n    SessionAccuracyDf=SessionAccuracyDf.withColumn(Assess_Title[0]+\'_accuracy_group\',\\\n         when((col(f\'sum({Assess_Title[0]}True)\')==0)\\\n          &(col(f\'sum({Assess_Title[0]}False)\')==0),\'NoAssess\'\\\n             ).otherwise(\\\n                       when((col(f\'sum({Assess_Title[0]}True)\')==1)\\\n                            &(col(f\'sum({Assess_Title[0]}False)\')==0),\'3\'\\\n                     ).otherwise(\\\n                            when((col(f\'sum({Assess_Title[0]}True)\')==1)\\\n                                 &(col(f\'sum({Assess_Title[0]}False)\')==1),\'2\'\\\n                                ).otherwise(when(col(f\'sum({Assess_Title[0]}True)\')==0,\'0\'\\\n                                                ).otherwise(\'1\')\\\n                                ))))\n#Create overall accuracy group \nSessionAccuracyDf=SessionAccuracyDf.withColumn(\'all_accuracy_group\',\\\n     when((col(f\'sum(true_attempts)\')==0)\\\n          &(col(f\'sum(false_attempts)\')==0),\'NoAssess\'\\\n         ).otherwise(\\\n                   when((col(f\'sum(true_attempts)\')==1).cast(BooleanType())\\\n                        &(col(f\'sum(false_attempts)\')==0),\'3\'\\\n                 ).otherwise(\\\n                        when((col(f\'sum(true_attempts)\')==1)\\\n                                 &(col(f\'sum(false_attempts)\')==1),\'2\'\\\n                                ).otherwise(when(col(f\'sum(true_attempts)\')==0,\'0\'\\\n                                                ).otherwise(\'1\')\\\n                                ))))\nColumnsToLag=[Title[0]+\'_accuracy_group\' for Title in Assess_Titles]\n\n\n#Add lagged accuracies\nCondition=F.lag(col(\'all_accuracy_group\'),1).over(windowval)!=\'NoAssess\'\nSessionAccuracyDf=SessionAccuracyDf.withColumn("Lag_all_accuracy", F.when(Condition\\\n                                    , F.lag(col(\'AllAssessmentAccy\'),1).over(windowval)))\nSessionAccuracyDf=SessionAccuracyDf.withColumn("Lag_all_accuracy"\\\n                            ,F.last(col(\'lag_all_accuracy\'),ignorenulls=True).over(windowval))\\\n                            .na.fill(-1)\n\nfor Assess_Title in Assess_Titles:\n    Condition=F.lag(col(f\'{Assess_Title[0]}_accuracy_group\'),1).over(windowval)!=\'NoAssess\'\n    SessionAccuracyDf=SessionAccuracyDf.withColumn(f\'Lag_{Assess_Title[0]}_accuracy\', F.when(Condition\\\n                                    , F.lag(col(f\'{Assess_Title[0]}Accy\'),1).over(windowval)))\n    SessionAccuracyDf=SessionAccuracyDf.withColumn(f\'Lag_{Assess_Title[0]}_accuracy\'\\\n                            ,F.last(col(f\'lag_{Assess_Title[0]}_accuracy\'),ignorenulls=True).over(windowval))\\\n                            .na.fill(-1)\n\n    \n#Create lagged accuracy group \nSessionAccuracyDf=SessionAccuracyDf.withColumn(\'Lag_all_accuracy_group\',\\\n     when((col(\'Lag_all_accuracy\')==0),0)\\\n            .otherwise(\\\n                   when((col(\'Lag_all_accuracy\')==1),3\\\n                 ).otherwise(\\\n                        when((col(\'Lag_all_accuracy\')==0.5),2\\\n                                ).otherwise(\\\n                                             when((col(\'Lag_all_accuracy\')==-1),-1\\\n                                                  ).otherwise(1)\\\n                                ))))\nColumnsToLag=[Title[0]+\'_accuracy_group\' for Title in Assess_Titles]')


# In[27]:


SessionAccuracyDf=SessionAccuracyDf.drop('StartTime','NumAssessmentAttemptsLagCum'                                        ,'PreviousSessStart','TimeSinceLastSess')


# In[28]:


get_ipython().run_cell_magic('time', '', '#Collate all details of assessment sessions - we will filter non-assessment sessions at the end.  \n\n#Join the frame with current session assessments\nSessionAccuracyDf.createOrReplaceTempView("SessionAccuracy")\nSessionTimingCumDf.createOrReplaceTempView("SessionTimingCum")\nreduce_CombinedDf=Spark.sql(f\'SELECT *,\\\nSessionTimingCum.installation_id as id1,SessionTimingCum.game_session as gs1 \\\nfrom SessionTimingCum LEFT JOIN SessionAccuracy \\\nON SessionTimingCum.installation_id=SessionAccuracy.installation_id \\\nAND SessionTimingCum.game_session=SessionAccuracy.game_session\').drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')\n\n\nreduce_CombinedDf=reduce_CombinedDf.na.drop()\n\nreduce_CombinedDf.createOrReplaceTempView("reduce_Combined")\n\n\n# # # #Remove non-assessment sessions - their data will be captured in the \'Cum\' fields, we only want to focus on assessment sessions\nreduce_CombinedDf=reduce_CombinedDf.filter(reduce_CombinedDf[\'NumAssessmentAttempts\']>0)')


# In[29]:


get_ipython().run_cell_magic('time', '', '#one-hot encode title \nSessionTitleDf=SessionTimingCumDf.groupBy("installation_id","game_session").pivot("r_session_title").count().na.fill(0)\nSessionTitleDf.createOrReplaceTempView("SessionTitle")\nreduce_CombinedDf.createOrReplaceTempView("reduce_Combined")\nreduce_CombinedDf=Spark.sql(f\'SELECT *,\\\nreduce_Combined.installation_id as id1,reduce_Combined.game_session as gs1 \\\nfrom reduce_Combined INNER JOIN SessionTitle \\\nON reduce_Combined.installation_id=SessionTitle.installation_id \\\nAND reduce_Combined.game_session=SessionTitle.game_session\').drop(\'installation_id\',\'game_session\')\\\n.withColumnRenamed(\'id1\',\'installation_id\').withColumnRenamed(\'gs1\',\'game_session\')')


# In[30]:


reduce_CombinedDf=reduce_CombinedDf.na.drop()


# In[31]:


#remove '()'  etc as parquet doen't like spaces or "()' in field names 
reduce_CombinedDf=reduce_CombinedDf.toDF(*(c.replace('(', '') for c in reduce_CombinedDf.columns))
reduce_CombinedDf=reduce_CombinedDf.toDF(*(c.replace(')', '') for c in reduce_CombinedDf.columns))
reduce_CombinedDf=reduce_CombinedDf.toDF(*(c.replace(',', '') for c in reduce_CombinedDf.columns))
reduce_CombinedDf=reduce_CombinedDf.toDF(*(c.replace(' ', '') for c in reduce_CombinedDf.columns))


# In[32]:


KeyColumns =['installation_id', 'game_session', 'StartTime', 'PreviousSessStart','TestOrTrain','Type','TestFlag','EndTime',
 'r_session_title','World']


# In[33]:


AccuracyGroupColumns =['CartBalancerAssessment_accuracy_group', 'CauldronFillerAssessment_accuracy_group', 'BirdMeasurerAssessment_accuracy_group', 'MushroomSorterAssessment_accuracy_group', 'ChestSorterAssessment_accuracy_group','all_accuracy_group']


# In[34]:


CurrentAssessmentColumns=['accuracy_group',
 'NumAssessmentAttempts',
 'sumtrue_attempts',
 'sumfalse_attempts',
 'sumCartBalancerAssessmentTrue',
 'sumCartBalancerAssessmentFalse',
 'sumCartBalancerAssessmentAllAttemps',
 'sumCauldronFillerAssessmentTrue',
 'sumCauldronFillerAssessmentFalse',
 'sumCauldronFillerAssessmentAllAttemps',
 'sumBirdMeasurerAssessmentTrue',
 'sumBirdMeasurerAssessmentFalse',
 'sumBirdMeasurerAssessmentAllAttemps',
 'sumMushroomSorterAssessmentTrue',
 'sumMushroomSorterAssessmentFalse',
 'sumMushroomSorterAssessmentAllAttemps',
 'sumChestSorterAssessmentTrue',
 'sumChestSorterAssessmentFalse',
 'sumChestSorterAssessmentAllAttemps',
 'CartBalancerAssessmentAccy',
 'CauldronFillerAssessmentAccy',
 'BirdMeasurerAssessmentAccy',
 'MushroomSorterAssessmentAccy',
 'ChestSorterAssessmentAccy',
 'AllAssessmentAccy',
  'NumEvents','misses_cnt', 'game_round',
 'game_level',
 'SessionDuration','sum12Monkeys',
 'sumAirShow',
 'sumAllStarSorting',
 'sumBalancingAct',
 'sumBirdMeasurerAssessment',
 'sumBottleFillerActivity',
 'sumBubbleBath',
 'sumBugMeasurerActivity',
 'sumCartBalancerAssessment',
 'sumCauldronFillerAssessment',
 'sumChestSorterAssessment',
 'sumChickenBalancerActivity',
 'sumChowTime',
 'sumCostumeBox',
 'sumCrystalCaves-Level1',
 'sumCrystalCaves-Level2',
 'sumCrystalCaves-Level3',
 'sumCrystalsRule',
 'sumDinoDive',
 'sumDinoDrink',
 'sumEggDropperActivity',
 'sumFireworksActivity',
 'sumFlowerWatererActivity',
 'sumHappyCamel',
 'sumHeavyHeavierHeaviest',
 'sumHoneyCake',
 'sumLeafLeader',
 'sumLiftingHeavyThings',
 'sumMagmaPeak-Level1',
 'sumMagmaPeak-Level2',
 'sumMushroomSorterAssessment',
 'sumOrderingSpheres',
 'sumPanBalance',
 "sumPirate'sTale",
 'sumRulers',
 'sumSandcastleBuilderActivity',
 'sumScrub-A-Dub',
 'sumSlopProblem',
 'sumTreasureMap',
 'sumTreeTopCity-Level1',
 'sumTreeTopCity-Level2',
 'sumTreeTopCity-Level3',
 'sumWateringHoleActivity',
 'sumWelcometoLostLagoon!', 'ActivityDur',
 'AssessmentDur',
 'ClipDur',
 'GameDur',
 'sumwin_code'
]      


# In[35]:


feature_cols=[x for x in reduce_CombinedDf.columns if (x not in AccuracyGroupColumns) and (x not in KeyColumns)               and (x not in CurrentAssessmentColumns) ]


# In[36]:


[x for x in feature_cols if not 'Lag' in x] 


# In[37]:


# vectorise the features
vectorAssembler = VectorAssembler(inputCols=feature_cols,outputCol='features',handleInvalid="skip")
assembledDf = vectorAssembler.transform(reduce_CombinedDf).drop(*feature_cols)


# In[38]:


#Convert target labels to integers
cols_to_convert = [col(Column).cast(IntegerType()) for Column in AccuracyGroupColumns]

missing_cols = [i for i in assembledDf.columns if i not in AccuracyGroupColumns]

assembledDf=assembledDf.select(missing_cols+cols_to_convert)


# In[39]:


get_ipython().run_cell_magic('time', '', '#This is the step that takes the longest and might test Kaggle machine\'s HDD\nassembledDf.write.mode("overwrite").save(\'assembledDf.parquet\')')


# In[ ]:




