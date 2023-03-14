#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('java -version')


# In[2]:


get_ipython().system('pip install --upgrade --quiet pyspark==3.0.0')

get_ipython().system('pip install --quiet koalas')


# In[3]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', 20)

import databricks.koalas as ks

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as sqlF
from pyspark import SparkContext, SparkConf


# In[4]:


#conf = SparkConf().setAll([('spark.executor.memory', '5g'), 
#                           ('spark.driver.memory','5g'),
#                           ('spark.driver.maxResultSize','0')])


spark = SparkSession             .builder.master('local[*]')            .appName("TutorialApp")            .getOrCreate()

sqlContext = SQLContext(sparkContext=spark.sparkContext, 
                        sparkSession=spark)


# In[5]:


train_pandas = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
train_koalas = ks.read_csv("../input/tweet-sentiment-extraction/train.csv", escape="_")

train_spark  = spark.read.csv("../input/tweet-sentiment-extraction/train.csv",
                             inferSchema="true", header="true", escape="_")

# another ways of attaching koalas api to spark dataframe
#train_koalas = train_spark.to_koalas()


# In[6]:


#This will be useful to show similarity between pandas and koalas dataframes
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# In[7]:


print("train_spark.show(5)")
train_spark.show(5)
display("train_koalas.head(5)","train_pandas.head(5)")


# In[8]:


print("Spark API:")
print(train_spark.dtypes)
print()
print("Koalas API:")
print(train_koalas.dtypes)
print()
print("Pandas API:")
print(train_pandas.dtypes)


# In[9]:


print("Spark API:")
print(train_spark.columns)
print()
print("Koalas API:")
print(train_koalas.columns)
print()
print("Pandas API:")
print(train_pandas.columns)


# In[10]:


print("SPARK doesn't use any index so there is no sort by index!")
display("train_koalas.sort_index(ascending=False).head(5)", "train_pandas.sort_index(ascending=False).head(5)")


# In[11]:


print("train_spark.sort('text', ascending=False).show(5)")
train_spark.sort("text", ascending=False).show(5)
display("train_koalas.sort_values(by='text',ascending=False).head(5)", 
        "train_pandas.sort_values(by='text',ascending=False).head(5)")


# In[12]:


print("train_spark.groupBy('sentiment').count().orderBy(sqlF.col('count').desc()).show()")
train_spark.groupBy("sentiment").count()    .orderBy(sqlF.col("count").desc())    .show()
display("train_koalas.groupby('sentiment')[['textID']].count().sort_values('textID', ascending=False)",
        "train_pandas.groupby('sentiment')[['textID']].count().sort_values('textID', ascending=False)")


# In[13]:


#to drop
spark_dropna  = train_spark.na.drop()
koalas_dropna = train_koalas.dropna()
pandas_dropna = train_pandas.dropna()


# In[14]:


#to fill
spark_fillna  = train_spark.na.fill('missing text')
koalas_fillna = train_koalas.fillna('missing text')
pandas_fillna = train_pandas.fillna('missing text')


# In[15]:


print("INITIALLY")
train_spark.filter(train_spark.text.isNull()).show()
print("AFTER DROP NA")
spark_dropna.filter(spark_dropna.text.isNull()).show()
print("AFTER FILL NA")
spark_fillna.filter(spark_fillna.text == "missing text").show()


# In[16]:


# First Create a tempView
train_spark.createOrReplaceTempView("train")


# In[17]:


spark.sql("SELECT * FROM train").show(5)


# In[18]:


spark.sql("""
            SELECT sentiment, count(*) AS total 
            FROM train 
            GROUP BY sentiment 
            ORDER BY total DESC
          """).show(5)


# In[19]:


spark.sql("""
            SELECT * 
            FROM train 
            WHERE text IS NULL
          """).show(5)


# In[20]:


data = train_koalas.groupby('sentiment')['textID'].count()

data.plot(kind="bar", figsize=(5,4))
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# In[21]:


sns.barplot(x=data.index.to_numpy(), y=data.to_numpy())
plt.ylabel("Frequency")
plt.xlabel("Sentiment")
plt.xticks(rotation=45)
plt.show()


# In[22]:


import wordcloud
import re
words = koalas_dropna.to_spark().rdd.flatMap(lambda x: re.split("\s+",x[2]))                  .map(lambda word: (word, 1))                  .reduceByKey(lambda a, b: a + b)

schema = StructType([StructField("words", StringType(), True),
                 StructField("count", IntegerType(), True)])

words_df = sqlContext.createDataFrame(words, schema=schema)


# In[23]:


print("Total number of words:")
words_df.groupBy().sum("count").show()


# In[24]:


words_df.groupBy('words')        .agg(sqlF.mean("count")/195177)        .orderBy(sqlF.desc("(avg(count) / 195177)"))        .show(50)

word_cloud = words_df.orderBy(sqlF.desc("count"))                     .limit(200)                     .toPandas()                     .set_index('words')                     .T                     .to_dict('records')


# In[25]:


wc = wordcloud.WordCloud(background_color="white", max_words=200)
wc.generate_from_frequencies(dict(*word_cloud))

plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
plt.show()


# In[26]:


from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, NGram
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, OneVsRest


# In[27]:


tokenizer = RegexTokenizer(inputCol="selected_text", outputCol="words", pattern="\\W")
ngram = NGram(inputCol="words", outputCol="n-gram").setN(1) #Unigram
tf = CountVectorizer(inputCol="n-gram", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features", minDocFreq=3)
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexer = StringIndexer(inputCol="sentiment", outputCol="sentiment_index")

pipeline = Pipeline(stages=[tokenizer, ngram, tf, idf, indexer])


# In[28]:


tf_idf = pipeline.fit(spark_dropna)
training_data = tf_idf.transform(koalas_dropna.to_spark()).select("features","sentiment_index")


# In[29]:


train, valid = training_data.randomSplit([0.7, 0.3], seed=41)
svc = LinearSVC()

classifierMod = OneVsRest(classifier=svc, featuresCol="features",
                         labelCol="sentiment_index")

model = classifierMod.fit(train)


# In[30]:


valid_prediction = model.transform(valid)
train_prediction = model.transform(train)


evaluator = MulticlassClassificationEvaluator(labelCol="sentiment_index", 
                                              predictionCol="prediction")
print("Train Accuracy achieved:",round(evaluator.evaluate(train_prediction.select("sentiment_index","prediction"), {evaluator.metricName: "accuracy"}),3))
print("Valid Accuracy achieved:",round(evaluator.evaluate(valid_prediction.select("sentiment_index","prediction"), {evaluator.metricName: "accuracy"}),3))


# In[31]:


#close the spark session when done
spark.stop()

