#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
sc


# In[2]:


df=spark.read.csv("train.csv",header=True)
df.show()


# In[3]:


def normalize_token(x) :
    tokenizer = TreebankWordTokenizer()
    lower=x.text.lower()
    text_token=tokenizer.tokenize(lower)
    count_char=len(lower)
    return x+(count_char,text_token,)

rdd_normalize_token=df.rdd.map(lambda x : normalize_token(x))
rdd_normalize_token.first()


# In[4]:


def remove_stop_word_and_changer_number(x,seuil,stop_words,list_punct) :
    tab_result=[]
    count_punct=0
    for elt in x[4] :
        if elt in list_punct :
            count_punct=count_punct+1
        try :
            number=float(elt)
            tab_result.append("number")
        except :
            if (len(elt)>seuil) &(elt not in stop_words) :
                tab_result.append(elt)            
    return x[:4]+(count_punct,tab_result,)
    
seuil=1
stop_words=set(stopwords.words('english'))
list_punct=list(string.punctuation)
rdd_stop_word=rdd_normalize_token.map(lambda x : remove_stop_word_and_changer_number(x,seuil,stop_words,list_punct))
rdd_stop_word.first()


# In[5]:


def lemmatisation(x,dict_cor) :
    tab_result=[]
    wordnet_lemmatizer = WordNetLemmatizer()
    pos_tmp = nltk.pos_tag(x[5])
    for elt in pos_tmp :
        if elt[1][0] in dict_cor :
            attrib=dict_cor[elt[1][0]]
        else :
            attrib = "n"
        tab_result.append(wordnet_lemmatizer.lemmatize(elt[0], pos=attrib))
    return x[:5]+(tab_result,)
        
dict_cor={
    "N" : "n",
    "V" : "v",
    "J" : "r",
    "A" : "a",
}

rdd_lemma=rdd_stop_word.map(lambda x : lemmatisation(x,dict_cor))
rdd_lemma.first()


# In[6]:


def size_word(x) :
    nb_word=len(x[5])
    return x+(nb_word,)

rdd_size=rdd_lemma.map(lambda x :size_word(x))
rdd_size.first()


# In[7]:


df_final=rdd_size.toDF(["id","phrase","Author","nb_carac","nb_punct","words","size"])
print df_final.count()
df_final.show()


# In[8]:


df_final.write.parquet("tokenize_03_12_v3",mode="overwrite")


# In[9]:


df_test=spark.read.csv("test.csv",header=True)
 #The map is important in order to have the same number of column as train set
df_test_1=df_test.rdd.map(lambda x : x+(1,)).toDF(["id","text","author"])
df_test_2=df_test_1.rdd.map(lambda x : normalize_token(x))
df_test_3=df_test_2.map(lambda x : remove_stop_word_and_changer_number(x,seuil,stop_words,list_punct))
df_test_4=df_test_3.map(lambda x : lemmatisation(x,dict_cor))
df_test_5=df_test_4.map(lambda x :size_word(x))
df_final_test=df_test_5.toDF(["id","phrase","Author","nb_carac","nb_punct","words","size"]).drop("Author")
print df_final_test.count()
print df_final_test.show()
df_final_test.write.parquet("tokenize_test_03_12_v3",mode="overwrite")


# In[10]:


import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,  RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors, VectorUDT


# In[11]:


df = spark.read.parquet("tokenize_03_12_v3")
df_test = spark.read.parquet("tokenize_test_03_12_v3")
df.show()


# In[12]:


def add_label(x) :
    if x.Author=="EAP" :
        EAP=1
        HPL=0
        MWS=0
    elif x.Author=="HPL" :
        EAP=0
        HPL=1
        MWS=0
    elif x.Author=="MWS" :
        EAP=0
        HPL=0
        MWS=1
    else :
        EAP=0
        HPL=0
        MWS=0
    return x+(EAP,HPL,MWS)

rdd_label=df.rdd.map(lambda x : add_label(x))
df_add_label=rdd_label.toDF(["id","phrase","Author","nb_carac","nb_punct","words","size","label_EAP","label_HPL","label_MWS"])
df_add_label.show()


# In[13]:


method="both"

if method=="tf-idf" :
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=256)
    featurizedData = hashingTF.transform(df_add_label)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

elif method=="w2v" :
    word2Vec = Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="features",seed=42, maxIter=20)
    df_all_words=df_test.select("words").union(df_add_label.select("words"))
    model_w2v = word2Vec.fit(df_all_words)

    rescaledData = model_w2v.transform(df_add_label)
    
elif method=="both" :
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=256)
    featurizedData = hashingTF.transform(df_add_label)

    idf = IDF(inputCol="rawFeatures", outputCol="features_tf_idf")
    idfModel = idf.fit(featurizedData)
    rescaledData_tmp = idfModel.transform(featurizedData)
    
    word2Vec = Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="features",seed=42, maxIter=20)
    df_all_words=df_test.select("words").union(df_add_label.select("words"))
    model_w2v = word2Vec.fit(df_all_words)

    rescaledData = model_w2v.transform(rescaledData_tmp)
    
rescaledData.show()


# In[14]:


if method=="both" :
    df_ML=rescaledData.rdd.map(lambda x : x[:-1]+(Vectors.dense(list(x.features)+list(x.features_tf_idf) + [x.size,x.nb_carac,x.nb_punct]),))    .toDF(rescaledData.columns)
else :
    df_ML=rescaledData.rdd.map(lambda x : x[:-1]+(Vectors.dense(list(x.features) + [x.size,x.nb_carac,x.nb_punct]),))    .toDF(rescaledData.columns)
df_ML.first()


# In[15]:


trainingData=rescaledData
#print trainingData.first()

nb_eap = RandomForestRegressor(featuresCol="features", predictionCol="EAP",labelCol="label_EAP", maxDepth=10, numTrees=20)

paramGrid = ParamGridBuilder()     .addGrid(nb_eap.numTrees, [ 40,50,60])     .addGrid(nb_eap.maxDepth, [ 5,10])     .build()

crossval = CrossValidator(estimator=nb_eap,
                          estimatorParamMaps=paramGrid,
                          evaluator= RegressionEvaluator(predictionCol="EAP",labelCol="label_EAP"),
                          numFolds=3)

model_eap=crossval.fit(trainingData)


# In[16]:


trainingData=rescaledData
#print trainingData.first()

nb_hpl = RandomForestRegressor(featuresCol="features",labelCol="label_HPL", predictionCol="HPL", maxDepth=10, numTrees=20)

paramGrid = ParamGridBuilder()     .addGrid(nb_hpl.numTrees, [40,50,60])     .addGrid(nb_hpl.maxDepth, [ 5,10])     .build()

crossval = CrossValidator(estimator=nb_hpl,
                          estimatorParamMaps=paramGrid,
                          evaluator= RegressionEvaluator(labelCol="label_HPL", predictionCol="HPL"),
                          numFolds=3)

model_hpl=crossval.fit(trainingData)


# In[17]:


trainingData=rescaledData
#print trainingData.first()

nb_mws = RandomForestRegressor(featuresCol="features",labelCol="label_MWS", predictionCol="MWS", maxDepth=10, numTrees=20)

paramGrid = ParamGridBuilder()     .addGrid(nb_mws.numTrees, [40,50,60])     .addGrid(nb_mws.maxDepth,  [5,10])     .build()

crossval = CrossValidator(estimator=nb_mws,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="label_MWS", predictionCol="MWS"),
                          numFolds=3)

model_mws=crossval.fit(trainingData)


# In[18]:


print model_eap.bestModel 
print model_hpl.bestModel 
print model_mws.bestModel 


# In[19]:


df_test.show()
if method=="tf-idf" :
    featurizedData_test = hashingTF.transform(df_test)
    listfeaturized=featurizedData_test.collect()
    rescaledData_test = idfModel.transform(featurizedData_test)
    listidfMode=rescaledData_test.collect()
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(rescaledData_test)
    df_to_predict=featureIndexer.transform(rescaledData_test)
     df_to_predict=df_to_predict_tmp.rdd.map(lambda x : x[:-1]+(Vectors.dense(list(x.features_tf_idf) + [x.size,x.nb_carac,x.nb_punct]),))    .toDF(df_to_predict_tmp.columns)

elif method=="w2v" :
    df_to_predict_tmp= model_w2v.transform(df_test)
    df_to_predict=df_to_predict_tmp.rdd.map(lambda x : x[:-1]+(Vectors.dense(list(x.features) + [x.size,x.nb_carac,x.nb_punct]),))    .toDF(df_to_predict_tmp.columns)
    print df_to_predict.first()
    
elif method=="both" :
    list0=df_test.collect()
    featurizedData_test = hashingTF.transform(df_test)
    rescaledData_test = idfModel.transform(featurizedData_test)
    featureIndexer = VectorIndexer(inputCol="features_tf_idf", outputCol="indexedFeatures").fit(rescaledData_test)
    df_to_predict=featureIndexer.transform(rescaledData_test)
    df_to_predict_tmp= model_w2v.transform(df_to_predict)
    df_to_predict=df_to_predict_tmp.rdd.map(lambda x : x[:-1]+(Vectors.dense(list(x.features)+list(x.features_tf_idf) + [x.size,x.nb_carac,x.nb_punct]),))    .toDF(df_to_predict_tmp.columns)
    print df_to_predict.first()

df_test_1 = model_eap.transform(df_to_predict).drop('rawPrediction').drop('probability')
df_test_2 = model_hpl.transform(df_test_1).drop('rawPrediction').drop('probability')
df_test_3 = model_mws.transform(df_test_2).drop('phrase').drop('words').drop('rawFeatures').drop('rawPrediction').drop('probability').drop('features').drop('indexedFeatures')
print df_test_3.show()


# In[20]:


def change_val(x) :
    EAP=x.EAP
    HPL=x.HPL
    MWS=x.MWS
    if x.EAP>1 :
        EAP=1
    if x.HPL>1 :
        HPL=1
    if x.MWS>1 :
        MWS=1
    if x.EAP<0 :
        EAP=0
    if x.HPL<0 :
        HPL=0
    if x.MWS<0 :
        MWS=0
    return (x.id,EAP,HPL,MWS)

df_save=df_test_3.rdd.map(lambda x : change_val(x)).toDF(['id','EAP','HPL','MWS'])


# In[21]:


df_save=df_test_3.coalesce(1)
print df_save.count()
print df_save.first()


# In[22]:


df_save.select(['id','EAP','HPL','MWS']).write.csv("result_test_03_12_v4",sep=",",header=True,mode="overwrite")


# In[23]:




