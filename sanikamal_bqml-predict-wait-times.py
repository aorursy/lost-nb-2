#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GCP Project Id
PROJECT_ID = 'bigquery-bqml-kaggle'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# In[2]:


dataset = client.create_dataset('bqml_intersection', exists_ok=True)


# In[3]:


# create a reference to our table
table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[4]:


# create a reference to our table
test_table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.test")
# look at five rows from test table
client.list_rows(test_table, max_results=5).to_dataframe()


# In[5]:


# Print information on all the columns in the "train" table
table.schema


# In[6]:


# Print information on all the columns in the "test" table
test_table.schema


# In[7]:


# Preview the first five entries in the "Latitude" and "Longitude" column of the "train" table
client.list_rows(table, selected_fields=table.schema[2:4], max_results=5).to_dataframe()


# In[8]:


# magic command
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[9]:


get_ipython().run_cell_magic('bigquery', 'total_street_name', 'SELECT\n    City,\n    COUNT(EntryStreetName) AS EntryStreetNameCount,\n    COUNT(ExitStreetName) AS ExitStreetNameCount\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nORDER BY City DESC')


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
total_street_name.plot(kind='bar', x='City', y=['EntryStreetNameCount','ExitStreetNameCount']);


# In[11]:


get_ipython().run_cell_magic('bigquery', 'total_heading', 'SELECT\n    City,\n    COUNT(EntryHeading) AS EntryHeadingCount,\n    COUNT(ExitHeading) AS ExitHeadingCount\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nORDER BY City DESC')


# In[12]:


total_heading.plot(kind='bar', x='City', y=['EntryHeadingCount','ExitHeadingCount']);


# In[13]:


get_ipython().run_cell_magic('bigquery', 'latitude_longitude', 'SELECT Latitude,Longitude\n\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`')


# In[14]:


sns.relplot(x="Latitude", y="Longitude", data=latitude_longitude);


# In[15]:


get_ipython().run_cell_magic('bigquery', 'count_IntersectionId', 'SELECT City,\nCOUNT(IntersectionId) AS total_IntersectionId\nFROM `kaggle-competition-datasets.geotab_intersection_congestion.train`\nGROUP BY City\nORDER BY City')


# In[16]:


count_IntersectionId.plot(kind='bar', x='City', y='total_IntersectionId');


# In[17]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.total_time_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p20 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[18]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.total_time_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p50 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[19]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.total_time_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p80 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[20]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.distance_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p20 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[21]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.distance_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p50 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[22]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_intersection.distance_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p80 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[23]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_intersection.total_time_p20`)\nORDER BY iteration ')


# In[24]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_intersection.distance_p20`)\nORDER BY iteration ')


# In[25]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_intersection.total_time_p20`, (\n  SELECT\n    TotalTimeStopped_p20 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[26]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_intersection.total_time_p50`, (\n  SELECT\n    TotalTimeStopped_p50 as label,\n    Hour,\n    Weekend,\n    Month,\n    EntryStreetName,\n    ExitStreetName,\n    EntryHeading,\n    ExitHeading,\n    Path,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[27]:


get_ipython().run_cell_magic('bigquery', 'df_1', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.distance_p20`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[28]:


get_ipython().run_cell_magic('bigquery', 'df_2', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.distance_p50`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[29]:


get_ipython().run_cell_magic('bigquery', 'df_3', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.distance_p80`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[30]:


get_ipython().run_cell_magic('bigquery', 'df_4', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.total_time_p20`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[31]:


get_ipython().run_cell_magic('bigquery', 'df_5', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.total_time_p50`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[32]:


get_ipython().run_cell_magic('bigquery', 'df_6', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_intersection.total_time_p80`,\n    (\n    SELECT\n        RowId,\n        Hour,\n        Weekend,\n        Month,\n        EntryStreetName,\n        ExitStreetName,\n        EntryHeading,\n        ExitHeading,\n        Path,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[33]:


import pandas as pd
df_1['RowId'] = df_1['RowId'].apply(str) + '_0'
df_2['RowId'] = df_2['RowId'].apply(str) + '_1'
df_3['RowId'] = df_3['RowId'].apply(str) + '_2'
df_4['RowId'] = df_4['RowId'].apply(str) + '_3'
df_5['RowId'] = df_5['RowId'].apply(str) + '_4'
df_6['RowId'] = df_6['RowId'].apply(str) + '_5'


# In[34]:


df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], axis=0)


# In[35]:


df.rename(columns={'RowId': 'TargetId'}, inplace=True)


# In[36]:


# df['RowId'] = df['RowId'].apply(str) + '_0'
# df.rename(columns={'RowId': 'TargetId', 'TotalTimeStopped_p20': 'Target'}, inplace=True)
# df


# In[37]:


# df.to_csv('submission.csv',index=False)
submission = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
submission = submission.merge(df, on='TargetId')
submission.rename(columns={'Target_y': 'Target'}, inplace=True)
submission = submission[['TargetId', 'Target']]
submission.to_csv('submission.csv', index=False)

