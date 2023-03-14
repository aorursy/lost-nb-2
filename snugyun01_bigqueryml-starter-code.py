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


PROJECT_ID = 'geultto'
# Not using kaggle-competitions-project!

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_example', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# In[3]:


table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[4]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[5]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.total_time_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p20 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[6]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.total_time_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p50 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[7]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.total_time_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p80 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[8]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.distance_p20`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p20 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[9]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.distance_p50`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p50 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[10]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.distance_p80`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    DistanceToFirstStop_p80 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[11]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.distance_p20`)\nORDER BY iteration ')


# In[12]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.total_time_p20`, (\n  SELECT\n    TotalTimeStopped_p20 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[13]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.total_time_p50`, (\n  SELECT\n    TotalTimeStopped_p50 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[14]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.total_time_p80`, (\n  SELECT\n    TotalTimeStopped_p80 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[15]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.distance_p20`, (\n  SELECT\n    DistanceToFirstStop_p20 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[16]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.distance_p50`, (\n  SELECT\n    DistanceToFirstStop_p50 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[17]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.distance_p80`, (\n  SELECT\n    DistanceToFirstStop_p80 as label,\n    Weekend,\n    Hour,\n    Month,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[18]:


get_ipython().run_cell_magic('bigquery', 'df_1', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.distance_p20`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[19]:


get_ipython().run_cell_magic('bigquery', 'df_2', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.distance_p50`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[20]:


get_ipython().run_cell_magic('bigquery', 'df_3', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.distance_p80`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[21]:


get_ipython().run_cell_magic('bigquery', 'df_4', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.total_time_p20`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[22]:


get_ipython().run_cell_magic('bigquery', 'df_5', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.total_time_p50`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[23]:


get_ipython().run_cell_magic('bigquery', 'df_6', 'SELECT\n  RowId,\n  predicted_label as Target\nFROM\n  ML.PREDICT(MODEL `bqml_example.total_time_p80`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        Month,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[24]:


df_6.head(2)


# In[25]:


df_1['RowId'] = df_1['RowId'].apply(str) + '_0'
df_2['RowId'] = df_2['RowId'].apply(str) + '_1'
df_3['RowId'] = df_3['RowId'].apply(str) + '_2'
df_4['RowId'] = df_4['RowId'].apply(str) + '_3'
df_5['RowId'] = df_5['RowId'].apply(str) + '_4'
df_6['RowId'] = df_6['RowId'].apply(str) + '_5'


# In[26]:


df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], axis=0)


# In[27]:


df.rename(columns={'RowId': 'TargetId'}, inplace=True)


# In[28]:


submission = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
submission = submission.merge(df, on='TargetId')
submission.rename(columns={'Target_y': 'Target'}, inplace=True)
submission = submission[['TargetId', 'Target']]


# In[29]:


submission.to_csv('submission.csv', index=False)


# In[30]:


submission.tail()


# In[ ]:




