#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Replace 'kaggle-competitions-project' with YOUR OWN project id here --  
PROJECT_ID = 'kaggle-competitions-project'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_example', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# create a reference to our table
table = client.get_table("kaggle-competition-datasets.geotab_intersection_congestion.train")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()


# In[2]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[3]:


get_ipython().run_cell_magic('bigquery', '', "CREATE MODEL IF NOT EXISTS `bqml_example.model1`\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    TotalTimeStopped_p20 as label,\n    Weekend,\n    Hour,\n    EntryHeading,\n    ExitHeading,\n    City\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train`\nWHERE\n    RowId < 2600000")


# In[4]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_example.model1`)\nORDER BY iteration ')


# In[5]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_example.model1`, (\n  SELECT\n    TotalTimeStopped_p20 as label,\n    Weekend,\n    Hour,\n    EntryHeading,\n    ExitHeading,\n    City\n  FROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train`\n  WHERE\n    RowId > 2600000))')


# In[6]:


get_ipython().run_cell_magic('bigquery', 'df', 'SELECT\n  RowId,\n  predicted_label as TotalTimeStopped_p20\nFROM\n  ML.PREDICT(MODEL `bqml_example.model1`,\n    (\n    SELECT\n        RowId,\n        Weekend,\n        Hour,\n        EntryHeading,\n        ExitHeading,\n        City\n    FROM\n      `kaggle-competition-datasets.geotab_intersection_congestion.test`))\n    ORDER BY RowId ASC')


# In[7]:


df['RowId'] = df['RowId'].apply(str) + '_0'
df.rename(columns={'RowId': 'TargetId', 'TotalTimeStopped_p20': 'Target'}, inplace=True)
df


# In[8]:


df.to_csv(r'submission.csv')

