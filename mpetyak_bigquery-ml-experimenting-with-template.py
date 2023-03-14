#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Replace 'kaggle-competitions-project' with YOUR OWN project id here --  
PROJECT_ID = <Project_ID>

import os
import pandas as pd

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID)
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


# Allow you to easily have Python variables in SQL query.
from IPython.core.magic import register_cell_magic
from IPython import get_ipython

LABEL_TO_NUM = {
    'TotalTimeStopped_p20' : '0',
    'TotalTimeStopped_p50' : '1',
    'TotalTimeStopped_p80' : '2',
    'DistanceToFirstStop_p20' : '3',
    'DistanceToFirstStop_p50' : '4',
    'DistanceToFirstStop_p80' : '5',
}
    
# Metrics for prediction
# TotalTimeStopped_p20, TotalTimeStopped_p50, TotalTimeStopped_p80, 
# DistanceToFirstStop_p20, DistanceToFirstStop_p50, DistanceToFirstStop_p80.
QUERY_CONFIG = {}
QUERY_CONFIG['LABEL_COLUMN'] = 'DistanceToFirstStop_p80'
QUERY_CONFIG['LABEL_NUM'] = LABEL_TO_NUM[QUERY_CONFIG['LABEL_COLUMN']]
QUERY_CONFIG['MODEL_NAME'] = '`bqml_example.model_' + QUERY_CONFIG['LABEL_NUM'] + "`"

@register_cell_magic('with_config')
def with_config(line, cell):
    contents = cell.format(**QUERY_CONFIG)
    get_ipython().run_cell(contents)


# In[3]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[4]:


# Helper tables containing information to JOIN

# CREATE TABLE IF NOT EXISTS helper.Directions (
#   direction STRING,
#   value FLOAT64
# );
           
# INSERT INTO helper.Directions
# VALUES ("N", 0),
#        ("NE", 1/4),
#        ("E", 1/2),
#        ("SE", 3/4),
#        ("S", 1),
#        ("SW", 5/4),
#        ("W", 3/2),
#        ("NW", 7/4);

# # Use this to later calculate distances from center
# CREATE TABLE IF NOT EXISTS helper.Cities (
#   city STRING,
#   centerLatitude FLOAT64,
#   centerLongitude FLOAT64
# );
                           
# INSERT INTO helper.Cities
# VALUES ("Atlanta", 33.753746, -84.386330),
#        ("Boston", 42.361145, -71.057083),
#        ("Chicago", 41.881832, -87.623177),
#        ("Philadelphia", 39.952583, -75.165222);


# In[5]:


get_ipython().run_cell_magic('with_config', '', "%%bigquery\nCREATE OR REPLACE MODEL {MODEL_NAME}\nOPTIONS(model_type='linear_reg') AS\nSELECT\n    {LABEL_COLUMN} as label,\n    Weekend,\n    Hour,\n    Month,\n    K.City,\n    IFNULL(EntryHeading, ExitHeading) as entryheading,\n    IFNULL(ExitHeading, EntryHeading) as exitheading,\n    IF( EntryStreetName = ExitStreetName , 1, 0) as samestreet,\n    SQRT( POW( C.centerLatitude - K.Latitude, 2) + POW(C.centerLongitude - K.Longitude, 2) ) as distance,\n    D1.value - D2.value as diffHeading\nFROM\n    `kaggle-competition-datasets.geotab_intersection_congestion.train` as K\n    LEFT JOIN `instant-medium-261000.helper.Directions` as D1 on D1.direction = EntryHeading\n    LEFT JOIN `instant-medium-261000.helper.Directions` as D2 on D2.direction = ExitHeading\n    LEFT JOIN `instant-medium-261000.helper.Cities` as C on C.city = K.City\nWHERE\n    RowId < 2600000")


# In[6]:


get_ipython().run_cell_magic('with_config', '', '%%bigquery\nSELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL {MODEL_NAME})\nORDER BY iteration ')


# In[7]:


get_ipython().run_cell_magic('with_config', '', '%%bigquery\nSELECT * \nFROM ML.EVALUATE(MODEL {MODEL_NAME},\n(SELECT\n    {LABEL_COLUMN} as label,\n    Weekend,\n    Hour,\n    Month,\n    K.City,\n    IFNULL(EntryHeading, ExitHeading) as entryheading,\n    IFNULL(ExitHeading, EntryHeading) as exitheading,\n    IF( EntryStreetName = ExitStreetName , 1, 0) as samestreet,\n    SQRT( POW( C.centerLatitude - K.Latitude, 2) + POW(C.centerLongitude - K.Longitude, 2) ) as distance,\n    D1.value - D2.value as diffHeading\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.train` as K\n  LEFT JOIN `instant-medium-261000.helper.Directions` as D1 on D1.direction = EntryHeading\n  LEFT JOIN `instant-medium-261000.helper.Directions` as D2 on D2.direction = ExitHeading\n  LEFT JOIN `instant-medium-261000.helper.Cities` as C on C.city = K.City\nWHERE\n    RowId > 2600000\n))')


# In[8]:


get_ipython().run_cell_magic('with_config', '', '%%bigquery df\nSELECT\n    RowId,\n    predicted_label as {LABEL_COLUMN}\nFROM\n  ML.PREDICT(MODEL {MODEL_NAME},\n    (\nSELECT\n    K.RowId,\n    K.Weekend,\n    K.Hour,\n    K.Month,\n    K.City,\n    IFNULL(EntryHeading, ExitHeading) as entryheading,\n    IFNULL(ExitHeading, EntryHeading) as exitheading,\n    IF( EntryStreetName = ExitStreetName , 1, 0) as samestreet,\n    SQRT( POW( C.centerLatitude - K.Latitude, 2) + POW(C.centerLongitude - K.Longitude, 2) ) as distance,\n    D1.value - D2.value as diffHeading\nFROM\n  `kaggle-competition-datasets.geotab_intersection_congestion.test` as K\n  LEFT JOIN `instant-medium-261000.helper.Directions` as D1 on D1.direction = EntryHeading\n  LEFT JOIN `instant-medium-261000.helper.Directions` as D2 on D2.direction = ExitHeading\n  LEFT JOIN `instant-medium-261000.helper.Cities` as C on C.city = K.City    \n    ))\nORDER BY RowId ASC')


# In[9]:


df['RowId'] = df['RowId'].apply(str) + '_' + QUERY_CONFIG['LABEL_NUM']
df.rename(columns={'RowId': 'TargetId', QUERY_CONFIG['LABEL_COLUMN']: 'Target'}, inplace=True)
df


# In[10]:


df.sort_values(by='TargetId', inplace=True)
df.to_csv('submission_{}.csv'.format(QUERY_CONFIG['LABEL_NUM']))


# In[11]:


# Combining resulting datasets
subm_df = pd.read_csv('submission_0.csv', index_col=[0])
for i in range(1, 6):
    temp_df = pd.read_csv('submission_{}.csv'.format(i), index_col=[0])
    subm_df = subm_df.append(temp_df)

subm_df.head()


# In[12]:


# arrange values in the right order for submission by sorting, reindexing, and 
def order_func(target_id):
    row, target_num = target_id.split("_")
    return (int(row), int(target_num))

subm_df['order'] = subm_df['TargetId'].apply(order_func)

subm_df.sort_values(by='order', inplace=True)
subm_df.drop(columns=['order'], inplace=True)
subm_df.reset_index(drop=True, inplace=True)


# In[13]:


subm_df.head(10) 


# In[14]:


subm_df.to_csv("submission.csv", index=False)


# In[ ]:




