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
#         print(dirname, len(filenames))
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import csv


# In[3]:


get_ipython().system('gcloud auth activate-service-account   kaggle-automl-vision-sa@trusty-mantra-251121.iam.gserviceaccount.com           --key-file=/kaggle/input/private-yufengguo556835-gcp-key/key.json           --project=trusty-mantra-251121')


# In[4]:


from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('/kaggle/input/private-yufengguo556835-gcp-key/key.json')


# In[5]:


PROJECT_ID = 'trusty-mantra-251121'
REGION='us-central1'           # this MUST be us-central1
BUCKET_NAME='my-kaggle-data1'  # This bucket should already exist in us-central1


# In[6]:


from google.cloud import automl_v1beta1 as automl
from google.cloud import storage


# In[7]:


storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME, credentials=credentials)


# In[8]:


tables_client = automl.TablesClient(project=PROJECT_ID, region=REGION, 
                                    gcs_client=tables_gcs_client, credentials=credentials)


# In[9]:


from kaggle.gcp import KaggleKernelCredentials
from kaggle_secrets import GcpTarget
import kaggle_gcp


# In[10]:


os.environ['KAGGLE_USER_SECRETS_TOKEN']


# In[11]:


KaggleKernelCredentials(GcpTarget.AUTOML).valid


# In[12]:


storage_client = storage.Client(project=PROJECT_ID, credentials=KaggleKernelCredentials(GcpTarget.GCS))
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME, credentials=KaggleKernelCredentials(GcpTarget.GCS))


# In[13]:


tables_client = automl.TablesClient(project=PROJECT_ID, region=REGION, 
                                    gcs_client=tables_gcs_client, credentials=KaggleKernelCredentials(GcpTarget.AUTOML))


# In[14]:


from google.cloud import storage, automl_v1beta1 as automl

storage_client = storage.Client(project=PROJECT_ID)
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME)
automl_client = automl.AutoMlClient()
prediction_client = automl.PredictionServiceClient()
tables_client = automl.TablesClient(project=PROJECT_ID, region=REGION, 
                                    client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)


# In[15]:


# dataset_metadata = {"primary_table_spec_id": "label"} 
# # Set dataset name and metadata of the dataset.
# my_dataset = {
#     "tables_dataset_metadata": dataset_metadata,
# }

dataset_display_name='kaggle_data_tester'
my_dataset = tables_client.create_dataset(dataset_display_name=dataset_display_name)


# In[16]:


list_datasets = tables_client.list_datasets()
# Using a dict is problematic since display_name does not have to be unique
# datasets = { dataset.display_name: dataset.name for dataset in list_datasets }
datasets = [ (dataset.display_name, dataset.name) for dataset in list_datasets ]

datasets


# In[17]:


def upload_blob(source_file_name, destination_blob_name, bucket_name=BUCKET_NAME):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    destination_gcs_uri = f'gs://{bucket_name}/{destination_blob_name}'
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_gcs_uri))
    return destination_gcs_uri


# In[18]:


gcs_upload_uri = upload_blob('/kaggle/input/Kannada-MNIST/train.csv', 'kannada_train.csv')
gcs_upload_uri


# In[19]:


import_data_operation = tables_client.import_data(
    dataset=my_dataset,
    gcs_input_uris=gcs_upload_uri
)
print('Dataset import operation: {}'.format(import_data_operation))


# In[20]:


# Synchronous check of operation status. Wait until import is done.
import_data_operation.result()
dataset = tables_client.get_dataset(dataset_name=my_dataset.name)
dataset


# In[21]:


automl_client = automl.AutoMlClient()


# In[22]:


project_location = automl_client.location_path(PROJECT_ID, REGION)
# Specify the image classification type for the dataset, MULTICLASS or MULTILABEL
dataset_metadata = {"primary_table_spec_id": "label"} 
# Set dataset name and metadata of the dataset.
my_dataset = {
    "display_name": 'kannada_kaggle',
    "tables_dataset_metadata": dataset_metadata,
}

# Create a dataset with the dataset metadata in the region.
# dataset = automl_client.create_dataset(project_location, my_dataset)


# In[23]:


automl_client = automl.AutoMlClient(credentials=credentials)


# In[24]:


project_location = automl_client.location_path(PROJECT_ID, REGION)
# Specify the image classification type for the dataset, MULTICLASS or MULTILABEL
dataset_metadata = {"primary_table_spec_id": "label"} 
# Set dataset name and metadata of the dataset.
my_dataset = {
    "display_name": 'kannadakaggle_manual',
    "tables_dataset_metadata": dataset_metadata,
}

# Create a dataset with the dataset metadata in the region.
# dataset = automl_client.create_dataset(project_location, my_dataset)


# In[ ]:





# In[25]:


data_train_file = "../input/Kannada-MNIST/train.csv"
data_test_file = "../input/Kannada-MNIST/test.csv" # test file for submission
data_dig_file = "../input/Kannada-MNIST/Dig-MNIST.csv" #labeled validation set

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)
df_dig = pd.read_csv(data_dig_file)


# In[26]:


df_train.head()


# In[ ]:




