#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch, torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import pydicom
from collections import Counter
import tqdm
os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection')

# Any results you write to the current directory are saved as output.


# In[2]:


input_path = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
df_train = pd.read_csv(input_path + 'stage_1_train.csv')
print(len(df_train))
df_train.head(10)


# In[3]:


df_test = pd.read_csv(input_path + 'stage_1_sample_submission.csv')
print(len(df_test))
df_test.head(10)


# In[4]:


ds_columns = ['ID',
              'PatientID',
              'Modality',
              'StudyInstance',
              'SeriesInstance',
                'PhotoInterpretation',
              'Position0', 'Position1', 'Position2',
              'Orientation0', 'Orientation1', 'Orientation2', 'Orientation3', 'Orientation4', 'Orientation5',
              'PixelSpacing0', 'PixelSpacing1']
def extract_dicom_features(ds):
    
    ds_items = [ds.SOPInstanceUID,
                ds.PatientID,
                ds.Modality,
                ds.StudyInstanceUID,
                ds.SeriesInstanceUID,
                ds.PhotometricInterpretation,
                ds.ImagePositionPatient,
                ds.ImageOrientationPatient,
                ds.PixelSpacing]

    line = []
    for item in ds_items:
        if type(item) is pydicom.multival.MultiValue:
            line += [float(x) for x in item]
        else:
            line.append(item)

    return line


# In[5]:


list_img = os.listdir(input_path + 'stage_1_test_images')
print(len(list_img))
df_features = []
for img in tqdm.tqdm(list_img):
    img_path = input_path + 'stage_1_test_images/' + img
    ds = pydicom.read_file(img_path)
    df_features.append(extract_dicom_features(ds))
df_features_test = pd.DataFrame(df_features, columns=ds_columns)


# In[6]:


list_img = os.listdir(input_path + 'stage_1_train_images')
print(len(list_img))
df_features = []
for img in tqdm.tqdm(list_img):
    img_path = input_path + 'stage_1_train_images/' + img
    ds = pydicom.read_file(img_path)
    df_features.append(extract_dicom_features(ds))
df_features_train = pd.DataFrame(df_features, columns=ds_columns)


# In[7]:


df_train[['ID', 'Subtype']] = df_train['ID'].str.rsplit(pat='_', n=1, expand=True)
df_train_new = df_train.pivot_table(index='ID', columns='Subtype').reset_index()
df_train_new.columns=['ID','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
df_train_merged = df_train_new.merge(df_features_train, how='right')


# In[8]:


df_test[['ID', 'Subtype']] = df_test['ID'].str.rsplit(pat='_', n=1, expand=True)
df_test_new = df_test.pivot_table(index='ID', columns='Subtype').reset_index()
df_test_new.columns=['ID','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
df_test_merged = df_test_new.merge(df_features_test, how='right')


# In[9]:


df_train_merged.to_csv('df_train_merged.csv', index=False)
df_test_merged.to_csv('df_test_merged.csv', index=False)


# In[10]:


for df in [df_train_merged, df_test_merged]:
    df['max_Position2']=0
    df['min_Position2']=0
    df['relative_Position2']=0
    
    for SeriesInstance in tqdm.tqdm(df["SeriesInstance"].unique()):
        sub_df=df[df['SeriesInstance']==SeriesInstance]
        df.loc[df['SeriesInstance']==SeriesInstance, 'max_Position2'] = max(sub_df['Position2'])
        df.loc[df['SeriesInstance']==SeriesInstance, 'min_Position2'] = min(sub_df['Position2'])

    df['relative_Position2'] = (df['Position2']-df['min_Position2'])/(df['max_Position2']-df['min_Position2'])


# In[11]:


df_train_merged=df_train_merged.sort_values(by=["PatientID","SeriesInstance",'Position2'])
df_train_merged.to_csv('df_train_merged.csv', index=False)
df_test_merged=df_test_merged.sort_values(by=["PatientID","SeriesInstance",'Position2'])
df_test_merged.to_csv('df_test_merged.csv', index=False)


# In[12]:


z_normal = df_train_merged[df_train_merged["any"] != 1]['relative_Position2']
z_epidural = df_train_merged[df_train_merged["epidural"] == 1]['relative_Position2']
z_subdural = df_train_merged[df_train_merged["subdural"] == 1]['relative_Position2']
z_intraventricular = df_train_merged[df_train_merged["intraventricular"] == 1]['relative_Position2']
z_intraparenchymal = df_train_merged[df_train_merged["intraparenchymal"] == 1]['relative_Position2']
z_subarachnoid = df_train_merged[df_train_merged["subarachnoid"] == 1]['relative_Position2']
z_any = df_train_merged[df_train_merged["any"] == 1]['relative_Position2']

z_describe=pd.DataFrame({'normal': z_normal.describe(),
                   'epidural': z_epidural.describe(),
                   'subdural': z_subdural.describe(),
                   'intraventricular': z_intraventricular.describe(),
                   'intraparenchymal': z_intraparenchymal.describe(),
                   'subarachnoid': z_subarachnoid.describe(),
                   'any': z_any.describe()})
print(z_describe)
bin_width=0.05
kwargs = dict(histtype='stepfilled', normed=False, bins=np.arange(0, 1+bin_width, bin_width))
fig, axes = plt.subplots(4, 2, figsize=(8, 16))
axes[0, 0].hist(z_normal, **kwargs)
axes[0, 0].set_title('normal')
axes[0, 1].hist(z_epidural, **kwargs) 
axes[0, 1].set_title('epidural')
axes[1, 0].hist(z_subdural, **kwargs) 
axes[1, 0].set_title('subdural')
axes[1, 1].hist(z_intraventricular, **kwargs) 
axes[1, 1].set_title('intraventricular')
axes[2, 0].hist(z_intraparenchymal, **kwargs) 
axes[2, 0].set_title('intraparenchymal')
axes[2, 1].hist(z_subarachnoid, **kwargs) 
axes[2, 1].set_title('subarachnoid')
axes[3, 0].hist(z_any, **kwargs) 
axes[3, 0].set_title('any')
plt.show()


# In[13]:


print(df_train_merged[df_train_merged["SeriesInstance"] == 'ID_2f48a87008'])

