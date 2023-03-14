#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pydicom


# In[2]:


# Get directory names/locations
data_root = os.path.abspath("../input/rsna-intracranial-hemorrhage-detection/")

train_img_root = data_root + "/stage_1_train_images/"
test_img_root  = data_root + "/stage_1_test_images/"

train_labels_path = data_root + "/stage_1_train.csv"
test_labels_path  = data_root + "/stage_1_test.csv"

# Create list of paths to actual training data
train_img_paths = os.listdir(train_img_root)
test_img_paths  = os.listdir(test_img_root)

# Dataset size
num_train = len(train_img_paths)
num_test  = len(test_img_paths)


# In[3]:


def create_efficient_df(data_path):
    
    # Define the datatypes we're going to use
    final_types = {
        "ID": "str",
        "Label": "float16"
    }
    features = list(final_types.keys())
    
    # Use chunks to import the data so that less efficient machines can only use a 
    # specific amount of chunks on import
    df_list = []

    chunksize = 1_000_000

    for df_chunk in pd.read_csv(data_path, dtype=final_types, chunksize=chunksize): 
        df_list.append(df_chunk)
        
    df = pd.concat(df_list)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    del df_list

    return df

train_labels_df = create_efficient_df(train_labels_path)
train_labels_df[train_labels_df["Label"] > 0].head()


# In[4]:


hem_types = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any"
]

new_cols = [
    "id",
    "type_0",
    "type_1",
    "type_2",
    "type_3",
    "type_4",
    "type_5"
]

num_ids = int(train_labels_df.shape[0] / len(hem_types))
print("Number of unique patient IDs: {}".format(num_ids))

empty_array = np.ones((num_ids, len(new_cols)))
hem_df = pd.DataFrame(data=empty_array, columns=new_cols)

# Fill in the ID of each image
hem_df["id"] = list(train_labels_df.iloc[::len(hem_types)]["ID"].str.split(pat="_").str[1])
    
# Fill in the categorical columns of each image
for hem_ix, hem_col in enumerate(list(hem_df)[1:]):
    hem_df[hem_col] = list(train_labels_df.iloc[hem_ix::len(hem_types), 1])
    
hem_df.info()
hem_df[hem_df["type_5"] > 0].head()


# In[5]:


def show_random_img(df, img_root):
    
    random_ix = random.randint(0, df.shape[0])
    random_record = df.iloc[random_ix, :]
    random_id = random_record[0]
    random_path = img_root + "ID_" + random_id + ".dcm"
    
    title = "Patient {}\nEpidural: {}\nIntraparenchymal: {}\nIntraventricular: {}\nSubarachnoid: {}\nSubdural: {}"        .format(random_id, random_record[1], random_record[2], random_record[3], random_record[4], random_record[5])
    
    dicom = pydicom.dcmread(random_path)
    img_array = dicom.pixel_array
    
    plt.imshow(img_array)
    plt.axis("off")
    plt.title(title)


# In[6]:


hem_counts = hem_df[new_cols[1:]].astype(bool).sum(axis=0)
hem_portions = [hem_counts[hem_type] / hem_counts[-1] for hem_type in range(6)]
hem_total_portions = [hem_counts[hem_type] / num_train for hem_type in range(6)]

# What percent chance is there that a patient has a hemorrhage at all?
print("\nProbability that a patient in the dataset has any of the 5 hemorrhage types:")
print("Number of patients with hemorrhage:  {}".format(hem_counts[-1]))
print("Total number of patients:           {}".format(num_train))
print("p(hemorrhage): %.2f%%" % (hem_counts[-1] / num_train * 100))

# Given that a patient has a hemorrhage what is the percent chance that it is each type of hemorrhage?
print("\nNumber of each type of hemorrhage found in dataset and share of total hemorrhages [p(hem_type | hemorrhage)]: ")
print("%21s | %7s | %9s" % ("hemorrhage type", "count", "portion"))
for hem_type in range(5):
    print("%21s | %7d | %8.2f%%" % (hem_types[hem_type], hem_counts[hem_type], hem_portions[hem_type] * 100))
    
# What is the chance of each type of hemorrhage?
print("\nNumber of each type of hemorrhage found in dataset and share of the raw total [p(hem_type)]: ")
print("%21s | %7s | %9s" % ("hemorrhage type", "count", "portion"))
for hem_type in range(5):
    print("%21s | %7d | %8.2f%%" % (hem_types[hem_type], hem_counts[hem_type], hem_total_portions[hem_type] * 100))


# In[7]:


# Set the certainty that we want to use to visualize each of our types of hemorrhage
CERTAINTY = 0.95


# In[8]:


epi_df = hem_df[hem_df["type_0"] > CERTAINTY]
show_random_img(epi_df, train_img_root)


# In[9]:


sub_df = hem_df[hem_df["type_4"] > CERTAINTY]
show_random_img(sub_df, train_img_root)


# In[10]:


iph_df = hem_df[hem_df["type_1"] > CERTAINTY]
show_random_img(iph_df, train_img_root)


# In[11]:


ivh_df = hem_df[hem_df["type_2"] > CERTAINTY]
show_random_img(ivh_df, train_img_root)


# In[12]:


sah_df = hem_df[hem_df["type_3"] > CERTAINTY]
show_random_img(sah_df, train_img_root)


# In[ ]:




