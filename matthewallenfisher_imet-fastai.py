#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import functools
import re
from fastai.vision import *

import seaborn as sns
sns.set(style="whitegrid")
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# First we load the csv files
path = Path('/kaggle/input/imet-2020-fgvc7/')
labels_df = pd.read_csv(path/'labels.csv')
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'sample_submission.csv')
train_df


# In[3]:


# Here we will visualize the number of tags per image
train_df["tag_count"] = train_df["attribute_ids"].apply(lambda x:len(x.split(' ')))
sns.countplot(x="tag_count",data=train_df,palette="Reds",log=True)
plt.ylabel('Number of images')
plt.xlabel('Tag Count')
plt.title('Tag Count per Image')
sns.despine()


# In[4]:


TAG_COUNTS = train_df['tag_count'].value_counts().reset_index().sort_values(by=['index']).set_index('index').style.background_gradient(cmap="cividis")
TAG_COUNTS


# In[5]:


# We are now replacing the attribute_type::attrbiute_value format in attribute_name with the two columns seperated, and deleting the original 
labels_df["attribute_type"]=labels_df["attribute_name"].apply(lambda x:x.split("::")[0])
labels_df["attribute_value"]=labels_df["attribute_name"].apply(lambda x:x.split("::")[1])
labels_df.drop("attribute_name",1,inplace=True)
labels_df


# In[6]:


# See which different types of attributes we have
unique_attributes = labels_df.attribute_type.unique()
unique_attributes
# Check range in indexes for each attribute type
print("Index Ranges For Each Attribute Type")
for attribute in unique_attributes:
    all_matches = labels_df.loc[labels_df['attribute_type']==attribute]
    print(attribute, ": ",all_matches.min().attribute_id,"-",all_matches.max().attribute_id)


# In[7]:


np.random.seed(69)
tfms = get_transforms()
data = (ImageList.from_csv(path, 'train.csv', folder='train', suffix='.png')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' ', cols="attribute_ids")
       .transform(tfms, size=128)
       .databunch(bs=32)
       .normalize(imagenet_stats))


# In[8]:


data.show_batch(3, figsize=(12,12))


# In[9]:


copy pretrained weights for resnet50 to the folder fastai will search by default
Path('/root/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/root/.cache/torch/checkpoints/resnet50-19c8e357.pth'")
arch = models.resnet50
acc_02 = partial(accuracy_thresh,thresh=0.2)
f_score = partial(fbeta,thresh=0.2)
learn=cnn_learner(data,arch,metrics=[acc_02,f_score], model_dir="/kaggle", pretrained=True)


# In[10]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:





# In[11]:


lr = 0.01
learn.fit_one_cycle(5,slice(lr))
learn.save('stage-1-rn50')
learn.export('resnet50_imet')


# In[12]:


# learn = load_learner('/kaggle/input/models/', 'resnet50_imet_2_f0.399.pkl')
# learn


# In[13]:


learn.data.add_test(ImageList.from_df(test_df,path,folder='test',suffix='.png'))


# In[14]:


preds,y = learn.get_preds(DatasetType.Test)


# In[15]:


pd.DataFrame(preds.numpy()).to_csv('preds.csv', index=False)


# In[16]:


# Use this one when doing predictions on a new model
preds_df = pd.DataFrame(preds.numpy())

# Use this one to load predictions calculated from a past model
# preds_df = pd.read_csv('/kaggle/input/imet-version17/preds.csv')
# preds_df


# In[17]:


def display_Predictions(image,display):
    #Selects top 20 attributes, display scores
    top20preds = preds_df.iloc[image].sort_values(ascending=False)[:20].reset_index()
    top20preds = top20preds.rename(columns={"index":"attribute_id",image:"acc_preds"})
    top20preds["attribute_type"] = top20preds["attribute_id"].apply(lambda x : labels_df.iloc[int(x)]["attribute_type"])
    top20preds["attribute_value"] = top20preds["attribute_id"].apply(lambda x : labels_df.iloc[int(x)]["attribute_value"])
    top20preds.reindex(columns=["attribute_id","acc_preds","attribute_type","attribute_value"])
    if display:
        image_path = path/'test'/(test_df.iloc[image].id+'.png')
        img = plt.imread(str(image_path))
        plt.imshow(img)
    return top20preds
display_Predictions(11, display=True).style.background_gradient(cmap='cividis')


# In[18]:


for img_index in range(len(preds_df)):
    values = display_Predictions(img_index,display=False)['acc_preds'].round(2)


# In[19]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
df = pd.DataFrame({'id':test_df['id'], 'attribute_ids':labelled_preds})


# In[20]:


df.to_csv('submission.csv', index=False)

