#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import cv2
import skimage.io
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")
def get_paths(index_location):
    index = os.listdir('../input/landmark-retrieval-2020/train/')
    paths = []
    a=index_location
    for b in index:
        for c in index:
            try:
                paths.extend([f"../input/landmark-retrieval-2020/train/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/train/{a}/{b}/{c}")])
            except:
                pass

    return paths

def show_sample(pathes):
    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(3, 3, figsize=(20, 20))
    axarr[0, 0].imshow(cv2.imread(pathes[0]))
    axarr[0, 1].imshow(cv2.imread(pathes[1]))
    axarr[0, 2].imshow(cv2.imread(pathes[2]))
    axarr[1, 0].imshow(cv2.imread(pathes[3]))
    axarr[1, 1].imshow(cv2.imread(pathes[4]))
    axarr[1, 2].imshow(cv2.imread(pathes[5]))
    axarr[2, 0].imshow(cv2.imread(pathes[6]))
    axarr[2, 1].imshow(cv2.imread(pathes[7]))
    axarr[2, 2].imshow(cv2.imread(pathes[8]))

show_sample(get_paths(2))
#train.describe()


# In[ ]:


# Landmark ID distribution
# from https://www.kaggle.com/machinesandi/google-landmark-retrieval-2020-extensive-eda/data
fig, axs = plt.subplots(ncols=3,figsize = (20, 5))
plt.title('Category Distribuition')
sns.kdeplot(train['landmark_id'], color="tomato", shade=True, ax=axs[0])

# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train.landmark_id.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']

plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,label="Count",ax=axs[1])

# Occurance of landmark_id in increasing order(Top categories)
temp = pd.DataFrame(train.landmark_id.value_counts().tail(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']

plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp, label="Count",ax=axs[2])
plt.show()


# In[ ]:


get_ipython().system('git clone https://github.com/tensorflow/models.git')
get_ipython().system('bash models/research/delf/delf/python/training/install_delf.sh')


# In[ ]:


#In case the installation was not successful:
get_ipython().system('cp -r models/research/delf/* ./')
get_ipython().system('protoc delf/protos/*.proto --python_out=.')


# In[ ]:


get_ipython().system('mkdir /tmp/data')
#!rm cleadn_data/*
#!ls -lh cleadn_data
get_ipython().system('python3 delf/python/training/build_image_dataset.py   --train_csv_path=../input/landmark-retrieval-2020/train.csv   --train_clean_csv_path=../input/train-clean-sample/train_clean_sample.csv \\#../input/cleaned-subsets-of-google-landmarks-v2/GLDv2_train_cleaned.csv \\')
  --train_directory=../input/landmark-retrieval-2020/train/*/*/*/   --output_directory=/tmp/data   --num_shards=12   --generate_train_validation_splits   --validation_split_size=0.2


# In[ ]:


get_ipython().system('curl -Os http://storage.googleapis.com/delf/resnet50_imagenet_weights.tar.gz')
get_ipython().system('tar -xzvf resnet50_imagenet_weights.tar.gz')


# In[ ]:


# add the delf to the pythonpath
os.environ['PYTHONPATH']+=':models/research/delf/:delf:protoc'
get_ipython().system('cp -r delf/protos/*  models/research/delf/delf/protos/')


# In[ ]:


# installing the object detection api, required by the delf model
get_ipython().system('pip install tensorflow-object-detection-api')


# In[ ]:


#-- dont forget to increase the number of iterations, default is max_iters=500.000
get_ipython().system('cp ../input/cleancode/train2.py models/research/delf/delf/python/training/train2.py')
get_ipython().system('python3 models/research/delf/delf/python/training/train2.py   --train_file_pattern=/tmp/data/train*   --seed=1   --max_iters=20000   --validation_file_pattern=/tmp/data/validation*   --imagenet_checkpoint=resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5   --dataset_version=gld_v2_clean   --logdir=gldv2_training/')


# In[ ]:





# In[ ]:


get_ipython().system('python3 models/research/delf/delf/python/training/model/export_global_model.py   --ckpt_path=gldv2_training/delf_weights   --export_path=gldv2_model_global   --input_scales_list=0.70710677,1.0,1.4142135   --multi_scale_pool_type=sum   --normalize_global_descriptor')

