#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
# #         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


train_data_df=pd.read_csv("../input/global-wheat-detection/train.csv")
train_data_df.head()


# In[4]:


image_id=[f"{i}.jpg" for i in train_data_df.image_id]
xmins,ymins,xmaxs,ymaxs=[],[],[],[]
for bbox in train_data_df.bbox:
    real_bbox=eval(bbox)
    
    xmin, ymin ,w ,h=real_bbox
    
    
    
    a=int(xmin+w)
    b=int(ymin+h)
    xmaxs.append(a)
    ymaxs.append(b)

    
    c=int(xmin)
    d=int(ymin)
    xmins.append(c)
    ymins.append(d)


# In[5]:


data=pd.DataFrame()
data["filename"]=image_id
data["width"]=train_data_df.width
data["width"]=train_data_df.height

data["class"]=["wheat"]*len(image_id)

data["xmin"]=xmins
data["ymin"]=ymins

data["xmax"]=xmaxs
data["ymax"]=ymaxs


# In[6]:


data.head()


# In[7]:


data.to_csv("train_labels.csv",index=False)


# In[8]:


pd.read_csv("/kaggle/working/train_labels.csv")


# In[9]:


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# In[10]:


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from collections import namedtuple, OrderedDict


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'wheat':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id':bytes_feature(filename),
        'image/encoded':bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text':bytes_list_feature(classes_text),
        'image/object/class/label':int64_list_feature(classes),
    }))
    return tf_example


def main(csv_input, output_path, image_dir):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    csv_input="train_labels.csv"
    output_path="train_label.record"
    image_dir="/kaggle/input/global-wheat-detection/train"
    main(csv_input, output_path, image_dir)


# In[11]:


cd "/kaggle/input/kerasversionefficientdet"


# In[12]:


os.mkdir("/kaggle/working/model")


# In[13]:


get_ipython().system('pip install pycocotools')


# In[14]:


# uncomment for training
# !python main.py --mode=train --training_file_pattern=train.record --model_name=efficientdet-d3 --model_dir=/kaggle/working/model --model_name=efficientdet-d3 --ckpt=/kaggle/input/effiecientdetd3-10k-epoch-checkpoints --train_batch_size=4 --num_epochs=3000 --num_examples_per_epoch=16

