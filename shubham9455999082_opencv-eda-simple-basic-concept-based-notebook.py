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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


sub=pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')
train_df=pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')


# In[3]:


sub.head


# In[ ]:





# In[4]:


from tqdm import tqdm
import ast


# In[5]:



# How many unique images?
len(train_df["image_id"].unique())


# In[6]:


# Separating out the coordinates
xmin, ymin, width, height = [], [], [], []

for i in tqdm(train_df["bbox"]):
    cooridinates_list = ast.literal_eval(i)
    xmin.append(cooridinates_list[0])
    ymin.append(cooridinates_list[1])
    width.append(cooridinates_list[2])
    height.append(cooridinates_list[3])


# In[7]:


train_df["xmin"] = xmin
train_df["ymin"] = ymin
train_df["width"] = width
train_df["height"] = height
train_df.head()


# In[13]:


# Visualizing some samples from the training set

sample_indices = np.random.choice(np.unique(train_df["image_id"].tolist()), 8)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
count=0

for row in ax:
    for col in row:
        img = plt.imread("/kaggle/input/global-wheat-detection/train/" + sample_indices[count] + ".jpg")
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(img)
        count += 1
plt.show()


# In[14]:


import matplotlib.patches as patches

def get_bbox(image_id, df, col, color='white'):
    bboxes = df[df['image_id'] == image_id]
    
    for i in range(len(bboxes)):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bboxes['xmin'].iloc[i], bboxes['ymin'].iloc[i]),
            bboxes['width'].iloc[i], 
            bboxes['height'].iloc[i], 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none')

        # Add the patch to the Axes
        col.add_patch(rect)


# In[15]:


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
count=0
for row in ax:
    for col in row:
        img = plt.imread("/kaggle/input/global-wheat-detection/train/" + sample_indices[count] + ".jpg")
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        get_bbox(sample_indices[count], train_df, col, color='red')
        col.imshow(img)
        count += 1
plt.show()


# In[16]:


# Images without bounding box
images_w_bbox = train_df["image_id"].unique()
images_w_bbox = ["/kaggle/input/global-wheat-detection/train/" + image_id + ".jpg" for image_id in images_w_bbox]

all_images = list(paths.list_images("/kaggle/input/global-wheat-detection/train/"))


# In[17]:


images_w_bbox[:5]


# In[18]:


all_images[:5]


# In[19]:


ax


# In[20]:


# Visualizing some images without any wheat heads

'''fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
count=0

for row in ax:
    for col in row:
        img = plt.imread(images_wo_bbox[count])
        col.grid(False)
        col.set_xticks([])
        col.set_yticks([])
        col.imshow(img)
        count += 1
plt.show()'''


# In[21]:


images_wo_bbox = list(set(all_images) - set(images_w_bbox))
images_wo_bbox[:5]


# In[ ]:





# In[12]:


import matplotlib.patches as patches
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import numpy as np 
import cv2
import os


# In[10]:


pip install imutils


# In[22]:


def showbbox(image_path, xy, width, height):
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(plt.imread(image_path))

    # Create a Rectangle patch
    rect = patches.Rectangle(xy, width, height, 
        linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


# In[23]:


train_df


# In[24]:


showbbox('/kaggle/input/global-wheat-detection/train/b6ab77fd7.jpg', (226.0, 548.0), 1024, 1024)


# In[29]:


p=[]
for i in (train_df['bbox']):
    d=i.split(',')
    p.append(d)
               


# In[26]:


p


# In[27]:


d=[]
for i in range(0,len(p)):
    s=p[i][0].strip('[')
    #s.strip('[')
    d.append(s)


# In[28]:


dd=[]
for i in range(0,len(p)):
    s1=p[i][1].strip('[')
    #s.strip('[')
    dd.append(s1)


# In[ ]:





# In[ ]:


#train['xmin']=pd.Series(d)
#train['ymin']=pd.Series(dd)


# In[30]:


train_df


# In[ ]:


#train['xmin']=train['xmin'].astype('float')
#train['ymin']=train['ymin'].astype('float')


# In[31]:


train_df["xmax"] = train_df["xmin"] + train_df["width"]
train_df["ymax"] = train_df["ymin"] + train_df["height"]
train_df.head()


# In[33]:


# Rename the image_id column to filename & add full paths
train_df.rename(columns={
        "image_id":"filename"
    }, inplace=True)

images_w_bbox = train_df["filename"]
images_w_bbox = ["/kaggle/input/global-wheat-detection/train/" + image_id + ".jpg" for image_id in images_w_bbox]
train_df["filename"] = images_w_bbox
train_df.head()


# In[34]:


# Drop the unnecessary columns, we will return to this step in a moment
train_df.drop(["source", "bbox"], axis=1, inplace=True)
train_df.head()


# In[35]:


import math
math.floor(2.3)


# In[36]:


# Add a class column 
train_df["class"] = "wheat_head"
train_df.head()


# In[37]:


# Prepare the splits
from sklearn.model_selection import train_test_split

train1, valid = train_test_split(train_df, test_size=0.15, random_state=666)
print("Training samples:", train1.shape[0])
print("Validation samples:", valid.shape[0])


# In[38]:



train1 = train1.reset_index(drop=True)
train1.head()


# In[39]:



valid = valid.reset_index(drop=True)
valid.head()


# In[40]:


# Serialize the dataframes
train1.to_csv("new_train_df.csv")
valid.to_csv("valid_df.csv")


# In[41]:



# Preparing the label maps
LABEL_ENCODINGS = {
    "wheat_head": 1
}

f = open("label_map.pbtxt", "w")

for (k, v) in LABEL_ENCODINGS.items():
    # construct the class information and write to file
    item = ("item {\n"
            "\tid: " + str(v) + "\n"
            "\tname: '" + k + "'\n"
            "}\n")
    f.write(item)

# close the output classes file
f.close()


# In[42]:


get_ipython().system('cat label_map.pbtxt')


# In[ ]:





# In[43]:


'''from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
#from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

LABEL_ENCODINGS = {
    "wheat_head": 1
}


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
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
        classes.append(LABEL_ENCODINGS[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd())
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()'''


# In[ ]:





# In[ ]:




