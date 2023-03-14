#!/usr/bin/env python
# coding: utf-8

# In[54]:


import re
import os
import cv2
import csv              
import random
import numpy as np
import pandas as pd
import seaborn as sns
from PIL  import  Image
import matplotlib.pyplot as plt


# In[55]:


train=pd.read_csv('../input/global-wheat-detection/train.csv')
sample_output = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')
train_dir = '../input/global-wheat-detection/train/'
test_dir ='../input/global-wheat-detection/test/'


# In[56]:


train.head()


# In[57]:


df = pd.read_csv('../input/global-wheat-detection/train.csv')
bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['xmin', 'ymin', 'w', 'h']):
    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=True)
df['xmax'] = df['xmin'] + df['w']
df['ymax'] = df['ymin'] + df['h']
# df['x_center'] = df['xmin'] + df['w']/2
# df['y_center'] = df['ymin'] + df['h']/2

df['classes'] = 0
df = df[['image_id','xmin', 'ymin','xmax', 'ymax', 'w', 'h','classes']]
df.head()


# In[58]:


# exporting cleaned dataframe
df.to_csv("data.csv")


# In[59]:


index = list(set(df.image_id))
# print(index)


# In[67]:


import shutil
try:
    shutil.copytree('../input/global-wheat-detection/train','./images','./images')
except:
  print("error or maybe file Already exsist")


# In[68]:


from pathlib import Path
Path("./annots").mkdir(parents=True, exist_ok=True)


# In[69]:


f = open('./data.csv')
csv_f = csv.reader(f)   
data = []

for row in csv_f: 
   data.append(row)
f.close()
print(data[:4])


# In[70]:


import xml.etree.ElementTree as gfg 
f = open('./data.csv')
csv_f = csv.reader(f)   
data = []

for row in csv_f: 
   data.append(row)
f.close()

def GenerateXML(annots_dir,img_dir,image_id) :
    fileName = image_id+".xml"
    root = gfg.Element("annotation") 

    b1 = gfg.SubElement(root, "folder")
    b1.text = "wheat_heads"
    b2 = gfg.SubElement(root, "filename") 
    b2.text = image_id+".jpg"
    b3 = gfg.SubElement(root, "path") 
    b3.text = img_dir+image_id+".jpg"
    
    m3 = gfg.Element("size") 
    root.append (m3) 
    
    d1 = gfg.SubElement(m3, "width") 
    d1.text = "1024"
    d2 = gfg.SubElement(m3, "height") 
    d2.text = "1024"
    d3 = gfg.SubElement(m3, "depth") 
    d3.text = "3"

    for row in data[1:] :
        if row[1]== image_id:

            m4 = gfg.Element("object") 
            root.append (m4) 
            
            d1 = gfg.SubElement(m4, "name") 
            d1.text = "wheat_head"
            
            m5=gfg.Element("bndbox")
            m4.append(m5)
            d1 = gfg.SubElement(m5,"xmin")
            d1.text= str(int(float(row[2])))
            d2 = gfg.SubElement(m5,"ymin")
            d2.text= str(int(float(row[3])))
            d3 = gfg.SubElement(m5,"xmax")
            d3.text= str(int(float(row[4])))
            d4 = gfg.SubElement(m5,"ymax")
            d4.text= str(int(float(row[5])))

    tree = gfg.ElementTree(root) 
    
    with open (annots_dir+fileName, "wb") as files :
        tree.write(files)


# In[73]:


# os.listdir('./images')


# In[72]:


#wait for a long time
for image_id in index:
    GenerateXML("./annots/","./images/", image_id) 


# In[74]:


## count xml files
# os.listdir('./annots')

i=0
for j in (os.listdir('./annots')):
    i+=1
print(i)


# In[75]:


from PIL  import  Image
img = Image.open('./images/7f01525b1.jpg')
plt.imshow(img)


# In[76]:


get_ipython().system('git clone https://github.com/mmaithani/Mask_RCNN.git')


# In[ ]:


cd Mask_RCNN


# In[ ]:


get_ipython().system('python setup.py install')


# In[77]:


get_ipython().system('pip install imutils')
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
get_ipython().run_line_magic('matplotlib', 'inline')
from os import listdir
from xml.etree import ElementTree


# In[78]:


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
     # kangaroo + BG
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 131
    
    # Learning rate
    LEARNING_RATE=0.006
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10


# In[79]:


config = myMaskRCNNConfig()


# In[80]:


config.display()


# In[ ]:


class WheatData(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "wheat_head")
        
        # define data locations for images and annotations
        images_dir = './images/'
        annotations_dir = './annots/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            image_id = filename[:-4]
            
#             # skip bad images
#             if image_id in ['00090']:
#                 continue
#             # skip all images after 150 if we are building the train set
#             if is_train and int(image_id) >= 150:
#                 continue

           # setting image file
            img_path = images_dir + filename
            
            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
# load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, w, h = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')
# load an image reference
#      """Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


# In[ ]:


cd ..


# In[ ]:


os.getcwd()


# In[ ]:


os.listdir(train_dir)


# In[ ]:


# prepare train set
train_set = WheatData()
train_set.load_dataset("./", is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_id))

# prepare test/val set
test_set = WheatData()
test_set.load_dataset("./", is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_id))


# In[ ]:


print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')


# In[ ]:


os.listdir('./Mask_RCNN')


# In[ ]:


#load the weights for COCO
model.load_weights('.\\Mask_RCNN\\mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


# In[ ]:


## train heads with higher lr to speedup the learning
model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=5, layers=’heads’)
history = model.keras_model.history.history

