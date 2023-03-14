#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# In[2]:


pip install 'git+https://github.com/facebookresearch/detectron2.git'


# In[3]:


import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(0))
else:
    device = torch.device("cpu")

print("-> Loading model")
cfg = get_cfg()
cfg.merge_from_file("../input/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.DEVICE = str(device)
cfg.MODEL.RPN.NMS_THRESH = 0.1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = "../input/parameters/model_final_f10217.pkl"

model = DefaultPredictor(cfg)


# In[4]:


import PIL.Image as Image

from torchvision import transforms

default_transform = transforms.Compose([transforms.ToTensor()])

def load_image(path, transform=default_transform):
    image = Image.open(path)
    return transform(image)


# In[5]:


image_path = '../input/pku-autonomous-driving/train_images/ID_7f6f07350.jpg'


# In[6]:


import cv2

image = cv2.imread(image_path)
outputs = model(image)


# In[7]:


from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8, instance_mode=ColorMode.IMAGE_BW)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
v = v.get_image()[:, :, ::-1]

plt.imshow(v)
plt.show()


# In[8]:


mask = outputs["instances"].pred_masks.sum(0) > 0


# In[9]:


import numpy as np

mask = torch.stack([mask, mask, mask], dim=2)
mask = mask.cpu().numpy().astype("uint8")

instances = cv2.multiply(image, mask)
plt.imshow(instances)
plt.show()


# In[10]:


import os
import cv2
import pdb
import glob
import argparse

import numpy as np


# In[11]:


def make_multi_channel_masks(source_dir='../input/pku-autonomous-driving/train_images',
                dist_dir='../input/pku-autonomous-driving/train_images_mask',
                ext='jpg'):
    """Function to predict for a single image or folder of images
    """

    # FINDING INPUT IMAGES
    if os.path.isdir(source_dir):
        # Searching folder for images
        paths = glob.glob(os.path.join(source_dir, '*.{}'.format(ext)))
        output_directory = dist_dir
    else:
        raise Exception("Can not find source_dir: {}".format(source_dir))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    print("-> Predicting on {:d} test images".format(len(paths)))

    for idx, image_path in enumerate(paths):
        image = cv2.imread(image_path)
        outputs = model(image)

        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(output_directory, "{}.npy".format(output_name))
        mask = outputs['instances'].pred_masks.cpu().numpy()
        np.save(name_dest_npy, mask)

        print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_npy))

    print('-> Done!')


# In[12]:


def make_single_channel_masks(source_dir='../input/pku-autonomous-driving/train_images',
                dist_dir='../input/pku-autonomous-driving/train_images_mask',
                ext='jpg'):
    """Function to predict for a single image or folder of images
    """

    # FINDING INPUT IMAGES
    if os.path.isdir(source_dir):
        # Searching folder for images
        paths = glob.glob(os.path.join(source_dir, '*.{}'.format(ext)))
        output_directory = dist_dir
    else:
        raise Exception("Can not find source_dir: {}".format(source_dir))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    print("-> Predicting on {:d} test images".format(len(paths)))

    for idx, image_path in enumerate(paths):
        image = cv2.imread(image_path)
        outputs = model(image)

        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(output_directory, "{}.npy".format(output_name))
        mask = outputs["instances"].pred_masks.sum(0) > 0
        mask = mask.float().unsqueeze(0)
        mask = mask.cpu().numpy()
        np.save(name_dest_npy, mask)

        print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_npy))

    print('-> Done!')


# In[13]:


def make_instances(source_dir='../input/pku-autonomous-driving/train_images',
                dist_dir='../input/pku-autonomous-driving/train_images_mask',
                ext='jpg'):
    """Function to predict for a single image or folder of images
    """

    # FINDING INPUT IMAGES
    if os.path.isdir(source_dir):
        # Searching folder for images
        paths = glob.glob(os.path.join(source_dir, '*.{}'.format(ext)))
        output_directory = dist_dir
    else:
        raise Exception("Can not find source_dir: {}".format(source_dir))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    print("-> Predicting on {:d} test images".format(len(paths)))

    for idx, image_path in enumerate(paths):
        image = cv2.imread(image_path)
        outputs = model(image)

        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_jpg = os.path.join(output_directory, "{}.jpg".format(output_name))
        mask = outputs["instances"].pred_masks.sum(0) > 0
        mask = torch.stack([mask, mask, mask], dim=2)
        mask = mask.cpu().numpy().astype("uint8")

        instances = cv2.multiply(image, mask)
        cv2.imwrite(name_dest_jpg, instances)

        print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_jpg))

    print('-> Done!')


# In[14]:


# make_multi_channel_masks()
# make_single_channel_masks()
# make_instances()

