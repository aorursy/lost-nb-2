#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' python ../input/mlcomp/mlcomp/setup.py')


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap


# In[3]:


unet = load('../input/severstaleffnet/traced_effnetb7_mixup_retrain.pth').cuda()
cls = load('../input/severstall-effnetb0-fimal-stage/traced_model.pth').cuda()
cls_alt = load('../input/severstaleffnet/traced_effnetb0_lovasz.pth').cuda()


# In[4]:


def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
batch_size = 1
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)],
    [A.VerticalFlip(p=1)],
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]


# In[5]:


class Classifier:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, loaders_batch):
        with torch.no_grad():
            preds = []
            image_file = []
            for i, batch in enumerate(loaders_batch):
                features = batch['features'].cuda()
                pred_raw, _ = model(features)
                p = torch.sigmoid(pred_raw)
                image_file = batch['image_file']

                # inverse operations for TTA
                p = datasets[i].inverse(p)
                preds.append(p)

            # TTA mean
            preds = torch.stack(preds)
            preds = torch.mean(preds, dim=0)
            preds = preds.detach().cpu().numpy()

            # Batch post processing
            p_img = []
            for p, file in zip(preds, image_file):
                file = os.path.basename(file)
                # Image postprocessing
                for i in range(4):
                    p_channel = p[i]
                    p_channel = (p_channel>thresholds[i]).astype(np.uint8)
                    if p_channel.sum() < min_area[i]:
                        p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
                    p_img.append(p_channel)
        return p_img


class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        labels = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                masks, label = m(x)
                res.append(masks)
                labels.append(label)
        res = torch.stack(res)
        labels = torch.stack(labels)
        return torch.mean(res, dim=0), torch.mean(labels, dim=0)

model = cls_alt
# cls2 = Classifier(cls_alt)


# In[6]:


import numpy as np
import torch
l = torch.tensor([1, 0])
if np.count_nonzero(l):
    print('yay')


# In[7]:


thresholds = [0.6, 0.99, 0.6, 0.6] # [0.5, 0.5, 0.5, 0.5] | [0.55, 0.55, 0.55, 0.55] | [0.6, 0.99, 0.6, 0.6]
min_area = [600, 600, 1000, 2000] # instead of 900 -> 1000 in old version

res = []
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
with torch.no_grad():
    for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
        preds = []
        image_file = []
        labels = []
        for i, batch in enumerate(loaders_batch):
            features = batch['features'].cuda()
            output = model(features)
            pred_raw, label = output
            labels.append(label)
            p = torch.sigmoid(pred_raw)
            image_file = batch['image_file']

            # inverse operations for TTA
            p = datasets[i].inverse(p)
            preds.append(p)
    
        # TTA mean
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        preds = preds.detach().cpu().numpy()
        labels = torch.stack(labels)
        labels = torch.mean(labels, dim=0)
        labels = labels.detach().cpu().numpy()  # has shape (1, 4)
        labels = labels[0] 
    
        # Batch post processing
        for p, file in zip(preds, image_file):
            file = os.path.basename(file)
            # Image postprocessing
            for i in range(4):
                p_channel = np.zeros((256, 1600), dtype=np.uint8)
                imageid_classid = file+'_'+str(i+1)
                if labels[i] > 0:
                    p_channel = p[i]
                    p_channel = (p_channel>thresholds[i]).astype(np.uint8)
                    if p_channel.sum() < min_area[i]:
                        p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

                res.append({
                    'ImageId_ClassId': imageid_classid,
                    'EncodedPixels': mask2rle(p_channel)
                })
        
df = pd.DataFrame(res)
df.to_csv('submission.csv', index=False)


# In[8]:


df = pd.DataFrame(res)
df = df.fillna('')
df.to_csv('submission.csv', index=False)


# In[9]:


df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
df['empty'] = df['EncodedPixels'].map(lambda x: not x)
df[df['empty'] == False]['Class'].value_counts()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('submission.csv')[:40]
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])

for row in df.itertuples():
    img_path = os.path.join(img_folder, row.Image)
    img = cv2.imread(img_path)
    mask = rle2mask(row.EncodedPixels, (1600, 256))         if isinstance(row.EncodedPixels, str) else np.zeros((256, 1600))
    if mask.sum() == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 60))
    axes[0].imshow(img/255)
    axes[1].imshow(mask*60)
    axes[0].set_title(row.Image)
    axes[1].set_title(row.Class)
    plt.show()


# In[ ]:




