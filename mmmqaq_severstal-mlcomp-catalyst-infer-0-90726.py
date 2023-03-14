#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('ls ../input')
import sys
sys.path.append('../input/resnet32-classifier-v5')
from resnet32_classifier import run_submit


# In[2]:


img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
run_submit('/kaggle/working', img_folder)


# In[3]:


import pandas as pd
df_label = pd.read_csv('./resnet34-cls-tta-0.50_test.csv').fillna('')
df_label[:16]
print(len(df_label))
get_ipython().system('ls -la /kaggle/input/severstal-steel-defect-detection/test_images | wc')
(1804-3)*4


# In[4]:


get_ipython().system(' ls ../input/severstalmodels')


# In[5]:


get_ipython().system(' python ../input/mlcomp/mlcomp/mlcomp/setup.py')


# In[6]:


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


# In[7]:


unet_se_resnext50_32x4d =     load('/kaggle/input/severstalmodels/unet_se_resnext50_32x4d.pth').cuda()
unet_mobilenet2 = load('/kaggle/input/severstalmodels/unet_mobilenet2.pth').cuda()
unet_resnet34 = load('/kaggle/input/severstalmodels/unet_resnet34.pth').cuda()


# In[8]:


class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

model = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])


# In[9]:


def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.230, 0.225, 0.223)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
batch_size = 2
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]


# In[10]:


thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [600, 600, 1000, 2000]

res = []
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
        features = batch['features'].cuda()
        p = torch.sigmoid(model(features))
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch['image_file']
    
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()
    
    # Batch post processing
    for p, file in zip(preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            imageid_classid = file+'_'+str(i+1)
            p_channel = (p_channel>thresholds[i]).astype(np.uint8)
            
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })


# In[11]:


get_ipython().system(' ls ../input/classification-results')


# In[12]:


df_mask = pd.DataFrame(res)
df_mask = df_mask.fillna('')

df_label_1 = pd.read_csv('./resnet34-cls-tta-0.50_test.csv').fillna('').sort_values(by=['ImageId_ClassId'])
#df_label_2 = pd.read_csv('../input/classification-results/resnet34-cls-tta-0.50.csv').fillna('').sort_values(by=['ImageId_ClassId'])

fp1 = list(df_label_1['EncodedPixels']=='')
#fp2 = list(df_label_2['EncodedPixels']=='')

assert(np.all(df_mask['ImageId_ClassId'].values == df_label_1['ImageId_ClassId'].values))

print("How many predictions in the segementation result will be removed?")
print((df_mask.loc[fp1,'EncodedPixels'] != '').sum() )
print("How many postive predictions in the segementation?")
print((df_mask['EncodedPixels'] != '').sum() )
print("How many postive predictions in the classification?")
print((df_label_1['EncodedPixels'] != '').sum() )

df_mask.loc[fp1,'EncodedPixels']=''
df_mask.to_csv('submission.csv', index=False)

df = df_mask


# In[13]:


df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
df['empty'] = df['EncodedPixels'].map(lambda x: not x)
df[df['empty'] == False]['Class'].value_counts()


# In[14]:


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




