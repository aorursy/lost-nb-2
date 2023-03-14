#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# In[2]:


TRAIN_CSV = '/kaggle/input/human-protein-atlas-train-val-split/train_df.csv'

TRAIN_DIR = '/kaggle/input/human-protein-atlas-image-classification/train'


# In[3]:


train_df = pd.read_csv(TRAIN_CSV)


# In[4]:


FILTERS = ['red', 'green', 'blue', 'yellow']


# In[5]:


def load_image(image_id, ddir):
    """
    return: 4-channel PIL Image
    """
    return Image.merge('RGBA', [Image.open(f"{TRAIN_DIR}/{image_id}_{f}.png") for f in FILTERS])


class ProteinLocalizationDataset(Dataset):
    def __init__(self, df, ddir, transform=None):
        self.df = df
        self.ddir = ddir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id = self.df.loc[idx]['Id']
        image = load_image(image_id, self.ddir)
        if self.transform:
            image = self.transform(image)
        return image


# In[6]:


train_ds = ProteinLocalizationDataset(train_df, TRAIN_DIR, transform=T.ToTensor())
train_dl = DataLoader(train_ds, batch_size=16)


# In[7]:


def get_stats(dl):
    
    # followed the calculations in http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    
    mean = 0
    var = 0
    size = 0

    for imgs in dl:
        # imgs.shape: (batch size, color channel, H, W)
        b_mean = torch.mean(imgs, [0, 2, 3])
        b_var = torch.var(imgs, [0, 2, 3])
        b_size = imgs.size(0)

        new_mean = size / (size + b_size) * mean                    + b_size / (size + b_size) * b_mean

        new_var = size / (size + b_size) * var                   + b_size / (size + b_size) * b_var                   + (size * b_size) / (size + b_size) ** 2 * (mean - b_mean) ** 2

        mean = new_mean
        var = new_var
        size += b_size
    return (mean, torch.sqrt(var))


# In[8]:


get_ipython().run_cell_magic('time', '', 'stats = get_stats(train_dl)\nstats')


# In[9]:


torch.save(stats, 'stats.pt')


# In[10]:


# torch.load('stats.pt')

