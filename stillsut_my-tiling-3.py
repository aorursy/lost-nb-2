#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


get_user = os.environ.get('USER', 'KAGGLE')

if get_user == 'KAGGLE':
    my_env = 'KAGGLE'
elif get_user == 'jupyter':
    my_env = 'GCP'
elif get_user == 'user':
    my_env = 'LOCAL'
else:
    my_env = None
    
assert my_env is not None    

env_input_fn = {
    'KAGGLE': '../input/prostate-cancer-grade-assessment/',
    'LOCAL':  '../data/',
    'GCP':    '../../',
}

input_fn = env_input_fn[my_env]


# In[3]:


train_df = pd.read_csv(input_fn + 'train.csv')


# In[4]:


TRAIN = input_fn + 'train_images/'
MASKS = input_fn + 'train_label_masks/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'
sz = 128
N = 16
x_tot,x2_tot = [],[]


# In[5]:


def build_tiles(img, mask):

    shape = img.shape
    
    # padding allows even modulo of SZ into WSI's
    # no overflow smaller tiles
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    
    # tile with broadcasting
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    # now, img is : (N_TILES, SZ, SZ, CHANNELS )
    
    return img, mask

def select_tiles(img, mask):
    
    result = []
    
    # deal with case where tiles < fully titled WSI
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    
    # tile selection when image is larger than tiles selected
    # idxs are indexes for the lowest N
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result

def tile(img, mask):
    
    img_tiled, mask_tiled = build_tiles(img, mask)
    
    return select_tiles(img_tiled, mask_tiled)


# In[6]:


x_tot,x2_tot = [],[]
names = [name[:-10] for name in os.listdir(MASKS)]


# In[7]:


# name = names[0]

# img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
# mask = skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[-1]
# tiles = tile(img,mask)


# In[8]:


with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for name in tqdm(names):
        img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
        mask = skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[-1]
        tiles = tile(img,mask)
        for t in tiles:
            img,mask,idx = t['img'],t['mask'],t['idx']
            x_tot.append((img/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 
            #if read with PIL RGB turns into BGR
            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}_{idx}.png', img)
            mask = cv2.imencode('.png',mask[:,:,0])[1]
            mask_out.writestr(f'{name}_{idx}.png', mask)


# In[9]:


img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))


# In[10]:


# !ls -lh1 


# In[11]:


# tz = zipfile.ZipFile('train.zip')


# In[12]:


# fns = [e.filename for e in tz.filelist]
# len(fns)


# In[ ]:





# In[ ]:




