#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
print(os.listdir("../input/fastai-pretrained"))
print(os.listdir("../input"))


# In[3]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
get_ipython().system('cp ../input/fastai-pretrained/densenet161-8d451a50.pth /tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth')
get_ipython().system('cp ../input/fastai-pretrained/densenet169-b2777c0a.pth /tmp/.cache/torch/checkpoints/densenet169-b2777c0a.pth')


# In[4]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import tqdm
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from collections import OrderedDict
import cv2

import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,     BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[5]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from torchvision.models import *


# In[7]:


# Loading...
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
from fastai.callbacks import *


# In[8]:


## Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/fastai-pretrained/densenet201/densenet201-4c113574.pth' '/tmp/.cache/torch/checkpoints/densenet201-c1103571.pth'")


# In[9]:


os.listdir('../input')


# In[10]:


print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)


# In[11]:


# Set seed fol all
def seed_everything(seed=1358):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# In[12]:


# Reading train_dataset
base_image_dir = os.path.join('..', 'input/humpback-whale-identification/')
base_image_dir2 = os.path.join('..', 'input/new_train_whale/')
train_dir = os.path.join(base_image_dir2,'new_train/')
df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
df['path'] = df['Id'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
df = df.drop(columns=['Id'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head()


# In[13]:


TRAIN ="../input/humpback-whale-identification/train"
TEST = "../input/humpback-whale-identification/test"
LABELS = "../input/humpback-whale-identification/train.csv"
SUB = "../input/humpback-whale-identification/sample_submission.csv"

def text2number(LABELS):
    df = pd.read_csv(LABELS)
    
    uni_labels = pd.DataFrame(df['Id'].unique(),columns=['Id'])
    uni_labels.index.name = 'ID_new'
    uni_labels = uni_labels.reset_index()


    df = df.merge(uni_labels,on='Id')
    
    return df
    
#submit = [p for _, p, _ in pd.read_csv(SUB).to_records()]
#join = list(tagged.keys()) + submit
df = text2number(LABELS)
df.head()


# In[14]:


tagged = dict([(p, w) for _, p,_, w in df.to_records()])
tagged.items()


# In[15]:


# Find fluke pics of whales that have less than 5 pics in training data

TRAIN ="../input/humpback-whale-identification/train"
TEST = "../input/humpback-whale-identification/test"
LABELS = "../input/humpback-whale-identification/train.csv"
SUB = "../input/humpback-whale-identification/sample_submission.csv"

train = pd.read_csv(LABELS)
train_count = train.groupby('Id').count().rename(columns={"Image":"image_count"})
train = train.merge(train_count,on=['Id'])
filelist = train['Image'].loc[(train['image_count']<5)].tolist()
filelist[:5]


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage
import keras
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from time import time

path = "data/train"       
gen = ImageDataGenerator(rotation_range=20, 
                         width_shift_range= 0.1, 
                         height_shift_range = 0.1)

tagged_new = tagged.copy() 

t = time()          
for img in filelist:
  try:
    class_num = tagged[img]
    image = np.expand_dims(mpimg.imread(os.path.join(path,img)),0)  
    aug_iter = gen.flow(image)
    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(5)]
    aug_image_names = [str(i)+"_"+img for i in range(5)]
    for i in range(5):
        plt.imsave(os.path.join(path,aug_image_names[i]),aug_images[i])
        tagged_new[aug_image_names[i]] = class_num
  except:
    pass
print(time()-t)


# In[ ]:





# In[17]:


# Set Batch Size and Image size
bs = 32 
sz=100


# In[18]:


value = {}
count = 0
for whale in df['path'].unique():
    value[whale] = count
    count += 1


# In[19]:


PATH = Path('../input/humpback-whale-identification/')
PATH2 = Path('../input/new_train_whale/')

df_train = pd.read_csv(PATH/'train.csv')


# In[20]:


df_small = df_train[:2500]


# In[21]:


data = ImageDataBunch.from_df(df=df_small,
                              path=PATH2, folder='new_train', 
                              valid_pct=0.1,
                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.1, max_zoom=1.15, max_rotate=45.),
                              size=100,
                              bs=200, 
                              num_workers=os.cpu_count()
                             ).normalize(imagenet_stats)


# In[22]:


print(f'Classes: {data.classes}')


# In[23]:


data.show_batch(rows=3, figsize=(7,6))


# In[24]:


# Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
        os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/fastai-pretrained/resnet50-19c8e357.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")
get_ipython().system("cp '../input/fastai-pretrained/densenet161-8d451a50.pth' '/tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth'")


# In[25]:


get_ipython().system("cp '../input/fastai-pretrained/densenet201-c1103571.pth' '/tmp/.cache/torch/checkpoints/densenet201-c1103571.pth'")


# In[26]:


# PATH = "../input/aptos2019-blindness-detection"
get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints/')
get_ipython().system('cp ../input/fastai-pretrained/densenet161-8d451a50.pth /tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth')
get_ipython().system('cp ../input/fastai-pretrained/densenet169-b2777c0a.pth /tmp/.cache/torch/checkpoints/densenet169-b2777c0a.pth')
get_ipython().system('cp ../input/fastai-pretrained/densenet169-b2777c0a.pth /tmp/.cache/torch/checkpoints/densenet169-b2777c0a.pth')
get_ipython().system('cp ../input/fastai-pretrained/densenet201-c1103571.pth /tmp/.cache/torch/checkpoints/densenet201-c1103571.pth')
# densenet201-c1103571.pth


# In[27]:


learn_1 = cnn_learner(data, models.densenet201, metrics=[accuracy], pretrained=True).mixup()


# In[28]:


learn_1.fit_one_cycle(10,1e-02)
learn_1.recorder.plot_losses()
learn_1.recorder.plot_metrics()


# In[29]:


learn_2 = cnn_learner(data, models.resnet50, metrics=[accuracy], pretrained=True).mixup()


# In[30]:


learn_2.fit_one_cycle(10,1e-02)
learn_2.recorder.plot_losses()
learn_2.recorder.plot_metrics()


# In[31]:


get_ipython().system('cp ../input/fastai-pretrained/resnet152-b121ed2ed.pth /tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth')

