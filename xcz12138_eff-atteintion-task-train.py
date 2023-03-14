#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
print(os.listdir('../input/xczdatasets'))


# In[2]:


# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy as sp
import sys
from functools import partial
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
sys.path.append('../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/')
from efficientnet_pytorch import EfficientNet
test_batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path =  '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'
class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef=[0.5, 1.5, 2.5, 3.5]):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
import matplotlib.pyplot as plt

def predict( X, coef=[0.5, 1.5, 2.5, 3.5]):
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance

    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
        #         print(img.shape)
        return img
def preprocess_image(image_path, desired_size=256):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)

    return img

def preprocess_image_old(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)

    return img

class MyDataset(Dataset):
    def __init__(self,dataframe, root_dir,transform=None,train = False):
        self.df =dataframe
        self.transform = transform
        self.train = train
        self.root_dir = root_dir
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.id_code.values[idx]
        if idx<10:
            p = os.path.join(self.root_dir,str(p)+'.jpg')
        else:
            p = os.path.join(self.root_dir,str(p)+'.jpg')
        if self.train:
            image = preprocess_image_old(str(p))
        else:
            image = preprocess_image(str(p))
        # image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)
        return image
from torchvision import transforms
test_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
root_dir = '../input/xczdatasets/Original_Images/Testing_Set'
train_df = pd.read_csv('../input/labels/test.csv')

testset        = MyDataset(train_df,root_dir,transform=test_transform)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
#show image test
image = testset[random.randint(0, len(testset)-1)]


class ClassifierModule(nn.Sequential):
    def __init__(self, n_features):
        super().__init__(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, n_features),
            nn.PReLU(),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.2),
            nn.Linear(n_features, 1),
        )

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', weights_path=None):
        assert model_name in ('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'e fficientnet-b4', 'efficientnet-b5')
        super().__init__()

        self.net = EfficientNet.from_name(model_name)
        self.net.load_state_dict(torch.load(weights_path))

        n_features = self.net._fc.in_features

        self.net._fc = ClassifierModule(n_features)

    def forward(self, x):
        return self.net(x)

model = CustomEfficientNet(model_name='efficientnet-b0', weights_path=weights_path)
model.load_state_dict(torch.load('../input/pretrained-model/efficientnet-b0_fold0_epoch12.pth'))
model.cuda()
# #


model.eval()
valid_preds = np.zeros((len(testset)))
avg_val_loss = 0.

for i, images in enumerate(test_loader):
    with torch.no_grad():
        y_preds = model(images.to(device)).detach()

    valid_preds[i * test_batch_size: (i+1) * test_batch_size] = y_preds[:, 0].to('cpu').numpy()
pred = predict(valid_preds)
print(np.squeeze(pred))
pred_label = pd.DataFrame({'id_code':pd.read_csv('../input/labels/test.csv').id_code.values,
                           'labels':np.squeeze(pred).astype(int)})
print(pred_label.head())
pred_label.to_csv('train.csv', index=False)


# In[3]:


train_params = {
    'n_splits': 5,
    'n_epochs': 12,
    'lr': 1e-3,
    'base_lr': 1e-4,
    'max_lr': 3e-3,
    'step_factor': 6,
    'train_batch_size': 32,
    'train_image_size': 512,
    'test_image_size': 512,
    'test_batch_size': 32,
    'accumulation_steps': 10,
    
}


# In[4]:


# ! pip install torch==1.1.0


# In[5]:


# ! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a


# In[6]:


import os
os.listdir('../input/efficientnet-pytorch/')
['efficientnet-b3-c8376fa2.pth',
 'efficientnet-b0-08094119.pth',
 'efficientnet-b5-586e6cc6.pth',
 'efficientnet-b2-27687264.pth',
 'efficientnet-b1-dbc7070a.pth',
 'efficientnet-b7-dcc49843.pth',
 'EfficientNet-PyTorch',
 'efficientnet-b6-c76e70fd.pth',
 'efficientnet-b4-e116e8b3.pth']


# In[7]:


os.listdir('../input/efficientnet-my/')


# In[8]:


import sys
# sys.path.append('../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/')

# 
sys.path.append('../input/efficientnet-my/')


# In[9]:


import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy as sp
from fastprogress import master_bar, progress_bar
from functools import partial
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from albumentations import (
    Compose, HorizontalFlip, IAAAdditiveGaussianNoise, Normalize, OneOf,
    RandomBrightness, RandomContrast, Resize, VerticalFlip, Rotate, ShiftScaleRotate,
    RandomBrightnessContrast, OpticalDistortion, GridDistortion, ElasticTransform, Cutout
)
from albumentations.pytorch import ToTensor

from apex import amp

from fastai.layers import Flatten, AdaptiveConcatPool2d


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[10]:


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


# In[11]:


def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('APTOS')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

LOG_FILE = 'aptos-train.log'
LOGGER = init_logger(LOG_FILE)


# In[12]:


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 777
seed_torch(SEED)


# In[13]:


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


# In[14]:


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[15]:


# NOTE: official CyclicLR implementation doesn't work now

from torch.optim.lr_scheduler import _LRScheduler

class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size, gamma=0.99, mode='triangular', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        assert mode in ['triangular', 'triangular2', 'exp_range']
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        # make sure that the length of base_lrs doesn't change. Dont care about the actual value
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (self.last_epoch))
            new_lr.append(lr)
        return new_lr


# In[16]:


# new_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
# old_train = pd.read_csv('../input/retinopathy-train-2015/rescaled_train_896/trainLabels.csv')
# print(new_train.shape)
# print(old_train.shape)


# In[17]:



old_train = old_train[['image','level']]
old_train.columns = new_train.columns
old_train.diagnosis.value_counts()

# # path columns
new_train['id_code'] = '../input/aptos2019-blindness-detection/train_images/' + new_train['id_code'].astype(str) + '.png'
old_train['id_code'] = '../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/' + old_train['id_code'].astype(str) + '.png'

train_df = old_train[:992].copy()
val_df = new_train[:480].copy()
train_df.head
print(train_df.shape)
print(val_df.shape)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
import matplotlib
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[18]:


root = '/kaggle/input/aptos2019-blindness-detection/'
train_df = pd.read_csv(os.path.join(root, 'train.csv'))


# In[19]:


import cv2
SEED = 125
fig = plt.figure(figsize=(30,30))
img_list = []
img_size = []
root = '../input/aptos2019-blindness-detection/'
# display 10 images from each class
#for class_id in sorted(train_y.unique()):
for class_id in [0, 1, 2, 3, 4]:
    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(1, random_state=SEED).iterrows()):
        ax = fig.add_subplot(1, 5, class_id+i+1 , xticks=[], yticks=[])
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.05)
        path = os.path.join(root, 'train_images', '{}.png'.format(row['id_code']))
        image = cv2.imread(path)
        image = cv2.resize(image,(256,256))
        img_size.append(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_list.append(row['id_code'])
        
        plt.imshow(image)
        ax.set_title('Grade: %d' % (class_id) )


# In[20]:


# train_df = train_df.sample(frac=1).reset_index(drop=True)
# val_df = val_df.sample(frac=1).reset_index(drop=True)
# print(train_df.shape)
# print(val_df.shape)


# In[21]:


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
def preprocess_image(image_path, desired_size=train_params['test_image_size']):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)
    
    return img

def preprocess_image_old(image_path, desired_size=train_params['train_image_size']):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)
    
    return img


# In[22]:


class MyDataset(Dataset): 
    def __init__(self, dataframe, transform=None,train = True):
        self.df = dataframe
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        label = torch.tensor(label).float()
        
        p = self.df.id_code.values[idx]
        if self.train:
            image = preprocess_image_old(str(p))
        else:
            image = preprocess_image(str(p))
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# In[23]:


from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# trainset     = MyDataset(train_df, transform =train_transform,train=True)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# valset     = MyDataset(val_df, transform=train_transform,train =False)
# val_loader   = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)


# In[24]:


# APTOS_DIR = Path('../input/aptos2019-blindness-detection')
# APTOS_TRAIN_DIR = Path('../input/aptos-train-dataset')

# APTOS_TRAIN_IMAGES = APTOS_TRAIN_DIR / 'aptos-train-images/aptos-train-images'

# #APTOS_FOLDS = Path('../input/aptos-folds/folds.csv')
# #APTOS_FOLDS = Path('../input/aptos-folds/jpeg_folds.csv')
# #APTOS_TRAIN_FOLDS = Path('../input/aptos-folds/jpeg_folds_all.csv')
# #APTOS_VALID_FOLDS = Path('../input/aptos-folds/png_folds_all.csv')
# APTOS_TRAIN_FOLDS = Path('../input/aptos-folds/2015_5folds.csv')
# APTOS_VALID_FOLDS = Path('../input/aptos-folds/2019_5folds.csv')

# ID_COLUMN = 'id_code'
# TARGET_COLUMN = 'diagnosis'


# In[25]:


PRETRAINED_DIR = Path('../input/pytorch-pretrained-models')
EFFICIENTNET_PRETRAINED_DIR = Path('../input/efficientnet-pytorch')
os.listdir('../input/efficientnet-pytorch/')
PRETRAINED_MAPPING = {
    # ResNet
    'resnet18': PRETRAINED_DIR / 'resnet18-5c106cde.pth', 
    'resnet34': PRETRAINED_DIR / 'resnet34-333f7ec4.pth',
    'resnet50': PRETRAINED_DIR / 'resnet50-19c8e357.pth',
    'resnet101': PRETRAINED_DIR / 'resnet101-5d3b4d8f.pth',
    'resnet152': PRETRAINED_DIR / 'resnet152-b121ed2d.pth',

    # ResNeXt
    'resnext101_32x4d': PRETRAINED_DIR / 'resnext101_32x4d-29e315fa.pth',
    'resnext101_64x4d': PRETRAINED_DIR / 'resnext101_64x4d-e77a0586.pth',

    # WideResNet
    #'wideresnet50'

    # DenseNet
    'densenet121': PRETRAINED_DIR / 'densenet121-fbdb23505.pth',
    'densenet169': PRETRAINED_DIR / 'densenet169-f470b90a4.pth',
    'densenet201': PRETRAINED_DIR / 'densenet201-5750cbb1e.pth',
    'densenet161': PRETRAINED_DIR / 'densenet161-347e6b360.pth',

    # SE-ResNet
    'se_resnet50': PRETRAINED_DIR / 'se_resnet50-ce0d4300.pth',
    'se_resnet101': PRETRAINED_DIR / 'se_resnet101-7e38fcc6.pth',
    'se_resnet152': PRETRAINED_DIR / 'se_resnet152-d17c99b7.pth',

    # SE-ResNeXt
    'se_resnext50_32x4d': PRETRAINED_DIR / 'se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': PRETRAINED_DIR / 'se_resnext101_32x4d-3b2fe3d8.pth',

    # SE-Net
    'senet154': PRETRAINED_DIR / 'senet154-c7b49a05.pth',

    # InceptionV3
    'inceptionv3': PRETRAINED_DIR / 'inception_v3_google-1a9a5a14.pth',

    # InceptionV4
    'inceptionv4': PRETRAINED_DIR / 'inceptionv4-8e4777a0.pth',

    # BNInception
    'bninception': PRETRAINED_DIR / 'bn_inception-52deb4733.pth',

    # InceptionResNetV2
    'inceptionresnetv2': PRETRAINED_DIR / 'inceptionresnetv2-520b38e4.pth',

    # Xception
    'xception': PRETRAINED_DIR / 'xception-43020ad28.pth',

    # DualPathNet
    'dpn68': PRETRAINED_DIR / 'dpn68-4af7d88d2.pth',
    'dpn98': PRETRAINED_DIR / 'dpn98-722954780.pth',
    'dpn131': PRETRAINED_DIR / 'dpn131-7af84be88.pth',
    'dpn68b': PRETRAINED_DIR / 'dpn68b_extra-363ab9c19.pth',
    'dpn92': PRETRAINED_DIR / 'dpn92_extra-fda993c95.pth',
    'dpn107': PRETRAINED_DIR / 'dpn107_extra-b7f9f4cc9.pth',

    # PolyNet
    'polynet': PRETRAINED_DIR / 'polynet-f71d82a5.pth',

    # NasNet-A-Large
    'nasnetalarge': PRETRAINED_DIR / 'nasnetalarge-a1897284.pth',

    # PNasNet-5-Large
    'pnasnet5large': PRETRAINED_DIR / 'pnasnet5large-bf079911.pth',

    # EfficientNet
    'efficientnet-b0': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b0-08094119.pth',
    'efficientnet-b1': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b2-27687264.pth',
    'efficientnet-b3': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': EFFICIENTNET_PRETRAINED_DIR / 'efficientnet-b5-586e6cc6.pth',
 
}


# In[26]:


# class APTOSTrainDataset(Dataset):
#     def __init__(self, image_dir, file_paths, labels, transform=None):
#         self.image_dir = image_dir
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file_path = f'{self.image_dir}/{self.file_paths[idx]}'
#         label = torch.tensor(self.labels[idx]).float()
        
#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
        
#         return image, label


# In[27]:


# from albumentations import ImageOnlyTransform

# def crop_image_from_gray(img, tol=7):
#     """
#     Crop out black borders
#     https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
#     """  
#     if img.ndim ==2:
#         mask = img>tol
#         return img[np.ix_(mask.any(1),mask.any(0))]
#     elif img.ndim==3:
#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         mask = gray_img>tol        
#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
#         if (check_shape == 0):
#             return img
#         else:
#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
#             img = np.stack([img1,img2,img3],axis=-1)
#         return img


# class CircleCrop(ImageOnlyTransform):
#     def __init__(self, tol=7, always_apply=False, p=1.0):
#         super().__init__(always_apply, p)
#         self.tol = tol
    
#     def apply(self, img, **params):
#         img = crop_image_from_gray(img)    
    
#         height, width, depth = img.shape    
    
#         x = int(width/2)
#         y = int(height/2)
#         r = np.amin((x,y))
    
#         circle_img = np.zeros((height, width), np.uint8)
#         cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
#         img = cv2.bitwise_and(img, img, mask=circle_img)
#         img = crop_image_from_gray(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#         return img 
    

# class CircleCropV2(ImageOnlyTransform):
#     def __init__(self, tol=7, always_apply=False, p=1.0):
#         super().__init__(always_apply, p)
#         self.tol = tol
    
#     def apply(self, img, **params):
#         img = crop_image_from_gray(img)
        
#         height, width, depth = img.shape
#         largest_side = np.max((height, width))
#         img = cv2.resize(img, (largest_side, largest_side))
    
#         height, width, depth = img.shape    
    
#         x = int(width/2)
#         y = int(height/2)
#         r = np.amin((x,y))
    
#         circle_img = np.zeros((height, width), np.uint8)
#         cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
#         img = cv2.bitwise_and(img, img, mask=circle_img)
#         img = crop_image_from_gray(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#         return img 


# In[28]:


# def get_transforms(*, data):
#     assert data in ('train', 'valid')
    
#     if data == 'train':
#         return Compose([
#             CircleCropV2(),
#             Resize(256, 256),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             Rotate(p=0.5), 
#             #ShiftScaleRotate(p=0.5),
#             #RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
#             #OpticalDistortion(distort_limit=(0.9,1.0), shift_limit=0.05, interpolation=1, border_mode=4, 
#             #                  value=None, always_apply=False, p=0.5),
#             #GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
#             #               value=None, always_apply=False, p=0.5),
#             #ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4,
#             #                 value=None, always_apply=True, approximate=False, p=0.5),
#             Cutout(p=0.25, max_h_size=25, max_w_size=25, num_holes=8),
#             #OneOf([
#             #    RandomBrightness(0.1, p=1),
#             #    RandomContrast(0.1, p=1),
#             #], p=0.25),
#             RandomContrast(0.5, p=0.5),
#             IAAAdditiveGaussianNoise(p=0.25),
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#             ToTensor(),
#         ])
    
#     elif data == 'valid':
#         return Compose([
#             CircleCropV2(),
#             Resize(256, 256),
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#             ToTensor(),
#         ])


# In[29]:


def basic_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=0, bn_momentum=0.1):
    basicconv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        nn.ReLU()
    )
    return basicconv
class Basic_cell(nn.Module):
    def __init__(self,in_channels=1280):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,64,(1,1),padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64,16,(1,1),padding=0)
        self.conv3 = nn.Conv2d(16,8,(1,1),padding=0)
        self.conv4 = nn.Conv2d(8,1,(1,1),padding=0)
        self.sig = nn.Sigmoid()
    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sig(self.conv4(x))
        return x

class Attention_Cell(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.bn2 =  nn.BatchNorm2d(in_channels)
        self.basic_cell = Basic_cell(in_channels)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, input):
        bn_x = self.bn2(input)
        x = self.basic_cell(bn_x)
      
        multi_x = bn_x*x
#         print('multi_x',multi_x.shape)
        x = self.GAP(x)
        multi_x = self.GAP(multi_x)
#         print("GAP-multi-x",multi_x.shape)
#         print("GAP-x",x.shape)
        output  = torch.div(multi_x,x)
#         print("output",output)

        return output

class RegressionDense(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(0.5),
            nn.Linear(in_channels, 1280),
            nn.PReLU(),
            nn.BatchNorm1d(1280),
            nn.Dropout(0.2),
            nn.Linear(1280, 1),
        )

class ClassfierDense(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(0.5),
            nn.Linear(in_channels, 1280),
            nn.PReLU(),
            nn.BatchNorm1d(1280),
            nn.Dropout(0.2),
            nn.Linear(1280, 5),
#             nn.Softmax(),
        )


# In[30]:


class CustomEfficientNet_Att(nn.Module):
    def __init__(self, model_name='efficientnet-b0', weights_path=None):
        assert model_name in ('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'e fficientnet-b4', 'efficientnet-b5')
        super().__init__()
        in_channels = 1280
        if model_name is 'efficientnet-b0':
            in_channels = 1280
        if model_name is 'efficientnet-b3':
            in_channels = 1536
        self.basic_net = EfficientNet.from_name(model_name)
        pretrained_dict = torch.load(weights_path)
#         n_features = self.basic_net._conv_head.in_features
        model_dict = self.basic_net.state_dict()
        state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                # state_dict.setdefult(k, v)
                state_dict[k] = v
        model_dict.update(state_dict)  # 更新(合并)模型的参数
        self.basic_net.load_state_dict(model_dict)
        self.attention_cell= Attention_Cell(in_channels)
        self.regress = RegressionDense(in_channels)
        self.classfier = ClassfierDense(in_channels)
    def forward(self, x):
        x = self.basic_net(x)
        x = self.attention_cell(x)
#         print("注意力",x.shape)
        x = x.squeeze(-1).squeeze(-1)
#         print("注意力",x.shape)
        pred_1 = self.regress(x)
        pred_2 = self.classfier(x)
        return pred_1,pred_2


# In[31]:


class CustomResnet_Att(nn.Module): 
    def __init__(self, model_name='resnet50', weights_path=None):
        assert model_name in ('resnet50', 'resnet101', 'resnet152')
        super().__init__()
        
        self.basic_net = pretrainedmodels.__dict__[model_name](pretrained=None)
        pretrained_dict = self.basic_net.load_state_dict(torch.load(weights_path))
#         self.basic_net = EfficientNet.from_name(model_name)
#         pretrained_dict = torch.load(weights_path)
#         n_features = self.basic_net._conv_head.in_features
        model_dict = self.basic_net.state_dict()
        state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                # state_dict.setdefult(k, v)
                state_dict[k] = v
        model_dict.update(state_dict)  # 更新(合并)模型的参数
        self.basic_net.load_state_dict(model_dict)
        
        self.attention_cell= Attention_Cell()
        self.regress = RegressionDense(1280)
        self.classfier = ClassfierDense(1280)
    def forward(self, x):
        x = self.basic_net(x)
        x = self.attention_cell(x)
#         print("注意力",x.shape)
        x = x.squeeze(-1).squeeze(-1)
#         print("注意力",x.shape)
        pred_1 = self.regress(x)
        pred_2 = self.classfier(x)
        return pred_1,pred_2


# In[32]:


class ClassifierModule(nn.Sequential):
    def __init__(self, n_features):
        super().__init__(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, n_features),
            nn.PReLU(),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.2),
            nn.Linear(n_features, 1),
        )


# In[33]:


class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet50', weights_path=None):
        assert model_name in ('resnet50', 'resnet101', 'resnet152')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        self.net.load_state_dict(torch.load(weights_path))
        
        n_features = self.net.last_linear.in_features
        
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.net.avgpool = AdaptiveConcatPool2d(1)
        self.net.last_linear = ClassifierModule(n_features)
        
    def forward(self, x):
        return self.net(x)


# In[34]:


class CustomResNeXt(nn.Module):
    def __init__(self, model_name='resnext101_32x4d', weights_path=None):
        assert model_name in ('resnext101_32x4d', 'resnext101_64x4d')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        self.net.load_state_dict(torch.load(weights_path))
        
        n_features = self.net.last_linear.in_features
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.net.avg_pool = AdaptiveConcatPool2d(1)
        self.net.last_linear = ClassifierModule(n_features)
        
    def forward(self, x):
        return self.net(x)


# In[35]:


class CustomSENet(nn.Module):
    def __init__(self, model_name='se_resnet50', weights_path=None):
        assert model_name in ('senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        self.net.load_state_dict(torch.load(weights_path))
        
        n_features = self.net.last_linear.in_features
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.net.avg_pool = AdaptiveConcatPool2d(1)
        self.net.last_linear = ClassifierModule(n_features)
        
    def forward(self, x):
        return self.net(x)


# In[36]:


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', weights_path=None):
        assert model_name in ('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'e fficientnet-b4', 'efficientnet-b5')
        super().__init__()
        
        self.net = EfficientNet.from_name(model_name)
        self.net.load_state_dict(torch.load(weights_path))

        n_features = self.net._fc.in_features
        
        self.net._fc = ClassifierModule(n_features)
        
    def forward(self, x):
        return self.net(x)


# In[37]:


# model = CustomEfficientNet_Att(model_name=MODEL, weights_path=PRETRAINED_MAPPING[MODEL])
# model.to(device)


# In[38]:


LOGGER.debug(f'Fold: {FOLD}')
LOGGER.debug(f'Model: {MODEL}')
LOGGER.debug(f'Train params: {train_params}')


# In[39]:


with timer('Prepare train and valid sets'):
#     with timer('  * load folds csv'):
#         #folds = pd.read_csv(APTOS_FOLDS)
#         #train_fold = folds[folds['fold'] != FOLD].reset_index(drop=True)
#         #valid_fold = folds[folds['fold'] == FOLD].reset_index(drop=True)
#         folds = pd.read_csv(APTOS_TRAIN_FOLDS)
#         train_fold = folds[folds['fold'] != FOLD].reset_index(drop=True)
#         #valid_fold2015 = folds[folds['fold'] == FOLD].reset_index(drop=True)
#         #valid_fold2019 = pd.read_csv(APTOS_VALID_FOLDS)
#         #valid_fold = pd.concat([valid_fold2015, valid_fold2019]).reset_index(drop=True)
#         valid_fold = pd.read_csv(APTOS_VALID_FOLDS)
    
    with timer('  * define dataset'):
        train_dataset     = MyDataset(train_df, transform =train_transform,train=True)
       
        valid_dataset     = MyDataset(val_df, transform=train_transform,train =False)

#         APTOSTrainDataset = partial(APTOSTrainDataset, image_dir=APTOS_TRAIN_IMAGES)
#         train_dataset = APTOSTrainDataset(file_paths=train_fold.id_code.values,
#                                           labels=train_fold.diagnosis.values[:, np.newaxis],
#                                           transform=get_transforms(data='train'))
#         valid_dataset = APTOSTrainDataset(file_paths=valid_fold.id_code.values,
#                                           labels=valid_fold.diagnosis.values[:, np.newaxis],
#                                           transform=get_transforms(data='valid'))
        
    with timer('  * define dataloader'):
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_params['train_batch_size'],
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=train_params['test_batch_size'],
                                  shuffle=False)
        
LOGGER.debug(f'train size: {len(train_dataset)}, valid size: {len(valid_dataset)}')


# In[40]:


def multi_loss(pred_reg,pred_classfier,y_true,multi_method='mse_ce',alpha = 1,beta =1):
    assert multi_method in ('mse_ce','')
    loss = 0;c1=0;c2=0
    if multi_method  =='mse_ce':
        c1 = nn.MSELoss()
        c2 = nn.CrossEntropyLoss()
    loss_reg =  c1(pred_reg,y_true)
    y_true = y_true.long()
#     print(pred_classfier.shape,y_true.shape)
    loss_class = c2(pred_classfier,y_true.squeeze())
    loss = alpha*loss_reg + beta*loss_class
    return loss


# In[41]:


# with timer('Train model'):
#     n_epochs = train_params['n_epochs']
#     lr = train_params['lr']
#     base_lr = train_params['base_lr']
#     max_lr = train_params['max_lr']
#     step_factor = train_params['step_factor']
#     test_batch_size = train_params['test_batch_size']
#     accumulation_steps = train_params['accumulation_steps']
    
#     model = CustomEfficientNet_Att(model_name=MODEL, weights_path=PRETRAINED_MAPPING[MODEL])
#     model.to(device)
    
#     optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)
#     #optimizer = SGD(model.parameters(), lr=lr, weight_decay=4e-5, momentum=0.9, nesterov=True)
#     scheduler = CyclicLR(optimizer,
#                          base_lr=base_lr,
#                          max_lr=max_lr,
#                          step_size=len(train_loader) * step_factor)

#     model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    
#     criterion = nn.MSELoss()
#     #criterion = nn.SmoothL1Loss()
    
    
#     optimized_rounder = OptimizedRounder()
# #     optimized_rounder2 = OptimizedRounder()
#     y_true = val_df.diagnosis.values
    
#     for epoch in range(n_epochs):
#         start_time = time.time()

#         model.train()
#         avg_loss = 0.

#         optimizer.zero_grad()

#         for i, (images, labels) in enumerate(train_loader):
#             if isinstance(scheduler, CyclicLR):
#                 scheduler.step()
#             images = images.to(device)
# #             images = images.float()
#             pred_reg,pred_cls = model(images)
# #             labels = labels.float()
# #             loss = criterion(y_preds, labels.to(device))
#             loss  = multi_loss(pred_reg,pred_cls,labels.to(device))
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()

#             if (i+1) % accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#             avg_loss += loss.item() / accumulation_steps / len(train_loader)

#         if not isinstance(scheduler, CyclicLR):
#             scheduler.step()

#         model.eval()
#         valid_preds = np.zeros((len(valid_dataset)))
#         avg_val_loss = 0.

#         for i, (images, labels) in enumerate(valid_loader):
#             with torch.no_grad():
# #                 y_preds = model(images.to(device)).detach()
#                   pred_reg,pred_cls = model(images.to(device))
#                   pred_reg = pred_reg.detach()
# #             loss = criterion(y_preds, labels.to(device))
#             loss =  multi_loss(pred_reg,pred_cls,labels.to(device))
#             valid_preds[i * test_batch_size: (i+1) * test_batch_size] = pred_reg[:, 0].to('cpu').numpy()
#             avg_val_loss += loss.item() / len(valid_loader)

# #         optimized_rounder.fit(valid_preds, y_true)
#         optimized_rounder.fit(valid_preds, y_true)
# #         optimized_rounder2.fit(pred_cls, y_true)
#         coefficients = optimized_rounder.coefficients()
# #         coefficients_cls = optimized_rounder2.coefficients()
#         final_preds = optimized_rounder.predict(valid_preds, coefficients)
#         qwk = quadratic_weighted_kappa(y_true, final_preds)

#         elapsed = time.time() - start_time

#         LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#         LOGGER.debug(f'          - qwk: {qwk:.6f}  coefficients: {coefficients}')

#         # FIXME: save all epochs for debug 
#         torch.save(model.state_dict(), f'{MODEL}_fold{FOLD}_epoch{epoch+1}.pth')


# In[42]:


.softmax(input, self.dim, _stacklevel=5)
2020-07-02 02:27:20,329 DEBUG   Epoch 1 - avg_train_loss: 0.3125  avg_val_loss: 4.7410  time: 112s
2020-07-02 02:27:20,330 DEBUG           - qwk: 0.000000  coefficients: [0.5 1.5 2.5 3.5]
2020-07-02 02:29:02,828 DEBUG   Epoch 2 - avg_train_loss: 0.3046  avg_val_loss: 5.4275  time: 102s
2020-07-02 02:29:02,830 DEBUG           - qwk: 0.271479  coefficients: [-0.542408  2.047389  3.62589   5.926311]
2020-07-02 02:30:45,871 DEBUG   Epoch 3 - avg_train_loss: 0.3203  avg_val_loss: 4.9421  time: 103s
2020-07-02 02:30:45,872 DEBUG           - qwk: 0.060628  coefficients: [0.5 1.5 2.5 3.5]
2020-07-02 02:32:29,153 DEBUG   Epoch 4 - avg_train_loss: 0.2932  avg_val_loss: 2.9958  time: 103s
2020-07-02 02:32:29,155 DEBUG           - qwk: 0.602941  coefficients: [0.276694 1.188741 0.554224 7.733649]
2020-07-02 02:34:12,737 DEBUG   Epoch 5 - avg_train_loss: 0.2489  avg_val_loss: 3.6941  time: 104s
2020-07-02 02:34:12,741 DEBUG           - qwk: 0.364468  coefficients: [0.40734  0.359478 2.882061 4.309653]
2020-07-02 02:35:55,522 DEBUG   Epoch 6 - avg_train_loss: 0.2240  avg_val_loss: 2.6687  time: 103s
2020-07-02 02:35:55,523 DEBUG           - qwk: 0.607598  coefficients: [0.494824 0.815553 2.177532 4.449734]
2020-07-02 02:37:39,348 DEBUG   Epoch 7 - avg_train_loss: 0.1965  avg_val_loss: 4.0795  time: 104s
2020-07-02 02:37:39,349 DEBUG           - qwk: 0.653501  coefficients: [0.596621 1.498057 1.842185 4.625971]
2020-07-02 02:39:22,939 DEBUG   Epoch 8 - avg_train_loss: 0.1642  avg_val_loss: 2.6657  time: 104s
2020-07-02 02:39:22,942 DEBUG           - qwk: 0.713007  coefficients: [0.461684 1.598026 1.734047 4.394338]
2020-07-02 02:41:06,369 DEBUG   Epoch 9 - avg_train_loss: 0.1579  avg_val_loss: 2.8600  time: 103s
2020-07-02 02:41:06,371 DEBUG           - qwk: 0.635940  coefficients: [0.519184 1.490187 2.401873 3.643743]
2020-07-02 02:42:49,061 DEBUG   Epoch 10 - avg_train_loss: 0.1606  avg_val_loss: 2.5784  time: 103s
2020-07-02 02:42:49,063 DEBUG           - qwk: 0.563788  coefficients: [0.516132 0.831461 2.752982 3.96373 ]
2020-07-02 02:44:32,402 DEBUG   Epoch 11 - avg_train_loss: 0.1453  avg_val_loss: 2.4899  time: 103s
2020-07-02 02:44:32,404 DEBUG           - qwk: 0.559555  coefficients: [0.602128 0.988473 2.593635 3.840373]
2020-07-02 02:46:15,584 DEBUG   Epoch 12 - avg_train_loss: 0.1299  avg_val_loss: 2.4178  time: 103s
2020-07-02 02:46:15,586 DEBUG           - qwk: 0.618733  coefficients: [0.715129 1.184878 1.110287 4.461294]
2020-07-02 02:46:15,646 INFO [Train model] done in 1247 s.


# In[ ]:




