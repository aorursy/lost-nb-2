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


import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as torchtransforms
import cv2
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
modelpath = "/kaggle/input/se-resnext50-32x4d-fold2/se_resnext50_32x4d_fold2.pkl"
root_path="/kaggle/input/bengaliai-cv19"


# In[4]:


simple_transform_valid = torchtransforms.Compose([
    torchtransforms.ToTensor(),
    torchtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[5]:


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
def crop_resize(img0, size=128, pad=16):
    HEIGHT = 137
    WIDTH = 236
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))
class ClsTestDataset(Dataset):
    def __init__(self, df, torchtransforms):
        self.df = df
        self.pathes = self.df.iloc[:,0].values
        self.data = self.df.iloc[:, 1:].values
        self.torchtransforms = torchtransforms

    def __getitem__(self, idx):
        HEIGHT = 137
        WIDTH = 236
        #row = self.df.iloc[idx].values
        path = self.pathes[idx]
        img = self.data[idx, :]
        img = 255 - img.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #img = crop_resize(img, size=128)
        #img = crop_resize(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)       
        img = torchtransforms.ToPILImage()(img)
        img = self.torchtransforms(img)
        return path, img
    def __len__(self):
        return len(self.df)

def make_loader(
        data_folder,
        batch_size=64,
        num_workers=2,
        is_shuffle = False,
):

    image_dataset = ClsTestDataset(df = data_folder,
                                    torchtransforms = simple_transform_valid)

    return DataLoader(
    image_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=is_shuffle
    )


# In[6]:


from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
__all__ = ['SENet', 'se_resnext50_32x4d']
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):        
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


# In[7]:


model = se_resnext50_32x4d(pretrained=None)
model.avg_pool = nn.AdaptiveAvgPool2d(1)
model.last_linear = nn.Linear(model.last_linear.in_features, 186)
modelvalue = torch.load(modelpath, map_location='cuda:0')
newmodelvalue = {}
for kv in modelvalue:
    newmodelvalue[kv[4:]]=modelvalue[kv]        
model.load_state_dict(newmodelvalue)
#model.load_state_dict(modelvalue)
model = model.to(device)


# In[8]:


def getmodeleval(model, dataloaders):
    model.eval()
    tbar = tqdm(dataloaders)
    pathes=[]

    alllogit1 = []
    alllogit2 = []
    alllogit3 = []
    for path, img in tbar:
        img = img.to(device)
        pathes.extend(path)
        with torch.no_grad():
            output = model(img)
        logit1, logit2, logit3 = output[:,: 168],                                    output[:,168: 168+11],                                    output[:,168+11:]
        logit1 = F.softmax(logit1, dim=1).cpu().numpy()  # 对每一行进行softmax
        logit2 = F.softmax(logit2, dim=1).cpu().numpy()
        logit3 = F.softmax(logit3, dim=1).cpu().numpy()
        alllogit1.extend(logit1.tolist())
        alllogit2.extend(logit2.tolist())
        alllogit3.extend(logit3.tolist())
    alllogit1 = np.array(alllogit1)
    alllogit2 = np.array(alllogit2)
    alllogit3 = np.array(alllogit3)
    
    print("getmodeleval::alllogit1.shape", alllogit1.shape)
    print("getmodeleval::alllogit2.shape", alllogit2.shape)
    print("getmodeleval::alllogit3.shape", alllogit3.shape)
    return pathes, alllogit1, alllogit2, alllogit3


# In[9]:


allpathes=[]
allpreds_root = []
allpreds_vowel = []
allpreds_consonant = []
tAllBegin = time.time()
for i in range(4):
    test_csv = pd.read_parquet(os.path.join(root_path, f'test_image_data_{i}.parquet'))
    tBegin = time.time()
    dataloaders = make_loader(data_folder = test_csv,
                                           batch_size=8,
                                           num_workers = 2,
                                           is_shuffle = False)
    pathes, logit1, logit2, logit3 = getmodeleval(model, dataloaders)
    preds_root = np.argmax(logit1, axis=1)# 其中，axis=1表示按行计算
    preds_vowel = np.argmax(logit2, axis=1)# 其中，axis=1表示按行计算
    preds_consonant = np.argmax(logit3, axis=1)# 其中，axis=1表示按行计算

    allpathes.extend(pathes)
    allpreds_root.extend(preds_root.tolist())
    allpreds_vowel.extend(preds_vowel.tolist())
    allpreds_consonant.extend(preds_consonant.tolist())
    tEnd = time.time()
    print(i, int(round(tEnd * 1000)) - int(round(tBegin * 1000)), "ms")
tAllEnd = time.time()
print(len(allpathes), len(allpreds_root), len(allpreds_vowel), len(allpreds_consonant),  int(round(tAllEnd * 1000)) - int(round(tAllBegin * 1000)), "ms")


# In[10]:


print(len(allpathes), len(allpreds_root), len(allpreds_vowel), len(allpreds_consonant))


# In[11]:


row_id=[]
target=[]
for idx, image_id in enumerate(allpathes):
    target.extend([allpreds_consonant[idx]])
    target.extend([allpreds_root[idx]])
    target.extend([allpreds_vowel[idx]])

    row_id.extend([str(image_id) + '_consonant_diacritic'])
    row_id.extend([str(image_id) + '_grapheme_root'])
    row_id.extend([str(image_id) + '_vowel_diacritic'])

#print(row_id)
#print(target)
submission_df = pd.read_csv(root_path + '/sample_submission.csv')
#print(submission_df.shape)
# print(len(target))
# print(len(row_id))
# print(target)
# print(row_id)
submission_df.target = np.hstack(np.array(target).astype(np.int))
#submission_df['target'] = np.array(target).astype(np.int)
#submission_df['row_id'] = row_id
print(submission_df.head(10))
submission_df.to_csv('submission.csv', index=False)
print("================end ======================")


# In[12]:


print("A")


# In[13]:


get_ipython().system('nvidia-smi')


# In[ ]:




