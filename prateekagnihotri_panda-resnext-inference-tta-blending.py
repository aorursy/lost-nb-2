#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn.functional as F
import os

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.utils.model_zoo as model_zoo
import cv2

import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
from sklearn.metrics import cohen_kappa_score
import albumentations
# General packages

# import PIL
from PIL import Image

# from IPython.display import Image, display


# In[2]:


BASE_PATH = '../input/prostate-cancer-grade-assessment'

# image and mask directories
data_dir = f'{BASE_PATH}/test_images'
# data_dir = f'{BASE_PATH}/train_images'


mask_dir = f'{BASE_PATH}/test_label_masks'

# Location of test labels
test = pd.read_csv(f'{BASE_PATH}/test.csv')
train = pd.read_csv(f'{BASE_PATH}/train.csv')
# test = pd.read_csv(f'{BASE_PATH}/train.csv').head(200)

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


# In[3]:


test.head()


# In[4]:


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)


# In[5]:


class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    TEST_BATCH_SIZE = 1
    CLASSES = 6


# In[6]:


from collections import OrderedDict
import math


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


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'],         'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = config.pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = config.pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


# In[7]:


class CustomSEResNeXt(nn.Module):

    def __init__(self, model_name='se_resnext50_32x4d'):
        assert model_name in ('se_resnext50_32x4d')
        super().__init__()
        
        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, config.CLASSES)
        
    def forward(self, x):
        x = self.model(x)
        return x


# In[8]:


class PandaDataset(Dataset):
    def __init__(self, images, img_height, img_width):
        self.images = images
        self.img_height = img_height
        self.img_width = img_width
        
        # we are in validation part
        self.aug = albumentations.Compose([
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(data_dir, f'{img_name}.tiff')

        img = skimage.io.MultiImage(img_path)
        img = cv2.resize(img[-1], (512, 512))
        save_path =  f'{img_name}.png'
        cv2.imwrite(save_path, img)
        img = skimage.io.MultiImage(save_path)
            
        img = cv2.resize(img[-1], (self.img_height, self.img_width))

        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img3 = cv2.rotate(img, cv2.ROTATE_180)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img1 = np.transpose(img1, (2, 0, 1)).astype(np.float32)
        img2 = np.transpose(img2, (2, 0, 1)).astype(np.float32)
        img3 = np.transpose(img3, (2, 0, 1)).astype(np.float32)

        return { 'image': torch.tensor(img, dtype=torch.float),'image1': torch.tensor(img1, dtype=torch.float),'image2': torch.tensor(img2, dtype=torch.float),'image3': torch.tensor(img3, dtype=torch.float) }
    


# In[9]:


model = CustomSEResNeXt(model_name='se_resnext50_32x4d')
weights_path = '../input/panda-resnext/resnext50_1.pth'
model.load_state_dict(torch.load(weights_path, map_location=config.device))


# In[10]:


model_1 = CustomSEResNeXt(model_name='se_resnext50_32x4d')
weights_path = '../input/panda-resnext/resnext50_2.pth'
model_1.load_state_dict(torch.load(weights_path, map_location=config.device))


# In[11]:


import cv2
model.eval()
predictions = []

device = config.device

if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
    
    test_dataset = PandaDataset(
        images=test.image_id.values,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
    )
    
    model.to(device)
    model_1.to(device)
    
    for idx, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
        preds_ = np.zeros((6,6))
        inputs = d["image"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds_[0,:] = outputs.cpu().detach().numpy()
        inputs = d["image1"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds_[1,:] = outputs.cpu().detach().numpy()
        inputs = d["image2"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds_[2,:] = outputs.cpu().detach().numpy()
        inputs = d["image3"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds_[3,:] = outputs.cpu().detach().numpy()
        inputs = d["image"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model_1(inputs)
        preds_[4,:] = outputs.cpu().detach().numpy()
        inputs = d["image3"]
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model_1(inputs)
        preds_[5,:] = outputs.cpu().detach().numpy()
        preds_ = preds_.sum(axis = 0)
        predictions.append(preds_.argmax(0))
#     predictions = np.concatenate(predictions)


# In[12]:


if len(predictions) > 0:
    submission.isup_grade = predictions
    submission.isup_grade = submission['isup_grade'].astype(int)


# In[13]:


submission.to_csv('submission.csv',index=False)


# In[14]:


submission.head()

