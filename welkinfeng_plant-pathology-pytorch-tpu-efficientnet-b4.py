#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AdamW(lr=1e-3) CosineLR(epoch=40, min=1e-7) CE loss
# iaa.Cutout(0.7) 320x512
# 1 outputs with dropout + 1 output without dropout
# (v6 efficientnet_b4) fold 2 and 4 are difficult to train


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


USE_COLAB, TRAIN_MODE, USE_TPU = 0, 1, 1
version_name = 'v6_tpu'
network_name = 'efficientnet_b4'  # 用作保存模型时的名字


# In[4]:


if USE_TPU:
    get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
    get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.data_parallel as dp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp


# In[5]:


import os, shutil, sys, time, gc
import copy, multiprocessing, functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdmauto
from PIL import Image
from collections import OrderedDict

import math, cv2, sklearn
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import StratifiedKFold

if not USE_COLAB:
    print(os.listdir('../input/'))
    # PATH = '../input/plant-pathology-2020-images-just-resize/'
    PATH = '../input/plant-pathology-image-processing/'
    SAVE_PATH = './'  # 模型要保存到的路径
    MODEL_PATH = '../input/plant-pathology-pytorch-tpu-efficientnet-b4/'
    get_ipython().system('git clone -q https://github.com/welkin-feng/ComputerVision.git')
    sys.path.append('./ComputerVision/')
    get_ipython().system('git clone -q https://github.com/rwightman/pytorch-image-models.git')
    sys.path.append('./pytorch-image-models/')
    # sys.path.append('../input/cvmodels/')
    # sys.path.append('../input/pytorch-image-models/')
    # # for training
    # if TRAIN_MODE:
    #     !git clone -q https://github.com/welkin-feng/ComputerVision.git
    #     sys.path.append('./ComputerVision/')
    #     !git clone -q https://github.com/rwightman/pytorch-image-models.git
    #     sys.path.append('./pytorch-image-models/')

else:
    from google.colab import drive

    drive.mount('/content/drive', force_remount=True)
    PATH = './drive/My Drive/Competition/plant-pathology-2020/plant-pathology-2020/'
    SAVE_PATH = f"./drive/My Drive/Competition/plant-pathology-2020/{version_name}_{network_name}/"
    MODEL_PATH = SAVE_PATH
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    get_ipython().system('git clone -q https://github.com/welkin-feng/ComputerVision.git')
    sys.path.append('./ComputerVision/')
    get_ipython().system('git clone -q https://github.com/rwightman/pytorch-image-models.git')
    sys.path.append('./pytorch-image-models/')

print("PATH: ", os.listdir(PATH))
print("SAVE_PATH: ", os.listdir(SAVE_PATH))
if os.path.isdir(MODEL_PATH):
    # 如果训练中断了就从这里重新读取模型
    print('MODEL_PATH: ', os.listdir(MODEL_PATH))

# Gets the GPU if there is one, otherwise the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
random_seed = 644
print(f'random_state: {random_seed}')


# In[6]:


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# if USE_TPU:
#     seed_everything(random_seed)


# In[7]:


network_name = network_name
version_name = version_name
DEVICE = DEVICE
PATH = PATH
SAVE_PATH = SAVE_PATH
MODEL_PATH = MODEL_PATH

IMAGE_FILE_NAME = ['train_images_320x512.npy', 'test_images_320x512.npy']
IMG_SHAPE = (1365, 2048, 3)
INPUT_IMG_SHAPE = (320, 512, 3)
IMG_MEAN = np.array([0])
IMG_STD = np.array([1])

# train_transforms = {
#     'mix_prob': 0.0, 'mixup_prob': 0.2, 'cutmix_prob': 0.35, 'fmix_prob': 0, 
#     'grid_prob': 0.2, 'erase_prob': 0, 'cutout_prob': 0, 
#     'cutout_ratio': (0.1, 0.5), 'cut_size': int(INPUT_IMG_SHAPE[0] * 0.7), # (0.1, 0.3), 
#     'brightness': (0.7, 1.1), 'noise_prob': 0, 'blur_prob': 0, 'drop_prob': 0, 'elastic_prob': 0,
#     'hflip_prob': 0.1, 'vflip_prob': 0, 'scale': (0.8, 1.1), 
#     'shear': (-10, 10), 'translate_percent': (-0.15, 0.15), 'rotate': (-20, 20)
# }

n_fold = 5
# just use one fold
fold = (0,) # (1, 2, 3, 4) # (0, 1, 2, 3, 4)
BATCH_SIZE = 16 if not USE_TPU else 64
TEST_BATCH_SIZE = 1
accumulation_steps = 1
loss_weights = (1, 1)

learning_rate = 1e-3
lr_ratio = np.sqrt(0.1)
reduce_lr_metric = ['loss', 'score', 'both'][0]
patience = 5
warm_up_steps = 1
warm_up_lr_factor = 0.1
num_classes = 4

n_epochs = 50
train_epochs = 50
resume = False
pretrained = not resume


# In[8]:


df_train = pd.read_csv(PATH + 'train.csv')
df_train['class'] = np.argmax(df_train.iloc[:, 1:].values, axis=1)

skf = StratifiedKFold(n_fold, shuffle = True, random_state = 644)
for i_fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['class'].values)):
    df_train.loc[val_idx, 'fold'] = i_fold
df_train['fold'] = df_train['fold'].astype(int)


# In[9]:


class PlantPathologyDataset(Dataset):
    def __init__(self, csv, idx, mode, transform = None, data = None):
        self.csv = csv.reset_index(drop = True)
        self.data = data
        self.filepath_format = PATH + 'images/{}.jpg'
        self.idx = np.asarray(idx).astype('int')
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        if self.data is not None:
            image = self.data[index]
        else:
            img_name = self.csv['image_id'].iloc[index]
            image = cv2.imread(self.filepath_format.format(img_name))
            if image.shape != IMG_SHAPE:
                image = cv2.rotate(image, rotateCode = cv2.ROTATE_90_CLOCKWISE)
        if image.shape != INPUT_IMG_SHAPE:
            image = cv2.resize(image, INPUT_IMG_SHAPE[1::-1], interpolation = cv2.INTER_AREA)

        image = image.astype('uint8')  
        image_origin = image.copy().astype('float32')
        image = self.transform(image).astype('float32') if self.transform is not None else image.astype('float32')
        image, image_origin =  np.rollaxis(image, 2, 0) / 255, np.rollaxis(image_origin, 2, 0) / 255

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            label = self.csv.iloc[index, 1:5].values.astype('float32') # len = 4
            return torch.tensor(image), torch.tensor(image_origin), torch.tensor(label)

def get_train_val_dataloader(i_fold, transforms_train, transforms_val):
    train_idx, valid_idx = np.where((df_train['fold'] != i_fold))[0], np.where((df_train['fold'] == i_fold))[0]
    train_data = np.load(PATH + IMAGE_FILE_NAME[0]) if os.path.isfile(PATH + IMAGE_FILE_NAME[0]) else None
    dataset_train = PlantPathologyDataset(df_train, train_idx, 'train', transform=transforms_train, data = train_data)
    dataset_valid = PlantPathologyDataset(df_train, valid_idx, 'val', transform=transforms_val, data = train_data)
    batch_size, train_sampler, drop_last = BATCH_SIZE, RandomSampler(dataset_train), False
    if USE_TPU and xm.xrt_world_size() > 1:
        batch_size, drop_last = BATCH_SIZE // xm.xrt_world_size(), True
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
        )
    train_loader = DataLoader(dataset_train, batch_size, sampler = train_sampler, drop_last = drop_last, num_workers=4)
    val_loader = DataLoader(dataset_valid, TEST_BATCH_SIZE, num_workers=4)

    return train_loader, val_loader

def get_test_dataloader():
    test_data = np.load(PATH + IMAGE_FILE_NAME[1]) if os.path.isfile(PATH + IMAGE_FILE_NAME[1]) else None
    dataset_test = PlantPathologyDataset(df_test, np.arange(len(df_test)), 'test', data = test_data)
    test_loader = DataLoader(dataset_test, TEST_BATCH_SIZE, sampler=None, num_workers=4)

    return test_loader


# In[10]:


from cvmodels.augment.grid_mask import GridMaskBatch
from cvmodels.augment.fmix import sample_mask

def rand_bbox(img_shape, lam):
    H, W = img_shape
    cut_rat = np.sqrt(1. - lam)  # (1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    # cx, cy = int(W * (np.random.randn()/8+0.5)), int(H * (np.random.randn()/8+0.5))
    cx, cy = np.random.randint(cut_w // 4, W - cut_w // 4), np.random.randint(cut_h // 4, H - cut_h // 4)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class MixupBatch(object):
    def __init__(self, mixup_prob, mixup_alpha, **kwargs):
        self.prob = mixup_prob
        self.alpha = mixup_alpha

    def set_prob(self, epoch, max_epoch):
        self.prob = min(1., epoch / max_epoch)

    def __call__(self, ori_img_batch, img_batch, label_batch):
        ''' 
        ori_img_batch: torch Tensor [N, C, H, W]
        img_batch: torch Tensor [N, C, H, W]
        label_batch: List[Tensor[N, cls]]
        '''
        label_batch_mix, lam = None, 1
        if self.alpha <= 0 or np.random.rand() > self.prob:
            return img_batch, label_batch, label_batch_mix, lam

        batch_size = ori_img_batch.shape[0]
        # lam = np.random.beta(self.alpha, self.alpha)
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = np.maximum(1 - lam, lam)
        lam = torch.from_numpy(lam).view(batch_size, 1, 1, 1).to(ori_img_batch)
        shuffled_idx = torch.randperm(batch_size)
        ori_img_batch = lam * ori_img_batch + (1 - lam) * ori_img_batch[shuffled_idx]
        label_batch_mix = [label[shuffled_idx] for label in label_batch]
        # label_batch = [lam * label + (1 - lam) * label[shuffled_idx] for label in label_batch]

        return ori_img_batch, label_batch, label_batch_mix, lam

class CutmixBatch(object):
    def __init__(self, cutmix_prob, cutmix_alpha, **kwargs):
        self.prob = cutmix_prob
        self.alpha = cutmix_alpha

    def set_prob(self, epoch, max_epoch):
        self.prob = min(1., epoch / max_epoch)

    def __call__(self, ori_img_batch, img_batch, label_batch):
        ''' 
        ori_img_batch: torch Tensor [N, C, H, W]
        img_batch: torch Tensor [N, C, H, W]
        label_batch: List[Tensor[N, cls]]
        '''
        label_batch_mix, lam = None, 1
        if self.alpha <= 0 or np.random.rand() > self.prob:
            return img_batch, label_batch, label_batch_mix, lam

        batch_size = ori_img_batch.shape[0]
        # lam = np.random.beta(self.alpha, self.alpha)
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = np.maximum(1 - lam, lam)
        lam = torch.from_numpy(lam).view(batch_size, 1, 1, 1).to(ori_img_batch)
        shuffled_idx = torch.randperm(batch_size)
        y = ori_img_batch[shuffled_idx].clone().detach()
        label_batch_mix = [label[shuffled_idx] for label in label_batch]
        # label_batch = [lam * label + (1 - lam) * label[shuffled_idx] for label in label_batch]
        for i in range(batch_size):
            bbx1, bby1, bbx2, bby2 = rand_bbox(ori_img_batch.shape[-2:], lam[i])
            ori_img_batch[i, :, bby1:bby2, bbx1:bbx2] = y[i, :, bby1:bby2, bbx1:bbx2]
            lam[i] = 1 - (bbx2 - bbx1) * (bby2 - bby1) / np.prod(ori_img_batch.shape[-2:])

        return ori_img_batch, label_batch, label_batch_mix, lam

class FMixBatch(object):
    def __init__(self, fmix_prob, fmix_alpha=1, decay_power=3, max_soft=0.0, reformulate=False):
        self.prob = fmix_prob
        self.alpha = fmix_alpha
        self.decay_power = decay_power
        self.max_soft = max_soft
        self.reformulate = reformulate

    def set_prob(self, epoch, max_epoch):
        self.prob = min(1., epoch / max_epoch)

    def __call__(self, ori_img_batch, img_batch, label_batch):
        ''' 
        ori_img_batch: torch Tensor [N, C, H, W]
        img_batch: torch Tensor [N, C, H, W]
        label_batch: List[Tensor[N, cls]]
        '''
        label_batch_mix, lam = None, 1
        if np.random.rand() > self.prob:
            return img_batch, label_batch, label_batch_mix, lam

        size = ori_img_batch.shape[-2:]
        lam, mask = sample_mask(self.alpha, self.decay_power, size, self.max_soft, self.reformulate)
        mask = torch.from_numpy(mask).to(ori_img_batch)
        shuffled_idx = torch.randperm(ori_img_batch.size(0))
        ori_img_batch = mask * ori_img_batch + (1 - mask) * ori_img_batch[shuffled_idx]
        label_batch_mix = [label[shuffled_idx] for label in label_batch]
        # label_batch = [lam * label + (1 - lam) * label[shuffled_idx] for label in label_batch]

        return ori_img_batch, label_batch, label_batch_mix, lam

class MixBatch(object):
    def __init__(self, transforms_dict = None, img_mean = np.zeros(1), img_std = np.ones(1), mixup_alpha = 0.4, cutmix_alpha = 1.0, **kwargs):
        transforms_dict = transforms_dict or {}
        self.mix_prob = transforms_dict.get('mix_prob', 0)
        probs = [transforms_dict.get('mixup_prob', 0), transforms_dict.get('cutmix_prob', 0),
                 transforms_dict.get('fmix_prob', 0), transforms_dict.get('grid_prob', 0)]
        self.probs = np.cumsum(probs)
        cutout_ratio = transforms_dict.get('cutout_ratio', (0.05, 0.25))

        self.mixup = MixupBatch(mixup_prob = 1, mixup_alpha = mixup_alpha)
        self.cutmix = CutmixBatch(cutmix_prob = 1, cutmix_alpha = cutmix_alpha)
        self.fmix = FMixBatch(fmix_prob = 1, fmix_alpha=1, decay_power=3, max_soft=0.0, reformulate=False)
        self.gridmask = [GridMaskBatch(num_grid = (3, 6), rotate = 15, mode = 0, prob = 1.),
                         GridMaskBatch(num_grid = (3, 6), rotate = 15, mode = 1, prob = 1.),
                         GridMaskBatch(num_grid = (3, 6), rotate = 15, mode = 2, prob = 1.),]

    def set_prob(self, epoch, max_epoch = 15):
        self.mixup.set_prob(epoch, max_epoch)
        self.cutmix.set_prob(epoch, max_epoch)
        self.fmix.set_prob(epoch, max_epoch)
        for gridmask in self.gridmask:
            gridmask.set_prob(epoch, max_epoch)

    def __call__(self, ori_img_batch, img_batch, label_batch):
        label_batch_mix, lam = None, 1
        if np.random.rand() < self.mix_prob:
            r = np.random.rand()
            if r < self.probs[0]:
                return self.mixup(ori_img_batch, img_batch, label_batch)
            elif r < self.probs[1]:
                return self.cutmix(ori_img_batch, img_batch, label_batch)
            elif r < self.probs[2]:
                return self.fmix(ori_img_batch, img_batch, label_batch)
            elif r < self.probs[3]:
                img_batch = np.random.choice(self.gridmask)(ori_img_batch, img_batch)

        return img_batch, label_batch, label_batch_mix, lam


# In[11]:


transforms_train, transforms_val = None, None
if TRAIN_MODE:
    get_ipython().system('pip install --upgrade -q albumentations')
    get_ipython().system('pip install --upgrade -q imgaug')

    import imgaug.augmenters as iaa
    import albumentations
    from torchvision.transforms import transforms
    
    transforms_train = transforms.Compose([
        lambda image: iaa.Sequential([
            # iaa.Sometimes(0.08, iaa.Rot90([1, 3])),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])(image=image),
        lambda image: albumentations.OneOf([
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=1),
            albumentations.RandomResizedCrop(INPUT_IMG_SHAPE[0], INPUT_IMG_SHAPE[1], scale=(0.9, 1.1), p=1),
        ])(image=image)['image'],
        # lambda image: iaa.SomeOf((3, 6),[
        #     iaa.Multiply((0.8, 1.2), per_channel = 0.5),
        #     iaa.LinearContrast((0.8, 1.2), per_channel = 0.5), 
        #     iaa.AddToHueAndSaturation(value_hue = (-10, 10), value_saturation = (-10, 10),per_channel=True), 
        #     iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.01*255, 0.1*255), per_channel=True)),
        #     iaa.Sometimes(0.5, iaa.OneOf([iaa.GaussianBlur(sigma = (0.1, 0.5)),
        #                                   iaa.AverageBlur(k = (2, 5))])),
        #     iaa.Sometimes(0.3, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        # ], random_order = True)(image=image),
        lambda image: iaa.Sequential([
            iaa.Sometimes(0.7, iaa.Cutout(nb_iterations=1, size=(0.6, 0.7), fill_mode="constant", cval=0))
        ])(image=image),
    ])

    # transforms_train = transforms.Compose([
    #     lambda image: iaa.Sequential([
    #         iaa.Sometimes(0.7, iaa.Cutout(nb_iterations=1, size=(0.6, 0.7), fill_mode="constant", cval=0))
    #     ])(image=image),
    # ])

mix = None


# In[12]:


df_show = df_train.iloc[:100]
data = np.load(PATH + IMAGE_FILE_NAME[0]) if os.path.isfile(PATH + IMAGE_FILE_NAME[0]) else None
dataset_show = PlantPathologyDataset(df_show, list(range(df_show.shape[0])), 'train', transform=transforms_train, data = data)

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
for i in range(1):
    f, axarr = plt.subplots(1,5)
    for p in range(5):
        idx = np.random.randint(0, len(dataset_show))
        t0 = time.time()
        img, img_org, label = dataset_show[idx]
        # print(f"{time.time()-t0:.4f}")
        axarr[p].imshow(img.transpose(0, 1).transpose(1,2).squeeze())
        axarr[p].set_title(idx)


# In[13]:


from timm.models.layers import Mish as MishJit, Swish as SwishJit

class RGBNorm(nn.Module):
    IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def forward(self, x):
        _, C, _, _ = x.shape
        if C == 1:
            x = x.repeat((1, 3, 1, 1))
        x = (x - self.IMAGENET_MEAN.to(x)) / self.IMAGENET_STD.to(x)
        return x

class Mish(nn.Module):
    def __init__(self, *args, **kwagrs):
        super(Mish, self).__init__()

    def forward(self,x):
        return x.mul(F.softplus(x).tanh())

def gem(x, kernel_size, stride = None, p = 3, eps = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size, stride).pow(1./p)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class Classifier(nn.Module):
    """
    backbone should output feature maps with [N, out_channel, H, W] shape,
    backbone should have attribute `out_channel`
    """
    def __init__(self, backbone, num_classes, **kwargs):
        super().__init__()
        # self.norm = nn.Conv2d(1, 3, 7, 1, 7//2)
        self.norm = RGBNorm()
        self.backbone = backbone
        self.gfc = nn.Sequential(*[nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   nn.Linear(backbone.out_channel, 2048),
                                   Mish()])
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.metric_classify = ArcMarginProduct(2048, num_classes)
        self.cls_head = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = self.norm(x)
        x = self.backbone(x)
        x = self.gfc(x)
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.cls_head(dropout(x))
            else:
                out += self.cls_head(dropout(x))
        out /= len(self.dropouts)
        metric_output = self.metric_classify(x)

        return out, metric_output

def load_pretrained_model(model, model_path = '', url = '', skip = (), conversion = ()):
    import os
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location = 'cpu')
        if not USE_TPU or xm.is_master_ordinal():
            print('=> loading pretrained model {}'.format(model_path))
    elif url != '':
        import torch.utils.model_zoo as model_zoo
        state_dict = model_zoo.load_url(url, progress = False, map_location = 'cpu')
        if not USE_TPU or xm.is_master_ordinal():
            print('=> loading pretrained model {}'.format(url))
    else:
        return

    conversion = np.array(conversion).reshape(-1, 2) if len(conversion) else []
    model_dict = model.state_dict()
    pretrained_state_dict = {}
    for ks in state_dict.keys():
        if ks in model_dict.keys() and all(s not in ks for s in skip):
            km = ks
            for _km, _ks in conversion:
                if ks == _ks:
                    km = _km
                    break
            pretrained_state_dict[km] = state_dict[ks]
    if not USE_TPU or xm.is_master_ordinal():
        print(f"=> loading pretrained model weight length {len(pretrained_state_dict)} / total_state_dict {len(state_dict)} / total_model_dict {len(model_dict)}")
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(model_dict, strict = False)


# In[14]:


from cvmodels.models.efficientnet import EfficientNet
from timm.models.efficientnet import _gen_efficientnet

def efficientnet_b4(pretrained = False, **kwargs):
    if not USE_TPU or xm.is_master_ordinal():
        print('=> create `efficientnet_b4`')
    backbone = EfficientNet.from_name('efficientnet-b4')
    backbone._fc = None
    backbone.out_channel = backbone.out_channels
    # backbone_kwargs = {'bn_eps': 1e-3, 'pad_type': 'same'}
    # backbone = _gen_efficientnet(
    #     'tf_efficientnet_b1', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **backbone_kwargs)
    # backbone.out_channel = backbone.num_features
    backbone.forward = backbone.forward_features
    if pretrained:
        pretrained_file_name = ''  # 'efficientnet-b1-f1951068.pth'
        url = 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth'
        load_pretrained_model(backbone, model_path = MODEL_PATH + pretrained_file_name, url = url,
                              skip = ('.num_batches_tracked', '_fc.', ))
    model = Classifier(backbone, num_classes)
    return model


# In[15]:


def cross_entropy(preds, trues, reduction = 'mean', **kwargs):
    if reduction == 'mean':
        return -torch.sum(trues * F.log_softmax(preds, dim = 1), dim = 1).mean()
    elif reduction == 'sum':
        return -torch.sum(trues * F.log_softmax(preds, dim = 1), dim = 1).sum()
    elif reduction == 'none':
        return -torch.sum(trues * F.log_softmax(preds, dim = 1), dim = 1)

def focol_loss(preds, trues, alpha = 4, gamma = 2,  reduction = 'mean', **kwargs):
    probs = F.softmax(preds, dim = 1)
    pos_loss = -torch.sum(trues * alpha * (1 - probs) ** gamma * torch.log(probs), dim = 1)
    # neg_loss = -torch.sum((trues == 0) * (1 - alpha) * probs ** gamma * torch.log(1 - probs), dim = 1)
    if reduction == 'mean':
        return pos_loss.mean()
    elif reduction == 'sum':
        return pos_loss.sum()
    elif reduction == 'none':
        return pos_loss

def ohem_loss(preds, trues, ohem_rate = 1, reduction = 'mean', **kwargs):
    ohem_rate = max(0, min(1, ohem_rate))
    loss = -torch.sum(trues * F.log_softmax(preds, dim = 1), dim = 1)
    loss = torch.topk(loss, int(ohem_rate * loss.size(0)))[0] if 0 <= ohem_rate < 1 else loss
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.s = s
        self.cos_m = math.cos(m)             #  0.87758
        self.sin_m = math.sin(m)             #  0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  #  0.23971

    def forward(self, logits, labels):
        logits = logits.float()  # float16 to float32 (if used float16)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = cross_entropy(output, labels, self.reduction)
        return loss / 2

def criterion(logits, metric_logits, trues, is_val = False, lam = 1, **kwargs):
    assert len(trues) == len(logits) == len(metric_logits)
    weights = loss_weights
    if not is_val:
        loss_0 = cross_entropy(logits, trues.float(), reduction='mean') * lam
        loss_metric = ArcFaceLoss(reduction='mean')(metric_logits, trues.float()) * lam
        loss = (loss_0 * weights[0] + loss_metric * weights[1]) / sum(weights)
    else:
        loss_0 = cross_entropy(logits, trues.float(), reduction='sum') * lam
        loss_metric = ArcFaceLoss(reduction='sum')(metric_logits, trues.float()) * lam 
        loss = (loss_0 * weights[0] + loss_metric * weights[1]) / sum(weights)

    return loss, (loss_0.detach(), loss_metric.detach())


# In[16]:


from sklearn.metrics import roc_auc_score

def get_score(submission, solution):
    roc_score = roc_auc_score(solution, submission)
    predictions = np.argmax(submission, axis = 1)
    trues = np.argmax(solution, axis = 1)
    hard_idx = np.where(np.max(submission, axis = 1) <= 0.9)[0]
    easy_idx = np.where(np.max(submission, axis = 1) > 0.9)[0]
    score = np.mean(predictions == trues)
    hard_score = np.mean(predictions[hard_idx] == trues[hard_idx]) if len(hard_idx) else 0
    easy_score = np.mean(predictions[easy_idx] == trues[easy_idx]) if len(easy_idx) else 0

    return roc_score, score, hard_score, easy_score

def show_running_time(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            t0 = time.time()
            res = func(*args, **kw)
            print(f"{text} time: {time.time()-t0:.1f} s")
            return res
        return wrapper
    return decorator
            
def clear_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        torch.cuda.empty_cache()
        gc.collect()
        return func(*args, **kw)
    return wrapper

@show_running_time('train')
@clear_cache
def train_epoch(model, loader, device, optimizer, verbose = False):
    model.train()
    train_loss = []
    optimizer.zero_grad()
    acc_steps = accumulation_steps
    steps = len(loader) // acc_steps * acc_steps # if not USE_TPU else None
    progress_bar = tqdmauto(loader) if (not USE_TPU or xm.is_master_ordinal()) and verbose else None
    for batch_idx, (img_batch, origin_img_batch, label_batch) in enumerate(loader):
        ### mixup & cutmix & cutout
        label_batch_mix, lam = None, 1
        if mix is not None:
            img_batch, label_batch, label_batch_mix, lam = mix(img_batch, origin_img_batch, label_batch)  # process from origin
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        label_batch_mix = None if label_batch_mix is None else label_batch_mix.to(device)
        lam = lam.to(device) if isinstance(lam, torch.Tensor) else lam
        if steps is not None and batch_idx >= steps:
            acc_steps = 1
        logits, metric_logits = model(img_batch)
        loss, _ = criterion(logits, metric_logits, label_batch, is_val = False, lam = lam)
        if label_batch_mix is not None:
            loss_mix = criterion(logits, metric_logits, label_batch_mix, is_val = False, lam = 1 - lam)
            loss = loss + loss_mix
        loss = loss / acc_steps
        loss.backward()
        if (batch_idx + 1) % acc_steps == 0:
            if USE_TPU:
                xm.optimizer_step(optimizer, barrier = True)
            else:
                optimizer.step()
            optimizer.zero_grad()

        loss_np = loss.detach().cpu().item() * acc_steps
        train_loss.append(loss_np)
        if (not USE_TPU or xm.is_master_ordinal()) and verbose and (batch_idx <= 10 or (batch_idx - 10) % verbose_step == 0):
            progress_bar.set_postfix_str(f"loss: {loss_np:.4f}, smooth_loss: {np.mean(train_loss[-20:]):.4f}")
            progress_bar.update(1 if batch_idx <= 10 else 30)

    return np.asarray(train_loss).mean()

@show_running_time('val')
@clear_cache
def val_epoch(model, loader, device):
    model.eval()
    val_loss, val_loss1, val_loss2 = 0, 0, 0
    preds, metric_preds, trues = [], [], []

    with torch.no_grad():
        for img_batch, origin_img_batch, label_batch in loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            logits, metric_logits = model(img_batch)
            loss, (loss1, loss2) = criterion(logits, metric_logits, label_batch, is_val = True)

            val_loss += loss.detach().cpu().item()
            val_loss1 += loss1.detach().cpu().item()
            val_loss2 += loss2.detach().cpu().item()
            preds.append(F.softmax(logits, dim = 1).cpu().numpy())
            metric_preds.append(F.softmax(metric_logits, dim = 1).cpu().numpy())
            trues.append(label_batch.cpu().numpy())

    preds = np.concatenate(preds)
    metric_preds = np.concatenate(metric_preds)
    trues = np.concatenate(trues)
    scores1 = get_score(preds, trues)
    scores2 = get_score(metric_preds, trues)
    val_result = (val_loss / len(trues), val_loss1 / len(trues), val_loss2 / len(trues), preds, metric_preds, trues, scores1, scores2)
    return val_result


# In[17]:


def resume_model(model, optimizer = None, lr_scheduler = None, resume_file_path = '', device = 'cpu'):
    other_param = {}
    history, last_epoch, best_val_loss, best_val_score = None, None, None, None
    if os.path.isfile(resume_file_path):
        checkpoint = torch.load(resume_file_path, map_location = device)
        model.load_state_dict(checkpoint.pop('state_dict'), strict = False)
        print(f"load model success!  last_epoch: {checkpoint['last_epoch']}")
        if optimizer is not None and lr_scheduler is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.2e}")
        checkpoint.pop('optimizer', None)
        checkpoint.pop('lr_scheduler', None)
        other_param.update(checkpoint)
        print(f"best_val_score: {other_param['best_val_score']:.4f}, " +
              f"best_val_loss: {other_param['best_val_loss']:.4f}\n")
    return model, optimizer, lr_scheduler, other_param


# In[18]:


model_class = efficientnet_b4


# In[19]:


@clear_cache
def train(i_fold, verbose = False):
    device = DEVICE if not USE_TPU else xm.xla_device()
    devices = [device] if not USE_TPU else [device]
    lr = learning_rate if not USE_TPU else learning_rate * max(len(devices), xm.xrt_world_size())
    batch_size = BATCH_SIZE if not USE_TPU else BATCH_SIZE // max(len(devices), xm.xrt_world_size())
    filename = f"{version_name}_{network_name}_fold_{i_fold}.pth"
    log_name = f"log_{version_name}_{network_name}_fold_{i_fold}.txt"

    train_loader, val_loader = get_train_val_dataloader(i_fold, transforms_train, transforms_val)

    model = model_class(pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr,momentum=0.9, weight_decay=1e-5)
    # if use_amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 1e-7)

    train_param = {}
    if resume:
        if SAVE_PATH != MODEL_PATH and os.path.isfile(MODEL_PATH + log_name):
            shutil.copy(MODEL_PATH + log_name, SAVE_PATH + log_name)
        model, optimizer, lr_scheduler, train_param =             resume_model(model, optimizer, lr_scheduler, MODEL_PATH + filename, device)
    last_epoch = train_param.get('last_epoch', -1)
    best_val_loss, best_val_score = train_param.get('best_val_loss', 2**20), train_param.get('best_val_score', 0)
    best_epoch, best_score_epoch = train_param.get('best_epoch', -1), train_param.get('best_score_epoch', -1)
    history = train_param.get('history', pd.DataFrame())

    def save_model(model, optimizer, lr_scheduler, is_best = False, is_best_score = False):
        model_state = {'state_dict': model.state_dict(),
                       'last_epoch': last_epoch, 'history': history, 'i_fold': i_fold,
                       'best_val_loss': best_val_loss, 'best_val_score': best_val_score,
                       'best_epoch': best_epoch, 'best_score_epoch': best_score_epoch,
                       'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}
        model_path = SAVE_PATH + f"{version_name}_{network_name}_fold_{i_fold}.pth"

        if not USE_TPU:
            torch.save(model_state, model_path)
        elif xm.is_master_ordinal():
            xm.save(model_state, model_path)
        else:
            return
        if is_best:
            best_model_path = SAVE_PATH + f"{version_name}_{network_name}_fold_{i_fold}_best.pth"
            shutil.copy(model_path, best_model_path)
        if is_best_score:
            best_model_path = SAVE_PATH + f"{version_name}_{network_name}_fold_{i_fold}_best_score.pth"
            shutil.copy(model_path, best_model_path)

    def show_attributes():
        attr = (f"train_set count: {len(train_loader.dataset)} val_set count: {len(val_loader.dataset)}\n" +
                f"save_path: {SAVE_PATH}\n" + f"batch_size: {batch_size}, accumulation_steps: {accumulation_steps}\n" +
                f"current lr: {optimizer.param_groups[0]['lr']:.2e}, reduce_lr_metric: {reduce_lr_metric}\n" +
                f"best_val_score: {best_val_score:.5f}, best_val_loss: {best_val_loss:.5f}\n" +
                f"best_epoch: {best_epoch}, best_score_epoch: {best_score_epoch}\n" +
                f"last_epoch: {last_epoch}, train_epochs: {train_epochs}, n_epochs: {n_epochs}\n")
        if hasattr(lr_scheduler, 'num_bad_epochs'):
            attr += f"num_bad_epochs: {lr_scheduler.num_bad_epochs}, min_lr: {lr_scheduler.min_lrs[0]:.2e}, patience: {lr_scheduler.patience}\n"
        print(attr + '\n')
        return attr

    try:
        if not USE_TPU or xm.is_master_ordinal():
            attr = show_attributes()
        if (not USE_TPU or xm.is_master_ordinal()) and (not resume or not os.path.isfile(SAVE_PATH + log_name)):
            content = (f"\nTraining fold {i_fold}\nfilename: {filename}\ntrain_data: {IMAGE_FILE_NAME[0]}\n" + 
                       f"train_img_size: {INPUT_IMG_SHAPE}\nbatch_size: {batch_size}\naccumulation_steps: {accumulation_steps}\nn_epochs: {n_epochs}\n"+
                       f"model_class: {model_class.__name__}\noptimizer: {optimizer}\nlr_scheduler: {lr_scheduler}\n")
            content += '\n\n' + f"{time.ctime()} Training start\n"
            print(content)
            with open(SAVE_PATH + log_name, 'a') as appender:
                appender.write(content)
        for epoch in range(last_epoch + 1, train_epochs):
            cur_lr = optimizer.param_groups[0]['lr']
            train_loss = train_epoch(model, train_loader, device, optimizer, verbose)
            val_result = val_epoch(model, val_loader, device)
            val_loss, val_loss1, val_loss2, preds, metric_preds, trues, scores1, scores2 = val_result

            last_epoch, val_score = epoch, scores1[1]
            is_best_loss, is_best_score = val_loss < best_val_loss, val_score > best_val_score
            is_best = (is_best_loss or reduce_lr_metric is 'score') and (is_best_score or reduce_lr_metric is 'loss')
            best_val_loss, best_val_score = min(val_loss, best_val_loss), max(val_score, best_val_score)
            lr_scheduler.step()

            content =( f"{time.ctime()} Epoch {epoch}, lr: {cur_lr:.2e}, " +
                    f"train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, val loss1: {val_loss1:.4f}, val loss2: {val_loss2:.4f}, " + 
                    f"roc_score1: {scores1[0]:.4f}, roc_score2: {scores2[0]:.4f}, " +
                    f"score1: {scores1[1]:.4f}, hard_score1: {scores1[2]:.4f}, easy_score1: {scores1[3]:.4f}, " + 
                    f"score2: {scores2[1]:.4f}, hard_score2: {scores2[2]:.4f}, easy_score2: {scores2[3]:.4f}")
            if is_best:
                best_epoch = epoch
                content += f"  => best metric"

            if not USE_TPU or xm.is_master_ordinal():
                print(content)
                with open(SAVE_PATH + log_name, 'a') as appender:
                    appender.write(content + '\n')
                _h = pd.DataFrame({'train_loss': [train_loss], 'val_loss': [val_loss], 
                                   'roc_score1': [scores1[0]], 'roc_score2': [scores2[0]],
                                   'score1': [scores1[1]], 'score2': [scores2[1]]})
                history = history.append(_h, ignore_index = True)
                save_model(model, optimizer, lr_scheduler, is_best, is_best_score)
                history.to_csv(SAVE_PATH + f"history_{version_name}_fold_{i_fold}.csv")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        if not USE_TPU or xm.is_master_ordinal():
            show_attributes()


# In[20]:


if TRAIN_MODE:
    if DEVICE == torch.device('cuda'):
        torch.backends.cudnn.benchmark = True
        print('cudnn.benchmark = True')
    if USE_TPU:
        torch.set_default_tensor_type('torch.FloatTensor')
    for i_fold in fold:
        seed_everything(random_seed)
        train(i_fold, verbose = False)


# In[21]:


def visualize_training_history(history, start_loc = 0):
    history = history.loc[start_loc:]
    plt.figure(figsize=(15,10))
    plt.subplot(311)
    train_loss = history['train_loss'].dropna()
    plt.plot(train_loss.index, train_loss, label = 'train_loss')
    plt.legend()

    plt.subplot(312)
    val_loss = history['val_loss'].dropna()
    plt.plot(val_loss.index, val_loss, label = 'val_loss')
    # plt.scatter(val_loss.index, val_loss)
    plt.legend()

    plt.subplot(313)
    val_score = history['score1'].dropna()
    plt.plot(val_score.index, val_score, label = 'score1')
    val_score = history['score2'].dropna()
    plt.plot(val_score.index, val_score, label = 'score2')
    # plt.scatter(val_score.index, val_score)
    plt.legend()
    plt.show()


# In[22]:


for i_fold in range(n_fold):
    if os.path.isfile(SAVE_PATH + f"history_{version_name}_fold_{i_fold}.csv"):
        history_file = SAVE_PATH + f"history_{version_name}_fold_{i_fold}.csv"
    elif os.path.isfile(MODEL_PATH + f"history_{version_name}_fold_{i_fold}.csv"):
        history_file = MODEL_PATH + f"history_{version_name}_fold_{i_fold}.csv"
    else:
        continue
    print(f"show {history_file}")
    history = pd.read_csv(history_file)
    visualize_training_history(history, start_loc = 2)


# In[23]:


def get_models(model_files, model_class):
    device = DEVICE if not USE_TPU else xm.xla_device()
    models = []
    params = []
    for model_f in model_files:
        if os.path.isfile(SAVE_PATH + model_f):
            resume_file_path = SAVE_PATH + model_f
        elif os.path.isfile(MODEL_PATH + model_f):
            resume_file_path = MODEL_PATH + model_f
        else:
            continue
        model = model_class()
        model, _, _, other_params = resume_model(model, resume_file_path = resume_file_path, device = 'cpu')
        model = model.to(device)
        model.eval()
        other_params['sub_mode'] = ''
        if 'best.' in model_f:
            other_params['sub_mode'] = '_best'
        elif 'best_score.' in model_f:
            other_params['sub_mode'] = '_best_score'
        models.append(model)
        params.append(other_params)
    return models, params

@show_running_time('test')
def predict(model, loader):
    device = DEVICE if not USE_TPU else xm.xla_device()
    model.eval()
    preds, metric_preds = [], []

    with torch.no_grad():
        for img_batch in loader:
            img_batch = img_batch.to(device)
            logits, metric_logits = model(img_batch)
            preds.append(F.softmax(logits, dim = 1).cpu().numpy())
            metric_preds.append(metric_logits.cpu().numpy())

    preds = np.concatenate(preds)
    metric_preds = np.concatenate(metric_preds)

    return preds, metric_preds


# In[24]:


@clear_cache
def predict_and_submission(test_loader, model_files):
    submission1 = submission.copy()
    submission2 = submission.copy()
    n_fold_preds, n_fold_metric_preds = [], []

    models, params = get_models(model_files, model_class)
    for model in models:
        preds, metric_preds = predict(model, test_loader)
        n_fold_preds.append(preds)
        n_fold_metric_preds.append(metric_preds)

    sub_name = '_'.join([f"fold_{p['i_fold']}_{p['last_epoch']+1}ep{p['sub_mode']}" for p in params])
    submission_name1 = f"{version_name}_{network_name}_{len(model_files)}_fold_submission1_{sub_name}.csv"
    submission_name2 = f"{version_name}_{network_name}_{len(model_files)}_fold_submission2_{sub_name}.csv"
    
    with open(SAVE_PATH + f'log_{version_name}_{network_name}_submission.txt', 'a') as appender:
        content = (f"\n{time.ctime()}, submission file: {submission_name1}\n" +
                   f"test_data: {IMAGE_FILE_NAME[0]}\ntest_img_size: {INPUT_IMG_SHAPE}\nfrom models:\n")
        for s, p in zip(model_files, params):
            content += s + f", i_fold: {p['i_fold']}, train_epochs: {p['last_epoch']+1}ep, best_val_score: {p['best_val_score']:.4f}, best_val_loss: {p['best_val_loss']:.4f}\n"
        print(content)
        appender.write(content + '\n\n')
    
    # dup_ids = submission[submission['image_id'].isin(['Test_1407', 'Test_829'])].index.values
    preds = np.stack(n_fold_preds).mean(axis = 0)
    submission1.iloc[:, 1:] = preds
    # submission1.iloc[dup_ids, 1:5] = df_train[df_train['image_id'].isin(['Train_1703', 'Train_1505'])].iloc[:, 1:5].values
    submission1.to_csv(SAVE_PATH + submission_name1, index=False)

    # metric_preds = np.stack(n_fold_metric_preds).mean(axis = 0)
    # submission2.iloc[:, 1:] = metric_preds
    # submission2.iloc[dup_ids, 1:5] = df_train[df_train['image_id'].isin(['Train_1703', 'Train_1505'])].iloc[:, 1:5].values
    # submission2.to_csv(SAVE_PATH + submission_name2, index=False)

    # submission_name1 = f"{version_name}_{network_name}_{len(model_files)}_fold_submission1_{sub_name}_max.csv"
    # submission_name2 = f"{version_name}_{network_name}_{len(model_files)}_fold_submission2_{sub_name}_max.csv"
    # max_preds = np.stack(n_fold_preds).max(axis = 0)
    # preds = np.stack(n_fold_preds).min(axis = 0)
    # preds[np.arange(len(preds)), np.argmax(max_preds, axis=1)] = np.max(max_preds, axis=1)
    # submission1.iloc[:, 1:] = preds
    # submission1.iloc[dup_ids, 1:5] = df_train[df_train['image_id'].isin(['Train_1703', 'Train_1505'])].iloc[:, 1:5].values
    # submission1.to_csv(SAVE_PATH + submission_name1, index=False)

    # max_metric_preds = np.stack(n_fold_metric_preds).max(axis = 0)
    # metric_preds = np.stack(n_fold_metric_preds).min(axis = 0)
    # metric_preds[np.arange(len(metric_preds)), np.argmax(max_metric_preds, axis=1)] = np.max(max_metric_preds, axis=1)
    # submission2.iloc[:, 1:] = metric_preds
    # submission2.iloc[dup_ids, 1:5] = df_train[df_train['image_id'].isin(['Train_1703', 'Train_1505'])].iloc[:, 1:5].values    # submission1.to_csv(SAVE_PATH + f"{version_name}_{network_name}_{len(model_files)}_fold_submission1_max.csv", index=False)
    # submission2.to_csv(SAVE_PATH + f"{version_name}_{network_name}_{len(model_files)}_fold_submission2_max.csv", index=False)
    display(submission1.tail(5))


# In[25]:


df_test = pd.read_csv(PATH + 'test.csv')
submission = pd.read_csv(PATH + 'sample_submission.csv')

test_loader = get_test_dataloader()

model_files = [
    f"{version_name}_{network_name}_fold_{0}_best.pth",
#     f"{version_name}_{network_name}_fold_{1}_best.pth",
#     f"{version_name}_{network_name}_fold_{2}_best.pth",
#     f"{version_name}_{network_name}_fold_{3}_best.pth",
#     f"{version_name}_{network_name}_fold_{4}_best.pth",
]
predict_and_submission(test_loader, model_files)

model_files = [
    f"{version_name}_{network_name}_fold_{0}_best_score.pth",
    # f"{version_name}_{network_name}_fold_{1}_best_score.pth",
    # f"{version_name}_{network_name}_fold_{2}_best_score.pth",
    # f"{version_name}_{network_name}_fold_{3}_best_score.pth",
    # f"{version_name}_{network_name}_fold_{4}_best_score.pth",
]
# predict_and_submission(test_loader, model_files)


# In[26]:


if os.path.isdir('./ComputerVision/'):
    shutil.rmtree('./ComputerVision/')
if os.path.isdir('./pytorch-image-models/'):
    shutil.rmtree('./pytorch-image-models/')
for file in os.listdir('./'):
    if any([(s in file) for s in ('torch-xla', 'torch-nightly', 'torchvision-nightly')]):
        os.remove('./' + file)

