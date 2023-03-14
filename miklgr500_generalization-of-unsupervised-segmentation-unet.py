#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import pdb
import time
import warnings
import random
import zipfile
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
import seaborn as sns
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
warnings.simplefilter("ignore")


# In[2]:


get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null 2>&1 # Install segmentations_models.pytorch, with no bash output')
import segmentation_models_pytorch as smp


# In[3]:


def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(121)


# In[4]:


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=5, # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms

def train_transform_creator():
    return get_transforms(
        'train',
        512,
        0.,
        1.
    )


# In[5]:


class OSICDataset(Dataset):
    def __init__(self, fnames, img_folder, mask_folder, transforms_creator):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transforms = transforms_creator()
        self.fnames = fnames

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        d = pydicom.dcmread(os.path.join(self.img_folder, image_id))
        image = (d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000)
        image[image < 1.5] = 0.
        mask = cv2.imread(os.path.join(self.mask_folder, f'{image_id[:-4]}.jpg'), cv2.IMREAD_GRAYSCALE)
        image = np.dstack([image, image, image])
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


# In[6]:


label = []
fnames = []
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

for i, p in tqdm(enumerate(train.Patient.unique())):
    for k in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/'):
        m = cv2.imread(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear/{p}/{k[:-4]}.jpg')
        if m is None:
            continue
        fnames.append(f'{p}/{k}')
        label.append(m.sum())


# In[7]:


_label = np.array(label) // 1000000
_label[_label > 40] = 40
sns.distplot(_label)


# In[8]:


tr_fnames, vl_fnames, _, _ = train_test_split(fnames, _label, train_size=0.85, random_state=41, shuffle=True, stratify=_label)


# In[9]:


tr_image_dataset = OSICDataset(tr_fnames, 
                            '../input/osic-pulmonary-fibrosis-progression/train', 
                            '../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear', 
                            train_transform_creator)
vl_image_dataset = OSICDataset(vl_fnames, 
                            '../input/osic-pulmonary-fibrosis-progression/train', 
                            '../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear', 
                            train_transform_creator)


# In[10]:


tr_dataloader = DataLoader(
        tr_image_dataset,
        batch_size=16,
        num_workers=5,
        pin_memory=True,
        shuffle=True,
    )

vl_dataloader = DataLoader(
        vl_image_dataset,
        batch_size=16,
        num_workers=5,
        pin_memory=True,
        shuffle=True,
    )


# In[11]:


batch = next(iter(tr_dataloader)) # get a batch from the dataloader
images, masks = batch


# In[12]:


idx = random.choice(range(32))
plt.figure(figsize=(20,20))
plt.imshow(images[idx].reshape((512, 512, 3)), cmap='bone')
plt.imshow(masks[idx][0], alpha=0.3, cmap='Reds')
plt.show()


# In[13]:


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


# In[14]:


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (epoch_loss, dice, dice_neg, dice_pos, iou))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


# In[15]:


model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
model


# In[16]:


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 6
        self.batch_size = {"train": 16, "val": 16}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 1e-3
        self.num_epochs = 7
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = FocalLoss(1.5)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: loader
            for phase, loader in zip(self.phases, [tr_dataloader, vl_dataloader])
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)

        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(self.net, 'model.pth')
            print()


# In[17]:


model_trainer = Trainer(model)
model_trainer.start()


# In[18]:


import gc
del tr_dataloader, vl_dataloader, Trainer, tr_image_dataset, vl_image_dataset
gc.collect()


# In[19]:


model = torch.load("model.pth")
model


# In[20]:


def zip_and_remove(path):
    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)
    
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            ziph.write(file_path)
            os.remove(file_path)
    
    ziph.close()
    
if not os.path.exists('train_mask_unet_prob/'):
    os.mkdir('train_mask_unet_prob/')
    
if not os.path.exists('train_mask_unet/'):
    os.mkdir('train_mask_unet/')


# In[21]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_data = {}
for p in train.Patient.values:
    train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')
    
keys = [k for k in list(train_data.keys()) if k not in ['ID00011637202177653955184', 'ID00052637202186188008618']]


# In[22]:


transforms = get_transforms('valid', 512, 0., 1.)

for k in tqdm(keys, total=len(keys)):
    x = []
    if not os.path.exists('train_mask_unet_prob/' + k):
        os.mkdir('train_mask_unet_prob/' + k)
        
    if not os.path.exists('train_mask_unet/' + k):
        os.mkdir('train_mask_unet/' + k)
        
    for i in train_data[k]:
        d =  pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')
        image = (d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000)
        image[image < 1.5] = 0.
        image = np.dstack([image, image, image])
        image = transforms(image=image)['image'].reshape((1, 3, 512, 512)).to('cuda')
        
        mask = model(image).detach().to('cpu').numpy()[0, 0, ...]
        
        cv2. imwrite('train_mask_unet_prob/' + k + f'/{i[:-4]}' + '.jpg', mask)
        cv2. imwrite('train_mask_unet/' + k + f'/{i[:-4]}' + '.jpg', np.uint8(mask > 0.5))


# In[23]:


if not os.path.exists('test_mask_unet_prob/'):
    os.mkdir('test_mask_unet_prob/')
    
if not os.path.exists('test_mask_unet/'):
    os.mkdir('test_mask_unet/')


# In[24]:


test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test_data = {}
for p in test.Patient.values:
    test_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')
    
keys = [k for k in list(test_data.keys()) if k not in ['ID00011637202177653955184', 'ID00052637202186188008618']]


# In[25]:


transforms = get_transforms('valid', 512, 0., 1.)

for k in tqdm(keys, total=len(keys)):
    x = []
    if not os.path.exists('test_mask_unet_prob/' + k):
        os.mkdir('test_mask_unet_prob/' + k)
        
    if not os.path.exists('test_mask_unet/' + k):
        os.mkdir('test_mask_unet/' + k)
        
    for i in train_data[k]:
        d =  pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/test/{k}/{i}')
        image = (d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000)
        image[image < 1.5] = 0.
        image = np.dstack([image, image, image])
        image = transforms(image=image)['image'].reshape((1, 3, 512, 512)).to('cuda')
        
        mask = model(image).detach().to('cpu').numpy()[0, 0, ...]
        
        cv2. imwrite('test_mask_unet_prob/' + k + f'/{i[:-4]}' + '.jpg', mask)
        cv2. imwrite('test_mask_unet/' + k + f'/{i[:-4]}' + '.jpg', np.uint8(mask > 0.5))


# In[26]:


masks = os.listdir('train_mask_unet_prob/ID00408637202308839708961')
masks = sorted(masks, key=lambda x: int(x[:-4]))
len(masks)


# In[27]:


_, axs = plt.subplots(6, 6, figsize=(24, 24))
axs = axs.flatten()
for m, ax in zip(masks, axs):
    img = cv2.imread(os.path.join('train_mask_unet_prob/ID00408637202308839708961', m), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img)
    ax.axis('off')
plt.show()


# In[28]:


zip_and_remove('train_mask_unet_prob')
zip_and_remove('train_mask_unet')
zip_and_remove('test_mask_unet_prob')
zip_and_remove('test_mask_unet')

