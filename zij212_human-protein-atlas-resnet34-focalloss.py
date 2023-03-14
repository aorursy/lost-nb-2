#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from typing import List, Dict


# In[2]:


TRAIN_CSV = '/kaggle/input/human-protein-atlas-train-val-split/train_df.csv'
VAL_CSV = '/kaggle/input/human-protein-atlas-train-val-split/test_df.csv'
TEST_CSV = '/kaggle/input/human-protein-atlas-image-classification/sample_submission.csv'

TRAIN_DIR = '/kaggle/input/human-protein-atlas-image-classification/train'
TEST_DIR = '/kaggle/input/human-protein-atlas-image-classification/test'

STATS_DIR = '/kaggle/input/human-protein-atlas-data-stats/stats.pt'


# In[3]:


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)


# In[4]:


text_labels = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Peroxisomes",
9:  "Endosomes",
10:  "Lysosomes",
11:  "Intermediate filaments",   
12:  "Actin filaments",
13:  "Focal adhesion sites",   
14:  "Microtubules",
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle",
18:  "Microtubule organizing center",  
19:  "Centrosome",
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions", 
23:  "Mitochondria",
24:  "Aggresome",
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}

NUM_LABELS = len(text_labels)
print(f"There are {NUM_LABELS} labels")


# In[5]:


FILTERS = ['red', 'green', 'blue', 'yellow']

def load_image(image_id, ddir):
    """
    return: 4-channel PIL Image
    """
    return Image.merge('RGBA', [Image.open(f"{TRAIN_DIR}/{image_id}_{f}.png") for f in FILTERS])


def encode(image_labels: str):
    """
    image_labels: label(s) of an image, e.g. "25 0"
    return: tensor of size (28)
    """
    target = torch.zeros(NUM_LABELS)
    for label in image_labels.split():
        target[int(label)] = 1
    return target


# In[6]:


class ProteinLocalizationDataset(Dataset):
    def __init__(self, df, ddir, transform=None):
        self.df = df
        self.ddir = ddir
        self.transform = transform
    
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        image_id, image_labels = self.df.loc[idx]
        image = load_image(image_id, self.ddir)
        if self.transform:
            image = self.transform(image)
        return image, encode(image_labels)


# In[7]:


stats = torch.load(STATS_DIR)
stats


# In[8]:


train_tfms = T.Compose([
    T.RandomRotation(10),
    T.RandomHorizontalFlip(),
    T.ToTensor(), 
    T.Normalize(*stats, inplace=True)
])

test_tfms = T.Compose([
    T.ToTensor(), 
    T.Normalize(*stats, inplace=True)
])


# In[9]:


train_ds = ProteinLocalizationDataset(
    train_df, TRAIN_DIR, transform=train_tfms)
val_ds = ProteinLocalizationDataset(
    val_df, TRAIN_DIR, transform=test_tfms)


# In[10]:


batch_size = 32


# In[11]:


train_dl = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(
    val_ds, batch_size*2, shuffle=False, num_workers=2, pin_memory=True)


# In[12]:


class MultiLocalizationClassification(nn.Module):
    def training_step(self, batch):
        imgs, targets = batch
        out = self(imgs)
        loss = CRITERION(out, targets)
        return loss
  

    def validation_step(self, batch):
        imgs, targets = batch
        out = self(imgs)
        loss = CRITERION(out, targets)
        score = f_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach()}
    
    
    @staticmethod
    def validation_epoch_end(outputs: List):
        val_losses = [o['val_loss'] for o in outputs]
        val_scores = [o['val_score'] for o in outputs]
        
        val_loss = torch.mean(torch.stack(val_losses))
        val_score = torch.mean(torch.stack(val_scores))
        return {'val_loss': val_loss.item(), 'val_score': val_score.item()}

    
    @staticmethod
    def epoch_end(epoch_num: int, result: Dict):
        train_loss, val_loss, val_score = result['train_loss'], result['val_loss'], result['val_score']
        print(f"Epoch {epoch_num}, train_loss: {train_loss}, val_loss: {val_loss}, val_score:{val_score}")

        
        
class Resnet34(MultiLocalizationClassification):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        # weight for RGB is from Resnet34, weight for Y is set to mean(weight of RGB)
        weight = self.network.conv1.weight.clone()
        self.network.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.network.conv1.weight[:,:3] = weight
            self.network.conv1.weight[:, 3] = torch.mean(weight, dim=1)
        # update out_features to NUM_LABELS
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, NUM_LABELS)
        
            
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    
    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True
        for param in self.network.conv1.parameters():
            param.requires_grad = True
    
    
    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True


# In[13]:


class FocalLoss(nn.Module):
    def __init__(self, gamma, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps  
        
    def forward(self, preds, targets):
        preds = preds.clamp(self.eps, 1 - self.eps)
        loss = (1 - preds) ** self.gamma * targets * torch.log(preds)                 + preds ** self.gamma * (1 - targets) * torch.log(1 - preds) 
    
        return -torch.mean(loss)


# In[14]:


# CRITERION = F.binary_cross_entropy
CRITERION = FocalLoss(gamma=1)


# In[15]:


EPSILON = 1e-6

def f_score(pred, target, threshold=0.5, beta=1):
    target = target > threshold
    pred = pred > threshold
    
    TP = (pred & target).sum(1, dtype=float)
    FP = (pred & ~target).sum(1, dtype=float)
    FN = (~pred & target).sum(1, dtype=float)
    
    precision = TP / (TP + FP + EPSILON)
    recall = TP / (TP + FN + EPSILON)
    f_scores = (1 + beta ** 2) * precision * recall / (
        beta ** 2 * precision + recall + EPSILON)
    
    return f_scores.mean()


# In[16]:


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)


# In[17]:


device = get_default_device()

model = to_device(Resnet34(), device)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


# In[18]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
def fit_one_cycle(epochs, model, train_dl, val_dl, max_lr,weight_decay=0, 
                  grad_clip=None, opt_func=torch.optim.SGD):
    
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        
        for batch in train_dl:
            loss = model.training_step(batch)
            train_losses.append(loss.detach())
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            
            scheduler.step()
        
        result = evaluate(model, val_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)    
    return history   
            


# In[19]:


get_ipython().run_cell_magic('time', '', 'history = [evaluate(model, val_dl)]\nhistory')


# In[20]:


model.freeze()


# In[21]:


epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[22]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, model, train_dl, val_dl, max_lr, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[23]:


model.unfreeze()


# In[24]:


epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[25]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, model, train_dl, val_dl, max_lr, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[26]:


epochs = 10
max_lr = 0.005
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[27]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, model, train_dl, val_dl, max_lr, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[28]:


def plot_lrs(history):
    scores = [x.get('lrs') for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning rate vs. No. of epochs');
    
def plot_scores(history):
    scores = [x.get('val_score') for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score vs. No. of epochs');

    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss') for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    


# In[29]:


plot_losses(history)


# In[30]:


plot_scores(history)


# In[31]:


plot_lrs(history)


# In[32]:


weights_fname = 'protein-resnet.pth'
torch.save(model.state_dict(), weights_fname)

