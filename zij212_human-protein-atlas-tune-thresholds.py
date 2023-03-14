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


VAL_CSV = '/kaggle/input/human-protein-atlas-train-val-split/test_df.csv'
TEST_CSV = '/kaggle/input/human-protein-atlas-image-classification/sample_submission.csv'

VAL_DIR = '/kaggle/input/human-protein-atlas-image-classification/train'
TEST_DIR = '/kaggle/input/human-protein-atlas-image-classification/test'

val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

TRAIN_STATS = '/kaggle/input/human-protein-atlas-data-stats/stats.pt'
stats = torch.load(TRAIN_STATS)

PRETAINED_WEIGHTS = '/kaggle/input/human-protein-atlas-resnet34-focalloss/protein-resnet.pth'


# In[3]:


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


# In[4]:


FILTERS = ['red', 'green', 'blue', 'yellow']

def load_image(image_id, ddir):
    """
    return: 4-channel PIL Image
    """
    return Image.merge('RGBA', [Image.open(f"{ddir}/{image_id}_{f}.png") for f in FILTERS])


def encode(image_labels: str):
    """
    image_labels: label(s) of an image, e.g. "25 0"
    return: tensor of size (28)
    """
    target = torch.zeros(NUM_LABELS)
    for label in image_labels.split():
        target[int(label)] = 1
    return target


def decode():
    pass


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, preds, targets):
        loss = targets * torch.log(preds) * (1 - preds) ** self.gamma                + (1 - targets) * torch.log(1 - preds) * preds ** self.gamma
    
        return -torch.mean(loss)

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


# In[5]:


tfms = T.Compose([T.ToTensor(), T.Normalize(*stats)])

val_ds = ProteinLocalizationDataset(val_df, VAL_DIR, transform=tfms)
test_ds = ProteinLocalizationDataset(test_df, TEST_DIR, transform=tfms)

batch_size = 32
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                    num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                     num_workers=2, pin_memory=True)

device = get_default_device()

val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(val_dl, device)

CRITERION = FocalLoss(gamma=1)
model = to_device(Resnet34(), device)
model.load_state_dict(torch.load(PRETAINED_WEIGHTS, map_location=device))
model.eval()


# In[6]:


get_ipython().run_cell_magic('time', '', "\n@torch.no_grad()\ndef predict_dl(dl, model):\n    torch.cuda.empty_cache()\n    batch_probs = []\n    for xb, _ in dl:\n        probs = model(xb)\n        batch_probs.append(probs.cpu().detach())\n    batch_probs = torch.cat(batch_probs)\n    return batch_probs\n\nval_probs = predict_dl(val_dl, model)\ntorch.save(val_probs, 'val_probs.pt')")


# In[7]:


from sklearn.metrics import precision_recall_curve


# In[8]:


def get_threshold(precision, recall, thresholds):
    f1 = precision * recall / (precision + recall + 1e-6)
    idx = np.argmax(f1)
    return thresholds[idx]


# val_probs.shape --> len(val_df), num_classes
# val_targets.shape --> len(val_df), num_classes
val_targets = torch.stack(list(map(encode, val_df['Target'])))  

th = [0] * NUM_LABELS

for i in range(NUM_LABELS):
    precision, recall, thresholds = precision_recall_curve(
        val_targets[:, i], val_probs[:, i])
    th[i] = get_threshold(precision, recall, thresholds)

print(th)


# In[9]:


torch.save(th, 'thresholds.pt')

