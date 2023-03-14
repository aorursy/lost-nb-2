#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install efficientnet_pytorch')
get_ipython().system(' pip install pytorch_lightning')


# In[ ]:


import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import torchvision.transforms as ttransforms 
import PIL
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations.pytorch
import albumentations
import math 

# At least fixing some random seeds. 
# It is still impossible to make results 100% reproducible when using GPU
warnings.simplefilter('ignore')
torch.manual_seed(47)
np.random.seed(47)
random_state=47
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


IMG_SIZE = 224
data_dir = Path('../input/')
csv_dir = data_dir/'siim-isic-melanoma-classification/'
im_dir_test = data_dir/'siic-isic-224x224-images/test'
im_dir_train = data_dir/'siic-isic-224x224-images/train'

if 'jpeg' in str(im_dir_test):
    ext = '.jpg'
else:
    ext='.png'
im_dir_train.is_dir()


# In[ ]:


train_df = pd.read_csv(csv_dir/'train.csv')
test_df = pd.read_csv(csv_dir/'test.csv')

# One-hot encoding of anatom_site_general_challenge feature
concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)
# Sex features
train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
train_df['sex'] = train_df['sex'].fillna(-1)
test_df['sex'] = test_df['sex'].fillna(-1)

# Age features
train_df['age_approx'] /= train_df['age_approx'].max()
test_df['age_approx'] /= test_df['age_approx'].max()
train_df['age_approx'] = train_df['age_approx'].fillna(0)
test_df['age_approx'] = test_df['age_approx'].fillna(0)

train_df['patient_id'] = train_df['patient_id'].fillna(0)

meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')
del train_df['patient_id'];del train_df['anatom_site_general_challenge']


# In[ ]:


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, 
                 box_trans=True, meta_features = None, ext='.png'):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.df['image_path'] = self.df['image_name'].apply(lambda x: os.path.join(self.imfolder, x + ext))
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
                
    def __getitem__(self, index):
        im_path = self.df.iloc[index]['image_path']
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        meta = np.array(self.df[self.meta_features].iloc[index].values, dtype=np.float32)
        if self.transforms:
            x = self.transforms(image=x)['image']

                
        if self.train:
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
        
    def __len__(self):
        return len(self.df)


# In[ ]:


class Microscope(albumentations.ImageOnlyTransform):
    def __init__(self, p: float = 0.3, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

cutout = albumentations.Cutout(num_holes=4, max_h_size=int(IMG_SIZE*0.05), 
                      max_w_size=int(IMG_SIZE*0.04), 
                      p=0.3)


# In[ ]:


MEL_DATASET_STATS = {'mean': [0.8016, 0.6186, 0.5849], 'std': [0.0916, 0.1036, 0.1139]}
IMAGNET_STATS = {'mean': [0.485, 0.456, 0.406],'std': [0.229, 0.224, 0.225]}
# normalization stats calculated this competition dataset 
dataset_stats = MEL_DATASET_STATS
height = 512
width = 512

train_transform = albumentations.Compose(transforms=[
    albumentations.Flip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=(-0.17, 0.17), scale_limit=0., rotate_limit=0, 
                      border_mode=cv2.BORDER_REFLECT_101, p=0.70),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.3, 0.15), contrast_limit=(-0.2, 0.15), p=1),
    albumentations.RGBShift(p=0.70, r_shift_limit=(-50, 50)),
    albumentations.GaussNoise(p=0.3),
    albumentations.core.composition.OneOf([cutout, Microscope()], p=0.3),
    albumentations.pytorch.transforms.ToTensor(normalize=dataset_stats),
])

valid_transform = albumentations.Compose([
    albumentations.Flip(p=0.5),
    albumentations.GaussNoise(p=0.3),
    albumentations.pytorch.transforms.ToTensor(normalize=dataset_stats)
])

test_transform = albumentations.Compose(transforms=[
    albumentations.Flip(p=0.5),
    albumentations.GaussNoise(p=0.3),
    albumentations.pytorch.transforms.ToTensor(normalize=dataset_stats),
])


# In[ ]:


# test_transform(False, image=np.ones((412,412, 3)).astype(np.uint8))


# In[ ]:


# new_im_dim = 300
# train_transform = transforms.Compose([
#     transforms.RandomApply(transforms.RandomResizedCrop(size=new_im_dim, scale=(0.7, 1.0)), p=0.6)
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness= (0.3, 1), contrast=(0.3, 1), saturation=0.5),
# #     Microscope(p=0.6),
# #     transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# ])

# test_transform = transforms.Compose([
# #     transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
# #     transforms.Resize((new_im_dim, new_im_dim)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# ])


# In[ ]:


# Functions to create weights for PyTorch RandomSampler - NOT USED

def generate_class_weights(df):
    B = 0.4
    C = np.array([(1 - B), B])*2
    ones = len(df.query('target == 1'))
    zeros = len(df.query('target == 0'))

    weightage_fn = {0: 1-zeros/len(df), 1: 1-ones/len(df)}
    return [weightage_fn[target] for target in df.target]
def covert_to_prob(r,NewMin=0.0,  NewMax=0.40):
    OldMax = r.max()
    OldMin = r.min()
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((r - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def generate_anat_weights(train):
    train['anatom_site_general_challenge']=train['anatom_site_general_challenge'].fillna('None')
    site_counts = train['anatom_site_general_challenge'].value_counts()
    print(site_counts)
    site_probs = (1-site_counts/len(train)).to_dict()
    anat_probs = train['anatom_site_general_challenge'].apply(lambda x: site_probs[x])
    anat_weights=torch.Tensor(anat_probs.to_numpy())
    
    return covert_to_prob(anat_weights, NewMax=1, NewMin=0.30)

def generate_weights(df):
    class_weights = torch.Tensor(generate_class_weights(df))
#     anat_weights = generate_anat_weights(df)
#     weights=class_weights * anat_weights
    
    return class_weights
    


# In[ ]:


# To resmple dataframe prior to training. Using this resampled dataset slightly increased performance because all samples
# are trained on here rather than in random sampling during training, where there is a chance that some examples are 
# are never or trained very less on 
    
def resample_df(df, targ_minority_frac=0.30, minority_class=1):
    minority = df[df['target']==minority_class]
    majority = df[df['target']!=minority_class]
    minority_frac = len(minority)/len(df)
    frac = targ_minority_frac/minority_frac
    
    minority = minority.sample(frac=frac, replace=True)
    df = pd.concat([minority, majority])
    return df.sample(frac=1).reset_index(drop=True)


# In[ ]:


from pytorch_lightning.core.lightning import LightningModule
from torch.functional import F
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

        
class LitModel(LightningModule):
    def __init__(self, backbone, n_meta_features: int, total_steps=None, pos_weight=None):
        super().__init__()
        self.total_steps=total_steps
        self.backbone = backbone
        if 'ResNet' in str(backbone.__class__):
            self.backbone.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        
        if 'EfficientNet' in str(backbone.__class__):
            print('Loading for EfficientNet')
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Linear(in_features=in_features, out_features=500, bias=True)
            
        
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        
        self.ouput = nn.Linear(500 + 250, 1)
#         self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).cuda())
        # changing gamma from 2 to 3 increased performance -> it penalizes the more frequent class lesser 
        self.criterion = FocalLoss(gamma=3)
        
    def forward(self, inputs):
        x, meta = inputs
        image_features = self.backbone(x)
        meta_features = self.meta(meta)
        features = torch.cat((image_features, meta_features), dim=-1)
        output = self.ouput(features)
        return output.squeeze(1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        d = {'loss': loss}
        d.update({'log': {'train_loss': loss}})

        return d

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=0.001)
        default_config = {'interval': 'epoch',  # default every epoch
                          'frequency': 1,  # default every epoch/batch
                          'reduce_on_plateau': True,  # most often not ReduceLROnPlateau scheduler
                          'monitor': 'val_auc'}  # default value to monitor for ReduceLROnPlateau
#         lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, 
#                                                        verbose=True, factor=0.50),
#                                                     'name': 'reduce_plateau', **default_config}
        
        lr_scheduler = {'scheduler': OneCycleLR(optimizer=optimizer, max_lr=1e-3, total_steps=self.total_steps,
                                                div_factor=15), 
                                                'name': 'OneCycleLR', **default_config}
        lr_scheduler.update({'interval': 'step', 'reduce_on_plateau': False})
        
        return [optimizer], [lr_scheduler]

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        
        y_hat = self(x)
        return (y_hat.cpu(), y.cpu())
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        
        return {'preds': torch.sigmoid(y_hat).cpu()}
    
    def test_epoch_end(self, outputs):
        self.preds = torch.cat([out['preds'] for out in outputs])
        return {'None': [1,2,3]}
        
    def validation_epoch_end(self, outputs):
        y_hat, y = [], []
        for o in outputs:
            y_hat.append(o[0])
            y.append(o[1])

        y_hat, y = torch.cat(y_hat, axis=0), torch.cat(y, axis=0)
#         y[-1] = 1
        loss = self.criterion(y_hat, y)
        d = calculate_metrics(y_hat,y)
        d.update({'val_loss': loss.item()}) 
        result = {'progress bar': d, 'log': d}
        return result
    
def calculate_metrics(outputs, true):
    pred = torch.sigmoid(outputs)
    pred, true = pred.cpu(), true.cpu().long()
    val_acc = accuracy_score(true, pred.round())
    val_roc = roc_auc_score(true,pred)
    
    return {f'val_acc': val_acc, 'val_auc': val_roc}

from tqdm import tqdm_notebook
def predict_model(model, test_loader):
    model.eval()
    epoch_preds = []
    for epochs in tqdm_notebook(range(4)):
        preds=[]
        for data in test_loader:
            data = (data[0].cuda(), data[1].cuda())
            with torch.no_grad(): 
                pred = torch.sigmoid(model(data))
                preds.append(pred.cpu())

        preds=torch.cat(preds, axis=0)
        epoch_preds.append(preds.numpy())

    preds = np.array(epoch_preds).mean(0)
    return preds 


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


import sklearn 
from pytorch_lightning import Trainer, callbacks, loggers

skf = StratifiedKFold(n_splits=6)

# test_df = test_df.iloc[:20]
test_dataset = MelanomaDataset(df=test_df,
                       imfolder=im_dir_test, 
                       train=False,
                       transforms=test_transform,
                       meta_features=meta_features,
                              ext=ext)
k_fold_preds = []
i = 0
BATCH_SIZE = 140//3
num_workers=0
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=num_workers)
epochs = 7
num_folds = 0

for train_idx, val_idx in skf.split(np.zeros(len(train_df)), y=np.zeros(len(train_df)), groups=train_df['target']):
    train = train_df.iloc[train_idx]
    valid = train_df.iloc[val_idx]
    
    train = resample_df(train)
    pos_weight = (train['target']==0).sum()/(train['target']==1).sum()
#     """Remove"""
#     valid = valid[:20]
#     train = train[:20]
    
    valid_dataset = MelanomaDataset(df=valid,
                        imfolder=im_dir_train, 
                        train=True,
                        transforms=valid_transform,
                        meta_features=meta_features, ext=ext)
    
    train_dataset = MelanomaDataset(df=train,
                        imfolder=im_dir_train, 
                        train=True,
                        transforms=train_transform,
                        meta_features=meta_features, ext=ext)
    
#     weights = generate_weights(train)
#     train_sampler = WeightedRandomSampler(weights, len(train))
    
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, sampler=None)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=num_workers)
    
    early_stop_callback = callbacks.EarlyStopping(monitor='val_auc',min_delta=0.00,patience=2,
                                                  verbose=False,mode='max')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(os.getcwd(), 'lightning_logs'),
        version=None,name='224, b1, cuttout and microscope, focal gamma 3, stacked', comment='sampling')

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=logger.log_dir,save_top_k=3,
        monitor='val_auc',mode='max',prefix='')
    lr_logger = callbacks.LearningRateLogger()
    # backbone = EfficientNet.from_name("efficientnet-b1")
    backbone = EfficientNet.from_pretrained("efficientnet-b1")
    model = LitModel(backbone=backbone, n_meta_features=len(meta_features), 
                     total_steps=math.ceil(len(train)/BATCH_SIZE)*epochs, pos_weight=pos_weight)
    trainer = Trainer(gpus=1, num_nodes=1, num_sanity_val_steps=0,  max_epochs=epochs,
                      callbacks=[early_stop_callback, lr_logger], 
                      checkpoint_callback=checkpoint_callback,
                      logger=logger, log_save_interval=10, check_val_every_n_epoch=1, precision=32,
                      weights_summary=None
                     )
    # train
    trainer.fit(model, train_loader, valid_loader)
    gc.collect()
    # predict 
    trainer.test(test_dataloaders=test_loader)
    preds = model.preds'
    k_fold_preds.append(preds)
    gc.collect()
    num_folds+=1
    if num_folds >= 3:
        break


# In[ ]:


final_preds = np.array(k_fold_preds).mean(0)


# In[ ]:


sub = pd.read_csv(csv_dir/'sample_submission.csv')
sub['target'] = final_preds
print('Number of 1s:', (sub['target'] > 0.5).sum())
print('Number of 0s:', (sub['target'] < 0.5).sum())
out_fname = os.path.join(logger.log_dir, 'submission_512.csv')
sub.to_csv(out_fname, index=False)

