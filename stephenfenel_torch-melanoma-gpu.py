#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 


# In[2]:


test_ohe = pd.read_csv('../input/combined-train/test.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train_ohe = pd.read_csv('../input/combined-train/train0.csv')


# In[3]:


test.head()


# In[4]:


test_ohe.head()


# In[5]:


train_ohe.drop(['target', 'strat'], axis = 1).head()


# In[6]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
train.head()


# In[7]:


pip install efficientnet_pytorch


# In[8]:


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as ttp
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import cv2 as cv 
import pickle
import random 
import albumentations
import gc 


# In[9]:


'''if mask:
    TRAIN_FOLDER ='../input/channel1/train_masks'
    TEST_FOLDER = '../input/channel1/test_masks'

elif nohair:
    TRAIN_FOLDER = '../input/nohair/train_nohair'
    TEST_FOLDER = '../input/nohair/test_nohair'
else:
    TRAIN_FOLDER = '/kaggle/input/melanoma-external-malignant-256/train/train'
    TEST_FOLDER = '/kaggle/input/melanoma-external-malignant-256/test/test'
TRAIN_CSV = '/kaggle/input/melanoma-external-malignant-256/train_concat.csv'
TEST_CSV = '../input/siim-isic-melanoma-classification/test.csv'''

MODELS_PATH = '../input/melanoma-models/'
#MODELS2 = '../input/practice-models/'


# In[10]:


DIR0 = '../input/jpeg-melanoma-384x384/train'
DIR1 = '../input/jpeg-isic2019-384x384/train'

TEST_FOLDER = '../input/jpeg-melanoma-384x384/test'
TEST_CSV = '../input/combined-train/test.csv'


# In[11]:


FOLD_CSVS ={0:'../input/combined-train/train0.csv',1:'../input/combined-train/train1.csv',2:'../input/combined-train/train2.csv',
            3:'../input/combined-train/train3.csv',4:'../input/combined-train/train4.csv'}


# In[12]:


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform_train = albumentations.Compose([
    albumentations.ShiftScaleRotate(shift_limit = 0.15, scale_limit = 0.1, rotate_limit = 25, p = 0.75),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomBrightness(limit=0.2, p=0.65),
    albumentations.RandomContrast(limit=0.2, p=0.65),
    albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        ], p=0.7),
    albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
    albumentations.Normalize(mean, std, always_apply = True),

])

transform_valid = albumentations.Compose([
    albumentations.Normalize(mean, std, always_apply = True),
])

transform_test = albumentations.Compose([
    albumentations.Normalize(mean, std, always_apply = True),
])


# In[13]:


class melanoma_dataset(Dataset):
    def __init__(self, root_dir, transform, df = pd.DataFrame() , csv_file = False, train = True):
        
        
        self.df = df
        
        if csv_file:
            self.csv = pd.read_csv(csv_file)
        
        self.directory = root_dir
        
        self.transform = transform
        
        self.train = train
        
        
        
    def __getitem__(self,idx):
        
        if not self.df.empty:
            tab = self.df
        else:
            tab = self.csv
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if tab.iloc[idx,1] == '-1':
            directory = self.directory[1]
        else:
            directory = self.directory[0]
        
        if not self.train:
            directory = TEST_FOLDER
        img_name = os.path.join(directory, tab.iloc[idx, 0]) + '.jpg'
        idd =  tab.iloc[idx, 0]
        
        img = cv.imread(img_name)

        target= tab.iloc[idx, 3] if self.train else 0
        
        if self.transform:
            #sample= self.transform(image = self['image'], target = self['target'])
            image = self.transform(image = img)
            flipped = image['image']
            image = np.transpose(flipped, (2, 0, 1)).astype(np.float32)
        if self.train:
            meta = np.asarray(tab.iloc[idx, [2,5,6,7,8,9,10,11,12,13,14]].values, dtype = np.float32)
        else:
            meta = np.asarray(tab.iloc[idx,[2,3,4,5,6,7,8,9,10,11,12]].values, dtype = np.float32)
        if self.train:
            return (image, meta), target
        else:
            return (image, meta)
        
        
        
        
    
    def __len__(self):
        if not self.df.empty:
            return len(self.df)
        else:
            return len(self.csv)
        
    
    


# In[14]:


#train = melanoma_dataset(pd.DataFrame(), TRAIN_CSV, TRAIN_FOLDER, transform_train)
#train_size = int(0.8*len(train))
#valid_size = len(train) - train_size

#train1, valid1 = torch.utils.data.random_split(train, [train_size, valid_size])
#dataset_sizes = {x:len(x) for x in [train1, valid1]}


# In[15]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(47)


# In[16]:


from efficientnet_pytorch import EfficientNet 
#model0 = EfficientNet.from_pretrained('efficientnet-b1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extract = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# In[17]:


class Net(nn.Module):
    def __init__(self, arch, n_meta_features = 11):
        super(Net, self).__init__()
        self.arch = arch

        self.arch._fc = nn.Linear(in_features=2304, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 1000),
                                  nn.BatchNorm1d(1000),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(1000, 500),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 500, 1)
        
    def forward(self, inputs):

        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output


# In[18]:


def kfold_train(folds=3, num_epochs = 3,train = True):


    #general variables
    since = time.time()
    val_acc_history = []
    roc = 0
    
    if train == True:
        for fold in range(folds):
            best_roc_auc = 0.0
            #model = EfficientNet.from_pretrained('efficientnet-b6')
            model = Net(EfficientNet.from_pretrained('efficientnet-b6'))
            #model.cuda()
            model_path = f'model_{fold}.pth'
            model._fc = nn.Linear(2304, 1)
            model.to(device)
            #each epoch has k folds
            print('-'*10)
            acc_dic = {}
            loss_dic = {}
            CSV = pd.read_csv(FOLD_CSVS[fold])
            train_df, valid_df, _,_ =  ttp(CSV, np.zeros(len(CSV)), train_size = .75)
        
            print('Fold {}/{}'.format(fold, folds-1))

            optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            criterion = nn.BCEWithLogitsLoss()



            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs-1))
                epoch_roc_auc = 0.0
                epoch_loss = 0.0
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                        dataset = melanoma_dataset([DIR0,DIR1], transform_train, train_df)
                        dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 2) 
                        count = 0
                        ts = time.time()
                    else:
                        model.eval()
                        dataset = melanoma_dataset([DIR0,DIR1], transform_valid, valid_df)
                        dataloader = DataLoader(dataset, batch_size = 8, shuffle = False, num_workers = 2)
                        count =0
                    dataset_size = len(dataset)
                    running_loss = 0.0
                    running_outputs = []

                    for inputs, labels in dataloader:
                        inputs[0] = torch.tensor(inputs[0], device = device, dtype = torch.float32)
                        inputs[1] = torch.tensor(inputs[1], device = device, dtype = torch.float32)
                        labels = torch.tensor(labels, device = device, dtype = torch.float32)
                        #print(len(labels))
                        if ((phase == 'train' )& (len(labels)!= 8)):
                            break
                        optimizer.zero_grad()
                        inputs_size = inputs[0].size(0)
                        #forward

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs[:2])
                            running_outputs.append(outputs.sigmoid().cpu())
                            #preds = outputs.sigmoid()
                            loss = criterion(outputs, labels.unsqueeze(1))
                            del inputs, labels
                            gc.collect()
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        count += inputs_size
                        #print(count)
                        #if count%80 == 0:
                        #   print(time.time() -ts)
                        running_loss += loss.item()*inputs_size
                        gc.collect()  


                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss/dataset_size
                    if phase == 'val':
                        epoch_roc_auc = roc_auc_score((np.asarray(valid_df.iloc[:len(np.concatenate(running_outputs).ravel()),3])) , np.concatenate(running_outputs).ravel())
                    
                    
                    if phase == 'val' and epoch_roc_auc > best_roc_auc:
                        best_roc_auc = epoch_roc_auc
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model,model_path)


                    print('{} Epoch: {} Loss: {:.4f} ROC-AUC: {:.4f}'.format(phase,epoch, epoch_loss, 0 if phase == 'train' else epoch_roc_auc))  




            print('Fold {} Best Val ROC-AUC: {:4f}, ({})'.format(fold, best_roc_auc, best_epoch))
            roc += best_roc_auc
        
        print('FINAL ROC-AUC: {}'.format(roc/folds))
            
    if train == False:
        dataset = melanoma_dataset(TEST_FOLDER, transform_test, csv_file = TEST_CSV, train = False)
        dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, num_workers = 2)

        outputs_df = pd.DataFrame()
        outputs_dic = {}
        means = []
        for fold in range(folds):
            ''''if fold < 4:
                model = torch.load(f'{MODELS_PATH}model_{fold}.pth')
            else:
                model = torch.load('../input/fold4-mod/model_0.pth')'''
            model = torch.load(f'{MODELS_PATH}model_{fold}.pth')
            model.eval()
            model.to(device)
            torch.set_grad_enabled(False)
            outputs_dic[fold] = []
            for inputs in dataloader:
                inputs[0] = torch.tensor(inputs[0], device = device, dtype = torch.float32)
                inputs[1] = torch.tensor(inputs[1], device = device, dtype = torch.float32)
                outputs = model(inputs).cpu().numpy()
                #preds = torch.round(outputs.sigmoid())
                outputs_dic[fold].append(outputs)
            outputs_df[fold] = np.concatenate(outputs_dic[fold]).ravel()

        for i in range(len(outputs_df)):
            means.append(np.mean(outputs_df.iloc[i,:]))
        means2 = torch.from_numpy(np.asarray(means))
        preds = means2.sigmoid().numpy()
        
        sub = pd.DataFrame()
        sub['image_name'] = pd.read_csv(TEST_CSV)['image_name']
        sub['target'] = preds
        sub.to_csv('melanoma_preds.csv', index = False)
    
                


# In[19]:


CSV = pd.read_csv(FOLD_CSVS[0])


# In[20]:


train_df


# In[21]:


kfold_train(folds = 1, num_epochs = 15)


# In[22]:


#kfold_train(folds = 1, num_epochs = 5, train = False)

