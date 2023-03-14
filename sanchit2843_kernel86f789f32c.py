#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('pip install -i https://test.pypi.org/simple/ supportlib')
import supportlib.gettingdata as getdata


# In[3]:


print(os.listdir('/kaggle/input'))


# In[4]:


import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import os
import cv2
import random
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.models as models
df = pd.read_csv('../input/train_v2.csv')


# In[5]:


df.head(10)


# In[6]:


tags = df['tags']
tags = tags.str.split()
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(tags)


# In[7]:


im_size = 256
img_dir = '../input/train-jpg'
batch_size = 16
epoch = 10
valid_size = 0.1
test_size = 0.2


# In[8]:


import os
a = os.listdir('../input/train-tif-v2')


# In[9]:


from fastai.vision import *


# In[10]:


class Amazon_dataset(Dataset):
    def __init__(self,image_dir,y_train,transform = None):

        self.img_dir = image_dir
        self.y_train = y_train
        self.transform = transform
        self.id = os.listdir(self.img_dir)
    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self,idx):
        img_name = os.path.join(self.img_dir, self.id[idx])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(self.y_train[idx])
        return image,label


# In[11]:


# Data transform
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((im_size,im_size)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.311, 0.340, 0.299], [0.167, 0.144, 0.138])
                    ])
inv_normalize = transforms.Normalize(
    mean=[-0.311/0.167, -0.340/0.144, -0.299/0.138],
    std=[1/0.167, 1/0.144, 1/0.138]
)

#Data laoder
amazon_data = Amazon_dataset(img_dir,y_train,transform)


# In[12]:


import numpy as np
data_len = len(amazon_data)
indices = list(range(data_len))
np.random.shuffle(indices)
split1 = int(np.floor(valid_size * data_len))
split2 = int(np.floor(test_size * data_len))
valid_idx , test_idx, train_idx = indices[:split1], indices[split1:split2] , indices[split2:] 
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=train_sampler)
valid_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=valid_sampler)
test_loader = DataLoader(amazon_data, batch_size=batch_size , sampler=test_sampler)


# In[13]:


image,label = amazon_data[0]
label = np.reshape(label,(1,17))
label = mlb.inverse_transform(label)
image = inv_normalize(image)
image = image.numpy()
image =  image.transpose(1,2,0)
random.randint(0,40479)


# In[14]:


plt.imshow(image)


# In[15]:





# In[15]:


for i in range(12):
    


# In[16]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.resnet34(pretrained = True)
        self.num_ftrs = self.resnet.fc.in_features
        self.l1 = nn.Linear(1000 , 512)
        self.l2 = nn.Linear(512,17)
    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return x


# In[17]:


get_ipython().system('pip install torchsummary')
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)
summary(classifier,(3,256,256))


# In[18]:


optimizer = optim.Adam(classifier.parameters(), lr=0.001)


# In[19]:


## from sklearn.metrics import f1_score
losses1 = []
f1score1 = []
dataloader = train_loader
for i in range(epoch):
    y_pred = []
    y_true = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        output = classifier(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        output = output.cpu().detach().numpy()
        y_pred.append(output)
        target = target.cpu().numpy()
        y_true.append(target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.item()))
    #y_pred = get_pred(y_pred)
    f_score = fbeta_score(y_true,y_pred)
    losses1.append(loss.item())
    f1score1.append(f_score)
    print('Train Epoch: {} \tf1_score: {:.6f}'.format(epoch , f_score))


# In[20]:



fbeta_score(y_true,y_pred)


# In[21]:


from sklearn.metrics import fbeta_score
def get_pred(y_pred):
    l = len(y_pred)
    y_pred = y_pred[0:l-1]
    y_pred = np.asarray(y_pred)
    for i in range(len(y_pred)):
        for j in range(16):
            for k in range(17):
                try:
                    if(y_pred[i][j][k]>=0.5):
                        y_pred[i][j][k] = 1
                    else:
                        y_pred[i][j][k] = 0
                except:
                    print(y_pred.shape)
    return y_pred


# In[22]:


from sklearn import metrics

import torch
import numpy as np


def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)


def fbeta_score(y_true, y_pred, beta=2, threshold=0.5, eps=1e-9):
    leng = len(y_true)
    y_pred = y_pred[0:leng-1]
    y_true = y_true[0:leng-1]
    beta2 = beta**2
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred = y_pred.astype(float)
    y_true = y_true.astype(float)
    
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)

    y_pred = torch.ge(y_pred, threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


# In[23]:


def get_fscore(y_true,y_pred):
    leng = len(y_true)
    y_true = y_true[0:leng-1]
    y_true = np.asarray(y_true)
    leng = len(y_true)
    siz = leng*16
    y_true = np.reshape(y_true,(siz,17))
    y_pred = np.reshape(y_pred,(siz,17))
    
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    f1 = fbeta_score(y_true=y_true, y_pred=y_pred, average='weighted')
    return f1

