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


f = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
f.head(5)


# In[3]:


def getDataLabels(csv, label):
    file = pd.read_csv(csv)
    labels = file[label].to_numpy()
    data = file.drop([label], axis=1).to_numpy(dtype=np.float32).reshape(file.shape[0],28,28)
    data = np.expand_dims(data, axis=1)
    return data, labels


# In[4]:


train_data, train_labels = getDataLabels('/kaggle/input/Kannada-MNIST/train.csv', 'label')
test_data, test_labels = getDataLabels('/kaggle/input/Kannada-MNIST/test.csv', 'id')
other_data, other_labels = getDataLabels('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv', 'label')


# In[5]:


print(f' Train:\tshape of data:{train_data.shape}\tshape of labels:{train_labels.shape}\n Test:\tshape of data:{test_data.shape} \tshape of labels:{test_labels.shape}\n Other:\tshape of data:{other_data.shape}\tshape of labels:{other_labels.shape}')


# In[6]:


len(train_data)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.title(f'Train Label: {train_labels[5]}')
plt.imshow(train_data[8,0])


# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[10]:


class Kannada(Dataset):
    
    def __init__(self, data, labels, transform=None):
        self.data =data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]


# In[11]:


BATCH_SIZE = 256
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')


# In[12]:


trans = transforms.Compose(transforms=[transforms.ToTensor()])

KannadaTrainSet = Kannada(train_data, train_labels, trans)
train_loader = DataLoader(dataset=KannadaTrainSet, batch_size=256, shuffle=True)


# In[13]:


KannadaTestSet = Kannada(test_data, test_labels, trans)
test_loader = DataLoader(dataset=KannadaTestSet, batch_size=100)


# In[14]:


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # b,c,t,g,w-->b,c,tgw
        g_x = g_x.permute(0, 2, 1)  # b,tgw,c

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # b,c,t,g,w-->b,c,tgw
        theta_x = theta_x.permute(0, 2, 1)  # b,tgw,c
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # b,c,t,g,w-->b,c,tgw
        f = torch.matmul(theta_x, phi_x)  # (tgw,c)x(c,tgw)-->(tgw,tgw)
        f_div_C = F.softmax(f, dim=-1)  # 行softmax   (tgw,tgw)-->(tgw,tgw)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output


# In[15]:


train_loader_len = len(KannadaTrainSet)


# In[16]:


CNN = Model().to(DEVICE)
optimizer = optim.Adam(CNN.parameters())
loss_func = nn.CrossEntropyLoss()


# In[17]:


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    correct = 0
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad() # zero_grad
        output = model(data) #forward
        loss = loss_func(output, target) # loss
        loss.backward() # backward   求导
        optimizer.step() # 更新
        
        predict = output.max(dim=1, keepdim=True)[1]
        correct += predict.eq(target.view_as(predict)).sum().item()
    print('Training epoch:{}\tTraining loss:{}\tAccuracy:{:.0f}%\n'.format(epoch, loss.item(), 100.*correct/train_loader_len))


# In[18]:


for epoch in range(1, EPOCHS+1):
    train(CNN, DEVICE, train_loader, optimizer, epoch)


# In[19]:


CNN.eval()

allPredictList = []
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        
        output = CNN(data)
        predict = output.max(dim=1)[1]
        predictList = list(predict.to('cpu').numpy())
        allPredictList += predictList


# In[20]:


submission = pd.DataFrame({'id':test_labels, 'label':np.array(allPredictList)})
submission.to_csv(path_or_buf ="submission.csv", index=False)


# In[21]:


submission


# In[ ]:




