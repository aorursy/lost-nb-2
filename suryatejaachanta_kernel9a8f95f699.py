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


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64 
imageSize = 64 

dataset = dset.ImageFolder(root="../input/all-dogs/",
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 1, 1,bias = True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
netG = G()
netG.apply(weights_init)
netG = netG.cuda()

class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1024, 3, 1, 1, bias = True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

def train(dataloader,netD,netG,optimizerD,optimizerG,criterion,total_epochs):
    for epoch in range(total_epochs):
        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()

            real = data[0]
            input = Variable(real.cuda())
            target = Variable(torch.ones(input.size()[0]).cuda())
            output = netD(input)
            errD_real = criterion(output, target)

            noise = Variable(torch.randn(input.size()[0], 100, 1, 1).cuda())
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]).cuda())
            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]).cuda())
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            if(i%20==0):
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, total_epochs, i, len(dataloader),(errD).item(), (errG).item()))
        
        #vutils.save_image(real, '%s/real_samples.png' % "./results1", normalize = True)
        #fake = netG(noise)
        #vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results1", epoch), normalize = True)
    
    #save the trained weights for future reference
    """torch.save({
                'epoch': epoch,
                'seg_state_dict': netG.state_dict(),
                'f_loss_d': errD_fake,
                'r_loss_d': errD_real,
                'r_loss_g': errG,
                'disc_state_dict': netD.state_dict(),
                }, 'checkpointWeights.pth')
    """
netD = D()
netD.apply(weights_init)
netD = netD.cuda()
total_epochs = 200

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

train(dataloader,netD,netG,optimizerD,optimizerG,criterion,total_epochs)


# In[ ]:





# In[3]:


import os
from torchvision.utils import save_image
if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
im_batch_size = 50
n_images=10000

for i_batch in range(0, n_images, im_batch_size):
    noise = Variable(torch.randn(im_batch_size, 100, 1, 1).cuda())
    gen_images = netG(noise)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
        
import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[4]:


mkdir results


# In[5]:


ls


# In[6]:


cd ..


# In[7]:


ls


# In[8]:


cd input/


# In[9]:


ls


# In[10]:


mkdir sample


# In[11]:


ls


# In[12]:


cd ..


# In[13]:


cd ..


# In[14]:


cd kaggle/


# In[15]:


cd input/


# In[16]:


cd all-dogs/


# In[17]:


ls


# In[18]:


cd all-dogs/


# In[ ]:





# In[ ]:




