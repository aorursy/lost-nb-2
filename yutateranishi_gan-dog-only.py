#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm_notebook as tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans


# In[2]:


class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False)
        # state size. (nchannels) x 64 x 64

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = torch.tanh(self.conv6(x))
        return x



class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        
        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4
        
        self.conv5 = nn.Conv2d(nfeats * 8, 1, 4, 1, 0, bias=False)
        # state size. 1 x 1 x 1
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        
        return x.view(-1, 1)


# In[3]:


imagesPath = os.listdir('../input/all-dogs/all-dogs/')
breedsPath = os.listdir('../input/annotation/Annotation/')


# In[4]:


trainData = []
idxIn = 0
for breed in tqdm(breedsPath):
    for dog in os.listdir('../input/annotation/Annotation/'+breed):
        try:
            img = Image.open('../input/all-dogs/all-dogs/'+dog+'.jpg') 
        except:
            continue           
        tree = ET.parse('../input/annotation/Annotation/'+breed+'/'+dog)
        root = tree.getroot()
        objects = root.findall('object')
        for o in objects:
            bndbox = o.find('bndbox') 
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            img2 = img.crop((xmin, ymin, xmax, ymax))
            img2 = img2.resize((64,64), Image.ANTIALIAS)
            trainData.append(np.asarray(img2))
trainData = np.array(trainData)


# In[5]:


for k in range(5):
    plt.figure(figsize=(15, 3))
    for j in range(5):
        x = np.random.choice(len(trainData))
        plt.subplot(1, 5, j + 1)
        img = Image.fromarray( (trainData[x]).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()


# In[6]:


trainData = trainData.transpose(0, 3, 1, 2)
trainData = trainData / 122.5 - 1
trainData = torch.tensor(trainData).float()
batch_size = 32
train_loader = torch.utils.data.DataLoader(trainData, shuffle=True,
                                           batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0005
beta1 = 0.5

netG = Generator(100, 32, 3).to(device)
netD = Discriminator(3, 48).to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

nz = 100
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

real_label = 0.9
fake_label = 0
batch_size = train_loader.batch_size



### training here

epochs = 25

step = 0
for epoch in range(epochs):
    for ii, real_images in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)
        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        if step % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            valid_image = netG(fixed_noise)
        step += 1
     


# In[7]:


im_batch_size = 50
n_images = 30000
out_images_cand = np.zeros([n_images, 3, 64, 64])
for i_batch in tqdm(range(0, n_images, im_batch_size)):
    gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
    gen_images = netG(gen_z)
    images = gen_images.to("cpu").clone().detach()
    out_images_cand[i_batch : i_batch + im_batch_size, :, :, :] = images.numpy()
out_images_cand = torch.tensor(out_images_cand).float()


# In[8]:


scores = np.zeros([n_images])
for i_batch in tqdm(range(0, n_images, im_batch_size)):
    scores[i_batch: i_batch + im_batch_size] = netD(out_images_cand[i_batch : i_batch + im_batch_size]).to("cpu").clone().detach().numpy().reshape([-1])


# In[9]:


pred = KMeans(n_clusters=10000, n_init=1, max_iter=50, tol=0.01).fit_predict(scores.reshape([-1, 1]))


# In[10]:


out_images = np.zeros([10000, 3, 64, 64])
for i in tqdm(range(10000)):
    out_images[i, :, :, :] = out_images_cand[np.where(pred==i)[0][0], :, :, :]
out_images = out_images.transpose(0, 2, 3, 1)
out_images = ((out_images + 1) * 122.5).astype(int)


# In[11]:


idx = 0
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray( (out_images[idx]).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
        idx += 1
    plt.show()


# In[12]:


if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
for i in tqdm(range(0, 10000)):
    pil_img = Image.fromarray(out_images[i, :, :, :].astype('uint8').reshape((64,64,3)))
    pil_img.save(os.path.join('../output_images', f'image_{i:05d}.png'))
import shutil
shutil.make_archive('images', 'zip', '../output_images')

