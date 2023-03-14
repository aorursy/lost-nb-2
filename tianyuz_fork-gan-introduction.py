#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# print(os.listdir("../input"))


# In[2]:


from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
from tqdm import tqdm_notebook as tqdm


# In[3]:


# PATH = "../input/all-dogs/all-dogs/"
# images = os.listdir(PATH)
# print(f"There are {len(images)} pictures of dogs.")

# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))

# for indx, axis in enumerate(axes.flatten()):
#     rnd_indx = np.random.randint(0, len(images))
#     img = plt.imread(PATH + images[rnd_indx])
#     imgplot = axis.imshow(img)
#     axis.set_title(images[rnd_indx])
#     axis.set_axis_off()
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[4]:


import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt, zipfile
from PIL import Image

DogsOnly = True

ROOT = "../input/"
IMAGES = os.listdir(ROOT + "all-dogs/all-dogs/")
breeds = os.listdir(ROOT + "annotation/Annotation/")

idxIn = 0; namesIn = []
imagesIn = np.zeros((25000, 64, 64, 3))

if DogsOnly:
    for breed in breeds:
        for dog in os.listdir(ROOT + "annotation/Annotation/" + breed):
            try: img = Image.open(ROOT + "all-dogs/all-dogs/" + dog + '.jpg')
            except: continue
            tree = ET.parse(ROOT + "annotation/Annotation/" + breed + '/' + dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                w = np.min((xmax-xmin, ymax-ymin))
                img2 = img.crop((xmin,ymin,xmin+w,ymin+w))
                img2 = img2.resize((64,64), Image.ANTIALIAS)
                imagesIn[idxIn, :, :, :] = np.asarray(img2)
                namesIn.append(breed)
                idxIn += 1
    idx = np.arange(idxIn)
    np.random.shuffle(idx)
    imagesIn = imagesIn[idx, :, :, :]
    namesIn = np.array(namesIn)[idx]


# In[5]:


# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader



batch_size = 32

import torch
import numpy as np
import torch.utils.data as utils

tensor_x = torch.stack([torch.Tensor(i) for i in imagesIn.transpose(0,3,1,2)]) # transform to torch tensors
print(type(tensor_x))
my_dataset = utils.TensorDataset(tensor_x) # create your datset
dataloader = utils.DataLoader(my_dataset, shuffle=True,batch_size=batch_size) # create your dataloader

# imgs = next(iter(dataloader))
# imgs = torch.stack(imgs)[0].numpy()
# imgs2 = imgs


# In[6]:


# batch_size = 32
# image_size = 64

# random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
# transform = transforms.Compose([
#     transforms.Resize(64), 
#     transforms.CenterCrop(64),
# #     transforms.RandomHorizontalFlip(p=0.5),
# #     transforms.RandomApply(random_transforms, p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5) )  
# ])

# train_data = datasets.ImageFolder("../input/all-dogs/", transform=transform)
# dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size = batch_size)

# imgs, label = next(iter(dataloader))
# imgs = imgs.numpy().transpose(0, 2, 3, 1)


# In[7]:


# for i in range(5):
#     plt.imshow(imgs[i])
#     plt.show()


# In[8]:


def weights_init(m):
    """
    Take a neural network m as input, weight_init will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    


# In[9]:


# class G(nn.Module):
#     def __init__(self):
#         # inherit torch.nn Module
#         super(G, self).__init__()
        
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(64, 3, 4, stride=2, padding=0, bias=False),
#             nn.Tanh()
#             )
        
#     def forward(self, input):
#         output = self.main(input)
#         return output

# netG = G()
# netG.apply(weights_init)


# In[10]:


# class D(nn.Module):
#     def __init__(self):
#         super(D,self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
#             nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
#             nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
#             nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
#             nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#             )
    
#     def forward(self, input):
#         output = self.main(input)
#         # .view(-1) = Flattens the output into 1D instead of 2D
#         return output.view(-1)
        
# netD = D()
# netD.apply(weights_init)


# In[11]:


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()
        
        self.nz=nz
        self.channels=channels
        
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(n_output),
                     nn.ReLU(inplace=True)
                    ]
            return block
        
        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0),
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()
            )
        
    def forward(self,z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img
    
    
    
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.channels = channels
        
        def convlayer(n_input, n_output, k_size = 4, stride = 2, padding = 0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding,bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits)
        return out.view(-1,1)
        


# In[12]:


get_ipython().system('mkdir results')
get_ipython().system('ls')


# In[13]:


# EPOCH = 0
# LR = 0.001
# criterion = nn.BCELoss()
# optimizerD = optim.Adam(netD.parameters(), lr=LR, betas = (0.5, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=LR, betas = (0.5, 0.999))


# In[14]:


# for epoch in range(EPOCH):
#     for i, data in enumerate(dataloader, 0):
#         # Updating the weights of the neural network of the discriminator
#         netD.zero_grad()
        
#         # Train discriminator with real image
#         real,_ = data
#         input = Variable(real)
#         target = Variable(torch.ones(input.size()[0]))
#         output = netD(input)
#         errD_real = criterion(output, target)
        
#         # Train discriminator with fake image
#         noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
#         fake = netG(noise)
#         target = Variable(torch.zeros(input.size()[0]))
#         output = netD(fake.detach())
#         errD_fake = criterion(output, target)
        
#         # Backpropagate the total error
#         errD = errD_real + errD_fake
#         errD.backward()
#         optimizerD.step()
        
#         # Train generator
#         netG.zero_grad()
#         target = Variable(torch.ones(input.size()[0]))
#         output = netD(fake)
#         errG = criterion(output, target)
#         errG.backward()
#         optimizerG.step()
        
#         # Print the losses and save the real images and the generated images of the minibatch every 100 steps
#         print("[%d/%d][%d/%d] Loss_D: %.4f; Loss_G: %.4f" % (epoch, EPOCH, i, len(dataloader), errD.item(), errG.item()))
#         if i % 100 == 0:
#             vutils.save_image(real, "%s/real_samples.png" % "./results", normalize = True)
#             fake = netG(noise)
#             vutils.save_image(fake.data, "%s/fake_samples_epoch_%03d.png" % ("./results", epoch), normalize = True)


# In[15]:


# batch_size = 32
LR_G = 0.001
LR_D = 0.0005

beta1 = 0.5
epochs = 500
real_label = 1
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[16]:


# def plot_loss(G_losses, D_losses, epoch):
#     plt.figure(figsize=(10,5))
#     plt.title("Generator and Discriminator Loss - EPOCH" + str(epoch))
#     plt.plot(G_losses,label='G')
#     plt.plot(D_losses,label='D')
#     plt.xlabel('iterations')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()


# In[17]:


def show_generated_img(n_images = 5):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device = device)
        gen_images = netG(noise).to("cpu").clone().detach().squeeze(0)
        gen_images = gen_images.numpy().transpose(1, 2, 0)
        sample.append(gen_images)
    
    figure, axes = plt.subplots(1, len(sample), figsize = (96,96))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample[index]
        axis.imshow(image_array)
        
    plt.show()
    plt.close()


# In[18]:


netG = Generator(nz).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, 1, 1, device=device)

G_losses = []
D_losses = []
epoch_time = []


# In[19]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(epochs):\n    start = time.time()\n    print("Epoch: {}".format(epoch+1))\n    for ii, real_images in tqdm(enumerate(dataloader), total=len(dataloader)):\n        # update D network\n        # train with real\n        netD.zero_grad()\n        real_images = real_images[0].to(device)\n        \n        batch_size = real_images.size(0)\n        labels = torch.full((batch_size,1), real_label, device=device)\n        \n        output = netD(real_images)\n        errD_real = criterion(output, labels)\n        errD_real.backward()\n        D_x = output.mean().item()\n        \n        # train with fake\n        noise = torch.randn(batch_size, nz, 1, 1, device = device)\n        fake = netG(noise)\n        labels.fill_(fake_label)\n        output = netD(fake.detach())\n        errD_fake = criterion(output, labels)\n        errD_fake.backward()\n        D_G_z1 = output.mean().item()\n        errD = errD_real + errD_fake\n        optimizerD.step()\n        \n        # update G network\n        netG.zero_grad()\n        labels.fill_(real_label)\n        output = netD(fake)\n        errG = criterion(output, labels)\n        errG.backward()\n        D_G_z2 = output.mean().item()\n        optimizerG.step()\n        \n        # save losses for plotting\n#         G_losses.append(errG.item())\n#         D_losses.append(errD.item())\n        \n        if (ii+1) % len(dataloader) == 0:\n            print("[%d/%d][%d/%d] Loss_D:%.4f Loss_G:%.4f D(x):%.4f D(G(z)):%.4f / %.4f"\n                 % (epoch+1, epochs, ii+1, len(dataloader),\n                   errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))\n            \n#     plot_loss(G_losses, D_losses, epoch)\n#     G_losses = []\n#     D_losses = []\n    show_generated_img()\n#     epoch_time.append(time.time()-start)')


# In[20]:


print(epoch)


# In[21]:


print (">> average EPOCH duration = ", np.mean(epoch_time))


# In[22]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("../output_images"):
    os.mkdir("../output_images")
    
im_batch_size = 50
n_images = 10000
for i_batch in tqdm(range(0, n_images, im_batch_size)):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = netG(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image,:,:,:], os.path.join("../output_images", f"image_{i_batch+i_image:05d}.png"))


# In[23]:


import shutil
shutil.make_archive("images", "zip", "../output_images")

