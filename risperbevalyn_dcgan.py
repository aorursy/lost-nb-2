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


import pickle as pkl
import matplotlib.pyplot as plt
#Data
import os
import xml.etree.ElementTree as ET

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

# for testing only
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[3]:


def doggo_loader(path):
    img = torchvision.datasets.folder.default_loader(path) # default loader
    # Get bounding box
    annotation_basename = os.path.splitext(os.path.basename(path))[0]
    annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))
    annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)
    tree = ET.parse(annotation_filename)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
    bbox = (xmin, ymin, xmax, ymax)
    
    # return cropped image
    return img.crop(bbox)


# In[4]:


#loading data 
def load_images(data_dir):
    trans=transforms.Compose([transforms.Resize(64),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor()])

    dog_trainset=datasets.ImageFolder(data_dir,transform=trans,loader=doggo_loader)
    train_loader=torch.utils.data.DataLoader(dog_trainset,batch_size=128,shuffle=True,num_workers=0)
    return train_loader
data_dir="../input/all-dogs/"
dog_trainloader=load_images(data_dir)


# In[5]:


#show images
def show_images(img):
    img=img.numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
#get one batch
dataiter =iter(dog_trainloader)
images,_=dataiter.next()


fig = plt.figure(figsize=(20, 4))
plot_size=30
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    show_images(images[idx])


# In[6]:


#scale images to be between -1 and 1
def scalling(img):
    img=(img*2)+(-1)
    return img   
maxi=scalling(images[0]).max()

mini=scalling(images[0]).min()
mini,maxi


# In[7]:


USE_GPU=True
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')


# In[8]:


#build model
#discriminator
conv_size=32

class Discriminator(nn.Module):
    
    def __init__(self, conv_size):
        super(Discriminator,self).__init__()
        self.conv_size=conv_size
        self.dconv_layers=nn.Sequential(nn.Conv2d(3,conv_size,4,stride=2,padding=1,bias=False), 
                                    nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                    
                                    nn.Conv2d(conv_size,conv_size*2,4,stride=2,padding=1,bias=False),
                                    nn.BatchNorm2d(conv_size*2),
                                    nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                          
                                    nn.Conv2d(conv_size*2,conv_size*4,4,stride=2,padding=1,bias=False),
                                    nn.BatchNorm2d(conv_size*4),
                                    nn.LeakyReLU(negative_slope=0.2,inplace=True)
                                    
                                       
                                   )

        self.fc=nn.Sequential(nn.Linear(conv_size*4*4*4,1),
                              nn.Sigmoid())

    def forward(self,x,feature=False):
       
        x=self.dconv_layers(x)
        features = x.view(-1,64)
        #flatten
        x=x.view(-1,self.conv_size*4*4*4)
       
        x=self.fc(x)
        if feature:
            return features,x
        else:
            return x
        
        
    


# In[9]:


#generator
class Generator(nn.Module):
    def __init__(self,conv_size,latent_size):
        super(Generator,self).__init__()
        self.conv_size=conv_size
        self.latent_size=latent_size
        self.gfc=nn.Sequential(nn.Linear(latent_size,(conv_size*4)*4*4))
        self.gconv_layers=nn.Sequential(nn.ConvTranspose2d(conv_size*4,conv_size*2,4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(conv_size*2),
                                        nn.ReLU(True),
                                        
                                        nn.ConvTranspose2d(conv_size*2,conv_size,4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(conv_size),
                                        nn.ReLU(True),
                                        
                                        nn.ConvTranspose2d(conv_size,3,4,stride=2,padding=1,bias=False))
    def forward(self,x):
        x=self.gfc(x)
        #unflatten
        x=x.view(-1, self.conv_size*4, 4, 4)
       
        x=self.gconv_layers(x)
        x=torch.tanh(x)
        return x


# In[10]:


G=Generator(conv_size,latent_size=100)
D=Discriminator(conv_size)
print('G And D loaded')


# In[11]:


def init_weights(m):
    
    
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    #weights for a convolutional layers and linear layer
    if hasattr(m,'weight') and (classname.find('Conv') != -1 or classname.find('Linear')  != -1):
        
        nn.init.normal_(m.weight.data,mean=0,std=0.02)
D.apply(init_weights),G.apply(init_weights)        


# In[12]:


#real and fake loss
#real loss
def real_loss(D_result,smooth=False):
    batch_size=D_result.size(0)
    #smoothen
    if smooth:
        labels=torch.ones(batch_size)*0.9
          
    else:
        labels=torch.ones(batch_size)
   
    criterion=nn.BCEWithLogitsLoss()
    if train_on_gpu and USE_GPU:
        labels=labels.cuda()  
        
    loss=criterion(D_result.squeeze(),labels)
    return loss
def fake_loss(D_result):
    batch_size=D_result.size(0)
    labels=torch.zeros(batch_size)
    criterion=nn.BCEWithLogitsLoss()
    if train_on_gpu and USE_GPU:
        labels=labels.cuda()
    loss=criterion(D_result.squeeze(),labels)
    return loss
    


# In[13]:


d_optim=optim.Adam(D.parameters(),lr=0.0002,betas=(0.5, 0.999))
g_optim=optim.Adam(G.parameters(),lr=0.0002,betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optim, 'min',factor=0.5,patience=2)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(g_optim, 'min',factor=0.5,patience=2)


# In[14]:


def gaussian_noise(inputs, mean=0, stddev=0.01):
    input = inputs.cpu()
    input_array = input.data.numpy()

    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(input_array))

    out = np.add(input_array, noise)

    output_tensor = torch.from_numpy(out)
    
    out = output_tensor.cuda()
    out = out.float()
    return out


# In[15]:


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))


# In[16]:


epochs=50
#move models 
criterionF=nn.MSELoss()
if train_on_gpu and USE_GPU:
    D.cuda()
    G.cuda()
#images generates
generated=[]
losses=[]
sample_size=10000
z_size=100
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
for e in range(epochs):
   
    for ii ,(images,dog_trainloaders) in enumerate(dog_trainloader,0):
        images=scalling(images)
        #move images to gpu
        
        if train_on_gpu and USE_GPU:
            images=images.cuda()
       
        
       
        #clear_gradients
        d_optim.zero_grad()
        #feed images to discriminator
        d_out=D(images)
        #compute discriminator loss on real images
        r_loss=real_loss(d_out,smooth=True)
        r_loss.backward()
        #generate fake images
        fixed_z = np.random.uniform(-1, 1, size=(images.size(0), z_size))
        fixed_z = torch.from_numpy(fixed_z).float()
        if train_on_gpu and USE_GPU:
            fixed_z=fixed_z.cuda()
        G_out=G(fixed_z)
        #compute discriminator loss on fake images
        df_out1=D(G_out)
        f_loss=fake_loss(df_out1)
        f_loss.backward()
        #total dicriminator loss
        D_loss=r_loss+f_loss  
        d_losses=D_loss.item()
        
        d_optim.step()
        
        z = np.random.uniform(-1, 1, size=(images.size(0), z_size))
        z = torch.from_numpy(z).float()
        if train_on_gpu and USE_GPU:
            z=z.cuda()
        #generator training
        g_optim.zero_grad()
        #generate fake images
        FakeImages=G(z)
        #feed to discriminator
        ####### feature matching ########
        feature_real,_ = D(images.detach(),feature=True)
        feature_fake,output = D( FakeImages,feature=True)
        feature_real = torch.mean(feature_real,0)
        feature_fake = torch.mean(feature_fake,0)
        G_loss = criterionF(feature_fake, feature_real.detach())
        G_loss.backward()
        g_optim.step()
        g_losses=G_loss.item()
        #scheduler.step(D_loss.item())
        #scheduler2.step(G_loss.item())
        
    losses.append((d_losses,g_losses))   
    print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        e+1, epochs, d_losses, g_losses))
    
    G.eval() # for generating samples    
    samples_z = G(fixed_z)
    generated.append(samples_z)
   
    G.train()


# In[17]:


with open('train_samples.pkl', 'wb') as f:
       pkl.dump(generated, f)
# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
   samples = pkl.load(f)  
_ = view_samples(-1, samples)    

