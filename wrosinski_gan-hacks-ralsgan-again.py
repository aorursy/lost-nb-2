#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


import os

import matplotlib.pyplot as plt  # 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from torchvision import datasets, transforms
from torchvision.utils import save_image


# In[3]:


batch_size = 32
LR_G = 0.0005
LR_D = 0.0005

beta1 = 0.5
epochs = 260

real_label = 0.5
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3, dropout=0.25):
        super(Generator, self).__init__()

        self.nz = nz
        self.channels = channels
        self.dropout = dropout

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(
                    n_input,
                    n_output,
                    kernel_size=k_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(n_output, momentum=0.9),
                # use LeakyRELU
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.dropout),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(
                self.nz, 1024, 4, 1, 0
            ),  # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=3, dropout=0.25, bn=True):
        super(Discriminator, self).__init__()

        self.channels = channels
        self.dropout = dropout
        # add BN by default
        self.bn = bn

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=self.bn):
            block = [
                nn.Conv2d(
                    n_input,
                    n_output,
                    kernel_size=k_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if bn:
                block.append(nn.BatchNorm2d(n_output, momentum=0.9))
            # use LeakyRELU
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout(self.dropout)),
            return block

        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits).view(-1, 1)
        # do not use sigmoid for usage of BCE with logits loss
        # out = logits.view(-1, 1)
        return out


# kaiming normal weights init
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)


# In[5]:


batch_size = 32
image_size = 64

random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=0)]
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(random_transforms, p=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_data = datasets.ImageFolder("../input/", transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True
)

imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)

fig = plt.figure(figsize=(20, 16))
for ii, img in enumerate(imgs):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    plt.imshow((img + 1.0) / 2.0)
# In[6]:


DROPOUT_RATE = 0.5

netG = Generator(nz, dropout=DROPOUT_RATE).to(device)
netD = Discriminator(dropout=DROPOUT_RATE).to(device)

# apply weights init
netG.apply(weights_init)
netD.apply(weights_init)

# use more stable version of BCE 
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

# SGD for discriminator, Adam for generator, according to 
# https://github.com/soumith/ganhacks
# optimizerD = optim.SGD(netD.parameters(), lr=LR_D, momentum=0.9)
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, 1, 1, device=device)


# In[7]:


def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow(gen_image)
    plt.show()
    
    
# https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
def smooth_labels(y, smooth_factor):
    # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
    y *= 1 - smooth_factor
    y += smooth_factor / y.shape[1]
    return y


# In[8]:


# parameters specifying whether to smooth and flip labels
USE_LABEL_SMOOTHING = True
USE_LABEL_FLIP = True

# smoothing factor range for discriminator
DISC_SMOOTH_FACTOR_REAL = (-20, 20)
DISC_SMOOTH_FACTOR_FAKE = (0, 30)
# smoothing factor range for generator
GEN_SMOOTH_FACTOR = (-20, 20)
# probability of flipping labels for discriminator
REAL_FLIP_PROB = 0.01
# probability of smoothing labels
SMOOTH_PROB = 0.5


for epoch in range(epochs):
    for ii, (real_images, train_labels) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):

        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)

        if USE_LABEL_FLIP:
            # occasional flip of real labels
            if np.random.random() < REAL_FLIP_PROB:
                # print("real labels flip for discriminator")
                labels = torch.full((batch_size, 1), fake_label, device=device)

        if USE_LABEL_SMOOTHING:
            if np.random.random() < SMOOTH_PROB:
                # print("real label smoothing for discriminator")
                disc_smoothing = np.random.randint(
                    DISC_SMOOTH_FACTOR_REAL[0], DISC_SMOOTH_FACTOR_REAL[1]
                )
                disc_smoothing /= 100
                labels = smooth_labels(labels, disc_smoothing)
                # print(labels)

        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)

        labels.fill_(fake_label)
        if USE_LABEL_SMOOTHING:
            if np.random.random() < SMOOTH_PROB:
                # print("fake label smoothing for discriminator")
                disc_smoothing = np.random.randint(
                    DISC_SMOOTH_FACTOR_FAKE[0], DISC_SMOOTH_FACTOR_FAKE[1]
                )
                disc_smoothing /= 100
                labels = smooth_labels(labels, disc_smoothing)
                # print(labels)

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
        if USE_LABEL_SMOOTHING:
            if np.random.random() < SMOOTH_PROB:
                # print("label smoothing for generator")
                gen_smoothing = np.random.randint(
                    GEN_SMOOTH_FACTOR[0], GEN_SMOOTH_FACTOR[1]
                )
                gen_smoothing /= 100
                labels = smooth_labels(labels, gen_smoothing)
                # print(labels)

        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if (ii + 1) % (len(train_loader) // 2) == 0:
            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (
                    epoch + 1,
                    epochs,
                    ii + 1,
                    len(train_loader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

#             valid_image = netG(fixed_noise)


# In[9]:


gen_z = torch.randn(32, nz, 1, 1, device=device)
gen_images = netG(gen_z).to("cpu").clone().detach()
gen_images = gen_images.numpy().transpose(0, 2, 3, 1)


# In[10]:


fig = plt.figure(figsize=(25, 16))
for ii, img in enumerate(gen_images):
    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
    plt.imshow(img)


# In[11]:


if not os.path.exists("../output_images"):
    os.mkdir("../output_images")

im_batch_size = 50
n_images = 10000

for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = (netG(gen_z) + 1.0) / 2.0
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(
            gen_images[i_image, :, :, :],
            os.path.join("../output_images", f"image_{i_batch+i_image:05d}.png"),
        )

import shutil

shutil.make_archive("images", "zip", "../output_images")

