#!/usr/bin/env python
# coding: utf-8

# In[14]:


WORKSPACE_PATH = '/kaggle'
# Define whether to use pretrained weights (ImageNet).
PRETRAINED_ON_IMAGENET = True 
# Base path to store the model checkpoint files.
CHECKPOINT_PATH = 'working/'
DATASETS_PATH = 'input/painter-by-numbers/'
# Model type name.
MODEL = 'VGG19' # ['VGG19', 'ResNet50', 'EfficientNet'] 
CHALLENGE = 'creator'
# How many epochs to train for.
EPOCHS_PER_SESSION = 5

training_type = 'FineTuning' if PRETRAINED_ON_IMAGENET else 'OffTheShelf'
EXPERIMENT_NAME = f'{MODEL}_{training_type}_PBN_{CHALLENGE}'
print(
    f'Running experiment with the following name: {EXPERIMENT_NAME}\n'
    f'Saving checkpoint files under {CHECKPOINT_PATH}\n'
    f'Training for {EPOCHS_PER_SESSION} epochs'
)


# In[2]:


cd $WORKSPACE_PATH


# In[3]:


pip install efficientnet-pytorch


# In[6]:


from PIL import Image
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from pathlib import Path
import random
import time

import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.optim import SGD
from efficientnet_pytorch import EfficientNet

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[7]:


metadata = pd.read_csv('input/painter-by-numbers/all_data_info.csv')

encoder = LabelEncoder()

metadata['artist_cat'] = encoder.fit_transform(metadata['artist'].astype(str))
metadata['style_cat'] = encoder.fit_transform(metadata['style'].astype(str))

metadata


# In[24]:


test_path = 'input/painter-by-numbers/test.zip'
train_path = 'input/painter-by-numbers/train.zip'
i = 123
with ZipFile(train_path) as myzip:
    files_in_zip = myzip.namelist()

len(files_in_zip)

with ZipFile(train_path) as myzip:
    with myzip.open(files_in_zip[i]) as myfile:
        name = (myfile.name).split('/')[-1]
        img = Image.open(myfile)
metadata.query('new_filename == @name')['artist_cat'].values.item()
img


# In[10]:


test_path = 'input/painter-by-numbers/test.zip'
train_path = 'input/painter-by-numbers/train.zip'

input_size = 224
batch_size = 32 if MODEL!='EfficientNet' else 16

labels = {
    'creator': 'artist',
    'type': 'style'    
}

classes = {
    'creator': max(metadata['artist_cat'].values),
    'type': max(metadata['style_cat'].values)
}

num_classes = classes[CHALLENGE]

norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
])

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unnormalize = UnNormalize(norm_mean, norm_std)

TensorToImage = transforms.ToPILImage(mode='RGB')

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, challenge):
        
        self.metadata = pd.read_csv('input/painter-by-numbers/all_data_info.csv')

        encoder = LabelEncoder()

        self.metadata[challenge+'_cat'] = encoder.fit_transform(metadata[challenge].astype(str))
        
        with ZipFile(path) as myzip:
            self.files_in_zip = myzip.namelist()

        self.n_images = len(self.files_in_zip)
        self.challenge = challenge

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        with ZipFile(train_path) as myzip:
            with myzip.open(files_in_zip[idx]) as myfile:
                name = (myfile.name).split('/')[-1]
                img = Image.open(myfile)

        return (
            transform(img),
            int(self.metadata.query('new_filename == @name')[self.challenge+'_cat'].values.item())
        )
    
    def random(self):
        return self.__getitem__(
            random.randint(0, self.n_images)
        )

    def __len__(self):
        return self.n_images

train_dataset = HDF5Dataset(train_path, labels[CHALLENGE])
test_dataset = HDF5Dataset(test_path, labels[CHALLENGE])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=1, 
                                        shuffle=True)


# In[15]:


def get_model(name, pretrained, nc):
    if name == 'VGG19':
        model = models.vgg19(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, nc)
        return model

    elif name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, nc)
        return model

    elif name == 'EfficientNet':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            model = EfficientNet.from_name('efficientnet-b7')
        model._fc = nn.Linear(2560, nc)
        return model
    else:
        print(f'Model called {name} not found...')


# In[16]:


checkpoint_path_full = CHECKPOINT_PATH+EXPERIMENT_NAME+'.pth'
checkpoint_load_path = 'input/artclassification/'+EXPERIMENT_NAME+'.pth'

loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-3
momentum = 0.9

if Path(checkpoint_load_path).is_file():
    print('Using checkpoint file...')
    if ~torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_load_path, map_location=torch.device('cpu'))
    else:    
        checkpoint = torch.load(checkpoint_load_path)

    model = get_model(MODEL, False, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Activate the GPU in the Colab runtime settings to utilize cuda acceleration
    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    avg_train_losses = checkpoint['avg_train_losses']
    avg_test_losses = checkpoint['avg_test_losses']
    accuracy = checkpoint['accuracy']

else:
    print(f'No checkpoint file named {EXPERIMENT_NAME}.pth found; Creating new model...')
    model = get_model(MODEL, PRETRAINED_ON_IMAGENET, num_classes)

    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epoch = 0
    avg_train_losses, avg_test_losses = [], []
    accuracy = []


# In[ ]:


epoch_list = range(epoch+1, epoch + (EPOCHS_PER_SESSION) + 1)

for e in epoch_list:
    print(f'\n-----Epoch {e} started.-----\n')

    since = time.time()
    
    train_losses, test_losses = [], []

    model.train()
    for batch, (images, labels) in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        logits = model(images)

        loss = loss_function(logits, labels)
        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()

        if batch % 200 == 0 or batch == 1 or batch==len(train_loader):
            time_elapsed = time.time() - since
            print(
                f'Training loss = {np.average(train_losses):8.3f} | ',
                f'Batch # {batch:6.0f} | [{time_elapsed//60:3.0f}m {time_elapsed%60:2.0f}s]')
        
    model.eval()
    score = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            logits = model(images)
            test_losses.append(loss_function(logits, labels).item())

            top_p, top_class = logits.topk(1, dim=1)
            correct = top_class.squeeze() == labels
            score += torch.sum(correct.float())

    # Save epoch stats
    avg_train_losses.append(np.average(train_losses))
    avg_test_losses.append(np.average(test_losses))
    accuracy.append(score/len(test_dataset))

    time_elapsed = time.time() - since
    print(
        f"\nSummary:\n",
        f"\tEpoch: {e}/{epoch_list[-1]}\n",
        f"\tLearning Rate: {learning_rate}\n",
        f"\tTraining Loss: {avg_train_losses[-1]:.5f}\n",
        f"\tTesting Loss: {avg_test_losses[-1]:.5f}\n",
        f"\tAccuracy: {accuracy[-1]*100:.2f}%\n",
        f"\tDuration: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s"
    )
    
    print(f'-----Epoch {e} finished.-----\n')

    print(f'Saving model as {checkpoint_path_full} at epoch #{e}')
    torch.save(
        {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_train_losses': avg_train_losses,
            'avg_test_losses': avg_test_losses,
            'accuracy': accuracy
        }, 
        checkpoint_path_full
    )

# Ensuring multiple runs of this cell stack up
epoch = e


# In[21]:


score = 0
model.eval()
with torch.no_grad():
    images, labels = test_dataset.random()
    images.unsqueeze_(0)
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
    logits = model(images)

    top_p, top_class = logits.topk(1, dim=1)
    correct = top_class.squeeze() == labels
    score += torch.sum(correct.float())

test_accuracy = score/len(test_dataset)

print(score)


# In[ ]:




