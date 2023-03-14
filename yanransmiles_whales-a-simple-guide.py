#!/usr/bin/env python
# coding: utf-8

# In[ ]:


We all like to play around with data and get things done. In this kernel I'll show you how you can do it yourself.

## Notebook Content
1. [Some libraries we need to get things done](#first-bullet)
2. [How to load the dataset](#second-bullet)
3. [Looking at 5 random beauties](#third-bullet)
4. [Preprocessing the data](#forth-bullet)<br/>
     4.1 [Using python OpenCV](#forth1-bullet)<br/>
     4.2 [Using torchvision](#forth2-bullet)<br/>
5. [Cleaning the Data](#fifth-bullet)
6. [Encoding](#fifth-bullet)
7. [Handling the dataset](#sixth-bullet)
8. [Building a very simple sequential model](#seventh-bullet)
9. [Conclusion](#eighth-bullet)


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os

from matplotlib.pyplot import imshow
from PIL import Image


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


img_train_path = os.path.abspath('../input/train')
img_test_path = os.path.abspath('../input/test')
csv_train_path = os.path.abspath('../input/train.csv')


# In[ ]:


df = pd.read_csv(csv_train_path)
df.head()


# In[ ]:


df['Image_path'] = [os.path.join(img_train_path,whale) for whale in df['Image']]


# In[ ]:


five_random_whales = np.random.choice(df['Image'],5)
full_path_random_whales = [os.path.join(img_train_path,random_beauty) for random_beauty in five_random_whales]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for whale in full_path_random_whales:
    img = Image.open(whale)
    plt.imshow(img)
    plt.show()


# In[ ]:


from torchvision import transforms


# In[ ]:


img = cv2.imread(full_path_random_whales[0])
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
plt.imshow(res,cmap='gray')
plt.show()


# In[ ]:


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Grayscale(num_output_channels=1),
   transforms.Resize(128),
   transforms.CenterCrop(128),
   transforms.ToTensor(),
   normalize
])
imgs = [Image.open(whale) for whale in full_path_random_whales]
imgs_tensor = [preprocess(whale) for whale in imgs]


# In[ ]:


imgs_tensor[0].shape


# In[ ]:


img = imgs_tensor[0]
plt.imshow(img[0],cmap='gray')
plt.show()


# In[ ]:


df.Id.value_counts().head()


# In[ ]:


I_dont_want_new_whales = df['Id'] != 'new_whale'
df = df[I_dont_want_new_whales]
df.Id.value_counts().head()


# In[ ]:


unique_classes = pd.unique(df['Id'])
encoding = dict(enumerate(unique_classes))
encoding = {value: key for key, value in encoding.items()}
df = df.replace(encoding)


# In[ ]:


df.head()


# In[ ]:


import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


# In[ ]:


test = df['Image_path'][:1000]
imgs = [Image.open(whale) for whale in test]
imgs_tensor = torch.stack([preprocess(whale) for whale in imgs])


# In[ ]:


labels = torch.tensor(df['Id'][:1000].values)
max_label = int(max(labels)) +1
max_label


# In[ ]:


plt.imshow(imgs_tensor[0].reshape(128,128),cmap='gray')


# In[ ]:


model = nn.Sequential(nn.Linear(128*128, 256),
                      nn.Sigmoid(),
                      nn.Linear(256, 128),
                      nn.Sigmoid(),
                      nn.Linear(128, max_label),
                      nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

model


# In[ ]:


epochs = 5
batch_size = 10
iters = int(len(imgs_tensor)/batch_size)
next_batch = 0
for e in range(epochs):
    running_loss = 0
    next_batch = 0
    for n in range(iters):
        batch_images = imgs_tensor[next_batch:next_batch+batch_size] 
        batch_images = batch_images.view(batch_images.shape[0], -1)
        batch_labels = labels[next_batch:next_batch+batch_size]
        
        optimizer.zero_grad()
        
        output = model(batch_images)
        loss = criterion(output, batch_labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        next_batch += batch_size
        
    print(running_loss)

