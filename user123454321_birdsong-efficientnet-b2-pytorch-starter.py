#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install efficientnet_pytorch')


# In[2]:


import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.models as models # resnet18 pretrained model
import librosa # for feature extraction
import scipy # to load wav files
from efficientnet_pytorch import EfficientNet


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


input_path = '/kaggle/input/'


# In[5]:


train_df = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')


# In[6]:


train_df = train_df[(train_df.filename != 'XC395021.mp3') & (train_df.filename != 'XC504005.mp3') & (train_df.filename != 'XC504006.mp3') & (train_df.filename != 'XC505006.mp3')]


# In[7]:


import os
wav_folders = [
    'birdsong-wav-1',
    'birdsong-wav-2',
    'birdsong-wav-3',
    'birdsong-wav-4',
    'birdsong-wav-5',
]
bird_folder = {
    bird: folder for folder in wav_folders for bird in os.listdir(os.path.join(input_path, folder)) 
}


# In[8]:


row = train_df.iloc[0]
ebird_code = row.ebird_code
file_name = row.filename
file_path = f'{input_path}/{bird_folder[ebird_code]}/{ebird_code}/{file_name.replace("mp3", "wav")}'
sr, audio = scipy.io.wavfile.read(file_path)
if len(audio.shape) == 2:
    audio = audio[:, 0]


# In[9]:


import IPython.display as ipd
ipd.Audio(file_path)


# In[10]:


bird_codes = sorted(list(set(train_df['ebird_code'])))
bird_to_idx = { bird: idx for idx, bird in enumerate(bird_codes) }


# In[11]:


class Dataset(data.Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):

        row = self.df.iloc[index]
        ebird_code = row.ebird_code
        file_name = row.filename
        file_path = f'{input_path}/{bird_folder[ebird_code]}/{ebird_code}/{file_name.replace("mp3", "wav")}'
        sr, audio = scipy.io.wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        i = np.random.randint(len(audio) - 480000) if len(audio) > 480000 else 0
        audio = audio[i:i+480000].astype('float')
        audio = np.pad(audio, (0, 480000 - len(audio)))
        # Generate a melspectrogram with 256 mels.
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=256)
        mel = (mel - mel.mean()) / (mel.std() + 1e-12)
        mel = mel[None, ...]
        return mel, bird_to_idx[row['ebird_code']]

    def __len__(self):
        return len(self.df)


# In[12]:


import sklearn.utils as utils 
train_df = utils.shuffle(train_df, random_state=42)


# In[13]:


train_df.shape


# In[14]:


train_dataset = Dataset(df=train_df.iloc[:20000].reset_index(drop=True))
val_dataset = Dataset(df=train_df.iloc[20000:].reset_index(drop=True))


# In[15]:


train_dataloader = data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=12, num_workers=2, pin_memory=True)
val_dataloader = data.DataLoader(dataset=val_dataset, shuffle=True, batch_size=6, num_workers=2, pin_memory=True)


# In[16]:


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        # Convert 1 channel to 3 channel to be able to send to resnet18
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.base_model = EfficientNet.from_pretrained('efficientnet-b2')
        self.fc2 = nn.Linear(1000, 264) # 264 different birds

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.base_model(x)
        x = self.fc2(x)
        
        return x
        


# In[17]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)


# In[18]:


def get_accuracy(y_pred, y_actual):
    y_pred_ = y_pred.argmax(1).detach().cpu().numpy()
    y_actual_ = y_actual.numpy()
    
    return np.mean(y_pred_ == y_actual_) * 100.0


# In[19]:


criterion = nn.CrossEntropyLoss()


# In[20]:


opt = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[21]:


EPOCHS = 5


# In[22]:


running_acc = 0
for epoch in range(0, EPOCHS):
    print('\nTraining: \n')
    model.train()
    for b, (x, y) in enumerate(train_dataloader):

        opt.zero_grad()

        y_pred = model(x.to(device))
        loss = criterion(y_pred, y.to(device))
        loss.backward()

        opt.step()
        
        acc = get_accuracy(y_pred, y.cpu())
        
        running_acc = running_acc * 0.9 + acc * 0.1

        print('\rEpoch: {}/{},         batch: {}/{},         loss: {:4f},         running_acc: {:.4f}'.format(epoch+1, EPOCHS, b+1, len(train_dataloader), loss.item(), running_acc),  end=' ')
    
    print('\nValidation: \n')
    running_acc = 0
    mean_acc = 0
    model.eval()
    for b, (x, y) in enumerate(val_dataloader):


        y_pred = model(x.to(device))

        loss = criterion(y_pred, y.to(device))
        acc = get_accuracy(y_pred, y)

        running_acc = running_acc * 0.9 + acc * 0.1
        mean_acc = mean_acc + acc

        print('\rbatch: {}/{},         loss: {:4f},         acc: {:.4f}'.format(b+1, len(val_dataloader), loss.item(), running_acc),  end=' ')
    mean_acc /= len(val_dataloader)
    print('Mean accuracy: ', mean_acc)


# In[23]:


torch.save(model.state_dict(), 'birdsong_model.pth')


# In[ ]:




