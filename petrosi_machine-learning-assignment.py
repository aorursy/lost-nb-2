#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import gzip
import os
import time
from functools import reduce # only in Python 3
from glob import glob

import librosa
from librosa.display import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from torch.nn import LSTM
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset, random_split

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[2]:


import matplotlib.pyplot as plt
print(plt.style.available)


# In[3]:


print(os.listdir('../input'))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/sample_submission.csv")


# In[4]:


train.head()


# In[5]:


print('The dataset consists of {} training examples'.format(len(train)))
print('The dataset contains audio files from {} different categories'.format(len(train.label.unique())))


# In[6]:


category_group = train.groupby(['label', 'manually_verified']).count()


# In[7]:


plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index)          .plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16,10))
plot.set_xlabel("Category")
plot.set_ylabel("Number of Samples")
plt.legend(['Not verified', 'Verified']);


# In[8]:


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


# In[9]:


import wave
class FGPA_Dataset(Dataset):
    
    def __init__(self, path, filenames, labels, use_mfcc=False):
        super(FGPA_Dataset, self).__init__()
        self.dir = path
        self.sr = 44100 if use_mfcc else 16000 
        self.max_duration_in_sec = 4
        self.max_length = self.sr * self.max_duration_in_sec
        self.use_mfcc = use_mfcc
        self.n_mfccs = 40
        
        self.filenames = filenames
        self.labels = labels
        
        self.sequences = np.array([self.load_data_from(filename) for filename in filenames])
        print(self.sequences.shape)
        
    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]
    
    def __len__(self):
        return len(self.sequences)
    
    def load_data_from(self, filename):

        original_samples = self.read_waveform(filename)

        if len(original_samples) > self.max_length:
            max_offset = len(original_samples) - self.max_length
            offset = np.random.randint(max_offset)
            samples = original_samples[offset:(self.max_length+offset)]
        else:
            if self.max_length > len(original_samples):
                max_offset = self.max_length - len(original_samples)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            samples = np.pad(original_samples, (offset, self.max_length - len(original_samples) - offset), "constant")

        if self.use_mfcc:
            samples = librosa.feature.mfcc(samples, sr=self.sr, n_mfcc=self.n_mfccs)
        else:
            pass
        
        return samples
    
    def read_waveform(self, filename):
        return librosa.core.load(self.dir+filename, sr=self.sr,res_type='kaiser_fast')[0]


# In[10]:


train_csv = pd.read_csv("../input/train.csv")
train_csv = train_csv.iloc[np.random.randint(low=len(train_csv), size=1000)]
train_filenames = train_csv['fname'].values
train_labels = train_csv['label'].values
len(np.unique(train_labels))


# In[11]:


label_transformer = LabelTransformer()
label_transformer = label_transformer.fit(train_labels)
train_label_ids = label_transformer.transform(train_labels)
len(np.unique(train_label_ids))


# In[12]:


test_csv = pd.read_csv("../input/test_post_competition.csv")
test_csv = test_csv[test_csv.usage != 'Ignored']
test_filenames = test_csv['fname'].values
test_labels = test_csv['label'].values
test_label_ids = label_transformer.transform(test_labels)
len(np.unique(test_label_ids))


# In[13]:


train_idx, validation_idx = next(iter(StratifiedKFold(n_splits=5).split(np.zeros_like(train_label_ids), train_label_ids)))
train_files = train_filenames[train_idx]
train_labels = train_label_ids[train_idx]
val_files = train_filenames[validation_idx]
val_labels = train_label_ids[validation_idx]


# In[14]:


d_train[0:2][0].shape


# In[15]:


label_df = pd.DataFrame({'labels':train_labels, 'count': np.ones_like(train_labels)}).groupby(['labels'], as_index=True).count()
label_count_dict = label_df.to_dict()['count']
plt.figure(num=None, figsize=(16,10))
plt.bar(label_count_dict.keys(), label_count_dict.values())
plt.xticks(rotation=90)
plt.show()


# In[16]:


def predict(model, test_dataset, device, batch_size=16):
    
    # Set the model to evaluation mode
    model = model.eval()
    
    # Wrap with no grad because the operations here
    # should not affect the gradient computations
    with torch.no_grad():
        
        predictions = np.array([])
        actual  = np.array([])
        correct = np.array([])
        
        for i, data in enumerate(DataLoader(test_dataset, batch_size)):
            
            # Load the batch
            X_batch, y_batch = data
            # Send to device for faster computations
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get the output of the model
            out = model(X_batch.float()).max(1)[1]
            # Send to device for faster computations
            out = out.to(device)
            
            actual = np.append(actual, y_batch.cpu().detach().numpy())
            correct = np.append(correct, (out == y_batch).cpu().detach().numpy())
            predictions = np.append(predictions, out.cpu().detach().numpy())
                            
    return predictions, correct, actual

def train(model, train_dataset, optimizer, criterion, device, epochs=30, batch_size=16, validation_dataset=None, model_name=None):

    if validation_dataset is not None:
        datasets = {
            'train': train_dataset,
            'validation': validation_dataset
        }
        previous_loss = 100
        phases = ['train', 'validation']
    else:
        train_dataset = train_dataset
        datasets = {
            'train': train_dataset,
            'validation': None
        }
        phases = ['train']
    
    if torch.cuda.device_count() > 1:
        print('Training computations are running on {} GPUs.'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    
    # Send to device for faster computations
    model = model.to(device) 

    train_loss = np.array([])
    validation_loss = np.array([])
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in phases:
            if phase == 'train':
                print('Entering the training phase..')
                # Set the model to training mode
                model = model.train()
            else:
                print('Entering the validation phase..')
                # Set the model to evaluation mode
                model = model.eval()

            # Clear loss for this epoch
            running_loss = 0.0

            for i, data in enumerate(DataLoader(datasets[phase], batch_size=batch_size, drop_last=True)):
                # Load the batch
                X_batch, y_batch = data
                # Send to device for faster computations
                X_batch, y_batch  = X_batch.to(device), y_batch.to(device)

                # Clear gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Get the output of the model
                    out = model(X_batch.float())
                    # Send to device for faster computations
                    out = out.to(device)
                       
                    # Compute loss
                    loss = criterion(out.float(), y_batch)

                    if phase == 'train':
                        # Compute the new gradients
                        loss.backward()
                        # Update the weights
                        optimizer.step()
                
                # Accumulate loss for this batch
                running_loss += loss.item()

            print('{} loss: {}'.format(phase, running_loss / (i+1)))

            if phase == 'validation':
                current_loss = running_loss / (i+1)
                if current_loss < previous_loss:
                    
                    print('Loss decreased. Saving the model..')
                    # If loss decreases,
                    # save the current model as the best-shot checkpoint
                    torch.save(model.state_dict(), '{}.pt'.format(model_name))

                    # update the value of the loss
                    previous_loss = current_loss
                else:
                    pass

            if phase=='train':
                train_loss = np.append(train_loss, running_loss / (i+1))
            else:
                validation_loss = np.append(validation_loss, current_loss)
                
            print()
    
    plt.figure(figsize=(8,6))
    plt.plot(train_loss, c='b')
    plt.plot(validation_loss, c='r')
    plt.xticks(np.arange(len(train_loss)))
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    model.load_state_dict(torch.load('{}.pt'.format(model_name), map_location=device))
    return model


# In[17]:


class CNN_1D(torch.nn.Module):

    def __init__(self, n_features, n_classes):
        
        super(CNN_1D, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.conv_layer11 = nn.Conv1d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 9
        )
        self.conv_layer12 = nn.Conv1d(
            in_channels = 16,
            out_channels = 16,
            kernel_size = 9
        )
        self.max_pool1 = nn.MaxPool1d(
            kernel_size=16
        )
        self.dropout1 = nn.Dropout(0.1)
        self.conv_layer21 = nn.Conv1d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3
        )
        self.conv_layer22 = nn.Conv1d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3
        )
        self.max_pool2 = nn.MaxPool1d(
            kernel_size=4
        )
        self.dropout2 = nn.Dropout(0.1)
        self.conv_layer31 = nn.Conv1d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3
        )
        self.conv_layer32 = nn.Conv1d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3
        )
        self.max_pool3 = nn.MaxPool1d(
            kernel_size=4
        )
        self.dropout3 = nn.Dropout(0.1)
        self.conv_layer41 = nn.Conv1d(
            in_channels = 32,
            out_channels = 256,
            kernel_size = 3
        )
        self.conv_layer42 = nn.Conv1d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = 3
        )
        self.max_pool4 = nn.MaxPool1d(
            kernel_size=1869
        )
        self.conv_layers = nn.Sequential(
            self.conv_layer11,
            self.conv_layer12,
            self.max_pool1,
            self.dropout1,
            self.conv_layer21,
            self.conv_layer22,
            self.max_pool2,
            self.dropout2,
            self.conv_layer31,
            self.conv_layer32,
            self.max_pool3,
            self.dropout3,
            self.conv_layer41,
            self.conv_layer42,
            self.max_pool4
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 41)
        )
    
    def forward(self, input):
        conv_out = self.conv_layers(input.unsqueeze(dim=1)).squeeze()
        dense_out = self.dense_layers(conv_out)
        return dense_out

cnn1d = CNN_1D(1, 41)
# cnn1d.forward(torch.FloatTensor([[d[0][0].numpy()]])).detach().numpy()


# In[18]:


class CNN_2D(torch.nn.Module):

    def __init__(self, n_features, n_classes):
        
        super(CNN_2D, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.conv_layer1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = (4,10)
        )
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        
        self.conv_layer2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (4,10)
        )
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.conv_layer3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (4,10)
        )
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.max_pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        
        self.conv_layers = nn.Sequential(
            self.conv_layer1,
            self.max_pool1,
            self.batch_norm1,
            self.relu1,
            self.conv_layer2,
            self.max_pool2,
            self.batch_norm2,
            self.relu2,
            self.conv_layer3,
            self.max_pool3,
            self.batch_norm3,
            self.relu3
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(32*2*35, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 41)
        )
    
    def forward(self, input):
        conv_out = self.conv_layers(input.unsqueeze(dim=1)).squeeze()
        dense_out = self.dense_layers(conv_out.view(conv_out.size()[0], -1))
        return dense_out

cnn2d = CNN_2D(1, 41)
# cnn1d.forward(torch.FloatTensor([[d[0][0].numpy()]])).detach().numpy()


# In[19]:


train_dataset_wave = FGPA_Dataset("../input/audio_train/audio_train/", train_files, train_labels, use_mfcc=False)
validation_dataset_wave = FGPA_Dataset("../input/audio_train/audio_train/", val_files, val_labels, use_mfcc=False)


# In[20]:


cnn1d = train(cnn1d, train_dataset_wave, torch.optim.Adam(cnn1d.parameters()), nn.CrossEntropyLoss(), torch.device('cuda'), epochs=20, batch_size=32, validation_dataset=validation_dataset_wave, model_name='cnn1d')


# In[21]:


import gc
del train_dataset_wave; del validation_dataset_wave; 
gc.collect()


# In[22]:


train_dataset_mfccs = FGPA_Dataset("../input/audio_train/audio_train/", train_files, train_labels, use_mfcc=True)
validation_dataset_mfccs = FGPA_Dataset("../input/audio_train/audio_train/", val_files, val_labels, use_mfcc=True)


# In[23]:


cnn2d = train(cnn2d, train_dataset_mfccs, torch.optim.Adam(cnn2d.parameters()), nn.CrossEntropyLoss(), torch.device('cuda'), epochs=10, batch_size=32, validation_dataset=validation_dataset_mfccs, model_name='cnn2d')


# In[24]:


test_dataset_wave = FGPA_Dataset("../input/audio_test/audio_test/", test_filenames, test_label_ids)


# In[25]:


predictions, correct, actual = predict(cnn1d, test_dataset_wave, device)


# In[26]:


test_dataset_mfccs = FGPA_Dataset("../input/audio_test/audio_test/", test_filenames, test_label_ids, use_mfcc=True)


# In[27]:


predictions, correct, actual= predict(cnn2d, test_dataset_mfccs, device)


# In[28]:


print(label_transformer.inverse_transform(predictions.astype('int64')))
print(correct.sum() / len(predictions))


# In[29]:


pred_df = pd.DataFrame({'prediction':label_transformer.inverse_transform(predictions.astype('int64')), 'count': np.ones_like(predictions)}).groupby(['prediction'], as_index=True).count()
pred_count_dict = pred_df.to_dict()['count']


# In[30]:


plt.figure(num=None, figsize=(16,10))
plt.bar(pred_count_dict.keys(), pred_count_dict.values())
plt.xticks(rotation=90)
plt.show()


# In[31]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[32]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(actual, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(16,16))
plot_confusion_matrix(cnf_matrix, classes=label_transformer.classes_,
                      title='Confusion matrix, without normalization (Acc: 54.875%)')

# Plot normalized confusion matrix
# plt.figure(figsize=(20,20))
# plot_confusion_matrix(cnf_matrix, classes=label_transformer.classes_, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()


# In[33]:




