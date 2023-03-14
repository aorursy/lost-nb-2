#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import PIL
import cv2
from PIL import Image, ImageOps

from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, Input, Conv2D, GlobalAveragePooling2D)
from keras.applications.densenet import DenseNet121
import keras
from keras.models import Model

SIZE = 224
NUM_CLASSES = 1108

train_csv = pd.read_csv("../input/recursion-cellular-image-classification/train.csv")
test_csv = pd.read_csv("../input/recursion-cellular-image-classification/test.csv")
sub = pd.read_csv("../input/recursion-cellular-keras-densenet/submission.csv")


# In[2]:


np.stack([train_csv.plate.values[train_csv.sirna == i] for i in range(10)]).transpose()


# In[3]:


# you will see the same output here for each sirna number
train_csv.loc[train_csv.sirna==0,'plate'].value_counts()


# In[4]:


plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
    
plate_groups[:10,:]


# In[5]:


all_test_exp = test_csv.experiment.unique()

group_plate_probs = np.zeros((len(all_test_exp),4))
for idx in range(len(all_test_exp)):
    preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values
    pp_mult = np.zeros((len(preds),1108))
    pp_mult[range(len(preds)),preds] = 1
    
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    
    for j in range(4):
        mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) ==                np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        
        group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)


# In[6]:


pd.DataFrame(group_plate_probs, index = all_test_exp)


# In[7]:


exp_to_group = group_plate_probs.argmax(1)
print(exp_to_group)


# In[8]:


from keras.applications.nasnet import NASNetLarge
def create_model(input_shape,n_out):
    input_tensor = Input(shape=input_shape)
    base_model = NASNetLarge(include_top=False,
                   weights='imagenet',
                   input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# In[9]:


model = create_model(input_shape=(SIZE,SIZE,3),n_out=NUM_CLASSES)


# In[10]:


predicted = []
for i, name in tqdm(enumerate(test_csv['id_code'])):
    path1 = os.path.join('../input/recursion-cellular-image-classification-224-jpg/test/test/', name+'_s1.jpeg')
    image1 = cv2.imread(path1)
    score_predict1 = model.predict((image1[np.newaxis])/255)
    
    path2 = os.path.join('../input/recursion-cellular-image-classification-224-jpg/test/test/', name+'_s2.jpeg')
    image2 = cv2.imread(path2)
    score_predict2 = model.predict((image2[np.newaxis])/255)
    
    predicted.append(0.5*(score_predict1 + score_predict2))
    #predicted.append(score_predict1)


# In[11]:


predicted = np.stack(predicted).squeeze()


# In[12]:


def select_plate_group(pp_mult, idx):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) !=            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult


# In[13]:


for idx in range(len(all_test_exp)):
    #print('Experiment', idx)
    indices = (test_csv.experiment == all_test_exp[idx])
    
    preds = predicted[indices,:].copy()
    
    preds = select_plate_group(preds, idx)
    sub.loc[indices,'sirna'] = preds.argmax(1)


# In[14]:


(sub.sirna == pd.read_csv("../input/recursion-cellular-keras-densenet/submission.csv").sirna).mean()


# In[15]:


sub.to_csv('../working/submission.csv', index=False, columns=['id_code','sirna'])

