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
# Any results you write to the current directory are saved as output.


from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import imageio
from skimage.data import imread
import math
from matplotlib.pylab import *
from tqdm import tqdm
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import load_img


# In[2]:


print(os.listdir("../input"))


# In[3]:


train_seg = pd.read_csv('../input/train_ship_segmentations_v2.csv')
#test_seg = pd.read_csv('../input/test_ship_segmentations_v2.csv')


# In[4]:


train = os.path.join('../input','train_v2')
#test = os.listdir('../input/test')


# In[5]:


#train = pd.read_csv(os.path.join('../input','train_v2'))


# In[6]:


train_seg.loc[0]


# In[7]:


train


# In[8]:


def run_len_decode(mask,shape=(768,768)):
    s = mask.split()
    start,length = [np.asarray(x,dtype=int) for x in (s[0:][::2],s[1:][::2])]
    start -=1
    end = start+length
    img = np.zeros(shape[0]*shape[1],dtype=np.uint8)
    for lo,hi in zip(start,end):
        img[lo:hi] = 1
    return img.reshape(shape).T


# In[9]:


def make_mask(img_mask):
    tot_mask = np.zeros((768,768))
    img_mask = img_mask.tolist()
    if not nan in img_mask:
        for mask in img_mask:
            tot_mask+=run_len_decode(mask)
    return(tot_mask)


# In[10]:


def show(num,imgage=False):
    idx = train_seg.loc[num]['ImageId']
    img = imread('../input/train_v2/'+idx)
    fig,ax = plt.subplots(1,3,figsize=(15,40))
    tot_mask = np.zeros((768,768))
    img_mask = train_seg.loc[train_seg['ImageId']==idx,'EncodedPixels'].tolist()
    if not nan in img_mask:
        for mask in img_mask:
            tot_mask+=run_len_decode(mask)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].imshow(img)
    ax[1].imshow(tot_mask)
    ax[2].imshow(img)
    ax[2].imshow(tot_mask, alpha = 0.4)
    print("image shape: ",img.shape)
    print("output shap: ",tot_mask.shape)
    if imgage:
        return tot_mask


# In[11]:


import random


# In[12]:


random.randint(1,100)


# In[13]:


k = show(random.randint(1,100),True)


# In[14]:





# In[14]:


def run_len_ecoding(img):
    px = img.T.flatten()
    px = np.concatenate([[0],px,[0]])
    px = where(px[1:]!=px[:-1])[0]+1
    px[1::2]-=px[::2]
    return ' '.join(str(x) for x in px)


# In[15]:


imshow(run_len_decode(run_len_ecoding(k)))


# In[16]:


idx = train_seg.loc[30]['ImageId']
img = imread('../input/train_v2/'+idx)
print(img.shape)


# In[17]:


img = load_img('../input/train/'+idx,grayscale=False)


# In[18]:


imshow(img)
print(img.size)


# In[19]:


def unet(pretrained_weights=None,input_size=(768,768,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
    
    conv5 = Conv2D(512,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
#     up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = Conv2D(256,(2,2),activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
#     merge6 = merge([drop4,up6],mode='concat',concat_axis=3)
    merge6 = concatenate([drop4,up6])
    conv6 = Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(128,(2,2),activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
#     merge7 = merge([conv3,up7],mode='concat',concat_axis=3)
    merge7 = concatenate([conv3,up7])
    conv7 = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(64,(2,2),activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
#     merge8 = merge([conv2,up8], mode='concat',concat_axis=3)
    merge8 = concatenate([conv2,up8])
    conv8 = Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2D(32,(2,2),activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
#     merge9 = merge([conv1,up9],mode='concat',concat_axis=3)
    merge9 = concatenate([conv1,up9])
    conv9 = Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1,1,activation='sigmoid')(conv9)
    
    model = Model(input=inputs,output=conv10)
    
    model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])
    
    #model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model
    
    


# In[20]:


k = unet()


# In[21]:


k.summary()


# In[22]:


os.listdir('../input')


# In[23]:


path = os.path.join('../input','train')


# In[24]:


imshow(imread(os.path.join(path,train_seg.loc[0]['ImageId'])))


# In[25]:


def givexy():
    X = np.zeros((4,768,768,3))
    Y = np.zeros((4,768,768,1))
    for i,id_ in tqdm(enumerate(train_seg['ImageId'])):
         X[i,...] = imread(os.path.join(path,id_))
         k = resize(make_mask(train_seg[train_seg['ImageId']==id_]['EncodedPixels']),(768,768,1))
         Y[i] = k
         if i==3:
                break
    return X,Y


# In[26]:


resize


# In[27]:


x,y = givexy()


# In[28]:


x.shape


# In[29]:


y.shape


# In[30]:


callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('Model1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

#results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
#                     validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))


# In[31]:


k.fit(x,y,batch_size=10,epochs=1)


# In[32]:


k.predict()


# In[33]:


img = 


# In[34]:


os.listdir('../input')


# In[35]:


os.path.join('../train','train')

