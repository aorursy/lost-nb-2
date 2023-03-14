#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd /kaggle/input/


# In[2]:


# Input data files are available in the "../input/" directory.
import numpy as np 
import pandas as pd 
import os
import cv2
import glob
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout,  Dense
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping


# In[3]:


weights_path = '/kaggle/i/tgs_Salt.h5'


# In[4]:


checkpoint_path = '/kaggle/i/tgs_salt_unet.h5'


# In[5]:


os.mkdir('/kaggle/i')


# In[6]:


cd /kaggle/


# In[7]:


ls


# In[8]:


cd /kaggle/i/


# In[9]:


ls


# In[10]:


weights_path


# In[11]:


img_width, img_height = 128,128
epochs = 1
batch_size = 20


# In[12]:


input_dir  = '/kaggle/input/'
train_dir = input_dir + 'train/'
train_img_dir = train_dir + 'images/'
train_masks_dir = train_dir + 'masks/'
test_dir = input_dir + 'test/'
test_img_dir = test_dir + 'images/'


# In[13]:


data_files = os.listdir(input_dir)
print('data_files in the directory', data_files)


# In[14]:


train_data = os.listdir(train_dir)


# In[15]:


train_data


# In[16]:


test_data = os.listdir(test_dir)


# In[17]:


test_data


# In[18]:


train_data_images = os.listdir(train_img_dir)
print('No. of images in train data:',len(train_data_images))


# In[19]:


train_data_masks = os.listdir(train_masks_dir)
print('No. of masks in train data:',len(train_data_masks))


# In[20]:


test_data_images = os.listdir(test_img_dir)
print('No. of images in test data:',len(test_data_images))


# In[21]:


#Display images and mask
def image_and_mask():
    plt.figure(figsize=(15,15))
    item = np.random.randint(0, len(train_data_images))
    img_name = train_data_images[item]
    img = cv2.imread(train_img_dir + train_data_images[item], 1)
    mask = cv2.imread(train_masks_dir + img_name, 1)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image'+' ' +img_name + ' ' + str(img.shape))
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title('Mask'+' '+img_name + ' ' + str(mask.shape))


# In[22]:


image_and_mask()


# In[23]:


#load csv files
train_csv = pd.read_csv(input_dir + 'train.csv', index_col = 'id', usecols = [0] )
print(len(train_csv))
train_csv.head()


# In[24]:


#tells the depth at which salt is present, for all train and test images
depths_csv = pd.read_csv(input_dir + 'depths.csv', index_col = 'id')
print(len(depths_csv))
depths_csv.head()


# In[25]:


training_df = train_csv.join(depths_csv)
print(len(training_df))
training_df.head()


# In[26]:


depths_csv.index


# In[27]:


print(sum(depths_csv.index.isin(train_csv.index)))
depths_csv.index.isin(train_csv.index)


# In[28]:


len(depths_csv.index.isin(train_csv.index))


# In[29]:


~depths_csv.index.isin(train_csv.index)


# In[30]:


len(~depths_csv.index.isin(train_csv.index))


# In[31]:


testing_df = depths_csv[~depths_csv.index.isin(train_csv.index)]
print(len(testing_df))
testing_df.head()


# In[32]:


#Exploring one image
idx = np.random.randint(0,len(train_data_images))
im = cv2.imread(glob.glob(train_img_dir + '*.png')[idx])
print(idx)
print(im.shape)


# In[33]:


im[22,0,:][0] == im[22,0,:][1] == im[22,0,:][2]


# In[34]:


#To check whether all channels have same values are not
sum1 = 0
for i in range(0,101):
    for j in range(0,101):
        if not im[i,j,:][0] == im[i,j,:][1] == im[i,j,:][2]:
            sum1+=1
            
if sum1==0:
    print('All channels have same pixel values')


# In[35]:


#load images and masks and reshape to 128*128
x = np.array([resize(cv2.imread(i)[:,:,0]/255, (128,128,1), mode = 'constant',preserve_range = True) for i in glob.glob(train_img_dir + '*.png') if cv2.imread(i).shape[0] != 128])
y = np.array([resize(cv2.imread(i)[:,:,0]/255, (128,128,1), mode = 'constant',preserve_range = True) for i in glob.glob(train_masks_dir + '*.png') if cv2.imread(i).shape[0] != 128])    


# In[36]:


x[0].shape


# In[37]:


y[0].shape


# In[38]:


print(x.shape)
print(y.shape)


# In[39]:


#unet
IMG_SIZE = 128
NO_OF_CHANNELS = 1

#Unet Blocks
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((IMG_SIZE, IMG_SIZE, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(NO_OF_CHANNELS, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


# In[40]:


def train():
    model = UNet()
    earlystopper = EarlyStopping(patience=5, verbose=1)
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x, y,validation_split = 0.2,batch_size = 20, epochs =20)#, callbacks = [checkpointer])
    model.save_weights('/kaggle/i/unet_tgs_salt.h5')
    return model


# In[41]:


trained_model = train()


# In[42]:


x_test = np.array([resize(cv2.imread(i)[:,:,0]/255, (128,128,1), mode = 'constant',preserve_range = True) for i in glob.glob(test_img_dir + '*.png') if cv2.imread(i).shape[0] != 128])


# In[43]:


preds = trained_model.predict(x_test)


# In[44]:


preds.shape


# In[45]:


x = preds[0]


# In[46]:


x.shape


# In[47]:


plt.imshow(np.resize(x,(128,128)), cmap = 'gray')


# In[48]:


np.unique(np.resize(x,(128,128)))


# In[49]:


y = (x>0.5)*1 #multiplying boolean array with 1


# In[50]:


np.unique(y)


# In[51]:


y.shape


# In[52]:


plt.imshow(np.resize(y, (101,101)), cmap = 'gray')


# In[ ]:





# In[ ]:




