#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 
print(os.listdir("../input/"))
box_path="../input/humpback-whale-identification-fluke-location/cropping.txt"
train_dataset="../input/whale-categorization-playground/train/train/"
test_dataset="../input/whale-categorization-playground/test/test/"


# In[2]:


boxs=[]
with open(box_path,"r") as f:
    for line in f.readlines():
        p,*coord = line.split(",")
        line=(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) 
        boxs.append(line)
print(len(boxs))
print(boxs[:2])


# In[3]:


import matplotlib.pyplot as plt
import cv2
def pathfile(path):
    from os.path import isfile
    p=train_dataset+path
    if isfile(p):
        return p
    p=test_dataset+path
    if isfile(p):
        return p
    return p

def bounding_rectangle(list,src_size,size=128):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
#     return x0,y0,x1,y1
    return x0/src_size[1],y0/src_size[0],x1/src_size[1],y1/src_size[0]

size=128
src=cv2.imread(pathfile(boxs[0][0]),3)
print(src.shape)
plt.imshow(src)
img_box= bounding_rectangle(boxs[0][1],src.shape[:2],size)
src=cv2.resize(src,(size,size))
print(img_box)
cv2.rectangle(src,(int(size*img_box[0]),int(size*img_box[1])),(int(size*img_box[2]),int(size*img_box[3])),(0,255,0), 1)
plt.imshow(src)
plt.show()


# In[4]:


import numpy as np
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import *
from keras.applications import *
from keras.utils import *
from keras.metrics import *
import keras.backend as K
import cv2
import pandas as pd


# In[5]:


anisotropy = 2.15
img_shape=(128,128,1)


# In[6]:


import random
import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
from numpy.linalg import inv as mat_inv
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
from os.path import isfile


def read_raw_image(p):
    return pil_image.open(pathfile(p))

def boundingrectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0,y0,x1,y1

# Read an image as black&white numpy array
def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

# Compute the coordinate transformation required to center the pictures, padding as required.
def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    if wi/hi/anisotropy < wo/ho: # input image too narrow, extend width
        w     = hi*wo/ho*anisotropy
        left  = (wi-w)/2
        right = left + w
    else: # input image too wide, extend height
        h      = wi*ho/wo/anisotropy
        top    = (hi-h)/2
        bottom = top + h
    center_matrix   = np.array([[1, 0, -ho/2], [0, 1, -wo/2], [0, 0, 1]])
    scale_matrix    = np.array([[(bottom - top)/ho, 0, 0], [0, (right - left)/wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi/2], [0, 1, wi/2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))

# Apply an affine transformation to an image represented as a numpy array.
def transform_img(x, affine):
    matrix   = affine[:2,:2]
    offset   = affine[:2,2]
    x        = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)

# Read an image for validation, i.e. without data augmentation.
def read_for_validation(p):
    x  = read_array(p)
    t  = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t 

# Read an image for training, i.e. including a random affine transformation
def read_for_training(p):
    x  = read_array(p)
    t  = build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.9, 1.0),
            random.uniform(0.9, 1.0),
            random.uniform(-0.05*img_shape[0], 0.05*img_shape[0]),
            random.uniform(-0.05*img_shape[1], 0.05*img_shape[1]))
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t   

# Transform corrdinates according to the provided affine transformation
def coord_transform(list, trans):
    result = []
    for x,y in list:
        y,x,_ = trans.dot([y,x,1]).astype(np.int)
        result.append((x,y))
    return result


# In[7]:


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  batch_size=32, dim=(128,128), n_channels=3,
                 n_classes=5005, shuffle=False,model=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.model = model
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        y = np.zeros((self.batch_size,self.n_classes))
        # Generate data
        if self.model:
            for i, ID in enumerate(list_IDs_temp):
                img,trans  = read_for_training(ID[0])
                coords  = coord_transform(ID[1], mat_inv(trans))
                X[i,]= img
                y[i,]= boundingrectangle(coords)
        else:
            for i, ID in enumerate(list_IDs_temp):
                img,trans  = read_for_validation(ID[0])
                coords  = coord_transform(ID[1], mat_inv(trans))
                X[i,]= img
                y[i,]= boundingrectangle(coords)
        return X, y


# In[8]:


images_size=128
params = {'dim': (images_size,images_size),
          'batch_size': 32,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': True,
         'model':True}

from sklearn.model_selection import train_test_split
train_txt, test_txt = train_test_split(boxs, test_size=200, random_state=1)
print(type(train_txt))
train_txt += train_txt
train_txt += train_txt
train_txt += train_txt
train_txt += train_txt
print(len(train_txt),len(test_txt))
# Generators
training_generator = DataGenerator(train_txt,  **params)
params = {'dim': (images_size,images_size),
          'batch_size': 32,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': True,
         'model':False}
validation_generator = DataGenerator(test_txt,  **params)


# In[9]:


for image,label in training_generator:
        print(image.shape,label.shape)
        break


# In[10]:


import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from keras import backend as K
from keras.preprocessing.image import array_to_img
from numpy.linalg import inv as mat_inv

def show_whale(imgs, per_row=5):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))

images = []
for image,label in validation_generator:
    for i in range(3):
        a         = image[i:i+1]
        rect1     = label[i]
        img       = array_to_img(a[0]).convert('RGB')
        draw      = Draw(img)
        draw.rectangle(list(rect1), outline='red')
        images.append(img)
    break

show_whale(images)


# In[11]:



img_shape  = (128,128,1)
def build_model(with_dropout=True):
    kwargs     = {'activation':'relu', 'padding':'same'}
    conv_drop  = 0.2
    dense_drop = 0.5
    inp        = Input(shape=img_shape)

    x = inp

    x = Conv2D(64, (9, 9), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    x = Conv2D(64, (2, 2), **kwargs, strides=2)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = Conv2D(64, (3, 3), **kwargs)(x)
    x = BatchNormalization()(x)
    if with_dropout: x = Dropout(conv_drop, noise_shape=(None, 1, 1, int(x.shape[-1])))(x)

    h = MaxPooling2D(pool_size=(1, int(x.shape[2])))(x)
    h = Flatten()(h)
    if with_dropout: h = Dropout(dense_drop)(h)
    h = Dense(16, activation='relu')(h)

    v = MaxPooling2D(pool_size=(int(x.shape[1]), 1))(x)
    v = Flatten()(v)
    if with_dropout: v = Dropout(dense_drop)(v)
    v = Dense(16, activation='relu')(v)

    x = Concatenate()([h,v])
    if with_dropout: x = Dropout(0.5)(x)
    x = Dense(4, activation='linear')(x)
    return Model(inp,x)

model = build_model(with_dropout=True)
model.summary()


# In[12]:


def calculate_iou(y_true, y_pred):
  """
  Input:
  Keras provides the input as numpy arrays with shape (batch_size, num_columns).

  Arguments:
  y_true -- first box, numpy array with format [x, y, width, height, conf_score]
  y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
  x any y are the coordinates of the top left corner of each box.

  Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)

  """
  import numpy as np
  
  results = []
  
  for i in range(0, y_true.shape[0]):
    # set the types so we are sure what type we are using
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    # boxTrue
    x_boxTrue_tleft = y_true[0, 0]  # numpy index selection
    y_boxTrue_tleft = y_true[0, 1]
    boxTrue_width = y_true[0, 2]
    boxTrue_height = y_true[0, 3]
    area_boxTrue = (boxTrue_width * boxTrue_height)
    # boxPred
    x_boxPred_tleft = y_pred[0, 0]
    y_boxPred_tleft = y_pred[0, 1]
    boxPred_width = y_pred[0, 2]
    boxPred_height = y_pred[0, 3]
    area_boxPred = (boxPred_width * boxPred_height)
    # calculate the bottom right coordinates for boxTrue and boxPred
    # boxTrue
    x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
    y_boxTrue_br = y_boxTrue_tleft + boxTrue_height  # Version 2 revision
    # boxPred
    x_boxPred_br = x_boxPred_tleft + boxPred_width
    y_boxPred_br = y_boxPred_tleft + boxPred_height  # Version 2 revision
    
    # calculate the top left and bottom right coordinates for the intersection box, boxInt
    
    # boxInt - top left coords
    x_boxInt_tleft = np.max([x_boxTrue_tleft, x_boxPred_tleft])
    y_boxInt_tleft = np.max([y_boxTrue_tleft, y_boxPred_tleft])  # Version 2 revision
    
    # boxInt - bottom right coords
    x_boxInt_br = np.min([x_boxTrue_br, x_boxPred_br])
    y_boxInt_br = np.min([y_boxTrue_br, y_boxPred_br])
    
    # Calculate the area of boxInt, i.e. the area of the intersection
    # between boxTrue and boxPred.
    # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
    
    
    # Version 2 revision
    area_of_intersection =       np.max([0, (x_boxInt_br - x_boxInt_tleft)]) * np.max([0, (y_boxInt_br - y_boxInt_tleft)])
    
    iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)
    
    # This must match the type used in py_func
    iou = iou.astype(np.float32)
    
    # append the result to a list at the end of each loop
    results.append(iou)
  # return the mean IoU score for the batch
  return np.mean(results)


def bbox_IoU(y_true, y_pred):
  # print(K.eval(bbox_IoU(np.array([[1, 2, 32, 33]], dtype=np.float32), np.array([[1, 2, 32, 33]], dtype=np.float32))))
  import tensorflow as tf
  iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)
  
  return iou


# In[13]:


def top_5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(loss='mse', optimizer=Adam(lr=0.032),metrics=[bbox_IoU])

checkpoint = ModelCheckpoint('weights.h5',  # model filename
                             monitor='val_bbox_IoU', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m
early_stopping = EarlyStopping(monitor='val_bbox_IoU',mode='max', patience=20, verbose=1)
backs=[checkpoint,early_stopping]
history = model.fit_generator(training_generator, epochs=100, validation_data=validation_generator,callbacks=backs,verbose=1)


# In[14]:


model.load_weights("weights.h5")
optimizer = Adam(lr=0.002) 
model.compile(loss='mse', optimizer=optimizer,metrics=[bbox_IoU])
checkpoint = ModelCheckpoint('weights.h5',  # model filename
                             monitor='val_bbox_IoU', # quantity to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='max') # The decision to overwrite model is m
early_stopping = EarlyStopping(monitor='val_bbox_IoU',mode='max', patience=30, verbose=1)
backs=[checkpoint,early_stopping]
history = model.fit_generator(training_generator, epochs=100, validation_data=validation_generator,callbacks=backs,verbose=1)


# In[15]:


model.load_weights("weights.h5")
print(model.evaluate_generator(validation_generator))  


# In[16]:


model2 = build_model(with_dropout=False)
model2.load_weights("weights.h5")


# In[17]:


for layer in model2.layers:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False
model2.compile(Adam(lr=0.002), loss='mean_squared_error',metrics=[bbox_IoU])
model2.fit_generator(training_generator, epochs=1, verbose=1, validation_data=validation_generator)
for layer in model2.layers:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True
model2.compile(Adam(lr=0.002), loss='mean_squared_error')
model2.save('cropping.model')


# In[18]:


print(model2.evaluate_generator(validation_generator))  


# In[19]:


y_=model2.predict_generator(validation_generator)
print(y_.shape)


# In[20]:


y_test =y_
print(y_test.shape)


# In[21]:


y_true=np.array([label for image,label in validation_generator])
y_true= y_true.reshape((-1,4))
print(y_true.shape) 


# In[22]:


def intersection(a,b):
    #求重合区左上角坐标
    x=max(a[0],b[0])
    y=max(a[1],b[1])
    #求出重合区右下角坐标，再求出重合区的宽度和高度
    w=min(a[2],b[2]) - x
    h=min(a[3],b[3]) - y
    if w<=0 or h<=0:
        return 0
    return w*h

def union(a,b,intersection_area):
    area1=(a[2]-a[0]) * (a[3]-a[1])
    area2=(b[2]-b[0]) * (b[3]-b[1])

    return area1+area2-intersection_area

def iou(a,b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0

    intersection_area=intersection(a,b)
    return float(intersection_area)/union(a,b,intersection_area)
all_ratio=[]
from tqdm import tqdm
for index in tqdm(range(y_true.shape[0])) :
    all_ratio.append(iou(y_true[index],y_test[index]))


# In[23]:


print("mean IOU:",np.mean(all_ratio))


# In[24]:


from PIL import Image as pil_image
from PIL.ImageDraw import Draw
def show_whale(imgs, per_row=5):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): 
        ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): 
        ax.imshow(img.convert('RGB'))


from keras.preprocessing.image import array_to_img
images = []
for image,label in validation_generator:
    for i in range(3):
        a         = image[i:i+1]
        rect1     = label[i]
        rect2     = model2.predict(a).squeeze()
        img       = array_to_img(a[0]).convert('RGB')
        draw      = Draw(img)
        draw.rectangle(list(rect1), outline='red')
        draw.rectangle(rect2, outline='yellow')
        images.append(img)
    break

show_whale(images)


# In[25]:





# In[25]:





# In[25]:




