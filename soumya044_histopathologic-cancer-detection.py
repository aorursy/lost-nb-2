#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Total Samples Available
print('Train Images = ',len(os.listdir('../input/train')))
print('Test Images = ',len(os.listdir('../input/test')))


# In[3]:


df = pd.read_csv('../input/train_labels.csv')
print('Shape of DataFrame',df.shape)
df.head()


# In[4]:


TRAIN_DIR = '../input/train/'


# In[5]:


fig = plt.figure(figsize = (20,8))
index = 1
for i in np.random.randint(low = 0, high = df.shape[0], size = 10):
    file = TRAIN_DIR + df.iloc[i]['id'] + '.tif'
    img = cv2.imread(file)
    ax = fig.add_subplot(2, 5, index)
    ax.imshow(img, cmap = 'gray')
    index = index + 1
    color = ['green' if df.iloc[i].label == 1 else 'red'][0]
    ax.set_title(df.iloc[i].label, fontsize = 18, color = color)
plt.tight_layout()
plt.show()


# In[6]:


# removing this image because it caused a training error previously
df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# removing this image because it's black
df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df.head()


# In[7]:


fig = plt.figure(figsize = (6,6)) 
ax = sns.countplot(df.label).set_title('Label Counts', fontsize = 18)
plt.annotate(df.label.value_counts()[0],
            xy = (0,df.label.value_counts()[0] + 2000),
            va = 'bottom',
            ha = 'center',
            fontsize = 12)
plt.annotate(df.label.value_counts()[1],
            xy = (1,df.label.value_counts()[1] + 2000),
            va = 'bottom',
            ha = 'center',
            fontsize = 12)
plt.ylim(0,150000)
plt.ylabel('Count', fontsize = 16)
plt.xlabel('Labels', fontsize = 16)
plt.show()


# In[8]:


SAMPLE_SIZE = 80000
# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df[df['label'] == 0].sample(SAMPLE_SIZE, random_state = 0)
# filter out class 1
df_1 = df[df['label'] == 1].sample(SAMPLE_SIZE, random_state = 0)

# concat the dataframes
df_train = pd.concat([df_0, df_1], axis = 0).reset_index(drop = True)
# shuffle
df_train = shuffle(df_train)

df_train['label'].value_counts()


# In[9]:


# train_test_split
# stratify=y creates a balanced validation set.
y = df_train['label']

df_train, df_val = train_test_split(df_train, test_size = 0.1, random_state = 0, stratify = y)


# In[10]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#Folder Structure

'''
    * base_dir
        |-- train_dir
            |-- 0   #No Tumor
            |-- 1   #Has Tumor
        |-- val_dir
            |-- 0
            |-- 1
'''
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
no_tumor = os.path.join(train_dir, '0')
os.mkdir(no_tumor)
has_tumor = os.path.join(train_dir, '1')
os.mkdir(has_tumor)


# create new folders inside val_dir
no_tumor = os.path.join(val_dir, '0')
os.mkdir(no_tumor)
has_tumor = os.path.join(val_dir, '1')
os.mkdir(has_tumor)


print(os.listdir('base_dir/train_dir'))
print(os.listdir('base_dir/val_dir'))


# In[11]:


# Set the id as the index in df_data
df.set_index('id', inplace=True)

# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])



# Transfer the train images

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = '0'
    elif target == 1:
        label = '1'
    
    # source path to image
    src = os.path.join('../input/train', file_name)
    # destination path to image
    dest = os.path.join(train_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


# Transfer the val images

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    file_name = image + '.tif'
    # get the label for a certain image
    target = df.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = '0'
    elif target == 1:
        label = '1'
    

    # source path to image
    src = os.path.join('../input/train', file_name)
    # destination path to image
    dest = os.path.join(val_dir, label, file_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dest)


# In[12]:


print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))


# In[13]:


from keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = 96
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = '../input/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32 #10
val_batch_size = 32 #10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[14]:


#Import Keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation


# In[15]:


class Net:
    @staticmethod
    def build(width, height, depth, classes):
            
            #initializa model
            model = Sequential()
            
            inputShape = (height, width, depth)
            
            #Add First Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 32, kernel_size = (5,5), padding="same", activation='relu', input_shape= inputShape))
            model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
                      
            #Add Second Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            #Add Third Layer CONV => ReLU => Pooling
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            
            #FC => ReLU
            model.add(Flatten())
            model.add(Dense(units = 500, activation = 'relu'))
            model.add(Dropout(0.2))
            #FC => Output
            model.add(Dense(classes, activation='softmax'))
            
            model.summary()
            
            return model


# In[16]:


class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 3
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        model.summary()

        # return the constructed network architecture
        return model


# In[17]:


model = Net.build(width = 96, height = 96, depth = 3, classes = 2)
#model = CancerNet.build(width = 96, height = 96, depth = 3, classes = 2)
from keras.optimizers import SGD, Adam, Adagrad
#Edit:: Adagrad(lr=1e-2, decay= 1e-2/10) was used previous;y
model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])


# In[18]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[19]:


# !wget 'https://www.kaggleusercontent.com/kf/10003609/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..lla3iIArVKorEUKxxzzMxg.Ju_WeWrdCHBebCvN-AdSwFCZRJIm1Ru5gJkP-v0jz212zkjh0ojBQ1uHu7Cv7eBXHx8xrBXQHAJpdEy8TQ59Z26Onub-OkbUbWmto-FcjuRGJfFHlxnehCU0fLVB3ZTye4beLcsar4TV_VlKHOic4QP0MW7ajdUimXs09qZhpwI.oZo9D1Huxk091PMK1QJslQ/checkpoint.h5'
# model.load_weights('checkpoint.h5')


# In[20]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, 
                             save_best_only = True, mode = 'max') #Save Best Epoch

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor = 0.5, patience = 2, verbose = 1, mode = 'max', min_lr = 0.00001)                              
callbacks_list = [checkpoint, reduce_lr] # LR Scheduler Used here

history = model.fit_generator(train_gen, steps_per_epoch = train_steps, 
                    validation_data = val_gen,
                    validation_steps = val_steps,
                    epochs = 11,
                    verbose = 1,
                    callbacks = callbacks_list)


# In[21]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()


# In[22]:


# Here the best epoch will be used.
model.load_weights('checkpoint.h5')

val_loss, val_acc = model.evaluate_generator(test_gen, steps=len(df_val))
print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[23]:


# make a prediction
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)


# In[24]:


# Put the predictions into a dataframe.
df_preds = pd.DataFrame(predictions, columns=['no_tumor', 'has_tumor'])
df_preds.head()


# In[25]:


# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor']


# In[26]:


from sklearn.metrics import roc_auc_score, roc_curve, auc
print('ROC AUC Score = ',roc_auc_score(y_true, y_pred))


# In[27]:


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)


# In[28]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.2f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[29]:


from sklearn.metrics import confusion_matrix
# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = predictions.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred_binary)

from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                               cmap = 'Dark2')
plt.show()


# In[30]:


from sklearn.metrics import classification_report
# Generate a classification report

report = classification_report(y_true, y_pred_binary, target_names = ['no_tumor', 'has_tumor'])
print(report)


# In[31]:


shutil.rmtree('base_dir')


# In[32]:


#Folder Structure

'''
    * test_dir
        |-- test_images
'''

# We will be feeding test images from a folder into predict_generator().

# create test_dir
test_dir = 'test_dir'
os.mkdir(test_dir)
    
# create test_images inside test_dir
test_images = os.path.join(test_dir, 'test_images')
os.mkdir(test_images)

# check that the directory we created exists
os.listdir('test_dir')


# In[33]:


# Transfer the test images into image_dir
test_list = os.listdir('../input/test')

for image in test_list:    
    fname = image
    # source path to image
    src = os.path.join('../input/test', fname)
    # destination path to image
    dst = os.path.join(test_images, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)
print('Total Test Images = ',len(os.listdir('test_dir/test_images')))


# In[34]:


test_path ='test_dir'
test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[35]:


num_test_images = 57458 #len(os.listdir('test_dir/test_images')

predictions = model.predict_generator(test_gen, steps=num_test_images, verbose=1)


# In[36]:


if predictions.shape[0] == num_test_images:
    print('All Predictions Done!')
else:
    print('Error!')


# In[37]:


# Put the predictions into a dataframe
df_preds = pd.DataFrame(predictions, columns=['no_tumor', 'has_tumor'])
df_preds.head()


# In[38]:


# This outputs the file names in the sequence in which the generator processed the test images.
test_filenames = test_gen.filenames

# add the filenames to the dataframe
df_preds['file_names'] = test_filenames

# Create an id column
# A file name now has this format: 
# images/00006537328c33e284c973d7b39d340809f7271b.tif

# This function will extract the id:
# 00006537328c33e284c973d7b39d340809f7271b
def extract_id(x):
    
    # split into a list
    a = x.split('/')
    # split into a list
    b = a[1].split('.')
    extracted_id = b[0]
    
    return extracted_id

df_preds['id'] = df_preds['file_names'].apply(extract_id)

df_preds.head()


# In[39]:


# Get the predicted labels.
# We were asked to predict a probability that the image has tumor tissue
y_pred = df_preds['has_tumor']

# get the id column
image_id = df_preds['id']


# In[40]:


submission = pd.DataFrame({'id':image_id, 
                           'label':y_pred, 
                          }).set_index('id')

submission.to_csv('submission.csv', columns=['label'])


# In[41]:


# Delete the test_dir directory we created to prevent a Kaggle error.
# Kaggle allows a max of 500 files to be saved.

shutil.rmtree('test_dir')

