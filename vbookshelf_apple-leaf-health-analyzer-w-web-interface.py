#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Check the tensorflow version

import tensorflow as tf
tf.__version__


# In[2]:


import pandas as pd
import numpy as np
import os

import cv2

import albumentations as albu
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report

import shutil

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


# In[4]:


os.listdir('../input/plant-pathology-2020-fgvc7')


# In[5]:


# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py


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
    plt.xticks(tick_marks, classes, rotation=45)
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



# In[6]:


path = '../input/plant-pathology-2020-fgvc7/train.csv'
df_train = pd.read_csv(path)

path = '../input/plant-pathology-2020-fgvc7/test.csv'
df_test = pd.read_csv(path)

path = '../input/plant-pathology-2020-fgvc7/sample_submission.csv'
df_sample = pd.read_csv(path)


print(df_train.shape)
print(df_test.shape)
print(df_sample.shape)


# In[7]:


# Identify the target class of each row in the train set

def get_class(row):
    
    if row['multiple_diseases'] == 1:
        return 'multiple_diseases'
    
    elif row['rust'] == 1:
        return 'rust'
    
    elif row['scab'] == 1:
        return 'scab'
    
    else:
        return 'healthy'
    
df_train['target'] = df_train.apply(get_class, axis=1)

df_train.head()


# In[ ]:





# In[8]:


# Filter out each class
df_healthy = df_train[df_train['target'] == 'healthy']
df_multiple_diseases = df_train[df_train['target'] == 'multiple_diseases']
df_rust = df_train[df_train['target'] == 'rust']
df_scab = df_train[df_train['target'] == 'scab']


# In[9]:


# Example
df_scab.head()


# In[10]:


path = '../input/plant-pathology-2020-fgvc7/images/'

image_list = list(df_healthy['image_id'])


# set up the canvas for the subplots
plt.figure(figsize=(25,10))

# Our subplot will contain 2 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)
plt.subplot(2,4,1)

# plt.imread reads an image from a path and converts it into an array

# starting from 1 makes the code easier to write
for i in range(1,9):
    
    plt.subplot(2,4,i)
    
    # get an image
    image = image_list[i]
    
    # display the image
    plt.imshow(plt.imread(path + image + '.jpg'))
    
    plt.xlabel('healthy', fontsize=20)


# In[11]:


path = '../input/plant-pathology-2020-fgvc7/images/'

image_list = list(df_multiple_diseases['image_id'])


# set up the canvas for the subplots
plt.figure(figsize=(25,10))

# Our subplot will contain 2 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)
plt.subplot(2,4,1)

# plt.imread reads an image from a path and converts it into an array

# starting from 1 makes the code easier to write
for i in range(1,9):
    
    plt.subplot(2,4,i)
    
    # get an image
    image = image_list[i]
    
    # display the image
    plt.imshow(plt.imread(path + image + '.jpg'))
    
    plt.xlabel('multiple_diseases', fontsize=20)


# In[12]:


path = '../input/plant-pathology-2020-fgvc7/images/'

image_list = list(df_rust['image_id'])


# set up the canvas for the subplots
plt.figure(figsize=(25,10))

# Our subplot will contain 2 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)
plt.subplot(2,4,1)

# plt.imread reads an image from a path and converts it into an array

# starting from 1 makes the code easier to write
for i in range(1,9):
    
    plt.subplot(2,4,i)
    
    # get an image
    image = image_list[i]
    
    # display the image
    plt.imshow(plt.imread(path + image + '.jpg'))
    
    plt.xlabel('rust', fontsize=20)


# In[13]:


path = '../input/plant-pathology-2020-fgvc7/images/'

image_list = list(df_scab['image_id'])


# set up the canvas for the subplots
plt.figure(figsize=(25,10))

# Our subplot will contain 2 rows and 4 columns
# plt.subplot(nrows, ncols, plot_number)
plt.subplot(2,4,1)

# plt.imread reads an image from a path and converts it into an array

# starting from 1 makes the code easier to write
for i in range(1,9):
    
    plt.subplot(2,4,i)
    
    # get an image
    image = image_list[i]
    
    # display the image
    plt.imshow(plt.imread(path + image + '.jpg'))
    
    plt.xlabel('scab', fontsize=20)


# In[14]:


df_train['target'].value_counts()


# In[15]:


# select the column that we will use for stratification
y = df_train['target']

# shuffle
df_train = shuffle(df_train)

df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=101, stratify=y)


print(df_train.shape)
print(df_val.shape)


# In[16]:


df_train['target'].value_counts()


# In[17]:


df_val['target'].value_counts()


# In[18]:


df_1 = df_train[df_train['target'] != 'multiple_diseases']

df_2 = df_train[df_train['target'] == 'multiple_diseases']

df_train_up = pd.concat([df_1, df_2,  df_2,  df_2,  df_2,  df_2], axis=0).reset_index(drop=True)

df_train = shuffle(df_train_up)

print(df_train.shape)

df_train.head()


# In[19]:


# This is the new class distribution of the train set

df_train['target'].value_counts()


# In[ ]:





# In[20]:


df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)
df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)
df_test.to_csv('df_test.csv.gz', compression='gzip', index=False)


# In[21]:


get_ipython().system('ls')


# In[22]:


# Albumentations

import albumentations as albu


def augment_image(augmentation, image):
    
    """
    Uses the Albumentations library.
    
    Inputs: 
    1. augmentation - this is the instance of type of augmentation to do 
    e.g. aug_type = HorizontalFlip(p=1) 
    # p=1 is the probability of the transform being executed.
    
    2. image - image with shape (h,w)
    
    Output:
    Augmented image as a numpy array.
    
    """
    # get the transform as a dict
    aug_image_dict =  augmentation(image=image)
    # retrieve the augmented matrix of the image
    image_matrix = aug_image_dict['image']
    
    
    return image_matrix


# In[23]:


# Define the transforms

# Modified from --> Pneumothorax - 1st place solution
# Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824#latest-620521


aug_types = albu.Compose([
            albu.HorizontalFlip(),
             albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                ], p=0.8),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
                ], p=0.3),
            albu.OneOf([
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            albu.ShiftScaleRotate()
            ])


# In[24]:


# Get an image to test the transformations

# get a list of train png images
path = '../input/plant-pathology-2020-fgvc7/images/'
image_list = os.listdir(path)

fname = image_list[1]
image_path = path + fname

print(fname)

image = plt.imread(image_path)
plt.imshow(image)

plt.show()


# In[25]:


# Test the transformation setup.
# The image will be different each time this cell is run.

aug_image = augment_image(aug_types, image)

plt.imshow(aug_image)

plt.show()


# In[ ]:





# In[26]:


#df_train.head()


# In[27]:


def train_generator(batch_size=8):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_train.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_train = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            

        
            
            # Create X_train
            #================
            
            for i in range(0, len(image_id_list)):
              
              
                # get the image and mask
                image_id = image_id_list[i] + '.jpg'


                # set the path to the image
                path = '../input/plant-pathology-2020-fgvc7/images/' + image_id

                # read the image
                image = cv2.imread(path)
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
                
                
            
            
            # Create y_train
            # ===============
                cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
                y_train = df[cols]
                y_train = np.asarray(y_train) 


       
              
            # Augment the image and mask
            # ===========================

                aug_image = augment_image(aug_types, image)
              
                # insert the image into X_train
                X_train[i] = aug_image
                
                          
                
            # Normalize the images
            X_train = X_train/255

            yield X_train, y_train


# In[28]:


# Test the generator

# initialize
train_gen = train_generator(batch_size=8)

# run the generator
X_train, y_train = next(train_gen)

print(X_train.shape)
print(y_train.shape)


# In[29]:


y_train


# In[30]:


# Print the first image in X_train
# Remember that train images have been augmented.

image = X_train[0,:,:,:]
plt.imshow(image)


# In[ ]:





# In[31]:


def val_generator(batch_size=5):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_val.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            

        
            
            # Create X_val
            #================
            
            for i in range(0, len(image_id_list)):
              
              
                # get the image and mask
                image_id = image_id_list[i] + '.jpg'


                # set the path to the image
                path = '../input/plant-pathology-2020-fgvc7/images/' + image_id

                # read the image
                image = cv2.imread(path)
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # insert the image into X_train
                X_val[i] = image
                
                
            
            
            # Create y_val
            # ===============

                cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
                y_val = df[cols]
                y_val = np.asarray(y_val) 

                       
                
            # Normalize the images
            X_val = X_val/255

            yield X_val, y_val


# In[32]:


# Test the generator

# initialize
val_gen = val_generator(batch_size=5)

# run the generator
X_val, y_val = next(val_gen)

print(X_val.shape)
print(y_val.shape)


# In[33]:


y_val


# In[34]:


# print the image from X_val
image = X_val[0,:,:,:]
plt.imshow(image)


# In[ ]:





# In[35]:


def test_generator(batch_size=1):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_test.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_test = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            

        
            
            # Create X_test
            #================
            
            for i in range(0, len(image_id_list)):
              
              
                # get the image and mask
                image_id = image_id_list[i] + '.jpg'


                # set the path to the image
                path = '../input/plant-pathology-2020-fgvc7/images/' + image_id

                # read the image
                image = cv2.imread(path)
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # insert the image into X_train
                X_test[i] = image
                
                 
                
            # Normalize the images
            X_test = X_test/255

            yield X_test


# In[36]:


# Test the generator

# initialize
test_gen = test_generator(batch_size=1)

# run the generator
X_test = next(test_gen)

print(X_test.shape)


# In[37]:


# print the image from X_test

image = X_test[0,:,:,:]
plt.imshow(image)


# In[ ]:





# In[38]:


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_accuracy

from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, CSVLogger, LearningRateScheduler)


# In[39]:


from tensorflow.keras.applications.mobilenet import MobileNet

model = MobileNet(weights='imagenet')

# Exclude the last 2 layers of the above model.
x = model.layers[-2].output

# Create a new dense layer for predictions
# 3 corresponds to the number of classes
predictions = Dense(4, activation='softmax')(x)

# inputs=model.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=model.input, outputs=predictions)

model.summary()


# In[40]:


TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 5

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = TRAIN_BATCH_SIZE
val_batch_size = VAL_BATCH_SIZE

# determine num train steps
train_steps = np.ceil(num_train_samples / train_batch_size)

# determine num val steps
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[41]:


# Initialize the generators
train_gen = train_generator(batch_size=TRAIN_BATCH_SIZE)
val_gen = val_generator(batch_size=VAL_BATCH_SIZE)

model.compile(
    Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])



filepath = "model.h5"

#earlystopper = EarlyStopping(patience=10, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, 
                                   verbose=1, mode='max')



log_fname = 'training_log.csv'
csv_logger = CSVLogger(filename=log_fname,
                       separator=',',
                       append=False)

callbacks_list = [checkpoint, csv_logger, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=50, 
                              validation_data=val_gen, validation_steps=val_steps,
                             verbose=1,
                             callbacks=callbacks_list)


# In[42]:


# Display the training log

train_log = pd.read_csv('training_log.csv')

train_log.head()


# In[43]:


# get the metric names so we can use evaulate_generator
model.metrics_names


# In[44]:


model.load_weights('model.h5')

val_gen = val_generator(batch_size=VAL_BATCH_SIZE)

val_loss, val_acc = model.evaluate_generator(val_gen, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_acc:', val_acc)


# In[ ]:





# In[45]:


# display the loss and accuracy curves

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()


# In[ ]:





# In[46]:


model.load_weights('model.h5')

val_gen = val_generator(batch_size=1)

preds = model.predict_generator(val_gen, steps=len(df_val), verbose=1)


# In[47]:


# get y_pred as index values

y_pred = np.argmax(preds, axis=1)


# In[48]:


# get y_true as index values

cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
y_true = df_val[cols]
y_true = np.asarray(y_true) 

y_true = np.argmax(y_true, axis=1)


# In[ ]:





# In[49]:


from sklearn.metrics import confusion_matrix
import itertools

cm = confusion_matrix(y_true, y_pred)


# In[50]:


cm_plot_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:





# In[51]:


from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_true, y_pred, target_names=['healthy', 'multiple_diseases', 'rust', 'scab'])

print(report)


# In[ ]:





# In[52]:


model.load_weights('model.h5')

# initialize the generator
test_gen = test_generator(batch_size=1)

preds = model.predict_generator(test_gen, steps=len(df_test), verbose=1)


# In[53]:


#preds


# In[54]:


# Put the preds into a dataframe

df_preds = pd.DataFrame(preds, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])

df_preds['image_id'] = df_test['image_id'].copy()

df_preds.head()


# In[55]:


df_test.head()


# In[56]:


#df_sample.head()


# In[57]:


# Create a submission csv file

df_results = pd.DataFrame({'image_id': df_preds.image_id,
                            'healthy': df_preds.healthy,
                               'multiple_diseases': df_preds.multiple_diseases,
                               'rust': df_preds.rust,
                               'scab': df_preds.scab
                           }).set_index('image_id')


# create a submission csv file
df_results.to_csv('submission.csv', 
                  columns=['healthy', 'multiple_diseases', 'rust', 'scab']) 


# In[58]:


df_results.head()


# In[59]:


get_ipython().system('ls')


# In[ ]:





# In[60]:


# --ignore-installed is added to fix an error.

# https://stackoverflow.com/questions/49932759/pip-10-and-apt-how-to-avoid-cannot-uninstall
# -x-errors-for-distutils-packages

get_ipython().system('pip install tensorflowjs --ignore-installed')


# In[61]:


# Use the command line conversion tool to convert the model

get_ipython().system('tensorflowjs_converter --input_format keras model.h5 tfjs/model')


# In[62]:


# check that the folder containing the tfjs model files has been created
get_ipython().system('ls')


# In[ ]:





# In[ ]:




