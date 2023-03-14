#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ls ../input/smalldata/smalldata/smalldata


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFile 
import numpy as np
import os 
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import pandas as pd
import random
from tqdm import tqdm
import seaborn as sns
import math


# In[ ]:


import keras
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback,ModelCheckpoint
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.image import load_img
from sklearn.metrics import classification_report


# In[ ]:


batch_size = 32
epochs = 10
num_classes = 2
num_t_samples = 20000
num_v_samples = 5000
path='../input/dogs-vs-cats-redux-kernels-edition/'
dir= '../input/dogscats/data/data/'
train_data_path = path+'train/'
test_data_path =  path+'test/'
train_data_dir = dir+'train/'
validation_data_dir=dir+'validation/'
test_dir='../input/dogscatstest/test1/test1/'
sample_test_dir='../input/sample-testdata/sample test data/sample test data'
img_size=224
#img_size=150


# In[ ]:


# Define the path for training and test image data.
train_images_path = [train_data_path+i for i in os.listdir(train_data_path)] 
train_images= os.listdir(train_data_path)
validation_images=os.listdir(train_data_path)
test_images = os.listdir(test_data_path)
test_images_path = [test_data_path+i for i in os.listdir(test_data_path)] 


# In[ ]:


# Visualize class distribution of traning data.
def image_plot(image_data):
    labels = []
    for img in image_data:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')


# In[ ]:


# Plotting training data to visualize class distribution by calling image_plot().
image_plot(train_images)


# In[ ]:


# Visualize image height and width of the test data.
testheight = []
testwidth=[]
for img in tqdm(test_images_path):
    im = Image.open(img)
    testheight.append(im.size[0])
    testwidth.append(im.size[1])

def plot_testheight():
    plt.hist(testheight)
    plt.title("Height")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
def plot_testwidth():
    plt.hist(testwidth)
    plt.title("Width")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

def plot_testimage_size(height=True):
    if (height):
        plot_testheight()
    else:
        plot_testwidth()


# In[ ]:


plot_testimage_size(True)


# In[ ]:


plot_testimage_size(False)


# In[ ]:


# Statistical analysis of the training image sizes
print ("Training images mean height: {}".format(np.mean(testheight)))
print ("Training images mean width: {}".format(np.mean(testwidth)))
print ("Training images max height is {} and min height is {}.".format(max(testheight),min(testheight)))
print ("Training images max width is {} and min width is {}.".format(max(testwidth),min(testwidth)))


# In[ ]:


# Visualize image height and width of the traning data.
trainheight = []
trainwidth=[]
for img in tqdm(train_images_path):
    im = Image.open(img)
    trainheight.append(im.size[0])
    trainwidth.append(im.size[1])

def plot_trainheight():
    plt.hist(trainheight)
    plt.title("Height")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
def plot_trainwidth():
    plt.hist(trainwidth)
    plt.title("Width")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

def plot_trainimage_size(height=True):
    if (height):
        plot_trainheight()
    else:
        plot_trainwidth()


# In[ ]:


plot_trainimage_size(True)


# In[ ]:


plot_trainimage_size(False)


# In[ ]:


# Statistical analysis of the training image sizes
print ("Training images mean height: {}".format(np.mean(height)))
print ("Training images mean width: {}".format(np.mean(width)))
print ("Training images max height is {} and min height is {}.".format(max(height),min(height)))
print ("Training images max width is {} and min width is {}.".format(max(width),min(width)))


# In[ ]:


def create_image_label(img):
    image = img.split('.')[-3]
    if image == 'cat': return 'cat'
    elif image == 'dog': return 'dog'
# Function to process training image data into numpy array.
def process_data(image_files,image_folder,Train=True):
    image_data = []
    for image in tqdm(image_files):
        path = os.path.join(image_folder,image)
        if(Train):
            label =create_image_label(image)
        else:
            label = img.split('.')[0]
        image_data.append([path,label])
    shuffle(image_data)
    return image_data


# In[ ]:


train=process_data(train_images,train_data_path)


# In[ ]:


test=process_data(test_images,test_data_path,False)


# In[ ]:


# Function to to display 25 sample images from train and test data
def display_images(input_data,Test=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i,data in enumerate(input_data[:25]):
        img_label = data[1]
        img_path = data[0]
        original = load_img('{}'.format(img_path))
        #label = np.argmax(img_num)
        if img_label  =='dog': 
            str_label='Dog'
        elif img_label == 'cat': 
            str_label='Cat'
        if(Test):
            str_label="None"
        ax[i//5, i%5].imshow(original)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()


# In[ ]:


display_images(train)


# In[ ]:


display_images(test,True)


# In[ ]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_size, img_size)
else:
    input_shape = (img_size, img_size, 3)


# In[ ]:


#Process training data to make it ready for fitting.
train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_size, img_size),
                                                    batch_size=batch_size,class_mode='categorical',shuffle=True)
validation_generator = val_datagen.flow_from_directory(validation_data_dir,target_size=(img_size, img_size),
                                                        batch_size=batch_size,class_mode='categorical',shuffle=True) 
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(img_size, img_size),batch_size=batch_size,
                                                  class_mode='categorical',shuffle=False)
sample_test_generator=test_datagen.flow_from_directory(sample_test_dir,
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',shuffle=False,seed=42)
filename=test_generator.filenames


# In[ ]:


print ('Creating model...')
new_model = Sequential()
new_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
new_model.add(Activation('relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Conv2D(32, (3, 3)))
new_model.add(Activation('relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Conv2D(64, (3, 3)))
new_model.add(Activation('relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Flatten())
new_model.add(Dense(64))
new_model.add(Activation('relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes))
new_model.add(Activation('softmax'))
print ('Summary of the model...')
new_model.summary()
print ('Compiling model...')
#new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#new_model.compile(optimizer=SGD(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
new_model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
print ('Model is ready to be fit with training data.')


# In[ ]:


# Create logs, filepath and checkpoints for the model.
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
history = LossHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=20,verbose=0, mode='auto')
checkpoint = ModelCheckpoint('new_model.h5',monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
callbacks_list = [checkpoint,history,early_stopping]


# In[ ]:


# Fit the model on batches of 20000 samples of training  data and validate on 5000 samples.
fitted_new_model=new_model.fit_generator(train_generator,
    steps_per_epoch=math.ceil(num_t_samples/batch_size),
    epochs=epochs,validation_data=validation_generator,
    validation_steps=math.ceil(num_v_samples/batch_size),callbacks=callbacks_list,verbose=1)


# In[ ]:


# Plot Val_loss,train_loss and val_acc and train_acc.
acc = fitted_new_model.history['acc']
val_acc = fitted_new_model.history['val_acc']
loss = fitted_new_model.history['loss']
val_loss =fitted_new_model.history['val_loss']
epochs = range(len(acc)) 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


ls '../input/dogs-and-cats-final-kernel-explore-cnn-scratch/new_model.h5'


# In[ ]:


new_model=load_model('../input/dogs-and-cats-final-kernel-explore-cnn-scratch/new_model.h5')


# In[ ]:


#Get the sample test data filenames from the generator
fnames = sample_test_generator.filenames
# Get the ground truth from generator
ground_truth = sample_test_generator.classes
# Get the label to class mapping from the generator
label2index = sample_test_generator.class_indices
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
# Get the predictions from the model using the generator
predictions = new_model.predict_generator(sample_test_generator,
steps=math.ceil(sample_test_generator.samples/sample_test_generator.batch_size),verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),sample_test_generator.samples))
# Display missclassified images
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(fnames[errors[i]].split('/')[0],pred_label,
        predictions[errors[i]][pred_class])
    original = load_img('{}/{}'.format(sample_test_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()


# In[ ]:


#get the predictions for the test data
#predicted_classes = model.predict_classes(Xt)
#get the indices to be plotted
#y_true = np.argmax(yt,axis=1)
correct = np.nonzero(predicted_classes==ground_truth)[0]
incorrect = np.nonzero(predicted_classes!=ground_truth)[0]
target_names = ["Class {}:".format(i) for i in range(num_classes)]
print(classification_report(ground_truth, predicted_classes, target_names=target_names))


# In[ ]:


# Proces test data to generate predictions on provided test data.
test_generator.reset()
pred=new_model.predict_generator(test_generator,steps=math.ceil(test_generator.samples/test_generator.batch_size),verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
new_preds=[]
for i in range(len(predictions)):
    if predictions[i]=='dogs':
        new_preds.append('dog')
    else:
        new_preds.append('cat')


# In[ ]:


# Display predictions with 25 pictures with their labels.
def display_testdata(testdata,filenames):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    i=0
    for a,b in zip(testdata,filenames):
        pred_label=a
        fname=b
        title = 'Prediction :{}'.format(pred_label)   
        original = load_img('{}/{}'.format(test_dir,fname))
        ax[i//5,i%5].axis('off')
        ax[i//5,i%5].set_title(title)
        ax[i//5,i%5].imshow(original)
        i=i+1
    plt.show()


# In[ ]:


display_testdata(new_preds[7700:7725],filename[7700:7725])

