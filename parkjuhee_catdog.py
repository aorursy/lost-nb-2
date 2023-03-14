#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os,shutil

in_path="../input"

print(os.listdir(in_path))


# In[2]:



IMAGE_SIZE=64
BATCH_SIZE=128


label=[]
data=[]
counter=0
path="../input/train/train"
for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)
    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
         print (counter," image data retreived")

data=np.array(data)
data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)
label=np.array(label)
print (data.shape)
print (label.shape)



# In[3]:


sns.countplot(label)
# 1이 dog 0이 cat


# In[4]:


from sklearn.model_selection import train_test_split
train_data, valid_data, train_label, valid_label = train_test_split(
    data, label, test_size=0.2, random_state=42)
print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)


# In[5]:


from keras import Sequential
from keras.layers import *
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
import keras.backend as K


# In[6]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D




my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(1, activation='sigmoid'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False


# In[7]:


ResNet50().summary()


# In[8]:


train_datagen = ImageDataGenerator(rescale=1./255, 
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=10,
                                   zoom_range=0.1,)

val_datagen = ImageDataGenerator(rescale=1./255, 
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 rotation_range=10,
                                 zoom_range=0.1,)



train_generator = train_datagen.flow(np.array(train_data), np.array(train_label), batch_size = 128)
validation_generator = val_datagen.flow(np.array(valid_data), np.array(valid_label), batch_size = 128)


# In[9]:


my_new_model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

callack_saver = ModelCheckpoint(
            "my_new_model.h5"
            , monitor='val_loss'
            , verbose=0
            , save_weights_only=True
            , mode='auto'
            , save_best_only=True
        )

train_history=my_new_model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=40,batch_size=128, callbacks=[callack_saver])


# In[10]:


import matplotlib.pyplot as plt
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'blue', label='Training acc')
plt.plot(epochs, val_acc, 'red', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, val_loss, 'red', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[11]:



IMAGE_SIZE=64
BATCH_SIZE=128


label=[]
data=[]
counter=0
path="../input/test1/test1"
for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)
    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
         print (counter," image data retreived")

data=np.array(data)
data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)
label=np.array(label)
print (data.shape)




# In[12]:


Y_pred = my_new_model.predict(data)


# In[13]:


print(Y_pred)


# In[14]:


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = test_datagen.flow_from_directory("../input/test1",target_size=(128, 128),batch_size=32,class_mode='binary')


# In[15]:



from tensorflow.python.keras.models import Sequential
from keras.models import load_model

print("-- Evaluate --")

scores = my_new_model.evaluate_generator(
            test_generator, 
            steps = 100)

print("%s: %.2f%%" %(my_new_model.metrics_names[1], scores[1]*100))


# In[16]:


steps = 12500 / 20
import numpy as np
# 모델 예측하기
print("-- Predict --")

output = my_new_model.predict_generator(test_generator,steps)


# In[17]:


print(output)


# In[18]:


print(len(output))

pred = np.array(output) 


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mlxtend.plotting import plot_confusion_matrix

# Get the confusion matrix

CM = confusion_matrix(test_generator.classes, pred.round())
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))
plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.show()


# In[20]:


Y_pred = my_new_model.predict(valid_data)


# In[21]:


# predicted_label=np.round(Y_pred,decimals=2)
# predicted_label=[1 if value>0.5 else 0 for value in predicted_label]
# confusion_mtx = confusion_matrix(valid_label, predicted_label) 
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = range(2)) 


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mlxtend.plotting import plot_confusion_matrix

# Get the confusion matrix

CM = confusion_matrix(valid_label, Y_pred.round())
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))
plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.show()


# In[22]:


test_data=[]
id=[]
counter=0
for file in os.listdir("../input/test1/test1"):
    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_COLOR)
    try:
        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("ek gaya")
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")

test_data=np.array(test_data)
print (test_data.shape)
test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],3)
dataframe_output=pd.DataFrame({"id":id})


# In[23]:


predicted_labels=my_new_model.predict(test_data)
predicted_labels=np.round(predicted_labels,decimals=2)
labels=[1 if value>0.5 else 0 for value in predicted_labels]


# In[24]:


dataframe_output["label"]=labels
print(dataframe_output)


# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for i in range(20):
    plt.figure()
    plt.imshow(test_data[i])
#     plt.xlabel('label')
    plt.xlabel(dataframe_output["label"][i])

    plt.show()


# In[26]:


0 이 cat 1 강아지

