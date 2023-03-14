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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df=pd.read_csv("../input/train.csv")


# In[3]:


train_df.shape


# In[4]:


train_df.head()


# In[5]:


import seaborn as sns 
import matplotlib.pyplot as plt


# In[6]:


sns.countplot(train_df["has_cactus"])


# In[7]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam


# In[8]:


train_dir = "../input/train/train/"
test_dir = "../input/test/test/"


# In[9]:


X_tr = []
Y_tr = []
imges = train_df['id'].values
for img_id in tqdm_notebook(imges):
    X_tr.append(cv2.imread(train_dir + img_id,0))    
    Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  
X_tr = np.asarray(X_tr)
X_tr = X_tr.astype('float32')
X_tr /= 255
Y_tr = np.asarray(Y_tr)


# In[10]:


X_tr=X_tr.reshape(-1,32,32,1)


# In[11]:


Y_tr.shape


# In[12]:


X_tr.shape


# In[13]:


target=train_df["has_cactus"]


# In[14]:


train_df=train_df.drop("has_cactus",axis=1)


# In[15]:


import os,array
import pandas as pd
import time
import dask as dd

from PIL import Image
def pixelconv(file_list,img_height,img_width,pixels):  
    columnNames = list()

    for i in range(pixels):
        pixel = 'pixel'
        pixel += str(i)
        columnNames.append(pixel)


    train_data = pd.DataFrame(columns = columnNames)
    start_time = time.time()
    for i in tqdm_notebook(file_list):
        t = i
        img_name = t
        img = Image.open('../input/train/train/'+img_name)
        rawData = img.load()
        #print rawData
        data = []
        for y in range(img_height):
            for x in range(img_width):
                data.append(rawData[x,y][0])
        print (i)
        k = 0
        #print data
        train_data.loc[i] = [data[k] for k in range(pixels)]
    #print train_data.loc[0]

    print ("Done pixel values conversion")
    print  (time.time()-start_time)
    print (train_data)
    train_data.to_csv("train_converted_new.csv",index = False)
    print ("Done data frame conversion")
    print  (time.time()-start_time)
pixelconv(train_df.id,32,32,1024) # pass pandas dataframe in which path of images only as column
                                    # in return csv file will save in working directory 


# In[16]:


new_data=pd.read_csv("../working/train_converted_new.csv")


# In[17]:


new_data.head()


# In[18]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:



pca = PCA(n_components=500)
pca.fit(new_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[20]:


NCOMPONENTS = 625

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(new_data)
pca_std = np.std(X_pca_train)
print(X_pca_train.shape)


# In[21]:


inv_pca = pca.inverse_transform(X_pca_train)
#inv_sc = scaler.inverse_transform(inv_pca)


# In[22]:


X_pca_train_new=X_pca_train.reshape(X_pca_train.shape[0],25,25,1)


# In[23]:


X_pca_train_new.shape


# In[24]:


X_pca_train.shape ### this shape will be used in MLP


# In[25]:


import keras
model = Sequential()
layers = 1
units = 128

model.add(Dense(units, input_dim=NCOMPONENTS, activation='relu'))
model.add(GaussianNoise(pca_std))
model.add(Dense(units, activation='relu'))
model.add(GaussianNoise(pca_std))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-5), metrics=['acc'])
history = model.fit(X_pca_train,target,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_split=0.15)

#model.fit(X_pca_train, Y_train, epochs=100, batch_size=256, validation_split=0.15, verbose=2)


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[27]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 1
epochs = 200

#input image dimensions
img_rows, img_cols = 25, 25

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(25,25,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])


# In[28]:


history1 = model.fit(X_pca_train_new,target,
          batch_size=batch_size,
          epochs=200,
          verbose=1,
          validation_split=0.15)


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history1.history['acc']
val_accuracy = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('CNN result Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('cnn Training and validation loss')
plt.legend()
plt.show()


# In[30]:


get_ipython().run_cell_magic('time', '', "X_tst = []\nTest_imgs = []\nfor img_id in tqdm_notebook(os.listdir(test_dir)):\n    X_tst.append(cv2.imread(test_dir + img_id,0))     \n    Test_imgs.append(img_id)\nX_tst = np.asarray(X_tst)\nX_tst = X_tst.astype('float32')\nX_tst /= 255")


# In[31]:


X_tst.shape


# In[32]:


X_tst=X_tst.reshape(-1,32,32,1)


# In[33]:


test_path=[]
for i in os.listdir(test_dir):
    test_path.append(i)


# In[34]:


test_dataframe=pd.DataFrame(data=test_path,columns=["id"])


# In[35]:


test_dataframe.head()


# In[36]:


import os,array
import pandas as pd
import time
import dask as dd

from PIL import Image
def pixelconv(file_list,img_height,img_width,pixels):  
    columnNames = list()

    for i in range(pixels):
        pixel = 'pixel'
        pixel += str(i)
        columnNames.append(pixel)


    train_data = pd.DataFrame(columns = columnNames)
    start_time = time.time()
    for i in file_list:
        t = i
        img_name = t
        img = Image.open('../input/test/test/'+img_name)
        rawData = img.load()
        #print rawData
        data = []
        for y in range(img_height):
            for x in range(img_width):
                data.append(rawData[x,y][0])
        print (i)
        k = 0
        #print data
        train_data.loc[i] = [data[k] for k in range(pixels)]
    #print train_data.loc[0]

    print ("Done pixel values conversion")
    print  (time.time()-start_time)
    print (train_data)
    train_data.to_csv("test_converted_new.csv",index = False)
    print ("Done data frame conversion")
    print  (time.time()-start_time)
pixelconv(test_dataframe.id,32,32,1024) # pass pandas dataframe in which path of images only as column
                                    # in return csv file will save in working directory 


# In[37]:


new_test=pd.read_csv("../working/test_converted_new.csv")


# In[38]:


X_tst=pca.transform(new_test)


# In[39]:


X_tst.shape


# In[40]:


X_tst=X_tst.reshape(-1,25,25,1)


# In[41]:


X_tst.shape


# In[42]:


train_df["image_location"]=train_dir+train_df["id"]


# In[43]:


import tensorflow as tf
from keras import backend as K


# In[44]:


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


# In[45]:


train_df.head()


# In[46]:


#from console_progressbar import ProgressBar
import shutil
import tqdm
import os
filenames=list(train_df["image_location"].values)
labels=list(train_df["has_cactus"].values)
folders_to_be_created = np.unique(list(train_df['has_cactus'].values))
files=[]
path="../working/trainset/"
for i in folders_to_be_created:
    if not os.path.exists(path+str(i)):
        os.makedirs(path+str(i)) 
#pb = ProgressBar(total=100, prefix='Save valid data', suffix='', decimals=3, length=50, fill='=')
for f in tqdm_notebook(range(len(filenames))):
    
    current_image=filenames[f]
    current_label=labels[f]
    src_path=current_image
   
    dst_path =path+str(current_label) 
    
    try :
        shutil.copy(src_path, dst_path)
        #pb.print_progress_bar((f + 1) * 100 / 4000)
    except Exception as e :
        files.append(src_path)


# In[47]:


import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
train_datagen = ImageDataGenerator(
    rescale = 1./255,
#     preprocessing_function= preprocess_input,
    #shear_range=0.2,
    zoom_range=0.2,
    fill_mode = 'reflect',
    #cval = 1,
    rotation_range = 30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,validation_split=.20)

valid_datagen = ImageDataGenerator(rescale=1./255)#,preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory='../working/trainset/',
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary',subset="training")

validation_generator = train_datagen.flow_from_directory(
    directory='../working/trainset/',
    target_size=(32,32),
    batch_size=32,
    class_mode='binary',subset="validation")


# In[48]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam


# In[49]:


vgg16_net = VGG16(weights='imagenet', 
                  include_top=False, 
                  input_shape=(32, 32, 3))


# In[50]:


vgg16_net.trainable = False
vgg16_net.summary()


# In[51]:


model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[52]:


model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=Adam(lr=1e-5))


# In[53]:


history=model.fit_generator(train_generator,
                    steps_per_epoch = 14001//32,
                    epochs=200,
                    validation_data = validation_generator,validation_steps=3499//32)


# In[54]:


import matplotlib.pyplot as plt
import numpy as np
#plot of model epochs what has been happened within 100 epochs baseline with vgg16 training 135 million params
# freezing first layer
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['acc'],'blue')
plt.plot(history.history['val_acc'],'orange')
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0,1,.1))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()
 
plt.figure(1)
plt.plot(history.history['loss'],'blue')
plt.plot(history.history['val_loss'],'orange')
plt.xticks(np.arange(0, 100, 10))
plt.rcParams['figure.figsize'] = (10, 10)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()


# In[55]:


get_ipython().run_cell_magic('time', '', "X_tst = []\nTest_imgs = []\nfor img_id in tqdm_notebook(os.listdir(test_dir)):\n    X_tst.append(cv2.imread(test_dir + img_id))     \n    Test_imgs.append(img_id)\nX_tst = np.asarray(X_tst)\nX_tst = X_tst.astype('float32')\nX_tst /= 255")


# In[56]:


# Prediction
test_predictions = model.predict(X_tst)


# In[57]:


sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])
sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.5 else 0)


# In[58]:


sub_df['id'] = ''
cols = sub_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub_df=sub_df[cols]


# In[59]:


for i, img in enumerate(Test_imgs):
    sub_df.set_value(i,'id',img)


# In[60]:


sub_df.head()


# In[61]:


sub_df.to_csv('submission.csv',index=False)


# In[62]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


# In[63]:


model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.layers[0].trainable = False


# In[64]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[65]:


history1=model.fit_generator(train_generator,
                    steps_per_epoch = 14001//32,
                    epochs=200,
                    validation_data = validation_generator,validation_steps=3499//32)

