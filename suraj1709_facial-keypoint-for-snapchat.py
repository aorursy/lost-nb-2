#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from zipfile import ZipFile
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import zipfile

Dataset = "training"
Dataset1="test"
# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/facial-keypoints-detection/"+Dataset+".zip","r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/facial-keypoints-detection/"+Dataset1+".zip","r") as z:
    z.extractall(".")


# In[3]:


test='../working/test.csv'
training='../working/training.csv'
lookid_dir = '../input/facial-keypoints-detection/IdLookupTable.csv'


# In[4]:



train_data = pd.read_csv(training)  
test_data = pd.read_csv(test)
lookid_data = pd.read_csv(lookid_dir)
os.listdir('../input')


# In[5]:


train_data.head().T

train_data.isnull().any().value_counts()
# In[6]:


train_data = train_data.dropna()
#train_data.fillna(method="ffill",inplace=True)


# In[7]:


train_data.isnull().any().value_counts()


# In[8]:


train_data.shape, type(train_data)
train_data['Image'] = train_data['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))


# In[9]:


def get_image_and_dots(df, index):
    image = plt.imshow(df['Image'][index],cmap='gray')
    l = []
    for i in range(1,31,2):
        l.append(plt.plot(df.loc[index][i-1], df.loc[index][i], 'ro'))
        
    return image, l


# In[10]:


fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    get_image_and_dots(train_data, i)

plt.show()


# In[11]:


X = np.asarray([train_data['Image']], dtype=np.uint8).reshape(train_data.shape[0],96,96,1)
y = train_data.drop(['Image'], axis=1)


# In[12]:


X.shape


# In[13]:


y.shape


# In[14]:


type(X), type(y)


# In[15]:


y2 = y.to_numpy()


# In[16]:


type(y2), y2.shape


# In[17]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3, random_state=42)


# In[18]:


train_data.count()


# In[19]:


train_data['Image']


# In[20]:


plt.imshow(X[2000].reshape(96,96),cmap='gray')
plt.show()


# In[21]:


Train_image_label=train_data.drop('Image',axis=1)


# In[22]:


Train_image_label


# In[23]:


y=Train_image_label.iloc[1,:]
y.count()


# In[24]:


y_train=[]
for i in range(0,1498):
    y=Train_image_label.iloc[i,:]
    y_train.append(y)
y_train=np.array(y_train)


# In[25]:


from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential
model = Sequential()


# In[26]:


from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


# In[27]:


model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()


# In[28]:


model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])


# In[29]:


model.fit(X_train,y_train,epochs = 500,batch_size = 256,validation_split = 0.2)
#model.fit(X_train, y_train, epochs=500)


# In[30]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[31]:


plt.imshow(X_train[2].reshape(96,96),cmap='gray')
plt.show()


# In[32]:


model.save('keypoint_model22.hdf5')


# In[33]:


plt.imshow(train_data['Image'][1].reshape(96,96),cmap='gray')
plt.show()


# In[34]:


img_model = np.reshape(train_data['Image'][1], (1,96,96,1))


# In[35]:


test_res = model.predict(img_model)


# In[36]:


test_res


# In[37]:


def show(train_data,j):
    img_model = np.reshape(train_data['Image'][j],(1,96,96,1))
    test_res = model.predict(img_model)
    xv = train_data['Image'][j].reshape((96,96))
    plt.imshow(xv,cmap='gray')

    for i in range(1,31,2):
        plt.plot(test_res[0][i-1], test_res[0][i], 'ro')

    plt.show()


# In[38]:


def showfig():
    for i in range(8):
        show(train_data,i)


# In[39]:


showfig()


# In[40]:


def show(X_test,j):
    img_model = np.reshape(train_data['Image'][j], (1,96,96,1))
    test_res = model.predict(img_model)
    xv = train_data['Image'][j].reshape((96,96))
    plt.imshow(xv,cmap='gray')

    for i in range(1,31,2):
        plt.plot(test_res[0][i-1], test_res[0][i], 'ro')

    plt.show()


# In[41]:


plt.imshow(X_train[2].reshape(96,96),cmap='gray')
plt.show()


# In[42]:


plt.imshow(train_data['Image'][15].reshape(96,96),cmap='gray')
plt.show()


# In[43]:


from skimage.transform import resize
test_image=X_test[2]
plt.imshow(test_image.reshape(96,96),cmap='gray')

test_img = resize(test_image, (96,96)) 
test_img = np.array(test_img)
test_img_input = np.reshape(test_img, (1,96,96,1)) 
prediction = model.predict(test_img_input)
for i in range(1,31,2):
        plt.plot(prediction[0][i-1], prediction[0][i], 'ro')
point=prediction[0]
plt.show()


# In[44]:


from skimage.transform import resize
def showalltrainimages(j):
    test_image=X_train[j]
    plt.imshow(test_image.reshape(96,96),cmap='gray')

    test_img = resize(test_image, (96,96)) 
    test_img = np.array(test_img)
    test_img_input = np.reshape(test_img, (1,96,96,1)) 
    prediction = model.predict(test_img_input)
    for i in range(1,31,2):
        plt.plot(prediction[0][i-1], prediction[0][i], 'ro')
    point=prediction[0]
    plt.show()


# In[45]:


from skimage.transform import resize
def showalltestimage(j):
    test_image=X_test[j]
    plt.imshow(test_image.reshape(96,96),cmap='gray')

    test_img = resize(test_image, (96,96)) 
    test_img = np.array(test_img)
    test_img_input = np.reshape(test_img, (1,96,96,1)) 
    prediction = model.predict(test_img_input)
    for i in range(1,31,2):
        plt.plot(prediction[0][i-1], prediction[0][i], 'ro')
    point=prediction[0]
    plt.show()


# In[46]:


def showim():
    for i in range(10):
        showalltrainimages(i)
        showalltestimage(i)


# In[47]:


showim()


# In[48]:


model.save("model.hdf5")

