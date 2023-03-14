#!/usr/bin/env python
# coding: utf-8
We are given three CSV files.
1)training.csv :Its has coordinates of facial keypoints like left eye, rigth eye etc and also the image.
2) test.csv : Its has image only and we have to give coordinates of various facial keypoints
3) We use a csv file to solve the problem which is IdLookupTable.csv
# In[1]:


#Importation of packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os


# In[2]:


# Input data files are available in the "../input/" directory.
# Any results we write to the current directory are saved as output.
train_data = pd.read_csv('data/training/training.csv')
test_data = pd.read_csv('data/test/test.csv')
lookid_data = pd.read_csv('data/IdLookupTable.csv')


# In[3]:


train_data.info()


# In[4]:


#Exploration of our dataset
lookid_data.head().T
print('size of training data {}'.format(len(train_data)))


# In[5]:


train_data.head().T


# In[6]:


#Checking for missing values
train_data.isnull().any().value_counts()

There are missing values in 28 columns.
We can do two things here one remove the rows having missing values and
another is the fill missing values with something.
We used two option as removing rows will reduce our dataset.
We filled the missing values with the previous values in that row.
# In[7]:


train_data.fillna(method = 'ffill',inplace = True)


# In[8]:


#Checking for missing values one more time
train_data.isnull().any().value_counts()


# In[9]:


len(train_data)


# In[10]:


len(test_data)

As there is no missing values we can now separate the labels and features.
As image column values are in string format and there is also some
missing values so we have to split the string by space and append it and
also handling missing values
# In[11]:


# conversion of image col to int and also check NaN
imag = []
for i in range(0,len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)


# In[12]:


# reshape the face images in [96,96] and convert it into float value.
image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96)


# In[13]:


#Lets see what is the first image.
plt.imshow(X_train[0],cmap='gray')
plt.show()


# In[14]:


# separate labels
training_y = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,len(train_data)):
    y = training_y.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')

As our data is ready for training , lets define our model.
We am using keras and simple dense layers.
For loss function we are using 'mse' ( mean squared error )
as we have to predict new values.
Our result evaluted on the basics of 'mae' ( mean absolute error ) .
# In[15]:


from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure Model
model = Sequential()
model.add(Flatten(input_shape=[96,96]))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(30))

# Compile model

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

Now our model is defined and we will train it by calling fit method.
We ran it for 500 iteration keeping batch size and validtion set size as
20% ( 20% of the training data will be kept for validating the model ).
# In[16]:


#Test of different methods: EarlyStopping, ModelCheckpoint

#k = EarlyStopping(patience = 10)
#k  = ModelCheckpoint(filepath = "/home/bogkosh/IdeaProjects/Python Ylii/my.h5", save_best_only = True)
hist = model.fit(X_train,y_train,epochs = 300,batch_size = 64,validation_split = 0.2) #callbacks = [k]


# In[17]:


#model = load_model("my.h5")


# In[18]:


def plot_learning_curves(hist):
    pd.DataFrame(hist.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 100)
    plt.show()


# In[19]:


plot_learning_curves(hist)


# In[20]:


#Preparing our testing data
# convert image col to int  also check NaN
#len(timag) = 1783 test data
timag = []
for i in range(0,len(test_data)):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    
    timag.append(timg)


# In[21]:


#Reshaping and converting the images back to 96*96 pixels
timages_list = np.array(timag,dtype = 'float')
X_test = timages_list.reshape(-1,96,96)


# In[22]:


# Preview result on test data with the first image
# We can check the performance of the model on the image dataset
plt.imshow(X_test[0])
plt.show()


# In[23]:


#predict our results
pred = model.predict(X_test)

Now the last step is the create our submission file keeping
in the mind required format. There should be two columns :
   - RowId and Location Location column values should be
  filled according the lookup table provided ( IdLookupTable.csv)
# In[24]:


lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)


# In[25]:


rowid = lookid_data['RowId']
rowid=list(rowid)


# In[26]:


feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))


# In[27]:


preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])


# In[28]:


rowid = pd.Series(rowid,name = 'RowId')


# In[29]:


loc = pd.Series(preded,name = 'Location')


# In[30]:


submission = pd.concat([rowid,loc],axis = 1)


# In[31]:


submission.Location=submission.Location.map(lambda x:0 if x<0 else x)
submission.Location=submission.Location.map(lambda x:96 if x>96 else x)


# In[32]:


submission.to_csv('face_key_detection_submission.csv',index = False)

