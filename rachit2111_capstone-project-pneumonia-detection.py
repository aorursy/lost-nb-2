#!/usr/bin/env python
# coding: utf-8

# In[154]:


import numpy as np 
import pandas as pd
import os        


# In[155]:


# Load detail dataset
detail_class_info_df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

# Load train dataset
train_labels_df = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')


# In[156]:


# Check no. of rows and columns of each dataset
print('Shape of Detail class dataset: ', detail_class_info_df.shape)
print('Shaoe of Train dataset:', train_labels_df.shape)


# In[157]:


# Read 7 data from datasets
detail_class_info_df.head(7)


# In[158]:


# Find number of occurences of different types of classes for patients
detail_class_info_df['class'].value_counts()


# In[159]:


import seaborn as sns
sns.countplot(x="class", hue="class", data=detail_class_info_df)


# In[160]:


train_labels_df.head(7)


# In[161]:


# Find number of occurences of different targets for patients
train_labels_df['Target'].value_counts()


# In[162]:


sns.countplot(x="Target", hue="Target", data=train_labels_df)


# In[163]:


# Check in each column how many null values are there in train label dataset
train_labels_df.isnull().sum()


# In[164]:


# Check how many patients has bounding box co-ordinates
train_labels_df.groupby(['Target']).count()


# In[165]:


import pydicom as dcm
from pydicom import dcmread


# In[166]:


# Get list of all dcm images
import glob 
train_image_list = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/*.dcm')
test_image_list = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_2_test_images/*.dcm')
print('Number of images in train image list: ', len(train_image_list))
print('Number of images in test image list: ', len(test_image_list))


# In[167]:


# Check unique patients in the dataset
print('Number of unique patients are: ', train_labels_df['patientId'].nunique())


# In[168]:


# Read one sample image information
sample_image_index = 4
sample_image_path = train_image_list[sample_image_index]
print('Sample image path is: ',sample_image_path)


# In[169]:


sample_image_dataset = dcm.read_file(sample_image_path)
sample_image_dataset


# In[ ]:


# Load sample image
import matplotlib.pyplot as plt
plt.imshow(sample_image_dataset.pixel_array, cmap=plt.cm.bone)
sample_patient = train_labels_df[train_labels_df['patientId'] == sample_image_dataset.PatientID]
sample_patient_data = list(sample_patient.T.to_dict().values())
print("Shape of the image: ", sample_image_dataset.pixel_array.shape)
print("Patient's Sex: ", sample_image_dataset.PatientSex)
print("Modality: ", sample_image_dataset.Modality)
print("Patient's Age: ", sample_image_dataset.PatientAge)
print("Body part examined: ", sample_image_dataset.BodyPartExamined)
print("Target: ", sample_patient_data[0]['Target'])


# In[170]:


# Draw a bounding box on the image

import matplotlib.patches as patches
imageArea, axes = plt.subplots(1)
x, y, width, height  = sample_patient_data[0]['x'], sample_patient_data[0]['y'], sample_patient_data[0]['width'], sample_patient_data[0]['height']

# Create a Rectangle patch
rect = patches.Rectangle((x, y), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
axes.imshow(sample_image_dataset.pixel_array, cmap=plt.cm.bone)

# Add the patch to the Axes
axes.add_patch(rect)

plt.show()


# In[171]:


# Merge two datasets

# train_class_df = pd.merge(detail_class_info_df, train_labels_df, on='patientId')
train_class_df = pd.concat([train_labels_df, detail_class_info_df["class"]], axis=1, sort=False)
train_class_df.head(7)
train_class_df.shape


# In[172]:


train_class_df.isna().apply(pd.value_counts)


# In[173]:


sns.countplot(x='Target', hue='class', data=train_class_df)


# In[174]:


index = 3        # This patient has class 'No Lung Opacity / Not Normal'
image_path_1 = train_image_list[index]
sample_image_data1 = dcm.read_file(image_path_1)
patient1 = train_class_df[train_class_df['patientId'] == sample_image_data1.PatientID]
sample_patient_data1 = list(patient1.T.to_dict().values())
print(sample_patient_data1[0]['class'])
imageArea, axes = plt.subplots(1)
x, y, width, height  = sample_patient_data1[0]['x'], sample_patient_data1[0]['y'], sample_patient_data1[0]['width'], sample_patient_data1[0]['height']

# Create a Rectangle patch
rect1 = patches.Rectangle((x, y), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
axes.imshow(sample_image_data1.pixel_array, cmap=plt.cm.bone)

# Add the patch to the Axes
axes.add_patch(rect1)

plt.show()


# In[175]:


index1 = 5        # This patient has class 'Normal'
image_path_2 = train_image_list[index1]
sample_image_data2 = dcm.read_file(image_path_2)
patient2 = train_class_df[train_class_df['patientId'] == sample_image_data2.PatientID]
sample_patient_data2 = list(patient2.T.to_dict().values())
print(sample_patient_data2[0]['class'])
imageArea, axes = plt.subplots(1)
x, y, width, height  = sample_patient_data2[0]['x'], sample_patient_data2[0]['y'], sample_patient_data2[0]['width'], sample_patient_data2[0]['height']

# Create a Rectangle patch
rect2 = patches.Rectangle((x, y), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
axes.imshow(sample_image_data2.pixel_array, cmap=plt.cm.bone)

# Add the patch to the Axes
axes.add_patch(rect2)

plt.show()


# In[176]:


print(len(train_class_df[(train_class_df['Target'] == 1) & (train_class_df['class'] == 'Lung Opacity')]))


# In[177]:


import tensorflow as tf
from keras import layers
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Add, Activation, Input, Flatten, Dense, AveragePooling2D, ZeroPadding2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
import sklearn
from sklearn.preprocessing import LabelEncoder

# Define Convolution block
def residual_convolutionBlock(layers, f, filters):
    f1, f2, f3 = filters
    x_shortcut = layers
    
    # First convolutional block
    x = Conv2D(filters = f1, kernel_size = (1, 1), padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional block
    x = Conv2D(filters = f2, kernel_size = (f, f), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third convolutional block
    x = Conv2D(filters = f3, kernel_size = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    x_shortcut = Conv2D(filters = f3, kernel_size = (1, 1), activation = 'relu', padding = 'same')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    # Add shortcut value to main path
    x = Add()([x, x_shortcut])

    # Pass it through RELU activation
    x = Activation('relu')(x)

    return x


# In[178]:


# Define Identity block
def residual_identityBlock(layers, f, filters):
    f1, f2, f3 = filters
    x_shortcut = layers
    
    # First convolutional block
    x = Conv2D(filters = f1, kernel_size = (1, 1), padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional block
    x = Conv2D(filters = f2, kernel_size = (f, f), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third convolutional block
    x = Conv2D(filters = f3, kernel_size = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut value to main path
    x = Add()([x, x_shortcut])

    # Pass it through RELU activation
    x = Activation('relu')(x)

    return x


# In[179]:


# Define ResNet Model (50-layer) 

inputs = Input(shape=(64, 64, 1)) # Size of the image

# Zero-Padding
x = ZeroPadding2D((3, 3))(inputs)

# Stage 1 (Initial convolution and max pooling)
x = Conv2D(64,(7, 7), strides = (2, 2))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Stage 2    
x = residual_convolutionBlock(inputs, f = 3, filters = [64, 64, 256])
x = residual_identityBlock(inputs, f = 3, filters = [64, 64, 256]) 
x = residual_identityBlock(inputs, f = 3, filters = [64, 64, 256]) 

# Stage 3    
x = residual_convolutionBlock(inputs, f = 3, filters = [128, 128, 512])
x = residual_identityBlock(inputs, f = 3, filters = [128, 128, 512]) 
x = residual_identityBlock(inputs, f = 3, filters = [128, 128, 512])
x = residual_identityBlock(inputs, f = 3, filters = [128, 128, 512]) 

# Stage 4    
x = residual_convolutionBlock(inputs, f = 3, filters = [256, 256, 1024])
x = residual_identityBlock(inputs, f = 3, filters = [256, 256, 1024]) 
x = residual_identityBlock(inputs, f = 3, filters = [256, 256, 1024])
x = residual_identityBlock(inputs, f = 3, filters = [256, 256, 1024]) 
x = residual_identityBlock(inputs, f = 3, filters = [256, 256, 1024]) 
x = residual_identityBlock(inputs, f = 3, filters = [256, 256, 1024]) 

# Stage 5
x = residual_convolutionBlock(inputs, f = 3, filters = [512, 512, 2048])
x = residual_identityBlock(inputs, f = 3, filters = [512, 512, 2048]) 
x = residual_identityBlock(inputs, f = 3, filters = [512, 512, 2048])

x = AveragePooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(units=1, activation='softmax')(x)

# create model
model = Model(inputs = inputs, outputs = x)
    
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize model
model.summary()


# In[180]:


trainSetImageMetadata_df = pd.DataFrame(train_image_list, columns=["Path"])
trainSetImageMetadata_df.head(2)

def getImgId(_imgData) :
    return str(_imgData).split(".dcm")[0].split("/")[4]

imageIdPaths = pd.DataFrame(columns=["patientId", "imgPath"])
imageIdPaths["patientId"] = trainSetImageMetadata_df["Path"].apply(getImgId)
imageIdPaths["imgPath"] = trainSetImageMetadata_df["Path"]

print("imageIdPaths", imageIdPaths.shape)
imageIdPaths.head(2)


# In[181]:


train_CombinedData = train_class_df[0:15000]
validate_CombinedData = train_class_df[15000:25000]
test_CombinedData = train_class_df[25000:30227]

train_imageIdPaths = imageIdPaths[0:13163]
validate_imageIdPaths = imageIdPaths[13163:21764]
test_imageIdPaths = imageIdPaths[21764:26684]

print("Train image path shape: ",train_imageIdPaths.shape)
print("Train Data shape: ",train_CombinedData.shape)


# In[182]:


import cv2
import math
from keras.utils import Sequence


CLSI_IMAGE_SIZE = 224    
CLSI_BATCH_SIZE = 64 
CLSI_IMG_PX_SIZE = 224
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

class ClassifierSequenceGenerator(Sequence):
    
    def __init__(self, _imageIdPaths, _CombinedData):
        self.pids = _CombinedData["patientId"].to_numpy()
        encoder = LabelEncoder()
        self.classes = encoder.fit_transform(_CombinedData["class"].to_numpy())
        self.samples = len(_CombinedData)
        self.imgIdPaths = _imageIdPaths
                
    def __len__(self):
        return math.ceil(len(self.classes) / CLSI_BATCH_SIZE)

    
    def __getitem__(self, idx): # Get a batch
        batch_pids = self.pids[idx * CLSI_BATCH_SIZE:(idx + 1) * CLSI_BATCH_SIZE] # Image pids
        batch_classes = self.classes[idx * CLSI_BATCH_SIZE:(idx + 1) * CLSI_BATCH_SIZE] # Image coords      
        
        batch_images = np.zeros((len(batch_pids), CLSI_IMAGE_SIZE, CLSI_IMAGE_SIZE, 3), dtype=np.float32)
        for _indx, _path in enumerate(batch_pids):
            _imgData = dcm.read_file(str(_path)) # Read image
            img = _imgData.pixel_array 
            
            # Resize image
            resized_img = cv2.resize(img, (CLSI_IMG_PX_SIZE, CLSI_IMG_PX_SIZE))
            
            # Preprocess image for the batch
            batch_images[_indx][:,:,0] = preprocess_input(np.array(resized_img, dtype=np.float32)) # Convert to float32 array
            batch_images[_indx][:,:,1] = preprocess_input(np.array(resized_img, dtype=np.float32)) # Convert to float32 array
            batch_images[_indx][:,:,2] = preprocess_input(np.array(resized_img, dtype=np.float32)) # Convert to float32 array

        return batch_images, batch_classes


# In[183]:


trainDataGen = ClassifierSequenceGenerator(train_imageIdPaths, train_CombinedData)
testDataGen = ClassifierSequenceGenerator(test_imageIdPaths, test_CombinedData)

print(len(trainDataGen))
print(len(testDataGen))


# In[ ]:


# model.fit_generator(
#           trainDataGen, steps_per_epoch = CLSI_BATCH_SIZE, 
#           epochs = 16, validation_data=testDataGen, 
#           validation_steps = CLSI_BATCH_SIZE)


# In[184]:


from keras.optimizers import Adam

# Use pretrained ResNet50 Model
base_model = ResNet50(weights= None, include_top=False, input_shape= (CLSI_IMAGE_SIZE,CLSI_IMAGE_SIZE,3))

X = base_model.output
X = GlobalAveragePooling2D()(X)
X = Dropout(0.7)(X)
predictions = Dense(1, activation= 'softmax')(X)
model1 = Model(inputs = base_model.input, outputs = predictions)

# compile model
adam = Adam(lr=0.0001)
model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# summarize model
model1.summary()


# In[ ]:


model.fit_generator(
          trainDataGen, steps_per_epoch = CLSI_BATCH_SIZE, 
          epochs = 16, validation_data=testDataGen, 
          validation_steps = CLSI_BATCH_SIZE)

