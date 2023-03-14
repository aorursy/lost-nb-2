#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

from PIL import Image

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# In[ ]:


get_ipython().system('pip install -U -q kaggle --force')


# In[ ]:


from google.colab import files
f=files.upload()


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')


# In[ ]:


get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[ ]:


get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle competitions download -c nnfl-cnn-lab2')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd /content\nunzip nnfl-cnn-lab2.zip')


# In[ ]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../content"))


# In[ ]:


FAST_RUN = False
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[ ]:


cd upload


# In[ ]:


ls


# In[ ]:


df = pd.read_csv("train_set.csv") 


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df['label'].value_counts().plot.bar()


# In[ ]:


df['label']=df['label'].astype(str)


# In[ ]:


transform = transforms.Compose([transforms.Resize(150),
                                transforms.ToTensor()                               
                                ])

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=transform):
        """
        Args:
            csv_file(string): path to csv file
            root_dir(string): directory with all train images
        """
        self.name_frame = pd.read_csv(csv_file, usecols=range(1))
        self.label_frame = pd.read_csv(csv_file, usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #sample = {'image': image, 'labels': labels}

        return image, labels

TrainSet = TrainingDataset(csv_file = './train_set.csv', root_dir = './train_images/train_images')

TrainLoader = torch.utils.data.DataLoader(TrainSet,batch_size=1, shuffle=True, num_workers=2)


# In[ ]:


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # transform to size x size x #channels
    plt.show()

# get some random training images
dataiter = iter(TrainLoader)
image, label = dataiter.next()

print(image.shape)

# show images
imshow(torchvision.utils.make_grid(image))
# print labels
print(label.item())


# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import cv2


from tensorflow.keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalMaxPooling2D
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

img_height,img_width = 150,150
num_classes = 6
base_model = applications.resnet_v2.ResNet50V2(weights= None, include_top=False, input_shape= (img_height,img_width,3))

x = base_model.output
x = GlobalMaxPooling2D()(x)

#x = Dense(32, activation='relu')(x)
#x = Dropout(0.7)(x)

predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


earlystop = EarlyStopping(patience=10)


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

callbacks = [earlystop, learning_rate_reduction,ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# In[ ]:


df['label'].head()


# In[ ]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[ ]:


train_df['label'].value_counts().plot.bar()


# In[ ]:


validate_df['label'].value_counts().plot.bar()


# In[ ]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# In[ ]:


print(total_train)
print(total_validate)


# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=-10,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
)


# In[ ]:


train_df['label']=train_df['label'].astype(str)


# In[ ]:


validate_df['label']=validate_df['label'].astype(str)


# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[ ]:


train_df['label']=train_df['label'].astype(str)


# In[ ]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "./train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[ ]:


epochs=3 if FAST_RUN else 40
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[ ]:


model.save_weights("model.h5")


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[ ]:


test_filenames = os.listdir("./test_images/test_images")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "./test_images/test_images", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[ ]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[ ]:


test_df['category'] = np.argmax(predict, axis=-1)


# In[ ]:


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)


# In[ ]:


test_df['category'].value_counts().plot.bar()


# In[ ]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./test_images/test_images/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[ ]:


submission_df = test_df.copy()
submission_df['image_name'] = submission_df['filename']
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)


# In[ ]:


from google.colab import files
files.download("submission.csv")


# In[ ]:


files.download("model.h5")


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


mv best_model.h5 /content/gdrive/'My Drive'


# In[ ]:


mv model.h5 /content/gdrive/'My Drive'


# In[ ]:




