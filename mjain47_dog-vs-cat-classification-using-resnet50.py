#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[2]:


from zipfile import ZipFile
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D


# In[3]:


import os
import shutil

dirpath='../output/kaggle/working/train/train'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')
    
    
dirpath='../output/kaggle/working/train/dog/train/'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')
    
dirpath='../output/kaggle/working/test/test'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')
    
dirpath='../output/kaggle/working/train'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')
    
    
dirpath='../output/kaggle/working/test'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')
    
dirpath='../kaggle/working/validation'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
    print('path deleted')
else:
    print('path doesnot exists')


# In[4]:


from zipfile import ZipFile

# specifying the zip file name 
test_file_name = "../input/dogs-vs-cats-redux-kernels-edition/test.zip"
  
# opening the zip file in READ mode 
with ZipFile(test_file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    #zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall('../output/kaggle/working/') 
    print('Done!') 
    
    


# In[5]:


#import os
f=[]
for root, dirs, files in os.walk("../output/kaggle/working/test/"):
    for filename in files:
        f.append(filename)

print(len(f))


# In[6]:


#import os

#for root, dirs, files in os.walk("../output/kaggle/working/"):
 #   for filename in files:
  #      print(filename)


# In[7]:


# specifying the zip file name 
train_file_name = "../input/dogs-vs-cats-redux-kernels-edition/train.zip"
  
# opening the zip file in READ mode 
with ZipFile(train_file_name, 'r') as zip: 
    # printing all the contents of the zip file 
   # zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall('../output/kaggle/working/') 
    print('Done!') 
    


# In[8]:


import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../output/kaggle/working/train/"):
    for filename in files:
        if filename.startswith('dog'):
            dogslist.append(filename)
        else:
            catslist.append(filename)
            
print(len(dogslist))
             
print(len(catslist))


# In[9]:



if not os.path.exists('../output/kaggle/working/train/dog'):
    os.makedirs('../output/kaggle/working/train/dog')

for e in [os.path.join('../output/kaggle/working/train/',name) for name in dogslist]:
    shutil.copy(e,'../output/kaggle/working/train/dog/')
   
    
    


# In[10]:


if not os.path.exists('../output/kaggle/working/train/cat'):
    os.makedirs('../output/kaggle/working/train/cat')

for e in [os.path.join('../output/kaggle/working/train/',name) for name in catslist]:
    shutil.copy(e,'../output/kaggle/working/train/cat/')
    


# In[11]:


import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../output/kaggle/working/train/dog"):
    for filename in files:
        if filename.startswith('dog'):
            dogslist.append(filename)
        else:
            catslist.append(filename)
            
print(len(dogslist))
             
print(len(catslist))


# In[12]:


import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../output/kaggle/working/train/cat"):
    for filename in files:
        if filename.startswith('cat'):
            catslist.append(filename)
        else:
            dogslist.append(filename)
            
print(len(catslist))
print(len(dogslist))
             


# In[13]:


num_classes=2


resnet_weights_path = '../input/weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_model=Sequential()

my_model.add(ResNet50(include_top=False,pooling='avg',weights=resnet_weights_path))

my_model.add(Dense(num_classes,activation='softmax'))

my_model.layers[0].trainable=False


# In[14]:


my_model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])


# In[15]:


if not os.path.exists('../output/kaggle/working/validation/dog'):
    os.makedirs('../output/kaggle/working/validation/dog')
    
if not os.path.exists('../output/kaggle/working/validation/cat'):
    os.makedirs('../output/kaggle/working/validation/cat')
    


# In[16]:


pip install split_folders


# In[17]:


import split_folders

split_folders.ratio('../output/kaggle/working/train', output="../output/kaggle/working/Val", seed=1337, ratio=(.8,.2))


# In[18]:


import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../working/validation/train/cat/"):
    for filename in files:
        if filename.startswith('cat'):
            catslist.append(filename)
        else:
            dogslist.append(filename)
            
print(len(catslist))
print(len(dogslist))


# In[19]:


import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../working/validation/train/dog/"):
    for filename in files:
        if filename.startswith('dog'):
            dogslist.append(filename)
        else:
            catslist.append(filename)
            

print(len(dogslist))
print(len(catslist))


# In[20]:




import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../working/validation/val/dog/"):
    for filename in files:
        if filename.startswith('dog'):
            dogslist.append(filename)
        else:
            catslist.append(filename)
            

print(len(dogslist))
print(len(catslist))


# In[21]:



import shutil

dogslist=[]
catslist=[]
for root, dirs, files in os.walk("../working/validation/val/cat/"):
    for filename in files:
        if filename.startswith('cat'):
            catslist.append(filename)
        else:
            dogslist.append(filename)
            


print(len(catslist))
print(len(dogslist))


# In[22]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size=224

data_generator = ImageDataGenerator(preprocess_input)


train_data_generator= data_generator.flow_from_directory(
        '/kaggle/output/kaggle/working/Val/train/',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')


validation_generator = data_generator.flow_from_directory(
        '/kaggle/output/kaggle/working/Val/val/',
        target_size=(image_size, image_size),
        class_mode='categorical')



# In[23]:


f=[]

for root, dirs, files in os.walk("../output/kaggle/working/test/"):
    for filename in files:
        f.append(filename)
    

if not os.path.exists('/kaggle/output/kaggle/working/Val/test/'):
    os.makedirs('/kaggle/output/kaggle/working/Val/test/')
else:
    print('folder exists')
    

    
if not os.path.exists('/kaggle/output/kaggle/working/Val/test/data/'):
    os.makedirs('/kaggle/output/kaggle/working/Val/test/data')
else:
    print('folder exists')

for e in [os.path.join('../output/kaggle/working/test/',name) for name in f]:
    shutil.copy(e,'/kaggle/output/kaggle/working/Val/test/data/')

    
d=[]

for root, dirs, files in os.walk("/kaggle/output/kaggle/working/Val/test/"):
    for filename in files:
        d.append(filename)
        
print(len(d))

test_generator = data_generator.flow_from_directory(
     '/kaggle/output/kaggle/working/Val/test',
    target_size = (224, 224),
    batch_size = 24,
    class_mode = None,
    shuffle = False,
    seed = 123
)


# In[24]:


my_model.fit_generator(
        train_data_generator,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=1)


# In[25]:


test_generator.reset()

pred = my_model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


# In[26]:


import cv2


from matplotlib import pyplot as plt

TEST_DIR = '/kaggle/output/kaggle/working/Val/test/'

f, ax = plt.subplots(10, 5, figsize = (15, 15))

for i in range(0,50):
    print(TEST_DIR+test_generator.filenames[i])
    imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    # a if condition else b
    predicted_class = "Dog" if predicted_class_indices[i] else "Cat"

    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    

plt.show()


# In[27]:


results_df = pd.DataFrame(
    {
        'id': pd.Series(test_generator.filenames), 
        'label': pd.Series(predicted_class_indices)
    })
results_df['id'] = results_df.id.str.extract('(\d+)')
results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
results_df.sort_values(by='id', inplace = True)

results_df.to_csv('submission.csv', index=False)
results_df.head()

