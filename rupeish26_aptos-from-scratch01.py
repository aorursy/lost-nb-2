#!/usr/bin/env python
# coding: utf-8

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[8]:


os.listdir('../input/aptos2019-blindness-detection/')


# In[9]:


base_image_dir = os.path.join('..','input/aptos2019-blindness-detection/')
train_dir = os.path.join(base_image_dir,'train_images/')
train_df = pd.read_csv(os.path.join(base_image_dir,'train.csv'))


# In[10]:


train_df.head()


# In[11]:


train_df['image']= train_df['id_code'].map(lambda x: '{}.png'.format(x))


# In[12]:


train_df['diagnosis']= train_df['diagnosis'].map(lambda x: str(x))


# In[13]:


train_df.head()


# In[14]:


train_df = train_df.drop(columns=['id_code'])


# In[15]:


train_df.head()


# In[16]:


train_df = train_df.sample(frac=1).reset_index(drop=True)


# In[17]:


train_df.head()


# In[18]:


print("Number of images: {}".format(len(train_df)))


# In[19]:


train_df['diagnosis'].hist(figsize=(10,5))


# In[87]:


from PIL import Image
from matplotlib import pyplot as plt
img = Image.open('../input/aptos2019-blindness-detection/train_images/'+train_df['image'][1])
w, h = img.size
print(w,h)


# In[88]:


plt.imshow(np.asarray(img))


# In[89]:


from sklearn.model_selection import train_test_split


# In[90]:


train, valid = train_test_split(train_df,test_size=0.2, random_state=42, shuffle=True)


# In[91]:


train.shape


# In[92]:


valid.shape


# In[93]:


from tensorflow import keras


# In[94]:


from keras.applications.inception_v3 import InceptionV3


# In[95]:


model = InceptionV3()


# In[96]:


model.summary()


# In[97]:


from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf


# In[98]:


flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(5, activation='softmax')(class1)


# In[99]:


model = Model(inputs=model.inputs, outputs=output)


# In[100]:


model.summary()


# In[101]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


# In[102]:


#history = model.fit(train_images, train_labels, epochs=10, 
 #                   validation_data=(test_images, test_labels))


# In[103]:


from keras_preprocessing.image import ImageDataGenerator


# In[104]:


datagen=ImageDataGenerator()


# In[105]:


train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory=train_dir,
x_col="image",
y_col="diagnosis",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(299,299))


# In[107]:


valid_generator=datagen.flow_from_dataframe(
dataframe=valid,
directory=train_dir,
x_col="image",
y_col="diagnosis",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(299,299))


# In[108]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


# In[109]:


model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=2
)


# In[42]:


test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="./test/",
x_col="id",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))


# In[ ]:


model.evaluate_generator(generator=valid_generator,
steps=1)


# In[111]:


mkdir model


# In[112]:


ls


# In[113]:


model.save('/kaggle/working/model')


# In[114]:


ls model/


# In[115]:


model_new = keras.models.load_model('/kaggle/working/model')


# In[116]:


model_new.summary()


# In[ ]:


valid_generator.reset()


# In[ ]:


pred=model.predict_generator(valid_generator,
steps=STEP_SIZE_VALID,
verbose=1)


# In[ ]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[ ]:


valid


# In[165]:


x = train_dir+'0024cdab0c1e.png'


# In[166]:


x


# In[147]:


from PIL import Image
from matplotlib import pyplot as plt
img = Image.open(x)
w, h = img.size
print(w,h)


# In[148]:


x= np.array(img)
x.shape


# In[149]:


x = np.resize(x,(299,299,3))


# In[150]:


x = np.expand_dims(x,0)


# In[151]:


x.shape


# In[152]:


model.summary()


# In[153]:


pred = model.predict(x,batch_size=1, verbose=0, steps=None, callbacks=None)


# In[154]:


pred


# In[155]:


np.argmax(pred,axis=1)


# In[156]:


pred_new = model_new.predict(x,batch_size=1, verbose=0, steps=None, callbacks=None)


# In[157]:


pred_new


# In[158]:


np.argmax(pred_new,axis=1)


# In[167]:


import cv2
import numpy as np
img_list = []
def prepare(filename):
  frame = cv2.imread(filename)
  im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  model_image_size = (299, 299)
  resized_image = cv2.resize(im, model_image_size, interpolation = cv2.INTER_CUBIC)
  resized_image = resized_image.astype(np.float32)
  resized_image /= 255.
  image_data = np.expand_dims(resized_image, 0)
  return image_data
img = prepare(x)
prediction =  model_new.predict([img])
print(prediction)


# In[168]:


np.argmax(prediction,axis=1)


# In[ ]:




