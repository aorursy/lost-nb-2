#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pydicom as dcm
import cv2

'''Customize visualization
Seaborn and matplotlib visualization.'''
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import HTML
from PIL import Image



'''Plotly visualization .'''
import plotly.express as px
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa




import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,MaxPool2D)
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import metrics,applications,optimizers
from keras import backend as K
from keras.models import Sequential
get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn


# In[2]:


IMAGE_PATH = "../input/siim-isic-melanoma-classification/"

train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


#Training data
print('Training data shape: ', train_df.shape)
print(' ')
print('Test data shape: ', test_df.shape)

display('**TRAINING DATA**')
display(train_df.info())
display('**TEST DATA**')
display(test_df.info())


# In[3]:


train_df.sample(20)


# In[4]:


fig = px.scatter_matrix(train_df,dimensions=["age_approx", "sex", "target",'anatom_site_general_challenge','diagnosis'], color="benign_malignant")
fig.show()


# In[5]:


print(train_df.benign_malignant.value_counts())
sns.countplot(x = 'benign_malignant',data=train_df)


# In[6]:


df_sub = train_df.anatom_site_general_challenge.value_counts().reset_index()
df_sub.columns = ['anatom_site_general_challenge', 'Counts']
fig = px.bar(df_sub, x="anatom_site_general_challenge", y="Counts", color='anatom_site_general_challenge', barmode='group',
             height=400)
fig.show()


# In[7]:


df_sub = train_df.diagnosis.value_counts().reset_index()
df_sub.columns = ['diagnosis', 'Counts']
fig = px.bar(df_sub, x="diagnosis", y="Counts", color='diagnosis', barmode='group',
             height=400)
fig.show()


# In[8]:


fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train_df.loc[(train_df['sex'] == 'male'),'age_approx'] , color='gray',shade=True,label='Male')
ax=sns.kdeplot(train_df.loc[(train_df['sex'] == 'female'),'age_approx'] , color='g',shade=True, label='Female')
plt.title('Age Distribution', fontsize = 25, pad = 40)
plt.ylabel("Frequency of Age", fontsize = 15, labelpad = 20)
plt.xlabel("Age", fontsize = 15, labelpad = 20);


# In[9]:


fig = px.histogram(train_df, x="age_approx", y="benign_malignant", color="benign_malignant", marginal="rug")
fig.show()


# In[10]:


df = train_df.copy()


# In[11]:


get_ipython().run_cell_magic('time', '', 'def get_df():\n    base_image_dir = \'../input/siim-isic-melanoma-classification/jpeg/\'\n    train_dir = os.path.join(base_image_dir,\'train/\')\n    train = pd.read_csv(\'../input/siim-isic-melanoma-classification/train.csv\')\n    df_0=train[train[\'target\']==0].sample(2000)\n    df_1=train[train[\'target\']==1]\n    df=pd.concat([df_0,df_1])\n    df[\'path\'] = df[\'image_name\'].map(lambda x: os.path.join(train_dir,\'{}.jpg\'.format(x)))\n    df["image_name"]=train_df["image_name"].apply(lambda x:x+".jpg")\n    df[\'image\'] = df[\'path\'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))\n    df[\'target\'] = df[\'target\'].astype(str)\n    return df\n\ndf = get_df()')


# In[12]:


df.head()


# In[13]:


df['image'].map(lambda x: x.shape).value_counts()


# In[14]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'image_name']
        image_id = df.loc[i,'image_name']
        img = cv2.imread(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_path}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# In[15]:


y = df.target

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
x_train_o, x_test_o, y_train, y_test = train_test_split(df, y, test_size=0.25)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[16]:


print('Shape of X_train : ',x_train.shape)
print('Shape of y_train : ',y_train.shape)
print('===============================================')
print('Shape of X_val : ',x_test.shape)
print('Shape of y_val : ',y_test.shape)


# In[17]:


input_shape = (75, 100, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[18]:


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


# In[19]:


checkpoint = ModelCheckpoint("model1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')


# In[20]:


batch_size = 32 # Todo: experiment with this variable more
epochs = 50

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [checkpoint, early],
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[21]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[22]:


submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')


# In[23]:


test_dir = IMAGE_PATH + 'jpeg/test/'
test_data=[]
for i in range(test_df.shape[0]):
    test_data.append(test_dir + test_df['image_name'].iloc[i]+'.jpg')
df_test=pd.DataFrame(test_data)
df_test.columns=['images']


# In[24]:


target=[]
for path in df_test['images']:
    img=cv2.imread(str(path))
    img = cv2.resize(img, (75,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img=np.reshape(img,(1,75,100,3))
    prediction=model.predict(img)
    target.append(prediction[0][0])

submission['target']=target


# In[25]:


submission.to_csv('submission.csv', index=False)
submission.head()


# In[26]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


# In[27]:


X = df.path.values
y = np.float32(df.target.values)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.1, random_state=43)
print('done!')


# In[28]:


AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[29]:


BATCH_SIZE = 4 * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE


# In[30]:


def decode_image(filename, label=None, image_size=(100,75)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    
    
    if label is None:
        return image
    else:
        return image, label


# In[31]:


train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image,num_parallel_calls=AUTO)
    .map(data_augment,num_parallel_calls=AUTO)
    .repeat()
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test, y_test))
    .map(decode_image,num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# In[32]:


def build_lrfn(lr_start=0.00001, lr_max=0.00005,lr_min=0.00001, lr_rampup_epochs=5,lr_sustain_epochs=0, lr_exp_decay=.8):
    
    lr_max = lr_max * strategy.num_replicas_in_sync
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs- lr_sustain_epochs) + lr_min
        return lr
    return lrfn


# In[33]:


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,verbose=True, mode="min")


# In[34]:


def Eff_B7_NS():
    model_EfficientNetB7_NS = Sequential([efn.EfficientNetB7(input_shape=(100,75,3),weights='noisy-student',include_top=False),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(64,activation='relu'),
                                 tf.keras.layers.Dense(1,activation='sigmoid')])               
    model_EfficientNetB7_NS.compile(optimizer='Adam',loss = 'binary_crossentropy',metrics=['binary_accuracy'])
    
    
    return model_EfficientNetB7_NS


# In[35]:


with strategy.scope():
    model_Eff_B7_NS=Eff_B7_NS()
    
model_Eff_B7_NS.summary()
#del model_Eff_B7_NS


# In[36]:


EfficientNetB7_NS = model_Eff_B7_NS.fit(train_dataset,
                    epochs=10,
                    callbacks=[lr_schedule,EarlyStopping],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


# In[37]:


plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(EfficientNetB7_NS.history['loss'])
ax1.plot(EfficientNetB7_NS.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epoch')

ax2.plot(EfficientNetB7_NS.history['binary_accuracy'])
ax2.plot(EfficientNetB7_NS.history['val_binary_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epoch')

