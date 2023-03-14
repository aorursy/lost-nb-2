#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm
from glob import glob
import time,gc
import cv2

from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class_map_df=pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
sample_sub_df=pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')


# In[3]:


train_df_.head()


# In[4]:


test_df_.head()


# In[5]:


sample_sub_df.head()


# In[6]:


class_map_df.head()


# In[7]:


print(f'Size of trainig Data: {train_df_.shape}')
print(f'Size of test Data : {test_df_.shape}')
print(f'Size of class map : {class_map_df.shape}')


# In[8]:


HEIGHT = 236
WIDTH = 236


# In[9]:


def get_n(df, field, n , top=True):
    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]
    top_grapheme_roots = top_graphemes.index
    top_grapheme_counts = top_graphemes.values
    top_graphemes = class_map_df[class_map_df['component_type']==field].reset_index().iloc[top_grapheme_roots]
    top_graphemes.drop(['component_type','label'],axis=1,inplace=True)
    top_graphemes.loc[:,'count'] = top_grapheme_counts
    return top_graphemes


# In[10]:


def image_from_char(char):
    image = Image.new('RGB',(WIDTH,HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont= ImageFont.truetype('/kaggle/input/kalpurush-fonts/kalpurush-2.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH-w)/2,(HEIGHT-h)/3),char,font=myfont)
    return image


# In[11]:


train_df_.head()
len(train_df_)


# In[12]:


print(f'Number of unique graheme roots : {train_df_["grapheme_root"].nunique()}')
print(f'Number of unique vowel diacritic : {train_df_["vowel_diacritic"].nunique()}')

print(f'Number of unique consonant diacritic : {train_df_["consonant_diacritic"].nunique()}')


# In[13]:


top_10_roots = get_n(train_df_,'grapheme_root',10)
top_10_roots


# In[14]:


top_graphemes = train_df_.groupby(['grapheme_root']).size().reset_index(name='counts')['counts'].sort_values(ascending=not True)[:10]


# In[15]:


type(top_graphemes)


# In[16]:


top_graphemes.values


# In[17]:


len(train_df_)


# In[18]:


len(top_graphemes)


# In[19]:


top_graphemes.head()


# In[20]:


def get_n(df, field, n , top=True):
    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]
    top_grapheme_roots = top_graphemes.index
    top_grapheme_counts = top_graphemes.values
    top_graphemes = class_map_df[class_map_df['component_type']==field].reset_index().iloc[top_grapheme_roots]
    top_graphemes.drop(['component_type','label'],axis=1,inplace=True)
    top_graphemes.loc[:,'count'] = top_grapheme_counts
    return top_graphemes


# In[21]:


a = pd.DataFrame([('bird', 'Falconiformes', 389.0),
	                    ('bird', 'Psittaciformes', 24.0),
	                    ('mammal', 'Carnivora', 80.2),
	                    ('mammal', 'Primates', np.nan),
	                    ('mammal', 'Carnivora', 58)],
	                  index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
	                  columns=('class', 'order', 'max_speed'))
 


# In[22]:


a


# In[23]:


b=a.groupby(['class'])


# In[24]:


b.groups


# In[25]:


b.indices


# In[26]:


f,ax = plt.subplots(2,5,figsize=(16,8))
ax = ax.flatten()

for i in range(10):
    ax[i].imshow(image_from_char(top_10_roots['component'].iloc[i]),cmap='Greys')


# In[27]:


bottom_10_roots = get_n(train_df_,'grapheme_root',10,False)
bottom_10_roots


# In[28]:


f,ax = plt.subplots(2,5, figsize=(16,8))
ax=ax.flatten()

for i in range(10):
    ax[i].imshow(image_from_char(bottom_10_roots['component'].iloc[i]),cmap='Greys')


# In[29]:


top_5_vowels = get_n(train_df_, 'vowel_diacritic',5)
top_5_vowels


# In[30]:


f,ax = plt.subplots(1,5,figsize=(16,8))
ax = ax.flatten()

for i in range(5):
    ax[i].imshow(image_from_char(top_5_vowels['component'].iloc[i]),cmap='Greys')


# In[31]:


top_5_consonants = get_n(train_df_,'consonant_diacritic',5)
top_5_consonants


# In[32]:


f,ax = plt.subplots(1,5,figsize=(16,8))
ax = ax.flatten()

for i in range(5):
    ax[i].imshow(image_from_char(top_5_consonants['component'].iloc[i]),cmap='Greys')


# In[33]:


train_df_ = train_df_.drop(['grapheme'],axis=1,inplace=False)


# In[34]:


train_df_[['grapheme_root','vowel_diacritic','consonant_diacritic']]=train_df_[['grapheme_root','vowel_diacritic','consonant_diacritic']].astype('uint8')


# In[35]:


IMG_SIZE=64
N_CHANNELS=1


# In[36]:


def resize(df, size=64, need_progress_bar=True):
    resized={}
    resize_size=64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image,30,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            idx=0
            is_xmin=[]
            is_ymin=[]
            is_xmax=[]
            is_ymax=[]
            for cnt in contours:
                idx+=1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x+w)
                ls_ymax.append(y+h)
            xmin=min(ls_xmin)
            ymin=min(ls_ymin)
            xmax=min(ls_xmax)
            ymax=max(ls_ymax)
            
            roi = image[ymin:ymax, xmin:xmax]
            resized_roi=cv2.resize(roi,(resize_size,resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]]=resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image,30,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            idx=0
            is_xmin=[]
            is_ymin=[]
            is_xmax=[]
            is_ymax=[]
            for cnt in contours:
                idx+=1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x+w)
                ls_ymax.append(y+h)
            xmin=min(ls_xmin)
            ymin=min(ls_ymin)
            xmax=min(ls_xmax)
            ymax=max(ls_ymax)
            
            roi = image[ymin:ymax, xmin:xmax]
            resized_roi=cv2.resize(roi,(resize_size,resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]]=resized_roi.reshape(-1)
    resized=pd.DataFrame(resized).T
    return resized


# In[37]:


def get_dummies(df):
    cols= []
    for col in df:
        cols.append(pd.get_dummieds(df[col].astype(str)))
    return pd.concat(cols,axis=1)


# In[38]:


inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Flatten()(model)
model = Dense(1024, activation = "relu")(model)
model = Dropout(rate=0.3)(model)
dense = Dense(512, activation = "relu")(model)

head_root = Dense(168, activation = 'softmax')(dense)
head_vowel = Dense(11, activation = 'softmax')(dense)
head_consonant = Dense(7, activation = 'softmax')(dense)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


# In[39]:


model.summary()


# In[40]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[41]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[42]:


#set a learning annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy',
                                                 patience=3,
                                                 verbose=1, #it may show progress bar 
                                                  factor=0.5,
                                                  min_lr=0.00001
                                                 )
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy',
                                                     patience=3,
                                                     verbose=1,
                                                     factor=0.5,
                                                     min_lr=0.00001)


# In[43]:


batch_size=256
epochs = 30


# In[44]:


class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        targets=None
        target_lengths={}
        ordered_outputs=[]
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets,target),axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)
        for flowx, flowy in super().flow(x,targets,batch_size=batch_size,
                                        shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:,i:i+target_length]
                i += target_length
                
            yield flowx, target_dict


# In[45]:


HEIGHT=137
WIDTH=236


# In[46]:


histories=[]
for i in range(4):
    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'),train_df_,on='image_id').drop(['image_id'],axis=1)
    #visualize few samples of current training dataset
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16,8))
    count=0
    for row in ax:
        for col in row:
            col.imshow(resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))            
            count+=1
    plt.show()

    X_train = train_df.drop(['grapheme_root','vowel_diacritic','consonant_diacritic'],axis=1)
    X_train = resize(X_train)/255
    #CNN takes images in shape '(batch_size,h,w,channels)', so reshape the images

    X_train=X_train.values.reshape(-1, IMG_SIZE,IMG_SIZE,N_CHANNELS)
    
    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
    
    print(f'Training Images : {X_train.shape}')
    print(f'Training labels root : {Y_train_root.shape}')
    print(f'Trainign labels vowel : {Y_train_vowel.shape}')
    print(f'Training labels consonants : {Y_train_consonant.shpae}')
    
    #Divide the data into training and validation set
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, 
        y_train_consonant, y_test_consonant = 
            train_test_split(X_train,Y_train_root,Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
    del train_df
    del X_train
    del Y_train_root, Y_train_vowel, Y_train_consonant
    
    # Data augmentation for creating more training data
    datagen = MultiOutputDataGenerator(
        featurewise_center=False, #set imput mean to 0 over the data set
        samplewise_cneter=False, #set each sample mean to 0
        featurewise_std_normalization=False , #divide inputs by std of the dataset
        samplewise_std_normalization=False , #divide each input by its std
        zca_whitening=False, #apply ZCA whitening
        rotation_range = 8, #randomly rotate images in the range(degrees, 0 to 180)
        zoom_range = 0.15, #Randomly zoom image
        width_shift_range = 0.15, #randomly shift images horizontally(fraction of total width)
        height_shift_range = 0.15, #randomly shift images vertically (fraction of total height)
        horizontal_flip = False, #randomly flip images
        vertical_flip = False ) #randomly flip images
    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    datagee.fix(x_train)
    
    # Fit the model
    history = model.fit_generator(datagen.flow(x_Train,{'dense_3':y_train_root,'dense_4':y_train_vowel,'dense_5':y_train_consonant},batch_size=batch_size),
                                 epochs=epochs, validation_data=(x_test,[y_test_root,y_test_vowel,y_test_consonant]),
                                  steps_per_epoch=x_train_shape[0]//batch_size,
                                  callbacks=[learning_rate_reduction_root,learning_rate_reduction_vowel,learning_rate_reduction_consonant])
    histories.append(history)
    
    #Delete to reduce memory usage
    
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
    
    gc.collect()
    
    


# In[ ]:




