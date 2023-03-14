#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gast==0.2.2')
get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models')


# In[2]:


import os
import gc
import numpy as np 
import pandas as pd 
import random as rn
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import segmentation_models as sm
sm.set_framework('tf.keras')

get_ipython().run_line_magic('matplotlib', 'inline')
print('TF version:',tf.__version__)
print('Eager execution:', tf.executing_eagerly())


# In[3]:


base_path = '../input/severstal-steel-defect-detection/'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))

train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['class_rle1'] = train['EncodedPixels'][::4].values
train2['class_rle2'] = train['EncodedPixels'][1::4].values
train2['class_rle3'] = train['EncodedPixels'][2::4].values
train2['class_rle4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna('',inplace=True); 
train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).map(str)
train2.head()


# In[4]:


# How many images are there with given defect in original training set
print(np.sum(train2[['class_rle1','class_rle2','class_rle3','class_rle4']]!=''))
print('Total images:', len(train2))


# In[5]:


# selecting images classified by binary classifier as 'with defect'
output_path = '../input/unet-dataset-for-binary-classification/'
filtered_train = pd.read_csv(os.path.join(output_path, 'train_filter.csv'))
train2 = train2[train2['ImageId'].isin(filtered_train['ImageId'].values)]


# In[6]:


# How many images are left after application of binary classifier
print(np.sum(train2[['class_rle1','class_rle2','class_rle3','class_rle4']]!=''))
print('Total images:', len(train2))


# In[7]:


np.random.seed(2019)
rn.seed(2019)

train2 = train2.sample(frac=1).reset_index(drop=True)
train2.head()


# In[8]:


# extract pixel positions and patch lengths - preprocessing before passing to Dataset
def convertEP(class_rle, idx):                       
    array = np.asarray([int(x) for x in class_rle.split()])
    starts = '' if len(array)==0 else " ".join(map(str, array[0::2] + 1600*256*idx ))+ " "
    lengths = '' if len(array)==0 else " ".join(map(str, array[1::2]))+ " "

    isPresent = '1' if len(starts)>0 else '0'
    
    return starts, lengths, isPresent


# In[9]:


depth = 4
train2['starts'], train2['lengths'] = "", ""

# as many defect classes that many layers would have our output image
for idx in range(1,depth+1):
    start, length, presence, class_rle = 'starts{}'.format(idx), 'lengths{}'.format(idx),                                          'isPresent{}'.format(idx), 'class_rle{}'.format(idx)    
    train2[start], train2[length], train2[presence] = zip(*train2[class_rle].apply(convertEP, idx=idx-1))

    train2['starts'] += train2[start]
    train2['lengths'] += train2[length]
    del train2[class_rle], train2[start], train2[length]    


# In[10]:


train2.head()


# In[11]:


def rle_tf(mydf):
    image_path, class_count, starts, lengths = mydf['ImageId'], mydf['count'], mydf['starts'], mydf['lengths']    
    class_count = tf.strings.to_number(class_count, out_type=tf.dtypes.int32)                        
    classes_one_hot = tf.strings.to_number([mydf['isPresent1'],mydf['isPresent2'],mydf['isPresent3'],mydf['isPresent4']], out_type=tf.dtypes.int32)

    height, width, factor = 256, 1600, 2                
    
    def generate_mask():                 
        starts_tensor = tf.strings.to_number( tf.strings.split([starts]).values, out_type=tf.dtypes.int32)
        lengths_tensor = tf.strings.to_number( tf.strings.split([lengths]).values, out_type=tf.dtypes.int32)    

        ranges = tf.ragged.range(starts_tensor, starts_tensor+lengths_tensor-1).values        
        ranges = tf.expand_dims(ranges, axis=-1)        
        
        mask = tf.scatter_nd(ranges,
                             tf.ones(tf.shape(ranges)[0], dtype=tf.dtypes.int16), 
                             tf.constant([width*height*depth]))                        

        mask = tf.reshape(mask, [depth, width, height])  
        mask = tf.transpose(mask, [2, 1, 0])
        mask = tf.image.resize(mask, [height//factor, width//factor])        
        mask = tf.dtypes.cast(mask, tf.dtypes.float32)        
        
        return mask                 
    
    def empty_mask():         
        return tf.zeros((height//factor, width//factor, depth), dtype=tf.dtypes.float32)
    
    return image_path, tf.cond(tf.math.greater(class_count, 0), generate_mask, empty_mask), class_count, classes_one_hot
    
def imag_proc_tf(image_path, mask, class_count, classes_one_hot):     
    height, width, channels, factor = 256, 1600, 3, 2
    
    im = tf.io.read_file(im_path+image_path)
    im = tf.image.decode_jpeg(im, channels=channels)    
    im = tf.image.resize(im, [height//factor, width//factor])            
    im = tf.dtypes.cast(im, tf.dtypes.int16)        
    
    return im, mask, class_count, classes_one_hot

#https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/    
def augmentations(im, mask, class_count, classes_one_hot):    
    im = tf.dtypes.cast(im, tf.dtypes.float32)        
    
    # rgb image only augmentations        
    im = tf.image.random_brightness(im, 0.001)
    im = tf.image.random_contrast(im, 0.7, 1.3)

    images = tf.concat([im, mask], axis=-1)

    # joint mask-image augmentations
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)                        
        
    im = tf.dtypes.cast(tf.clip_by_value(images[:,:,:3], 0, 255), tf.dtypes.int16)
    mask = tf.clip_by_value(images[:,:,3:], 0, 1)
      
    return im, mask, class_count, classes_one_hot


# In[12]:


def get_dataset(mydf, batch_size = 16, do_augment = False, filter_specific = None, isTrain = True):

    dataset = tf.data.Dataset.from_tensor_slices(dict(mydf))
    dataset = dataset.map(rle_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)            
    dataset = dataset.map(imag_proc_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)         
    
    if do_augment:
        dataset = dataset.map(augmentations, num_parallel_calls=tf.data.experimental.AUTOTUNE)                                   
                   
    if filter_specific is not None:
        dataset = dataset.filter(lambda image, mask, class_count, cl1h: 
                                 tf.reshape( tf.reduce_all( tf.equal( cl1h, filter_specific)),[]))
        
    # image-110 part is a preprocessing required by the selected segmentation model
    dataset = dataset.map(lambda image, mask, class_count, cl1h:
                          (image-110, mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)          
    if isTrain:
        dataset = dataset.shuffle(batch_size*10, reshuffle_each_iteration=True).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))  
        
    return dataset


# In[13]:


im_path = os.path.join(base_path,'train_images/')

# Repeat selected image with defect
selected_image = train2[train2['count']=='1'].iloc[0:1]
test_df = pd.concat([selected_image]*16, ignore_index=True)

minibatch = 16
ds = get_dataset(test_df, batch_size=minibatch, do_augment=True)

for images, masks in ds.take(1):
    columns, rows = 1, minibatch
    fig = plt.figure(figsize=(10,40))

    for idx, (img, mask) in enumerate(zip(images.numpy(), masks.numpy())):    
        fig.add_subplot(rows, columns, idx+1)        

        img = (img+110).astype(int)            
        emask = np.zeros(img.shape, dtype=int)  

        for nc in range(depth):
            if nc==0:
                emask[mask[:,:,0]==1,0] = 255
            elif nc==1:
                emask[mask[:,:,1]==1,1] = 255        
            elif nc==2:
                emask[mask[:,:,2]==1,2] = 255        
            elif nc==3:
                emask[mask[:,:,3]==1,0] = 255        
                emask[mask[:,:,3]==1,1] = 255        

        plt.imshow(img)      
        plt.imshow(emask, alpha=0.25)      


# In[14]:


# Get training and validation sets
total_samples = len(train2)
ratio = 0.8

train_df = train2.iloc[:int(ratio*total_samples)]
valid_df = train2.iloc[int(ratio*total_samples):]

del ds, test_df, train2
gc.collect()


# In[15]:


def get_ds(tr_df, val_df, minibatch=32, filter_specific=None, do_augment=False):    
    dstr = get_dataset(tr_df, batch_size=minibatch, filter_specific=filter_specific, isTrain=True, do_augment=do_augment)
    dsva = get_dataset(val_df, batch_size=minibatch, filter_specific=filter_specific, isTrain=False, do_augment=False)    
        
    return dstr, dsva


# In[16]:


def dice_coef2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#https://github.com/nabsabraham/focal-tversky-unet
def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss_r(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


# https://github.com/nabsabraham/focal-tversky-unet/issues/3
def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

# channels sensitive loss function
def focal_tversky_loss_c(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1-pt_1), gamma))


# In[17]:


def get_callbacks(model_name, val_metric='val_dice_coef2'):

    RRc = ReduceLROnPlateau(monitor = val_metric, 
                            factor = 0.5, 
                            patience = 15, 
                            min_lr=0.000001, 
                            verbose=1, 
                            mode='max')

    MCc = ModelCheckpoint(model_name,
                          monitor=val_metric,
                          save_best_only=True, 
                          verbose=1,                       
                          mode='max')    
    return [RRc, MCc]


# In[18]:


from segmentation_models import Unet
model = Unet('resnet34', input_shape=(128, 800, 3), classes=depth, activation='sigmoid', encoder_weights='imagenet')
model.save_weights('imagenet.h5')

settings = zip(['regular_ftl.h5', 'channeles_ftl.h5'],
               [focal_tversky_loss_r, focal_tversky_loss_c])

hist=[]
for model_name, loss_fcn in settings:    
    opt = tf.keras.optimizers.Adam(0.0001,  clipnorm=1.0)
    model.load_weights('imagenet.h5')
    model.compile(optimizer=opt, loss=loss_fcn, metrics=[dice_coef2])    
    
    minibatch = 30
    dstr, dsva = get_ds(tr_df=train_df, val_df=valid_df, minibatch=minibatch, do_augment=False)
            
    history = model.fit(dstr,                  
                      epochs=60,
                      verbose=2,
                      validation_data=dsva,                   
                      callbacks = get_callbacks(model_name))
                        
    hist.append(history)        


# In[19]:


plt.plot(1+np.arange(len(hist[0].history['dice_coef2'])), hist[0].history['dice_coef2'], '-xr', label='train regular')
plt.plot(1+np.arange(len(hist[0].history['val_dice_coef2'])), hist[0].history['val_dice_coef2'], '-b', label='valid regular')
plt.plot(1+np.arange(len(hist[1].history['dice_coef2'])), hist[1].history['dice_coef2'], '-xk', label='train channels')
plt.plot(1+np.arange(len(hist[1].history['val_dice_coef2'])), hist[1].history['val_dice_coef2'], '-m', label='valid channels')
plt.ylabel('dice score [a.u.]')
plt.xlabel('epoch [n]')
plt.legend()
plt.grid()

print('Best valid dice score regular: {0:.4f}'.format(max(hist[0].history['val_dice_coef2'])))
print('Best valid dice score channels: {0:.4f}'.format(max(hist[1].history['val_dice_coef2'])))


# In[20]:


minibatch = 4
config_list = []
for gr, l in valid_df.groupby(['isPresent1','isPresent2','isPresent3','isPresent4']):
    if len(l)>=minibatch:        
        config_list.append([[int(c) for c in gr], len(l)])
        print(*config_list[-1])


# In[21]:


from collections import defaultdict
scores = defaultdict(list)

settings = zip(['regular_ftl.h5', 'channeles_ftl.h5'],
               [focal_tversky_loss_r, focal_tversky_loss_c])

for model_name, loss_fcn in settings: 
    print('\n\n')
    model.load_weights(model_name)          
    for cur_filt, quant in config_list:        
        _, dsva = get_ds(tr_df=train_df, val_df=valid_df, minibatch=minibatch, filter_specific=cur_filt)
        loss, dice = model.evaluate(dsva, verbose = 0)         
        scores[model_name].append((cur_filt, dice, quant))
        
    print('\nLoss:',loss_fcn.__name__)
    scores[model_name] = sorted(scores[model_name], key=lambda x: x[2], reverse=True)
    for cur_filt, dice, step in scores[model_name]:
        print('Config: {0} (steps={1}) =>\t dice_coef: {2:.5f}'.format(cur_filt, step, dice))


# In[22]:


settings = zip(['regular_ftl.h5', 'channeles_ftl.h5'],
               [focal_tversky_loss_r, focal_tversky_loss_c],
               ['.','x'])
fig = plt.figure(figsize=(10,5))

for model_name, loss_fcn, marker in settings:     
    x = [rec[2] for rec in scores[model_name]]
    y = [rec[1] for rec in scores[model_name]]    
    plt.semilogx(x,y, marker,  markersize=22, label = loss_fcn.__name__)
    
plt.ylabel('dice score [a.u.]')
plt.xlabel('quantity [n]')
plt.legend()
plt.grid()    

