#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Dataset    : 2019 and 2020
Mode       : BiT 
Image size : 384

'''


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import re
import seaborn as sns
import numpy as np
import pandas as pd
import math
import tensorflow_hub as hub
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow.keras.layers as L

from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
#import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[3]:


img_2019_dir = '../input/isic2019-384x384/train'
img_2019_csv_loc = '../input/isic2019-384x384/train.csv'
img_2020_dir = '.../input/melanoma-384x384'
img_2020_train_csv = '../input/siim-isic-melanoma-classification/train.csv'


# In[4]:


img_2019_csv = pd.read_csv(img_2019_csv_loc)
img_2019_csv.target.value_counts()


# In[5]:


img_2020_csv = pd.read_csv(img_2020_train_csv)
img_2020_csv.target.value_counts()


# In[6]:


DEVICE = 'TPU'
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


# In[7]:


SEED = 545454

# Configuration
EPOCH = 11
BATCH_SIZE = 4 * strategy.num_replicas_in_sync

FOLDS = 5

# WHICH IMAGE SIZES TO LOAD EACH FOLD
# CHOOSE 128, 192, 256, 384, 512, 768
IMAGE_SIZE = 384
IMG_SIZES = [IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE]

# INCLUDE OLD COMP DATA? YES=1 NO=0
INC2019 = [1,1,1,1,1]
#INC2018 = [1,1,1,1,1]

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [BATCH_SIZE]*FOLDS
EPOCHS = [EPOCH]*FOLDS


# In[8]:


print(BATCH_SIZES)
print(EPOCHS)


# In[ ]:





# In[9]:


GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS

for i,k in enumerate(IMG_SIZES):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('melanoma-%ix%i'%(k,k))
    GCS_PATH2[i] = KaggleDatasets().get_gcs_path('isic2019-%ix%i'%(k,k))
    
files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))


# In[10]:


print(GCS_PATH)
print(GCS_PATH2)


# In[11]:


#KaggleDatasets().get_gcs_path('jpeg-isic2019-384x384')


# In[12]:


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# In[13]:


# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
VERBOSE = 0
DISPLAY_PLOT = True

TRAINING_FILENAMES   = []
VALIDATION_FILENAMES = []

# Data access
MAIN_TEST_GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TEST_FILENAMES = tf.io.gfile.glob(MAIN_TEST_GCS_PATH + '/tfrecords/test*.tfrec')

skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

oof_pred = []; oof_tar = []; oof_val = []; oof_names = []; oof_folds = [] 
preds = np.zeros((count_data_items(files_test),1))

for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

    if DEVICE=='TPU':
        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
            
    print('#'*25); print('#### FOLD',fold+1)
    print('#### Image Size %i and batch_size %i'%(IMG_SIZES[fold],BATCH_SIZES[fold]*REPLICAS))
    
    # CREATE TRAIN SUBSETS
    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])
    
    if INC2019[fold]:
        files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+1])
        print('#### Using 2019 external data')
        
    np.random.shuffle(files_train); print('#'*25)
    
    TRAINING_FILENAMES.append(files_train)
    
    # CREATE VALIDATION SUBSETS
    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])
    VALIDATION_FILENAMES.append(files_valid)
    
    # CREATE TEST SUBSETS
    #files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))


# In[14]:


#print((TRAINING_FILENAMES[5]))


# In[15]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    return image

def read_labeled_tfrecord(example):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    
    return image, label

def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    # To add if using Bit
    image = decode_image(example['image'])
    idnum = example['image_name']

    return image, idnum if return_image_name else 0


# In[16]:


def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.transpose(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)    
    # used in Christ's notebook
    #image = tf.image.random_saturation(image, 0, 2)
    #imgage = tf.image.random_contrast(img, 0.8, 1.2)
    #imgage = tf.image.random_brightness(img, 0.1)

    return image, label

def get_training_dataset(fold_index):
    dataset = load_dataset(TRAINING_FILENAMES[fold_index], labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(fold_index):
    dataset = load_dataset(VALIDATION_FILENAMES[fold_index], labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


# In[17]:


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES[1])
NUM_VALID_IMAGES = count_data_items(VALIDATION_FILENAMES[1])
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images,{} vaid images, {} unlabeled test images'.
       format(NUM_TRAINING_IMAGES,NUM_VALID_IMAGES, NUM_TEST_IMAGES))
print('Steps per epoch :{}'.format(STEPS_PER_EPOCH))


# In[18]:


def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=1, 
               lr_sustain_epochs=0, lr_exp_decay=.78):
    lr_max = lr_max #* strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


# In[19]:


lr = 0.003 * BATCH_SIZE / 512 

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5,10,15], 
                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])


# In[20]:


MODELPATH = KaggleDatasets().get_gcs_path('big-transfer-models-without-top')
# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r101x1_1/')
# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r101x3_1/')
# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r152x4_1/')
# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r50x1_1/')
#module = hub.KerasLayer(f'{MODELPATH}/bit_m-r50x3_1/')


# In[21]:


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


# In[22]:


with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))
    
    MODELPATH = KaggleDatasets().get_gcs_path('big-transfer-models-without-top')
    module = hub.KerasLayer(f'{MODELPATH}/bit_m-r152x4_1/')
    back_bone = module
    back_bone.trainable = True
    
    logits = back_bone(inputs)
    #logits = tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2()
    #                               , dtype='float32')(logits)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(logits)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    #focal_loss = tfa.losses.sigmoid_focal_crossentropy(gamma = 2.0, alpha = 0.80)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = tfa.losses.SigmoidFocalCrossEntropy(gamma = 2.0, alpha = 0.80),
        #tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.01),
        metrics=[tf.keras.metrics.AUC()]
        #['binary_crossentropy',tf.keras.metrics.AUC()]
    )
    model.summary()


# In[23]:


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
class_weight = {0: 1, 1: 2}


# In[24]:


rng = [i for i in range(EPOCH)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


# In[25]:


history = model.fit(
    get_training_dataset(0), 
    epochs=EPOCH, 
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    class_weight=class_weight,
    validation_data=get_validation_dataset(0)
)


# In[26]:


#bce = history.history['binary_crossentropy']
#val_bce = history.history['val_binary_crossentropy']

auc = history.history['auc']
val_auc = history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(bce))

plt.plot(epochs, auc, 'b', label='Training AUC')
plt.plot(epochs, val_auc, 'r', label='Validation AUC')
plt.title('Training and validation AUC')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation Loss')
plt.legend()

#plt.figure()
 
#plt.plot(epochs, bce, 'b', label='Training BCE')
#plt.plot(epochs, val_bce, 'r', label='Validation BCE')
#plt.title('Training and validation BCE')
#plt.legend()


plt.show()


# In[27]:


get_ipython().system('mkdir -p /tmp/siim-model')


# In[28]:


print('test')
#/kaggle/temp/


# In[29]:


os.environ['KAGGLE_USERNAME'] ='rajnishe'
os.environ['KAGGLE_KEY'] = '35ac284ce25621f8c0caf11d4974879b'


# In[30]:


model.save('/tmp/siim-model/Bit-f1-focal-loss-152-epoch9.h5')


# In[31]:


get_ipython().system('ls -ltr /tmp/siim-model')


# In[32]:


data = '''{
  "title": "siim-bit-model-v4",
  "id": "rajnishe/siim-Bit-Model-v4",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
'''
text_file = open("/tmp/siim-model/dataset-metadata.json", 'w+')
n = text_file.write(data)
text_file.close()


# In[33]:


get_ipython().system('kaggle datasets create -p /tmp/siim-model')


# In[ ]:




