#!/usr/bin/env python
# coding: utf-8

# In[1]:


#References
#https://www.groundai.com/project/environment-sound-classification-using-multiple-feature-channels-and-deep-convolutional-neural-networks/1
#https://keunwoochoi.wordpress.com/2019/09/28/log-melspectrogram-layer-using-tensorflow-keras/


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os     
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import os
import librosa
import librosa.display
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
#!pip install python_speech_features
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import glob
import glob
import librosa
from librosa import feature
import numpy as np
from pathlib import Path
import cv2
AUTO = tf.data.experimental.AUTOTUNE
from kaggle_datasets import KaggleDatasets
import scipy
import pickle
from sklearn.model_selection import train_test_split
import time
import statistics


# In[3]:


# Detect hardware, return appropriate distribution strategy
def get_strategy():
    TFREC_GCS_DIR = ""
    gpu = ""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())     
    except ValueError:
        tpu = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu = tf.config.list_physical_devices("GPU")
        if len(gpu) == 1:
            print('Running on GPU ', gpu)
    if tpu:
        print("Running in TPU")
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        GCS_PATH = KaggleDatasets().get_gcs_path('birdsong-recognition')
        TFREC_GCS_DIR = KaggleDatasets().get_gcs_path('birdcall-tfrec-2s')
        print()
    elif len(gpu) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision":True})
        GCS_PATH = "/kaggle/input/birdsong-recognition/"
    else:
        strategy = tf.distribute.get_strategy()
        GCS_PATH = "/kaggle/input/birdsong-recognition/"

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    base_dir = "../input/birdsong-recognition/"
    print(base_dir)
    return strategy, GCS_PATH, base_dir, TFREC_GCS_DIR

strategy,GCS_PATH, base_dir, TFREC_GCS_DIR = get_strategy()
sns.set_palette("pastel")
palette = sns.color_palette()
CACHE = {}
tfrec_dir = "../input/birdcall-tfrec-2s/"


# In[ ]:





# In[4]:


def get_ebird_filename_dic():
    ebird_code_list = all_train_data["ebird_code"].unique()
    if os.path.exists(tfrec_dir + "dic_ebird.pkl"):
        with open(tfrec_dir + "dic_ebird.pkl","rb") as f:
            dic_ebird_code = pickle.load(f)
    else:
        dic_ebird_code = {k:v for v,k in enumerate(ebird_code_list)}
    
    dic_ebird_code_rev = [v for v,k in dic_ebird_code.items()]
    all_train_data["int_ebird_code"] = all_train_data["ebird_code"].map(dic_ebird_code)

    filename_list = all_train_data["filename"].unique()
    if os.path.exists(tfrec_dir + "dic_filename.pkl"):
        with open(tfrec_dir + "dic_filename.pkl","rb") as f:
            dic_filename = pickle.load(f)
    else:
        dic_filename = {k:v for v,k in enumerate(filename_list)}
    dic_filename_rev = [v for v,k in dic_filename.items()]
    all_train_data["int_filename"] = all_train_data["filename"].map(dic_filename)

    with open("dic_ebird.pkl","wb") as f:
        pickle.dump(dic_ebird_code, f)

    with open("dic_filename.pkl","wb") as f:
        pickle.dump(dic_filename, f)
        
    return dic_ebird_code_rev, dic_filename_rev


# In[5]:


def parse_rec_train(data):           
    feature_set = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'int_ebird_code': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(data, features= feature_set )
    img = features["img"]
    img = tf.image.decode_image(img)
    img = tf.ensure_shape(img, (img_sz1, img_sz2, 1))
    y = features['int_ebird_code']
    return tf.cast(img, tf.float32), tf.one_hot(y, 264)


def parse_waveform_rec_train(data):           
    feature_set = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'int_ebird_code': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(data, features= feature_set )
    img = features["img"]
    img = tf.image.decode_image(img)
    img = tf.ensure_shape(img, (img_sz1, img_sz2, 1))
    y = features['int_ebird_code']
    return tf.cast(img, tf.float32), tf.one_hot(y, 264)


# In[6]:


all_train_data = pd.read_csv(base_dir + "train.csv")
all_train_data = all_train_data[all_train_data["duration"]>=5]
all_train_data = all_train_data[all_train_data["filename"]!="XC195038.mp3"]


# In[7]:


img_sz1 = 64
img_sz2 = 512


# In[ ]:





# In[8]:


dic_ebird_code_rev, dic_filename_rev = get_ebird_filename_dic()


# In[9]:


def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.000020   * strategy.num_replicas_in_sync
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


# In[10]:


from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import GlobalMaxPooling2D, Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras import Model

import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[11]:


if 1==2:
    get_ipython().system('pip install soundfile')
    get_ipython().system('git clone https://github.com/tensorflow/models.git')
    get_ipython().run_line_magic('cd', 'models/research/audioset/yamnet')

    # Download YAMNet data
    get_ipython().system('curl -O https://storage.googleapis.com/audioset/yamnet.h5')


    # Imports.
    import numpy as np
    import soundfile as sf

    import matplotlib.pyplot as plt

    import params
    import yamnet as yamnet_model
    import tensorflow as tf


# In[12]:


class Params:
    sample_rate: float = 16000.0
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    patch_hop_seconds: float = 0.48

    @property
    def patch_frames(self):
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))

    @property
    def patch_bands(self):
        return self.mel_bands

    num_classes: int = 521
    conv_padding: str = 'same'
    batchnorm_center: bool = True
    batchnorm_scale: bool = False
    batchnorm_epsilon: float = 1e-4
    classifier_activation: str = 'sigmoid'

    tflite_compatible: bool = False


# In[13]:


def get_yamnet_model(params):
    # Install required packages.
    
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')

    x = tf.keras.layers.Dense(1024, activation='relu')(yamnet.layers[-3].output)
    o = tf.keras.layers.Dropout(0.5)(x)
    o = tf.keras.layers.Dense(264, activation='softmax')(o)

    model = Model(inputs=yamnet.input, outputs=o)

    for layer in model.layers:
        layer.trainable = True


    checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss', 
                                 verbose=1,
                                 save_best_only=True, 
                                 mode='auto')

    reducelr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.5, 
                                  patience=3, 
                                  verbose=1)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.CategoricalCrossentropy() 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    
#yamnet = get_yamnet_model(Params())


# In[14]:


if 1==2:
    with strategy.scope():
        if 1==2:
            base_model = ResNet50(weights='imagenet', include_top=False)

            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalMaxPooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='tanh')(x)
            x = Dropout(0.2)(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.2)(x)
            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(264, activation='sigmoid')(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])
            #model.summary()
            callback = get_lr_callback(BATCH_SIZE)
            sv = tf.keras.callbacks.ModelCheckpoint(
                'model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=False, mode='min', save_freq='epoch')

            model.fit(train_dataset, epochs=5, verbose=1, callbacks=[sv, callback], steps_per_epoch=200, validation_data = valid_dataset, validation_steps=10)    #steps_per_epoch=steps_per_epoch, 
        else:
            model = tf.keras.models.load_model('../input/multichannel-spectrogram/model.h5')
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])


# In[15]:


def get_conv_block(net, filters, kernel_size, activation="relu"):
    net = Conv2D(filters=RESNET_K*FILTERS[0], kernel_size=KERNEL_SIZES[0], padding='same')(net)
    net = BatchNormalization()(net)
    if activation is not None:
        net = Activation("relu")(net)
    return net

def resblock(net_in, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):
    # Show input shape
    #log.p(("\t\t" + name + " IN SHAPE:", l.get_output_shape(net_in)), new_line=False)

    # Pre-activation
    if block_id > 1:
        net_pre = Activation("relu")(net_in)
    else:
        net_pre = net_in

    # Pre-activated shortcut?
    if preactivated:
        net_in = net_pre

    # Bottleneck Convolution
    if stride > 1:
        net_pre = get_conv_block(net_pre, net_pre.shape[1], 1)
       
    # First Convolution     
    net = get_conv_block(net_pre, net_pre.shape[1], kernel_size)

    # Pooling layer
   
    if stride > 1:
        net = MaxPooling2D(pool_size=(stride, stride))(net)

    # Dropout Layer
    net = Dropout(0.5)(net)       

    # Second Convolution
    net = get_conv_block(net, filters, kernel_size)

    # Shortcut Layer
    if not net.shape == net_in.shape:
        # Average pooling
        shortcut = AveragePooling2D(pool_size=(stride, stride))(net_in)

        # Shortcut convolution
        shortcut = get_conv_block(shortcut, filters, 1, None)  
    else:

        # Shortcut = input
        shortcut = net_in
    
    # Merge Layer
    out = tf.math.add_n([net, shortcut])

    # Show output shape
    #log.p(("OUT SHAPE:", l.get_output_shape(out), "LAYER:", len(l.get_all_layers(out)) - 1))

    return out


# In[16]:


def classificationBranch(net, kernel_size):

    # Post Convolution
    branch = get_conv_block(net, int(FILTERS[-1] * RESNET_K), kernel_size)  

    #log.p(("\t\tPOST  CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))

    # Dropout Layer
    branch = Dropout(0.5)(branch)
    
    # Dense Convolution
    branch = get_conv_block(net, int(FILTERS[-1] * RESNET_K * 2), 1) 

    #log.p(("\t\tDENSE CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))
    
    # Dropout Layer
    branch = Dropout(0.5)(branch)
    
    # Class Convolution
    branch = Conv2D(filters=CLASSES,kernel_size=1)(branch)
    return branch


# In[17]:


import keras.backend as K
def f1_loss(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# In[18]:


CLASSES=264
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
RESNET_K = 4
RESNET_N = 3

def buildNet(fold, model_path=None):
    
    inp = Input(shape=(img_sz1, img_sz2, 1))
    
    net = get_conv_block(inp, RESNET_K*FILTERS[0], KERNEL_SIZES[0])
    net = MaxPooling2D(pool_size=(1,2), padding="same")(net)
    for i in range(1, len(FILTERS)):
            #log.p(("\tRES STACK", i, ':'))
            net = resblock(net,
                           filters=int(FILTERS[i] * RESNET_K),
                           kernel_size=KERNEL_SIZES[i],
                           stride=2,
                           preactivated=True,
                           block_id=i,
                           name='BLOCK ' + str(i) + '-1')

            for j in range(1, RESNET_N):
                net = resblock(net,
                               filters=int(FILTERS[i] * RESNET_K),
                               kernel_size=KERNEL_SIZES[i],
                               preactivated=False,
                               block_id=i+j,
                               name='BLOCK ' + str(i) + '-' + str(j + 1))
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    # Classification branch
    #log.p(("\tCLASS BRANCH:"))
    net = classificationBranch(net,  (4, 10)) 
    #log.p(("\t\tBRANCH OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Pooling
    net = GlobalAveragePooling2D()(net)
    #log.p(("\tGLOBAL POOLING SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Sigmoid output
    net = Activation("softmax")(net)
    print(net.shape)
    model = Model(inputs=inp, outputs=net)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    #model.summary()
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
    model_file = "model_" + str(fold) + ".h5"
    if model_path is None:
        model_path = "../input/multichannel-spectrogram/"
    if os.path.exists(model_path + model_file):
        print("Found model!", model_path + model_file)
        model.load_weights(model_path + model_file)
        blnTrain = False
    else:
        blnTrain = True
    
    sv = tf.keras.callbacks.ModelCheckpoint(
        model_file, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    return model, [learning_rate_reduction, sv, es], model_file, True


# In[19]:


all_train_data.head(1)


# In[20]:


def get_train_val_ds(train_idx, val_idx, train_val_data):
    dic_train = get_dic_for_data(train_idx, train_val_data)
    dic_val = get_dic_for_data(val_idx, train_val_data)
    
    train_dataset_raw = tf.data.Dataset.from_generator(
         DataGenerator, args=[dic_train], \
         output_types=(tf.float32, tf.int64),
         output_shapes=(tf.TensorShape((img_sz1, img_sz2, 1)), tf.TensorShape((264))))
    
    valid_dataset_raw = tf.data.Dataset.from_generator(
         DataGenerator, args=[dic_val],\
         output_types=(tf.float32, tf.int64),
         output_shapes=(tf.TensorShape((img_sz1, img_sz2, 1)), tf.TensorShape((264) )))


    

    train_dataset = train_dataset_raw.batch(BATCH_SIZE).prefetch(AUTO)
    valid_dataset = valid_dataset_raw.batch(BATCH_SIZE).prefetch(AUTO)
    for X,y in train_dataset.take(1):
       print(X.shape)

    fix, ax = plt.subplots(2,3, figsize=(18,3))
    for row in range(2):
        for col in range(3):
            ax[row,col].imshow(tf.reshape(X[row*col + col], (img_sz1, img_sz2)))
            
    return train_dataset, valid_dataset


# In[21]:


def plot_metrics(num_data, metric_name, color_train, color_valid, loc, min_max_arg_func, min_max_func, train_metric_data, val_metric_data):
    plt.plot(np.arange(num_data),train_metric_data,'-o',label=metric_name,color=color_train)
    plt.plot(np.arange(num_data),val_metric_data,'-o',label='Val ' + metric_name ,color=color_valid)
    x = min_max_arg_func( val_metric_data); y = min_max_func( val_metric_data )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color=color_valid); plt.text(x-0.03*xdist,y-0.13*ydist,'min/max %s\n%.2f'%(metric_name, y),size=14)
    plt.ylabel(metric_name,size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=loc)
    
def get_f1(history, precision_key, recall_key):
    train_f1 = []
    train_precision = history.history[precision_key]
    train_recall = history.history[recall_key]
    for i in range(len(train_precision)):
        f1 = statistics.harmonic_mean([train_precision[i], train_recall[i]])
        train_f1.append(f1)
        
    valid_f1 = []
    valid_precision = history.history["val_" + precision_key]
    valid_recall = history.history["val_" + recall_key]
    for i in range(len(valid_precision)):
        f1 = statistics.harmonic_mean([valid_precision[i], valid_recall[i]])
        valid_f1.append(f1)
        
    return train_f1, valid_f1
        
        
        
    
def plot_history(history):
    num_data = len(history.history['loss'])
    plt.figure(figsize=(15,5))
    
    for key in history.history.keys():
        if 'precision' in key and 'val' not in key:
            precision_key = key
            
    for key in history.history.keys():
        if 'recall' in key and 'val' not in key:
            recall_key = key
            
    if 1==2:
        plot_metrics(num_data, 'precision', '#ff7f0e', '#1f77b4', 2, np.argmax, np.max, history.history[precision_key], history.history['val_' + precision_key])

        plt2 = plt.gca().twinx()

        plot_metrics(num_data, 'recall', '#2ca02c', '#d62728', 3, np.argmax, np.max, history.history[recall_key], history.history['val_' + recall_key])
   
    plt.show()  
    
    train_f1, valid_f1 = get_f1(history, precision_key, recall_key)
        
        
    plot_metrics(num_data, 'F1', '#ff7f0e', '#1f77b4', 2, np.argmax, np.max, train_f1, valid_f1)
    
    plt2 = plt.gca().twinx()
    
    plot_metrics(num_data, 'loss', '#2ca02c', '#d62728', 3, np.argmin, np.min, history.history["loss"], history.history['val_' + "loss"])
   
    plt.show();


# In[22]:


def get_file_list():
    dataset_list = ["birdcall-tfrec-2s", "birdcall-tfrec-2s-28","birdcall-tfrec-2s-34","birdcall-tfrec-2s-40",
                   "birdcall-tfrec-2s-50","birdcall-tfrec-2s-60"]
    file_list = []
    for dataset in dataset_list:
        gcs_dir = KaggleDatasets().get_gcs_path(dataset)
        gcs_file = [file for file in tf.io.gfile.glob(TFREC_GCS_DIR + '/train*') if ".pkl" not in file  if ".csv" not in file ]
        file_list = file_list + gcs_file

    tfrec_files_train_all = np.sort(np.array(file_list))
    return tfrec_files_train_all
    


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
BLN_TRAIN = False

KFOLD_SPLITS = 10
BATCH_SIZE = 128 * strategy.num_replicas_in_sync

if BLN_TRAIN:
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True,random_state=42)
    
    tfrec_files_train_all = get_file_list()
    dataset_train = tf.data.TFRecordDataset(tfrec_files_train_all).map(parse_rec_train, num_parallel_calls=AUTO)
    for X,y in dataset_train.skip(14).take(1):
        plt.imshow(255 * X[:,:,0])

    for fold , (train_idx, val_idx) in enumerate(kf.split(list(range(len(tfrec_files_train_all))))):
        if fold < 1:
            train_dataset_raw = tf.data.TFRecordDataset(tfrec_files_train_all[train_idx]).map(parse_rec_train, num_parallel_calls=AUTO).cache()
            valid_dataset_raw = tf.data.TFRecordDataset(tfrec_files_train_all[val_idx]).map(parse_rec_train, num_parallel_calls=AUTO)
            train_dataset = train_dataset_raw.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO)
            valid_dataset = valid_dataset_raw.cache().batch(BATCH_SIZE).prefetch(AUTO)
            with strategy.scope():
                model, callbacks, model_file, bln_train = buildNet(fold)
            if bln_train:
                history = model.fit(train_dataset, epochs=25, verbose=1, callbacks=callbacks, validation_data = valid_dataset)    #steps_per_epoch=steps_per_epoch, 
                model.load_weights(model_file)
                plot_history(history)
            else:
                model.save_weights(model_file)


# In[24]:


KFOLD_SPLITS = 1


# In[25]:



n_fft1 = int(0.0025 * 22050)
hop_length1 = int(0.001 * 22050)

n_fft2 = int(0.005 * 22050)
hop_length2 = int(0.0025 * 22050)

n_fft3 = int(0.01 * 22050)
hop_length3 = int(0.005 * 22050)
n_mels = 128
fmin=150
fmax=15000
img_sz1 = 64
img_sz2 = 512



    
def buildBandpassFilter(rate, fmin, fmax, order=4):

    global CACHE

    fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')
        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]

def applyBandpassFilter(sig, rate, fmin, fmax):
    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate, fmin, fmax)

    return scipy.signal.sosfiltfilt(filter_sos, sig)

def get_mel_filterbanks(num_banks, fmin, fmax, f_vec, dtype=np.float32):
    '''
    An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
    an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
    flooring down the bin indices.
    '''

    global CACHE

    # Filterbank already in cache?
    fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        
        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

        # Compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # Convert mels to Hz
        banks_ends = (f_break * (10 ** (melpoints / A) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) /                 (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) /                 (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]

def get_spectrogram(sig, rate, shape=(img_sz1, img_sz2), win_len=512, fmin=150, fmax=15000, magnitude_scale='nonlinear', bandpass=True, decompose=False):

    # Compute overlap
    hop_len = int(len(sig) / (shape[1] - 1)) 
    win_overlap = win_len - hop_len + 2
    #print 'WIN_LEN:', win_len, 'HOP_LEN:', hop_len, 'OVERLAP:', win_overlap

    
    n_fft = win_len
    

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig, rate, fmin, fmax)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(win_len),
                                          nperseg=win_len,
                                          noverlap=win_overlap,
                                          nfft=n_fft,
                                          detrend=False,
                                          mode='magnitude')

    # Scale frequency?
   

    # Determine the indices of where to clip the spec
    valid_f_idx_start = f.searchsorted(fmin, side='left')
    valid_f_idx_end = f.searchsorted(fmax, side='right') - 1

    # Get mel filter banks
    mel_filterbank, mel_f = get_mel_filterbanks(shape[0], fmin, fmax, f, dtype=spec.dtype)

    # Clip to non-zero range so that unnecessary multiplications can be avoided
    mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

    # Clip the spec representation and apply the mel filterbank.
    # Due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
    spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
    spec = np.dot(spec, mel_filterbank)
    spec = np.transpose(spec, [1, 0])        

    # Magnitude transformation
    if magnitude_scale == 'pcen':
        
        # Convert scale using per-channel energy normalization as proposed by Wang et al., 2017
        # We adjust the parameters for bird voice recognition based on Lostanlen, 2019
        spec = pcen(spec, rate, hop_len)
        
    elif magnitude_scale == 'log':
        
        # Convert power spec to dB scale (compute dB relative to peak power)
        spec = spec ** 2
        spec = 10.0 * np.log10(np.maximum(1e-10, spec) / np.max(spec))
        spec = np.maximum(spec, spec.max() - 100) # top_db = 100

    elif magnitude_scale == 'nonlinear':

        # Convert magnitudes using nonlinearity as proposed by Schlüter, 2018
        a = -1.2 # Higher values yield better noise suppression
        s = 1.0 / (1.0 + np.exp(-a))
        spec = spec ** s

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()
    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)
    spec = (spec * 255).astype(np.int64)

    return spec


# In[26]:


### Create predictions
last_file_name = "" 
last_data = []
last_sr = 0
def load_test_clip(path, start_time):
    global last_file_name, last_data, last_sr
    try:
        path = path + "*.*"
        #path = "../input/birdsong-recognition/train_audio/bkhgro/XC109305.mp3"
        path = [file for file in glob.glob(path)][0]
        if last_file_name != path:
            data, sr = librosa.load(path, sr=48000, mono=True)
            last_file_name = path
            last_data = data
            last_sr = sr
        start = int(start_time*last_sr)
        end = start + int(2*last_sr)
        data = last_data[start:end]
        print(data[0:10])
        return data, last_sr
    except Exception as e:
        print("Exception:", e)
        return None, 0
    

dic_model = {}
def make_prediction(block, sr):
    test_feature_data = get_spectrogram(block, sr)
    test_feature_data = test_feature_data.reshape(1,img_sz1, img_sz2, 1)
    list_pred = []
    
    for i in range(KFOLD_SPLITS):
        if i not in dic_model.keys():
            model, _, _, _ = buildNet(i)
            dic_model[i] = model
        else:
            model = dic_model[i]
        list_pred.append(model.predict(test_feature_data))
    
        
    pred = np.stack(list_pred, axis=1).mean(axis=1)
    return pred[0]


# In[27]:


TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
if not os.path.exists(TEST_FOLDER):
    TEST_FOLDER = "../input/birdsong-recognition/example_test_audio/"
    test_info = pd.read_csv('../input/birdsong-recognition/example_test_audio_summary.csv').tail(50)
    test_info["site"] = "site_1"
    test_info.rename(columns={"filename_seconds":"row_id","filename":"audio_id"}, inplace=True)
else:
    test_info = pd.read_csv('../input/birdsong-recognition/test.csv')
test_info.head(50)


# In[28]:


def get_clip_pred(path, start_time):
    sound_clip, sr = load_test_clip(TEST_FOLDER + audio_id , start_time)
    if sr == 0:
        pred = np.zeros((264))
        print("Zero:", pred.shape)
    else:
        pred = make_prediction(sound_clip, sr)
        pred_class = np.argmax(pred)
        pred_prob = pred[pred_class]
        pred = pred * 0
        if pred_prob > 0.5:
            pred[pred_class] = 1
    return pred
        
def get_file_pred(path, start_time, duration):
    if duration is None:
        duration = get_file_duration(path)
    list_pred = []
    start_time = 0
    while start_time <= duration-2:
        pred = get_clip_pred(path, start_time)
        list_pred.append(pred)
        start_time = start_time + 1.5
    pred = np.stack(list_pred, axis=1).sum(axis=1)
    return pred


def get_file_duration(path):
    #if not os.path.exists(TEST_FOLDER):
    #    path = base_dir + "train_audio/aldfly/XC134874.mp3"
    data, sr = librosa.load(path, mono=True)
    duration = len(data)//sr
    return duration


# In[29]:


#try:
if 1==1:
    preds = []
    for index, row in test_info.iterrows():
        site = row['site']
        start_time = row['seconds'] - 5
        row_id = row['row_id']
        audio_id = row['audio_id']
        path = TEST_FOLDER + audio_id + '.mp3'
       
        if (site == 'site_1' or site == 'site_2'):
            pred = get_file_pred(path, start_time, 5)
        else:
            pred = get_file_pred(path, 0, None)

        preds.append([row_id, pred])

    preds = pd.DataFrame(preds, columns=['row_id', 'pred'])
    preds["pred2"] = preds["pred"].map(lambda x: [i for i in range(x.shape[0]) if x[i]>0])
    preds["birds"] = preds["pred2"].map(lambda x: " ".join(list(np.sort([dic_ebird_code_rev[i] for i in x]))))
    preds["birds"] = preds["birds"].map(lambda x: "nocall" if x=="" else x)

    preds[["row_id","birds"]].to_csv('submission.csv', index=False)
#except Exception as e:
if 1==2:
    e = None
    print("exception",e)
    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
    preds[["row_id","birds"]].head(1).to_csv('submission.csv', index=False)


# In[30]:


preds.head(50)


# In[31]:


preds.tail(50)


# In[ ]:





# In[ ]:




