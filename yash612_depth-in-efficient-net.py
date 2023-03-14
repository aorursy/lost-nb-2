#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().system('pip install -q efficientnet')
import numpy as np
import pandas as pd 
import efficientnet.tfkeras as efn
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
import tensorflow as tf



# In[ ]:


img_size=384


# In[ ]:


def binary_focal_loss(gamma=2., alpha=.75):
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


# In[ ]:



def basic_model():
    inp=layers.Input(shape=(img_size,img_size,3),name='inp')
    efnetb3 = efn.EfficientNetB3(weights = 'imagenet', include_top = False)
    x=efnetb3(inp)
    output=layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [output])
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
        # opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt,loss = [binary_focal_loss(gamma = 2.0, alpha = 0.80)],metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    return model
    

    
    
    


# In[ ]:


def dense_added_model():
    inp=layers.Input(shape=(img_size,img_size,3),name='inp')
    efnetb3 = efn.EfficientNetB3(weights = 'imagenet', include_top = False)
    x=efnetb3(inp)
    x=layers.Dense(256,activation='relu')(x)
    x=layers.Dropout(0.6)(x)
    x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(0.6)(x)
    x=layers.Dense(64,activation='relu')(x)
    x=layers.Dropout(0.3)(x)
    output=layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [output])
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
        # opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt,loss = [binary_focal_loss(gamma = 2.0, alpha = 0.80)],metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    return model


# In[ ]:


def residual_block(y,nb_channels_in,nb_channels_out,strides=(1,1)):
    def conv_block(feat_maps_out, prev):
        y = layers.BatchNormalization(prev)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        return y
    def skip_block(feat_maps_in, feat_maps_out, prev):
        if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = layers.Conv2D(feat_maps_out,kernel_size=(1, 1), padding='same')(prev)
        return prev 
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(nb_channels_in,nb_channels_out, y)
    conv = conv_block(nb_channels_out,y)
    
    merger=layers.add([skip, conv])
    output = layers.LeakyReLU()(merger)
    return output
    
def eff_res():
    inp=layers.Input(shape=(img_size,img_size,3),name='inp')
    efnetb3 = efn.EfficientNetB3(weights = 'imagenet', include_top = False)
    x=efnetb3(inp)
    x=layers.GlobalAveragePooling2D()(x)
    x=residual_block(x,1536,512)
    x= layers.AveragePooling2D(pool_size=(4, 4))(x)
    x=layers.Flatten()(x)
    output=layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [output])
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
        # opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt,loss = [binary_focal_loss(gamma = 2.0, alpha = 0.80)],metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model
    
    
    

 
    


# In[ ]:




