#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import dicom
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

INPUT_FOLDER = '../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)

patients[0]


# In[2]:


label = labels_df.get_value(patients[0], 'cancer')


# In[3]:


# load dicom files and add slice thickness

def load_slices(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try: 
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices


# In[4]:


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(int16)

    image[ image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slices.number].RescaleSclope
    
        if slope!=1:
            image[slice_number] = slope*image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        
    
    return np.array(image, dtype = np.int16)


# In[5]:


def resample(image, scan, new_spacing=[1,1,1]):
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode = 'nearest')
    
    return image, new_spacing
    


# In[6]:


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


# In[7]:


PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


# In[8]:


x = tf.placeholder('float')
y = tf.placeholder('float')

def conv_3d(x, W):
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], ksize, padding = 'SAME')

def max_pool3d(x):
    return tf.nn.maxpool3d(x, ksize = [1,2,2,2,1], ksize = [1,2,2,2,1], padding = 'SAME')

def convolutionl_3d(x):
    
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1)),
               'W_conv2':tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1)),
               'W_fc':tf.Variable(tf.truncated_normal([50480,1024], stddev = 0.1)),
               'out':tf.Variable(tf.truncated_normal([1024, n_classes], stddev = 0.1))}
    
    biases = { 'b_conv1':tf.Variable(tf.constant(0.1, shape = [32])),
               'b_conv2':tf.Variable(tf.constant(0.1, shape = [64])),
               'b_fc':tf.Variable(tf.constant(0.1, shape = [1024])),
               'out':tf.Variable(tf.constant(0.1, shape = [n_classes]))}
    
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 50480])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output
    


# In[9]:


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    sess= tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_xs, y: batch_ys})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    
    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y : mnist.test.labels}))


train_neural_network(x)

