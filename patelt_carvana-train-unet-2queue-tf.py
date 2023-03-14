#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# partial code for unet architecture is taken from https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py

import tensorflow as tf
import glob
import os
import cv2

# Input data files are available in the "../input/" directory.
from subprocess import check_output

WIDTH = 1840
HEIGHT = 1200


# In[3]:


def prepare_queue(input_dir, mask_dir):
    files = glob.glob(input_dir + "/*.jpg")
    base_files = []
    for file in files:
        base_files.append(os.path.basename(file).split(".")[0])

    base_tensor = tf.convert_to_tensor(base_files)
    
    input_queue = tf.train.string_input_producer(input_dir + base_tensor + ".jpg", shuffle=True, seed=123)
    mask_queue = tf.train.string_input_producer(mask_dir + base_tensor + "_mask.gif", shuffle=True, seed=123)
    
    return input_queue, mask_queue


# In[4]:


def read_input_image(input_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(input_queue)
    
    input_image = tf.image.decode_jpeg(value)
    
    input_image = tf.image.resize_images(input_image, (HEIGHT,WIDTH))
    input_image = tf.reshape(input_image, (HEIGHT,WIDTH,3))
#     input_image = tf.cast(input_image, dtype=tf.uint8)
    
    return key, input_image
    

def read_mask_image(mask_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(mask_queue)
    
    mask_image = tf.image.decode_gif(value)
    mask_image = tf.image.rgb_to_grayscale(mask_image)
    mask_image = tf.image.resize_images(mask_image, (HEIGHT,WIDTH))
    mask_image = tf.reshape(mask_image, (HEIGHT,WIDTH))
    return key, mask_image


# In[5]:


def get_input_mask_batch(input_queue, mask_queue):
    input_name, input_image = read_input_image(input_queue)
    mask_name, mask_image = read_mask_image(mask_queue)
    
    input_image_batch, mask_image_batch, input_name_batch, mask_name_batch =                 tf.train.shuffle_batch([input_image, mask_image, input_name, mask_name],                batch_size=1, capacity=4, min_after_dequeue=2)
    
    
    
    return input_image_batch, mask_image_batch, input_name_batch, mask_name_batch


# In[6]:


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


# In[7]:


def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


# In[8]:


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))
    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


# In[9]:


def make_unet(X, training):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
#     net = X / 127.5 - 1
    net = X
    net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool(pool4, [128, 128], training, name=5, pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)
    
    final = tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')
    
    return final


# In[10]:


def IOU_(y_pred, y_true):
    
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


# In[11]:


def make_train_op(y_pred, y_true, n_iteration):
    """Returns a training operation
    Loss function = - IOU(y_pred, y_true)
    IOU is
        (the area of intersection)
        --------------------------
        (the area of two boxes)
    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)
    Returns:
        train_op: minimize operation
    """
    starter_learning_rate = 0.001
    loss = -IOU_(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()
    
    learning_rate = tf.train.exponential_decay(starter_learning_rate,                         global_step,n_iteration, decay_rate=0.5, staircase=False)
    tf.summary.scalar("learning_rate", learning_rate)
    
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optim.minimize(loss, global_step=global_step, name="train_op")
    
    return train_op


# In[12]:


def train():
#     os.chdir("/home/tejas/Documents/selfstudy/carvana/")
#     checkpoint_dir = "../checkpoints/"
#     summary_dir = "../summary/"

    n_iteration = 1000
    tf.reset_default_graph()

    # prepare the queue for reading input and mask images
    input_queue, mask_queue = prepare_queue(input_dir = "../input/train/", mask_dir = "../input/train_masks/")
    input_image_batch, mask_image_batch, _, _ = get_input_mask_batch(input_queue, mask_queue)

    x = input_image_batch
    y = mask_image_batch

    pred = make_unet(x, training=True)
    
    tf.add_to_collection("inputs", x)
    tf.add_to_collection("outputs", pred)
    
    tf.summary.histogram("predicted_mask", pred)
    tf.summary.image("predicted_mask", pred)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y, n_iteration)

    IOU_op = IOU_(pred, y)
    IOU_op = tf.Print(IOU_op, [IOU_op])
    tf.summary.scalar("IOU", IOU_op)
    
    summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

#         saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)
#         summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

#         if os.path.exists(checkpoint_dir) and tf.train.checkpoint_exists(checkpoint_dir):
#             latest_check_point = tf.train.latest_checkpoint(checkpoint_dir)
#             saver.restore(sess, latest_check_point)
#         else:
#             try:
#                 os.rmdir(checkpoint_dir)
#             except FileNotFoundError:
#                 pass
#             os.mkdir(checkpoint_dir)

#         if not os.path.exists(summary_dir):
#             os.mkdir(checkpoint_dir)
        
        global_step = tf.train.get_global_step(sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(n_iteration):
            if epoch%2 == 0:
#                 iou_val, summary_val, global_step_val = sess.run([IOU_op, summary_op, global_step])
#                 saver.save(sess, checkpoints_dir, global_step=i)
#                 summary_writer.add_summary(summary_val, epoch)
                iou_val, global_step_val = sess.run([IOU_op, global_step]) #comment this line
                print(iou_val, global_step_val)
    
            train_op, _ = sess.run([train_op, global_step])
        
#         saver.save(sess, checkpoints_dir, global_step=i)
        
        coord.request_stop()
        coord.join(threads)


# In[13]:


if __name__ == '__main__':
    train()


# In[14]:




