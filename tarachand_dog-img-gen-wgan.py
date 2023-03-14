#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, zipfile
import matplotlib.image as mpimg

import tensorflow as tf

import os
import sys
from tqdm import tqdm, tqdm_notebook
import shutil

# import xml.etree.ElementTree as ET
import time # time the execution of codeblocks
# from IPython.display import FileLink, FileLinks


# In[2]:


os.listdir('../input/')
path = ('../input/')
img_output = path + 'img_output'

try:
    if os.path.exists(img_output):
        shutil.rmtree(img_output)
    if not os.path.exists(img_output):
        os.mkdir(img_output)
except:
    print('Unable to create the directory' + "'"+ img_output + "'")
    print('Permission Denied!')


# In[3]:


# List of dog breeds from annotation
dog_breeds = os.listdir(path + 'annotation/Annotation')
print('total number of breeds %d'%len(dog_breeds))
# pd.Series(dog_breeds).value_counts()
dog_breeds = pd.DataFrame({'Annotation': dog_breeds})
dog_breeds.loc[:,'Breed_Code'] = dog_breeds['Annotation'].apply(lambda x: x.split('-')[0])
dog_breeds.loc[:,'Breed_Name'] = dog_breeds['Annotation'].apply(lambda x: x.split('-')[1])
dog_breeds.head()


# In[4]:


dog_img = os.listdir( path + 'all-dogs/all-dogs')
dog_img = pd.DataFrame({'Image_Name': dog_img})
dog_img.loc[:,'Breed_Code'] = dog_img['Image_Name'].apply(lambda x: x.split('_')[0])
dog_img.loc[:,'Image_Num'] = dog_img['Image_Name'].apply(lambda x: x.split('_')[1]).apply(lambda x: x.split('.')[0])
dog_img.loc[:,'Img_Path'] = dog_img['Image_Name'].apply(lambda x: path + 'all-dogs/all-dogs/' + x)
dog_img.head()


# In[5]:



import random
from PIL import Image
def plot_random_image(img_input, path, n_img = None, rows = None, img_size = None):
    img_rand = random.sample(set(img_input), n_img)
    cols = np.floor(len(img_rand)/rows)
    plt.figure(figsize = (12,10))
    for num, x in enumerate(img_rand):
        img = Image.open(x)
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        plt.subplot(rows, cols, num + 1)
        plt.axis('off')
        plt.imshow(img)
        plt.tight_layout()
#         plt.show();
#         if not os.path.exists(path + 'sample_dog_img/'):
#             os.makedirs(path + 'sample_dog_img/')
#         img.save(path + 'sample_dog_img/' + 'dog_' + str(num) + '.jpg')

plot_random_image(dog_img.Img_Path.values, path, n_img = 10, rows = 2, img_size = 256)


# In[6]:


###############################################################################
# Function to read image, resize and output image in sample batch
###############################################################################

# def image_batch_process(image_path = dog_img['Img_Path'].values):
def image_batch_process():
    img_path_tf = tf.convert_to_tensor(dog_img['Img_Path'].values, dtype = tf.string)   
    img_path_tf = tf.train.slice_input_producer([img_path_tf])
    raw_img = tf.read_file(img_path_tf[0])

    img = tf.image.decode_jpeg(raw_img, channels = CHANNEL)

    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_brightness(img, max_delta = 0.1)
    # img = tf.image.random_contrast(img, lower = 0.9, upper = 1.1)

    img = tf.image.resize_images(img, size = [HEIGHT, WIDTH])
    img.set_shape([HEIGHT, WIDTH, CHANNEL])

    img = tf.cast(img, tf.float32)
    img = img / 255.0
    
    min_after_dequeue = 10000
    capacity = min_after_dequeue + (THREADS + 1) * BATCH_SIZE

    img_batch = tf.train.shuffle_batch([img], batch_size = BATCH_SIZE, 
                    num_threads = 10, capacity = capacity,
                    min_after_dequeue = min_after_dequeue)
    n_img = len(dog_img['Img_Path'].values)
    return img_batch, n_img


# In[7]:


###############################################################################
# Function to train the Generator
###############################################################################

# Generator Function
def generator(input, RANDOM_DIM, is_train):
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen', reuse = tf.AUTO_REUSE):
        
        # Layer 1 = Input * W1 + b1 --> batch norm --> lrelu
        gen_w1 = tf.get_variable('gen_w1', shape = [RANDOM_DIM, 4 * 4 * 256], dtype = tf.float32,                             initializer = tf.truncated_normal_initializer(stddev = 0.2))
        gen_b1 = tf.get_variable('gen_b1', shape = [256 * 4 * 4], dtype = tf.float32,                             initializer = tf.constant_initializer(0.001))
        layer_1 = tf.add(tf.matmul(input, gen_w1), gen_b1, name = 'layer_1')
        
        # Add Convolution layer1 --> batch norm --->relu
        gen_conv1 = tf.reshape(layer_1, shape=[-1, 4, 4, 256], name = 'conv1')
        gen_act1 = tf.nn.relu(gen_conv1, name = 'gen_act1')

        # Add Convolution Layer2 --> batch norm --> relu
        gen_conv2 = tf.layers.conv2d_transpose(gen_act1, 128, kernel_size = [5, 5], strides=[2, 2], padding="same",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='gen_conv2')        
        gen_act2 = tf.nn.relu(gen_conv2, name = 'gen_act2')

        # Add Convolution Layer3 --> batch norm --> relu
        gen_conv3 = tf.layers.conv2d_transpose(gen_act2, 64, kernel_size=[5, 5], strides=[2, 2], padding="same",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name = 'gen_conv3')
        gen_act3 = tf.nn.relu(gen_conv3, name='gen_act3')
        
        # Add Convolution Layer4 --> batch norm --> relu
        gen_conv4 = tf.layers.conv2d_transpose(gen_act3, 32, kernel_size=[5, 5], strides=[2, 2], padding="same",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='gen_conv4')
        gen_act4 = tf.nn.relu(gen_conv4, name = 'gen_act4')
    
        # Add Convolution Layer5 --> tanh
        gen_conv5 = tf.layers.conv2d_transpose(gen_act4, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="same",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='gen_conv5')
        gen_act5 = tf.nn.tanh(gen_conv5, name = 'gen_act5')
        print('Gen Activation 5 shape:', gen_act5.shape)
        return gen_act5


# In[8]:


###############################################################################
# Function to train the Discriminator
###############################################################################

# Discriminator Function
def discriminator(input, is_train, reuse = False):
    print('Inside Discriminator...')
    with tf.variable_scope('dis', reuse = tf.AUTO_REUSE):
#     with tf.variable_scope('dis') as scope:
#         if reuse:
#             scope.reuse_variables()

        # Add Convolution layer1 --> batch norm --->relu
        dis_conv1 = tf.layers.conv2d(input, 64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='dis_conv1')        
        dis_act1 = tf.nn.leaky_relu(dis_conv1, name='dis_act1')

        # Add Convolution Layer2 --> batch norm --> relu
        dis_conv2 = tf.layers.conv2d(dis_act1, 128, kernel_size=[5, 5], strides=[2, 2], padding="SAME",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='dis_conv2')
        dis_act2 = tf.nn.leaky_relu(dis_conv2, name='dis_act2')

        # Add Convolution Layer3 --> batch norm --> relu
        dis_conv3 = tf.layers.conv2d(dis_act2, 256, kernel_size=[5, 5], strides=[2, 2], padding="SAME",                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),                                           name='dis_conv3')
        dis_act3 = tf.nn.leaky_relu(dis_conv3, name='dis_act3')

        # Here we are re-shaping the tensor to feed into Wasserstein function
        # to check and compete between - Fake and Original image by generator and discriminator
        dim = np.int(np.prod(dis_act3.get_shape()[1:]))
        conv_vect = tf.reshape(dis_act3, shape = [-1, dim], name = 'conv_vect')

        # Weight Initialization
        dis_w2 = tf.get_variable('dis_w2', shape = [conv_vect.shape[-1], 1], dtype = tf.float32,                             initializer=tf.truncated_normal_initializer(stddev = 0.02))
        dis_b2 = tf.get_variable('dis_b2', shape = [1], dtype = tf.float32,                             initializer = tf.constant_initializer(0.001))

        # Wasserstein GAN - wgan to caculate Wasserstein Distance to check generator 
        dis_wgan = tf.add(tf.matmul(conv_vect, dis_w2), dis_b2, name = 'dis_wgan')
        print('Shape of Linear Wgan:', dis_wgan.shape)

        return dis_wgan


# In[9]:


###############################################################################
# Function to train the Discriminator/Generator Model
###############################################################################

# Main Train function
def train():

    with tf.variable_scope('inputs', reuse = tf.AUTO_REUSE):
        real_data = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name = 'real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, RANDOM_DIM], name = 'rand_input')
        is_train = tf.placeholder(tf.bool, name = 'is_train')
    
    # Generator function call
    fake_data = generator(random_input, RANDOM_DIM, is_train = True)
    
    # Training two discriminators to discriminate both real and fake data
    real_result = discriminator(real_data, is_train)
    fake_result = discriminator(fake_data, is_train, reuse = True)
    
    # Loss functions of Discriminator and Generator
    # d_loss --> will optimize discriminator
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    
    # g_loss --> will optimize generator
    g_loss = -tf.reduce_mean(fake_result)
            
    # Getting the list of trainable variables/weights from generator and descriminator
    t_vars = tf.trainable_variables()

    # List of Trainable Variables

    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
#    print('Trainable variables in Dis: ', d_vars)
#    print('Trainable variables in Gen: ', g_vars)
    
# running optimization on trainble variables
    with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
#         trainer_d = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, \
#                        name = 'dis_rmsprop').minimize(d_loss, var_list = d_vars)
        
        trainer_d = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE, #                     beta1 = 0.5, beta2 = 0.9,\
                    name = 'dis_adam').minimize(d_loss, var_list = d_vars)
        
#         trainer_g = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, \
#                        name = 'gen_rmsprop').minimize(g_loss, var_list = g_vars)
        
        trainer_g = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE,                      #   beta1 = 0.5, beta2 = 0.9,\
                        name = 'gen_adam').minimize(g_loss, var_list = g_vars)
    
    # clip discriminator weights. Restricting the weights of discriminator
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    # Fetch image data from image process function
    img_batch, samples_num = image_batch_process()
    batch_num = int(samples_num / BATCH_SIZE)
    
    # TF Session start
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    print('\ntotal training sample:%d' % samples_num)
    print('\nbatch size: %d, batch num per epoch: %d, epoch num: %d'           % (BATCH_SIZE, batch_num, EPOCH))
    print('\nstart training...')
    
    # List to store avg generator and discriminator loss
    avg_gen_loss = []
    avg_dis_loss = []
    for i in range(EPOCH):
        print("\nRunning epoch {}/{}...".format(i, EPOCH))
        
        gen_loss = []
        dis_loss = []
        for j in range(batch_num):
#             print('Training on Batch number %d'%j)
            d_iters = 3
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size = [BATCH_SIZE,                                             RANDOM_DIM]).astype(np.float32)

#            train_noise = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, \
#                                            RANDOM_DIM]).astype(np.float32)

    # Training a Discriminator 3 times for a single Generator
            for k in range(d_iters):
#                 print('Training Discriminator: ', k)
                img_train = sess.run(img_batch)
#                 print('Image Training Batch Size: ', img_train.shape)
                
                #wgan clip weights
                sess.run(d_clip)
                    
                    # Update the discriminator
                _, disLoss = sess.run([trainer_d, d_loss],                         feed_dict = {random_input: train_noise, real_data: img_train,                                                    is_train: True})
#             print('Discriminator Loss:', disLoss)
            dis_loss.append(disLoss)

            # Update the generator
            for k in range(g_iters):
#                 print('Training Generator: ',k)
#                 print('Generator Noise input shape: ', train_noise.shape)
               # train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, RANDOM_DIM]).astype(np.float32)
                _, genLoss = sess.run([trainer_g, g_loss],                            feed_dict = {random_input: train_noise,                                                    is_train: True})
#             print('Generator Loss: ', genLoss)
            gen_loss.append(genLoss)
        
        # Caculating avg Generator and Discriminator loss
        avg_gen_loss.append(np.mean(gen_loss))
        avg_dis_loss.append(np.mean(dis_loss))

        print('Epochs:[%d/%d], avg_dis_loss:%f, avg_gen_loss:%f' % (i, EPOCH, np.mean(dis_loss),               np.mean(gen_loss)))
            
        # save check point every 500 epoch
#         if i%2 == 0:
#             try:
#                 if not os.path.exists(dog_img_path +  version +'model/'):
#                     os.makedirs(path +  version + 'model/')
#                 saver.save(sess, path +  version + 'model/'+ str(i) + '_dog_img_gen.ckpt')
#             except:
#                 print('Unable to save the trained session. permission denied!')
        final_epoch = i + 1
        if final_epoch%EPOCH == 0:   
            # save Generated Samples
#             try:
#                 if not os.path.exists(path +  version + 'Dog_Gen_Images/'):
#                     os.makedirs(path +  version + 'Dog_Gen_Images/')
#             except:
#                 print('Unable to create path for saving image!')
            
            sample_noise = np.random.uniform(-1.0, 1.0, size = [SAMPLE_SIZE, RANDOM_DIM]).astype(np.float32)
            sample_pred = sess.run(fake_data, feed_dict = {random_input: sample_noise, is_train: False})
            
            print('Sample Generated Shape: ', sample_pred.shape)
            print('Showing some random image generated...')
            
            xxx = 0
            for smp_img in sample_pred:
                imgs = smp_img * 255.0
#                 imgs = imgs.numpy()
#                 print('SMP_IMG: ', smp_img[0])
#                 imgs = Image.fromarray(smp_img.astype('uint8').reshape((64,64)), 'RGB')
                imgs = Image.fromarray(smp_img.astype('uint8'), 'RGB')
#                 sample_images(imgs, rows = HEIGHT, cols=WIDTH)
                plt.figure(figsize = (8,5))
                plt.title('Reconstructed')
                plt.imshow(imgs)
                plt.axis('off')
                xxx +=1
        
        final_epoch = i + 1
        if final_epoch%EPOCH == 0:
            # save Generated Samples
            sample_noise = np.random.uniform(-1.0, 1.0, size = [10000, RANDOM_DIM]).astype(np.float32)
            sample_pred = sess.run(fake_data, feed_dict = {random_input: sample_noise, is_train: False})
            
#             print('Sample Generated Shape: ', sample_pred.shape)
            
            # SAVE TO ZIP FILE NAMED IMAGES.ZIP
            z = zipfile.PyZipFile('images.zip', mode='w')
            lop = 0
            try:
                for smp_img in sample_pred:
                    imgs = smp_img * 255.0
#                     imgs = imgs.numpy()
                    imgs = Image.fromarray(imgs.astype('uint8').reshape((64,64,3)), 'RGB')
#                     imgs.save(img_output + '/dog_' + str(lop).zfill(3) + '.jpg')
                    f = str(lop).zfill(3) + '.png'
                    tf.keras.preprocessing.image.save_img(f, imgs, scale=True)
                    lop +=1
#                 print('Generated Image in Pixels: ', np.asarray(imgs, dtype="int32" ))
                    z.write(f); os.remove(f)
                z.close()
                
            except:
                print('Unable to create and save image in directory!')
            # Plot some random image
#             plot_random_image(sample_pred, path, n_img = 10, rows = 2, img_size = HEIGHT)
        
    coord.request_stop()
    coord.join(threads)
    
    print('Plotting Average Generator and Discriminator Loss...')
    plt.figure(figsize = (12,10))
    plt.plot(avg_gen_loss)
    plt.plot(avg_dis_loss)
    plt.title('Generator & Discriminator Loss by Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Generator Loss', 'Discriminator Loss'], loc = 'upper right')
    plt.show()
    
    # Saving the complete trained session/model
    
    try:
        if not os.path.exists(path + version + 'final_model/'):
            os.makedirs(path + version +'final_model/')
        saver.save(sess, path + version +'final_model/' + 'wgan_dog_img_gen.ckpt')
        print('\nFinal WGAN Model Stored path: ')
    except:
        print('Unable to save the final model!')

    print('\nFinal Generator Loss: ', np.mean(avg_gen_loss))
    print('Final Discriminator Loss: ', np.mean(avg_dis_loss))


# In[10]:


###############################################################################
# Main Function to call all the above functions
###############################################################################

if __name__ == "__main__":
    # Major Hyper-parameters
    THREADS = 4
    BATCH_SIZE = 128
    EPOCH = 300
    RANDOM_DIM = 128
    LEARNING_RATE = 0.0003
    SAMPLE_SIZE = 128
    version = 'GAN_DOG_IMG_V1/'
    HEIGHT, WIDTH, CHANNEL = 64, 64, 3
    import time
    t1 = time.time()
    train()
    print('total time taken in mins: ', (time.time()  - t1)/60)
#    data_generator(nrows = 10000)

