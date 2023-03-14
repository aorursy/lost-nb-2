#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import os
import time
from glob import glob
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import xml
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import IPython.display as display

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import TruncatedNormal, RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


tf.enable_eager_execution()
print(tf.test.gpu_device_name())


# In[2]:


print(os.listdir("../input/"))


# In[3]:


# filepaths for image loading code
ROOT = '../input/'
# list of all image file names in all-dogs
IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs')
# list of all the annotation directories, each directory is a dog breed
BREEDS = os.listdir(ROOT + 'annotation/Annotation/') 

# variables that determine how tensorflow will create batches after data load
BUFFER_SIZE = 20000
BATCH_SIZE = 32

# weight initializers for the generator network
WEIGHT_INIT = RandomNormal(mean=0.0, stddev=0.05)

# generate/classify 64x64 images
IMG_SIZE = 64

# for training
EPOCHS = 1000
NOISE_SIZE = 128
NB_EXAMPLES_TO_GENERATE = 16

# for animated GIF
seed = tf.random.normal([NB_EXAMPLES_TO_GENERATE, NOISE_SIZE])


# In[4]:


# dom = xml.dom.minidom.parse('../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98') 
# pretty_xml_as_string = dom.toprettyxml()
# print(pretty_xml_as_string)


# In[5]:


# Code slightly modified from user: cdeotte | https://www.kaggle.com/cdeotte/supervised-generative-dog-net

imgs = []
names = []

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY
# iterate through each directory in annotation
for breed in BREEDS:
    # iterate through each file in the directory
    for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):
        try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg')
        except: continue
        # Element Tree library allows for parsing xml and getting specific tag values
        tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)
        # take a look at the print out of an xml previously to get what is going on
        root = tree.getroot() # <annotation>
        objects = root.findall('object') # <object>
        for o in objects:
            bndbox = o.find('bndbox') # <bndbox>
            xmin = int(bndbox.find('xmin').text) # <xmin>
            ymin = int(bndbox.find('ymin').text) # <ymin>
            xmax = int(bndbox.find('xmax').text) # <xmax>
            ymax = int(bndbox.find('ymax').text) # <ymax>
            w = np.min((xmax - xmin, ymax - ymin))
            img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
            img2 = img2.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            imgs.append(np.asarray(img2))
            names.append(breed)

imgs = np.array(imgs)
names[:] = map(str.lower, names)


# In[6]:


print("imgs.shape: {}".format(imgs.shape))


# In[7]:


# DISPLAY CROPPED IMAGES

# list of 25 random numbers in range 0, idxIn
# this allows for displaying random images in the for loop using x[k*5+j]
x = np.random.randint(0, len(imgs), 25)

for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray(imgs[x[k*5+j], :, :, :].astype('uint8'))
        plt.axis('off')
        plt.title(names[x[k*5+j]].split('-')[1], fontsize=11)
        plt.imshow(img)
plt.show()


# In[8]:


# normalize the pixel values
imgs = (imgs - 127.5) / 127.5


# In[9]:


# view some images after normalization 
# plt.figure(figsize=(8,8))
# for image in range(4):
#     plt.subplot(2,2, image+1)
#     plt.imshow((imgs[image]))


# In[10]:


# this is needed because the gradient functions from TF require float32 instead of float64
imgs = imgs.astype(np.float32)


# In[11]:


ds = tf.data.Dataset.from_tensor_slices(imgs).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(ds)


# In[12]:


def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm


# In[13]:


class PixelwiseNorm(tf.keras.layers.Layer):

    def __init__(self):
        super(PixelwiseNorm, self).__init__()


    def call(self, x, eps=1e-8):
        """
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = tf.sqrt(tf.reduce_mean(x**2, axis=3, keepdims=True) + eps)
        y = x / y  # normalize the input x volume
        return y


# In[14]:


# adapted from keras.optimizers.Adam
class AdamWithWeightnorm(Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.floatx())))

        t = K.cast(self.iterations + 1, K.floatx())
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = K.get_variable_shape(p)
            if len(ps)>1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)
                v_g = K.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * K.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (K.sqrt(v_g_t) + self.epsilon)
                self.updates.append(K.update(m_g, m_g_t))
                self.updates.append(K.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(grad_V)
                new_V_param = V - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if getattr(p, 'constraint', None) is not None:
                    new_V_param = p.constraint(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                new_p = p_t
                # apply constraints
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)
                self.updates.append(K.update(p, new_p))
        return self.updates

def get_weightnorm_params_and_grads(p, g):
    ps = K.get_variable_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    V_scaler = K.ones(V_scaler_shape)  # init to ones, so effective parameters don't change

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) *              (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V

def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = K.get_variable_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(K.update(W, new_W))
    updates.append(K.update(V_scaler, new_V_scaler))

# data based initialization for a given Keras model
def data_based_init(model, input):
    # input can be dict, numpy array, or list of numpy arrays
    if type(input) is dict:
        feed_dict = input
    elif type(input) is list:
        feed_dict = {tf_inp: np_inp for tf_inp,np_inp in zip(model.inputs,input)}
    else:
        feed_dict = {model.inputs[0]: input}

    # add learning phase if required
    if model.uses_learning_phase and K.learning_phase() not in feed_dict:
        feed_dict.update({K.learning_phase(): 1})

    # get all layer name, output, weight, bias tuples
    layer_output_weight_bias = []
    for l in model.layers:
        trainable_weights = l.trainable_weights
        if len(trainable_weights) == 2:
            W,b = trainable_weights
            assert(l.built)
            layer_output_weight_bias.append((l.name,l.get_output_at(0),W,b)) # if more than one node, only use the first

    # iterate over our list and do data dependent init
    sess = K.get_session()
    for l,o,W,b in layer_output_weight_bias:
        print('Performing data dependent initialization for layer ' + l)
        m,v = tf.nn.moments(o, [i for i in range(len(o.get_shape())-1)])
        s = tf.sqrt(v + 1e-10)
        updates = tf.group(W.assign(W/tf.reshape(s,[1]*(len(W.get_shape())-1)+[-1])), b.assign((b-m)/s))
        sess.run(updates, feed_dict)


# In[15]:


# function will return a generator model
def make_generator():
    model = Sequential([

        Input(shape=(NOISE_SIZE,)),
        # first layer with 32, 768 nodes expecting an input of vector size NOISE_SIZE (random noise)
        Dense(4*4*64, use_bias=False),
        # apply leaky relu activation: f(x) = {x if x > 0 : 0.01*x}
        LeakyReLU(),
        # Normalize the activations of the previous layer at each batch
#         BatchNormalization(),
        # reshape input to (8,8,512)
        Reshape((4, 4, 64)),
        # added later
#         PixelwiseNorm(),

        # 8 x 8
        UpSampling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_initializer=WEIGHT_INIT),
#         Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', use_bias=False,
#                         kernel_initializer=WEIGHT_INIT),
#                         kernel_regularizer=spectral_norm),
        LeakyReLU(),
#         BatchNormalization(),
#         Dropout(0.1),
#         PixelwiseNorm(),
        

        # 16 x 16
        UpSampling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_initializer=WEIGHT_INIT),
#         Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', use_bias=False,
#                         kernel_initializer=WEIGHT_INIT),
#                         kernel_regularizer=spectral_norm),
        LeakyReLU(),
#         BatchNormalization(),
#         Dropout(0.1),
#         PixelwiseNorm(),
        
        
        # 32 x 32
        UpSampling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_initializer=WEIGHT_INIT),
#         Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', use_bias=False,
#                         kernel_initializer=WEIGHT_INIT),
#                         kernel_regularizer=spectral_norm),
        LeakyReLU(),
#         BatchNormalization(),
#         PixelwiseNorm(),

        # 64 x 64
        UpSampling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_initializer=WEIGHT_INIT),
#         Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', use_bias=False,
#                         kernel_initializer=WEIGHT_INIT),
#                         kernel_regularizer=spectral_norm),
        LeakyReLU(),
        
        
        Conv2D(3, (3, 3), padding='same', use_bias=False, activation='tanh', kernel_initializer=WEIGHT_INIT)
        
        
#         Dense(3, activation='tanh', use_bias=False,
#               kernel_initializer=WEIGHT_INIT),
#               kernel_regularizer=spectral_norm)
        
#         Conv2D(3, (1, 1), padding='same', use_bias=False,
#                kernel_initializer=WEIGHT_INIT)

    ])
    
    return model


# create an instance of the generator model defined
generator = make_generator()

# random noise vector
noise = tf.random.normal([1, NOISE_SIZE])

# run the generator model with the noise vector as input
generated_image = generator(noise, training=False)

print(generated_image.dtype)

# display output
plt.imshow(generated_image[0, :, :, :])
print(generated_image.shape)


# In[16]:


def make_discriminator():
    model = tf.keras.Sequential([
        
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # downsample to 32 x 32
        Conv2D(128, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer=WEIGHT_INIT),
        LeakyReLU(0.2),
#         BatchNormalization(),
        Dropout(0.25),
    
        # downsample to 16 x 16
        Conv2D(128, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer=WEIGHT_INIT),
#                kernel_regularizer=spectral_norm),
        LeakyReLU(0.2),
#         BatchNormalization(),
        Dropout(0.25),

        # downsample to 8 x 8
        Conv2D(128, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer=WEIGHT_INIT),
#                kernel_regularizer=spectral_norm),
        LeakyReLU(0.2),
#         BatchNormalization(),
        Dropout(0.25),
    
        # downsample to 4 x 4
        Conv2D(128, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer=WEIGHT_INIT),
#                kernel_regularizer=spectral_norm),
        LeakyReLU(0.2),
#         BatchNormalization(),
        Dropout(0.25),

        # flatten input into 1-D and output a single a number from the last layer using sigmoid activation
        Flatten(),
        Dense(1, activation='sigmoid', kernel_initializer=WEIGHT_INIT)
    ])

    return model


discriminator = make_discriminator()

decision = discriminator(generated_image)
print(decision)


# In[17]:


# Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, we assign a random integer in range [0.7, 1.0] for positive class
# and [0.0, 0.3] for negative class

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3


# Recomended to introduce some noise to the labels, so out of 1000 real labels, approximately 50 should be flipped to 0 (5%)
# randomly flip some labels

def noisy_labels(y, p_flip=0.05):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0].value)
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0].value)], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y


# In[18]:


# This method returns a helper function to compute cross entropy loss
# code from tf dcgan tutorial
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# The Discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_output_smooth = smooth_positive_labels(tf.ones_like(real_output))
#     real_output_noisy = noisy_labels(real_output_smooth, 0.05)
    fake_output_smooth = smooth_negative_labels(tf.zeros_like(fake_output))
#     fake_output_noisy = noisy_labels(fake_output_smooth, 0.05)
    real_loss = cross_entropy(real_output_smooth, real_output)
    fake_loss = cross_entropy(fake_output_smooth, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# The Generator loss function
def generator_loss(fake_output):
    fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
#     fake_output_noisy = noisy_labels(fake_output_smooth, 0.05)
    return cross_entropy(fake_output_smooth, fake_output)


# optimizers -- Adam
generator_optimizer = AdamWithWeightnorm(lr=0.0002, beta_1=0.5)
discriminator_optimizer = AdamWithWeightnorm(lr=0.00006, beta_1=0.5)
# generator_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
# discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=0.00006, beta1=0.5)
# discriminator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00006)


# In[19]:


# code from tf dcgan tutorial
def train_step(images, G_loss_list, D_loss_list):
    noise = tf.random.normal([BATCH_SIZE, NOISE_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    G_loss_list.append(gen_loss.numpy().mean())
    D_loss_list.append(disc_loss.numpy().mean())
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[20]:


def train(dataset, epochs, time_limit=32000):
    training_start = time.time()

    for epoch in range(epochs):
        G_loss = []
        D_loss = []

        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, G_loss, D_loss)
        if (epoch % 10 == 0):
#             display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
        print('epoch {:3d} - G loss: {:.4f} - D loss: {:.4f} - {:.2f} sec'.format(epoch + 1, G_loss[-1], D_loss[-1], time.time()-start))
        if(time.time() - training_start > time_limit):
            print(f"Reached training time limit ({time_limit} s) at {time.time() - training_start:.2f}")
            break

    # Generate after the final epoch
    print("Final Epoch")
    generate_and_save_images(generator,
                             epochs,
                             seed)


# In[21]:


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(8,8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :] + 1.) / 2.)
        plt.axis('off')
    plt.savefig("img_at_epoch_{}.png".format(epoch+1))
    plt.show()


# In[22]:


get_ipython().run_cell_magic('time', '', "\nprint('Starting training')\ntrain(ds, EPOCHS)")


# In[23]:


get_ipython().run_cell_magic('time', '', "\n# SAVE TO ZIP FILE NAMED IMAGES.ZIP\nz = zipfile.PyZipFile('images.zip', mode='w')\n\nfilename = 'gen_model.h5'\ntf.keras.models.save_model(\n    generator,\n    filename,\n    overwrite=True,\n    include_optimizer=True,\n    save_format=None\n)\n\nfor k in range(10000):\n    generated_image = generator(tf.random.normal([1, NOISE_SIZE]), training=False)\n    f = str(k) + '.png'\n    img = ((generated_image[0,:,:,:] + 1.) / 2.).numpy()\n    tf.keras.preprocessing.image.save_img(\n        f,\n        img,\n        scale=True\n    )\n    z.write(f)\n    os.remove(f)\n\nz.close()")


# In[ ]:




