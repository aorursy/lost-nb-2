#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import psutil #useful to see memory usage

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


psutil.virtual_memory()


# In[3]:


import io
import os
import bson
import matplotlib.pyplot as plt

INPUT_PATH = os.path.join('..', 'input')
CATEGORY_NAMES_DF = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train.bson'), 'rb'))
TRAIN_EXAMPLE_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train_example.bson'), 'rb'))
TEST_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'test.bson'), 'rb'))


# In[4]:


CATEGORY_NAMES_DF.head(5)


# In[5]:


CAT = pd.DataFrame(CATEGORY_NAMES_DF.category_id)
CAT['category_nb'] = CAT.index
CATEGORY_NAMES_DF = pd.merge(CAT, CATEGORY_NAMES_DF, on = ['category_id'])
CATEGORY_NAMES_DF.head()


# In[6]:


psutil.virtual_memory()


# In[7]:


for item in TRAIN_DB:
    break
print(type(item))
print(item.keys())
print(item['_id'], item['category_id'], type(item['imgs']), len(item['imgs']))


# In[8]:


import cv2
from PIL import Image

def decode_image(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def decode_pil(data):
    return Image.open(io.BytesIO(data))

for img_dict in item['imgs']:
    img = decode_image(img_dict['picture'])
    plt.figure()
    plt.imshow(img)
    plt.grid(False)
    
#Alternatively
#for e, pic in enumerate(item['imgs']):
#    picture = imread(io.BytesIO(pic['picture']))
#    plt.imshow(picture)
#    plt.show()


# In[9]:


level_tags = CATEGORY_NAMES_DF.columns[2:]
CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'] == item['category_id']][level_tags]


# In[10]:


def decode_images(item_imgs):
    nb_imgs = len(item_imgs)
    nx = 2 if nb_imgs > 1 else 1
    ny = 2 if nb_imgs > 2 else 1
    set_imgs = np.zeros((180*ny, 180*nx,3), dtype = np.uint8)
    for i,img_dict in enumerate(item_imgs):
        img = decode_image(img_dict['picture'])
        h, w, _ = img.shape        
        xstart = (i % nx) * 180
        xend = xstart + w
        ystart = (i // nx) * 180
        yend = ystart + h
        set_imgs[ystart:yend, xstart:xend] = img
    return set_imgs


# In[11]:


#Reset the iterator
TRAIN_EXAMPLE_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train_example.bson'), 'rb'))
prod_to_category = dict()
k = 0

rand_rows = np.random.permutation(82)
fig, axArray = plt.subplots(nrows=2,ncols=2, figsize=(16,8))
plt.subplots_adjust(wspace=0.1, hspace=0.6)
for c, d in enumerate(TRAIN_EXAMPLE_DB):
    product_id = d['_id']
    category_id = d['category_id']
    prod_to_category[product_id] = category_id
    picture = decode_images(d['imgs'])
    if(c in rand_rows[0:4]):
        mask = CATEGORY_NAMES_DF['category_id'] == d['category_id']
        cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]
        cat_levels = [c[:25] for c in cat_levels]
        title = str(d['category_id']) + '\n'
        title += '\n'.join(cat_levels)
        nx = 1 if k % 2 == 0 else 0
        ny = 1 if k // 2 == 0 else 0
        k += 1
        axArray[ny][nx].set_title(title)
        axArray[ny][nx].imshow(picture)
plt.show()

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)


# In[12]:


prod_to_category.head(5)


# In[13]:


psutil.virtual_memory()


# In[14]:


#Reset the iterator
TEST_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'test.bson'), 'rb'))
n = 4

maxcounter = 15
for c, item in enumerate(TEST_DB):
    if c % n == 0:
        plt.figure(figsize=(14,6))
    
    plt.subplot(1, n, c % n + 1)
    title = str(item['_id'])
    plt.imshow(decode_images(item['imgs']))
    plt.title(title)
    plt.axis('off')
    
    if c==maxcounter:
        break


# In[15]:


import struct
from tqdm import tqdm_notebook

num_dicts = 7069896 # according to data page
length_size = 4
IDS_MAPPING = {}

with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f, tqdm_notebook(total=num_dicts) as bar:
    item_data = []
    offset = 0
    while True:        
        bar.update()
        f.seek(offset)
        
        item_length_bytes = f.read(length_size)     
        if len(item_length_bytes) == 0:
            break                
        # Decode item length:
        length = struct.unpack("<i", item_length_bytes)[0]
        
        f.seek(offset)
        item_data = f.read(length)
        assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
        
        # Check if we can decode
        item = bson.BSON.decode(item_data)
        
        IDS_MAPPING[item['_id']] = (offset, length)        
        offset += length            
            
def get_item(item_id):
    assert item_id in IDS_MAPPING
    with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f:
        offset, length = IDS_MAPPING[item_id]
        f.seek(offset)
        item_data = f.read(length)
        return bson.BSON.decode(item_data)


# In[16]:


print(psutil.virtual_memory())
print(psutil.cpu_times())


# In[17]:


item = get_item(1234)

mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']
cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]
cat_levels = [c[:25] for c in cat_levels]
title = str(item['category_id']) + '\n'
title += '\n'.join(cat_levels)
plt.imshow(decode_images(item['imgs']))
plt.title(title)
plt.axis('off')
plt.show()


# In[18]:


print("Number of categories %i"% CATEGORY_NAMES_DF.category_id.nunique())
print("Number of level 1 categories %i"% CATEGORY_NAMES_DF.category_level1.nunique())
print("Number of level 2 categories %i"% CATEGORY_NAMES_DF.category_level2.nunique())
print("Number of level 3 categories %i"% CATEGORY_NAMES_DF.category_level3.nunique())


# In[19]:


import seaborn as sns

#Histogram of level1 categories
plt.figure(figsize=(12,12))
sns.countplot(y=CATEGORY_NAMES_DF.category_level1)
plt.show()


# In[20]:


psutil.virtual_memory()


# In[21]:


num_dicts = 7069896
prod_to_category = [None] * num_dicts
TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train.bson'), 'rb'))

with tqdm_notebook(total=num_dicts) as bar:
    for i, item in enumerate(TRAIN_DB):
        bar.update()
        prod_to_category[i] = (item['_id'],item['category_id'])


# In[22]:


psutil.virtual_memory()


# In[23]:


TRAIN_CATEGORIES_DF = pd.DataFrame(prod_to_category, columns=['_id', 'category_id'])
TRAIN_CATEGORIES_DF.head()


# In[24]:


TRAIN_DF = pd.merge(TRAIN_CATEGORIES_DF, CATEGORY_NAMES_DF, on = ['category_id'])


# In[25]:


TRAIN_DF.head(5)


# In[26]:


TRAIN_DF._id.unique().sort() == TRAIN_CATEGORIES_DF._id.unique().sort()


# In[27]:


#Histogram of level1 categories
plt.figure(figsize=(12,12))
sns.countplot(y=TRAIN_DF.category_level1)
plt.title("Train set level1 histogram")
plt.show()


# In[28]:


train_gb = TRAIN_DF.groupby('category_id')
train_count = train_gb['category_id'].count()

most_freq_cats = train_count[train_count == train_count.max()]
print("Most frequent category: ", CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'].isin(most_freq_cats.index)].values)


# In[29]:


most_freq_cat = most_freq_cats.index[0]
TRAIN_MOST_FREQ_DF = TRAIN_DF[TRAIN_DF['category_id']==most_freq_cat]

mask = CATEGORY_NAMES_DF['category_id'] == most_freq_cat   
cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]
title = str(item['category_id']) + '\n'
title += '\n'.join(cat_levels)

maxcounter = 50
n = 10
c = 0
for item_id in TRAIN_MOST_FREQ_DF['_id'][:maxcounter]:
    if c % n == 0:
        plt.figure(figsize=(14,4))
        if c == 0:
            plt.suptitle(title)
    
    item = get_item(item_id)
    plt.subplot(1, n, c % n + 1)
    plt.imshow(decode_images(item['imgs']))
    plt.axis('off')
    
    c += 1
    if c==maxcounter:
        break
plt.show()


# In[30]:


item = get_item(1234)
img = decode_images(item['imgs'])
plt.imshow(img)
plt.axis('off')
plt.show()


# In[31]:


psutil.virtual_memory()


# In[32]:


import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

rng = np.random.RandomState(23455)

#instantiate 4D tensor for input
input = T.tensor4(name='input')

#initialize shared variables for weights
w_shp = (4,3,9,9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shp),
                            dtype = input.dtype), name='W')

# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefore initialize
# them to random values to "simulate" learning.
b_shp = (4,)
b = theano.shared(np.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
# create theano function to compute filtered images
f = theano.function([input], output)


# In[33]:


from PIL import Image

item = get_item(1)
img = decode_images(item['imgs'])
img = img/256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 180,180)
filtered_img = f(img_)
# plot original image and first and second components of output)
plt.subplot(1, 5, 1); plt.axis('off'); plt.imshow(img)
plt.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
plt.subplot(1, 5, 2); plt.axis('off'); plt.imshow(filtered_img[0, 0, :, :])
plt.subplot(1, 5, 3); plt.axis('off'); plt.imshow(filtered_img[0, 1, :, :])
plt.subplot(1, 5, 4); plt.axis('off'); plt.imshow(filtered_img[0, 2, :, :])
plt.subplot(1, 5, 5); plt.axis('off'); plt.imshow(filtered_img[0, 3, :, :])
plt.show()


# In[34]:


from theano.tensor.signal import pool

input = T.dtensor4('input')
maxpool_shape = (2,2)
pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input], pool_out)

img_pool = f(img_)
plt.subplot(1,2,1) ; plt.axis('off') ; plt.imshow(img)
plt.subplot(1,2,2) ; plt.axis('off') ; plt.imshow(img_pool[0].transpose(1,2,0))


# In[35]:


import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype = theano.config.floatX),
                               name = 'W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype = theano.config.floatX),
                               name = 'b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
        
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            #1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        if activation is None:
            output = lin_output
        else:
            output = activation(lin_output)
          
        self.params = [self.W, self.b]
        self.output = output


# In[36]:


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input


# In[37]:


class CNN(object):
    def __init__(self,rng,input,batch_size,n_out, n_kerns, n_hidden):    
        
        # filter_shape is (n_output_channels,n_input_channels, filter_height, filter_width)
        # filtering reduces the image size to (180-5+1 , 180-5+1) = (176, 176)
        # poolsize = (2,2)  reduces it further to (176/2,176/2) = (88,88)
        
        
        self.layer0=LeNetConvPoolLayer(
                rng=rng,
                input=input.reshape((batch_size,3,180,180)),
                image_shape=(batch_size, 3, 180, 180),
                filter_shape=(n_kerns[0], 3, 5, 5),
                poolsize=(2, 2)
        )
        
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (88-5+1, 88-5+1) = (84,84)
        # maxpooling reduces this further to (84/2, 84/2) = (42, 42)
        # 4D output tensor is thus of shape (batch_size, 1, 42, 42)
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, n_kerns[0], 88, 88),
            filter_shape=(n_kerns[1], n_kerns[0], 5, 5),
            poolsize=(2, 2)
        )
        
        # filter_shape is (n_output_channels,n_input_channels, filter_height, filter_width)
        # filtering reduces the image size to (42-3+1 , 42-3+1) = (40, 40)
        # poolsize = (2,2)  reduces it further to (40/2,40/2) = (20,20)
        
        
        self.layer2=LeNetConvPoolLayer(
                rng=rng,
                input=input.reshape((batch_size,3,42,42)),
                image_shape=(batch_size, 3, 42, 42),
                filter_shape=(n_kerns[0], 3, 3, 3),
                poolsize=(2, 2)
        )
        
        
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 20 * 20),
        # or (batch_size, 3 * 20 * 20) = (batch_size, 1200) with the default values.
        # construct a fully-connected sigmoidal layer
        
        self.layer3 = HiddenLayer(
            rng,
            input=self.layer2.output.flatten(2),
            n_in=n_kerns[1] * 20 * 20,
            n_out=n_hidden,
            activation=T.tanh
        )
        
        #n_out is the number of categories
        self.layer4 = LogisticRegression(
            input=self.layer3.output, 
            n_in=n_hidden, 
            n_out=n_out
        )
        
        self.negative_log_likelihood = (
            self.layer4.negative_log_likelihood
        )

        self.errors = self.layer4.errors
        self.params = self.layer4.params + self.layer3.params + self.layer2.params             + self.layer1.params + self.layer0.params
        self.input = input


# In[38]:


def load_dataset(rand_rows, offset, length):
    
    n_train = np.int(0.6*length)
    n_valid = np.int(0.2*length)
    n_test = np.int(0.2*length)
     
    train_set_x = np.zeros((n_train,3,180,180), dtype=float)
    train_set_y = np.zeros((n_train,), dtype=float)
        
    valid_set_x = np.zeros((n_valid,3,180,180), dtype=float)
    valid_set_y = np.zeros((n_valid,), dtype=float)
    
    test_set_x = np.zeros((n_test,3,180,180), dtype=float)
    test_set_y = np.zeros((n_test,), dtype=float)
       
    #with tqdm_notebook(total=n_train) as bar:
    for iter in range(offset,n_train+offset):
        item = get_item(TRAIN_DF._id[rand_rows[iter]])
        img = decode_images([item['imgs'][0]])
        mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']
        train_set_x[iter-offset] = img.transpose(2, 0, 1).reshape(3, 180,180)
        train_set_y[iter-offset] = CATEGORY_NAMES_DF[mask]['category_nb'].values.tolist()[0]
    
    #with tqdm_notebook(total=n_valid) as bar:
    for iter in range(offset+n_train, offset+n_train + n_valid):
        item = get_item(TRAIN_DF._id[rand_rows[iter]])
        img = decode_images([item['imgs'][0]])
        mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']
        valid_set_x[iter-n_train-offset] = img.transpose(2, 0, 1).reshape(3, 180,180)
        valid_set_y[iter-n_train-offset] = CATEGORY_NAMES_DF[mask]['category_nb'].values.tolist()[0]
    
    #with tqdm_notebook(total=n_test) as bar:
    for iter in range(offset+n_train+n_valid, offset+n_train + n_valid+n_test):
        item = get_item(TRAIN_DF._id[rand_rows[iter]])
        img = decode_images([item['imgs'][0]])
        mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']
        test_set_x[iter-n_train-n_valid-offset] = img.transpose(2, 0, 1).reshape(3, 180,180)
        test_set_y[iter-n_train-n_valid-offset] = CATEGORY_NAMES_DF[mask]['category_nb'].values.tolist()[0]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    
    def shared_dataset(data_xy, borrow=True):
        data_x,data_y = data_xy
        
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


# In[39]:


psutil.virtual_memory()


# In[40]:


def test_cnn(learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 100,
             batch_size = 10000, mini_batch_size = 100, n_kerns=(3,3),
             n_hidden=750, n_out = 5270):
    
    #batch_size is the total number of images loaded in memory
    
    n_train_batches = np.int((0.6*batch_size)//mini_batch_size)
    n_valid_batches = np.int((0.2*batch_size)//mini_batch_size)
    n_test_batches = np.int((0.2*batch_size)//mini_batch_size)


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    index = T.lscalar()
    size = T.lscalar()
    x = T.tensor4('x')
    y = T.ivector('y')
    
    rng = np.random.RandomState(1234)
    
    classifier = CNN(rng=rng,
                     input = x.reshape((mini_batch_size,3,180,180)),
                     batch_size = mini_batch_size,
                     n_out = n_out,
                     n_kerns = n_kerns,
                     n_hidden = n_hidden
    )
    
    cost = classifier.negative_log_likelihood(y)
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param-learning_rate*gparam) for param,gparam in zip(classifier.params, gparams)]
    
    print('... training')

    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatch before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    rand_rows = np.random.permutation(7069896)
    offset = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        dataset = load_dataset(rand_rows=rand_rows, offset=0, length=batch_size)
        
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]
        
        test_model = theano.function(
                inputs=[index],
                outputs = classifier.errors(y),
                givens = {
                        x: test_set_x[index*mini_batch_size:(index+1)*mini_batch_size],
                        y: test_set_y[index*mini_batch_size:(index+1)*mini_batch_size]
                        }
        )
        
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * mini_batch_size:(index + 1) * mini_batch_size],
                y: valid_set_y[index * mini_batch_size:(index + 1) * mini_batch_size]
            }
        )
        
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * mini_batch_size: (index + 1) * mini_batch_size],
                y: train_set_y[index * mini_batch_size: (index + 1) * mini_batch_size]
            }
        )
            
        
        
        for minibatch_index in range(n_train_batches):

            #There are n_train_batches*mini_batch_size used for the training
            minibatch_avg_cost = train_model(minibatch_index)
            
            print("Error function for minibatch %i/%i is %.5f"%(minibatch_index+1, n_train_batches, minibatch_avg_cost))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


# In[41]:


learning_rate = 0.01
L1_reg = 0.00
L2_reg = 0.0001
n_epochs = 100
batch_size = 10000
mini_batch_size = 100
n_kerns = (3,3)
n_hidden = 750
n_out = 5270

test_cnn(learning_rate = learning_rate,
         L1_reg = L1_reg,
         L2_reg = L2_reg,
         n_epochs = n_epochs,
         batch_size = batch_size,
         mini_batch_size = mini_batch_size,
         n_kerns = n_kerns,
         n_hidden = n_hidden,
         n_out = n_out)


# In[42]:




