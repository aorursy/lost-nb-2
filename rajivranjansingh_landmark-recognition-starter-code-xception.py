#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import cv2
import skimage.io
from cv2 import imread as cv2_imread
from cv2 import resize as cv2_resize
from cv2 import cvtColor
from skimage.io import imread
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3,Xception
from tensorflow.keras.initializers import glorot_uniform,he_uniform
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(False)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


# In[3]:


train = pd.read_csv("../input/landmark-recognition-2020/train.csv")
train.head()


# In[4]:


train=train.sample(n=100000,random_state=0)
train.head()


# In[5]:


train["path"] = train.id.map(lambda path: f"../input/landmark-recognition-2020/train/{path[0]}/{path[1]}/{path[2]}/{path}.jpg")


# In[6]:


def get_paths(sub='train'):
    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]

    paths = []

    for a in index:
        for b in index:
            for c in index:
                try:
                    paths.extend([f"../input/landmark-recognition-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-recognition-2020/{sub}/{a}/{b}/{c}")])
                except:
                    pass

    return paths


# In[7]:


train.head()


# In[8]:


batch_size = 4
seed = 42
IMAGE_SIZE = 128
shape = (IMAGE_SIZE, IMAGE_SIZE, 3) ##desired shape of the image for resizing purposes


MIN_SAMPLE_IMAGE = 10
val_sample = 0.2 # % of validation sample

DENSE_UNITS = 1024

epochs = 20


# In[9]:


k =train[['id','landmark_id']].groupby(['landmark_id']).agg({'id':'count'})
k.rename(columns={'id':'Count_class'}, inplace=True)
k.reset_index(level=(0), inplace=True)
freq_ct_df = pd.DataFrame(k)
freq_ct_df.head()


# In[10]:


freq_ct_df.shape


# In[11]:


train_labels = pd.merge(train,freq_ct_df, on = ['landmark_id'], how='left')
train_labels.head()


# In[12]:


train_labels.describe()


# In[13]:


# freq_ct_df.sort_values(by=['Count_class'],ascending=False,inplace=True)
# freq_ct_df.head()


# In[14]:


# TOP_FRACTION = 0.005
# TOP = math.floor(freq_ct_df['landmark_id'].nunique()*TOP_FRACTION)
# TOP


# In[15]:


freq_ct_df_top = freq_ct_df[freq_ct_df['Count_class']>MIN_SAMPLE_IMAGE]
top_class = freq_ct_df_top['landmark_id'].tolist()


# In[16]:


class_weights = freq_ct_df_top['Count_class'].sum()/freq_ct_df_top['Count_class']
class_weights/=class_weights.min()
class_weights = dict(zip(list(range(0,len(class_weights))),class_weights))


# In[17]:


topclass_train = train[train['landmark_id'].isin (top_class) ]
topclass_train.shape


# In[18]:


topclass_train.head()


# In[19]:


def getTrainParams():
    
    print("Encoding labels")
    data = topclass_train.copy()
    le = preprocessing.LabelEncoder()
    
    print("fitting LabelENcoder")
    data['label'] = le.fit_transform(data['landmark_id'])
    print("Success in LabelENcoder")
    
    lbls = data['label'].tolist()
    lb = LabelBinarizer(sparse_output=False)
    
    print("fitting LabelBinarizer")
    labels = lb.fit_transform(lbls)
    print("Success in LabelBinarizer")
    
    x = np.array(topclass_train['path'].tolist())
    print("Converting Paths to array")
    
    y = labels
    print("Converting labels to array")
    
    return x,y,le


# In[20]:


paths, labels,label_encoder = getTrainParams()
print(paths.shape, labels.shape)


# In[21]:


np.random.seed(seed)
pathsTrain,pathsVal, labelsTrain, labelsVal = train_test_split(paths,labels,test_size = val_sample,random_state=42)


print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)


# In[22]:


# Some garbage collection
import gc
gc.collect()


# In[23]:


class Landmark2020_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
                
        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    
                    iaa.ContrastNormalization((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    
                    iaa.Affine(rotate=0),
                    #iaa.Affine(rotate=90),
                    #iaa.Affine(rotate=180),
                    #iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    #iaa.Flipud(0.5),
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        im = cv2_imread(path)
        im = cv2_resize(im,(IMAGE_SIZE,IMAGE_SIZE))
        im = cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im/255.0
        return im


# In[24]:


base_model = Xception(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
                       weights=None, include_top=False)
base_model.load_weights("../input/keraspretrainedmodel/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")

input_image = Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype = tf.uint8)

x = base_model(input_image)
x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dense(DENSE_UNITS,activation='relu', kernel_initializer = he_uniform(seed=0))(x)
x = BatchNormalization()(x)

embedding_model = Model(inputs = input_image,outputs = x,name='embedding_model')

for layer in base_model.layers:
    layer.trainable = False
    
embedding_model.compile()
embedding_model.summary()


# In[25]:


nlabls = topclass_train['landmark_id'].nunique()
nlabls


# In[26]:


NN_branches = nlabls//100 +1
# DENSE_UNITS = 1024


# In[27]:


NN_branches


# In[28]:


cat_out_layers = []
output = None


for i in range(1,NN_branches+1):
    cat = Dense(DENSE_UNITS//NN_branches,activation = 'relu',name = 'cat'+str(i), kernel_initializer = he_uniform(seed=1))(embedding_model.output)
#     cat = Dropout(0.2)(cat)
    cat = Dense(DENSE_UNITS//NN_branches,activation = 'relu',name = 'cat'+str(i)+'cat', kernel_initializer = he_uniform(seed=2))(cat)
    bat = BatchNormalization()(cat)    
    
    if i != NN_branches:
        cat_out = Dense(nlabls//NN_branches, activation=None, name = 'cat'+str(i)+'_out',  kernel_initializer = he_uniform(seed=3))(bat)
    else:
        cat_out = Dense(nlabls//NN_branches+nlabls%NN_branches, activation=None, name = 'cat'+str(i)+'_out', kernel_initializer = he_uniform(seed=3))(bat)

    cat_out_layers.append(cat_out)    
        #     if output == None:
#         output = cat_out
#     else:
#         output = Concatenate()([output,cat_out])

if len(cat_out_layers)==1:
    concat_out = cat_out_layers
else:
    concat_out = Concatenate()(cat_out_layers)
    
output = Activation('softmax')(concat_out)

# cat1 = Dense(512,activation = 'relu')(embedding_model.output)
# cat2 = Dense(512,activation = 'relu')(embedding_model.output)

# cat1_out = Dense(nlabls/2, activation='softmax',  kernel_initializer = glorot_uniform(seed=0))(cat1)
# cat2_out = Dense(nlabls/2, activation='softmax',  kernel_initializer = glorot_uniform(seed=0))(cat2)

# output = Concatenate()([cat1_out,cat2_out])
# output = Dense(nlabls, activation='softmax', name='fc' + str(nlabls), kernel_initializer = glorot_uniform(seed=0))(embedding_model.output)

model = Model(inputs=[input_image], outputs=[output])
model.summary()


# In[29]:


from tensorflow.keras.utils import plot_model

plot_model(model,show_shapes=True,)


# In[30]:


from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy
def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# In[31]:


# initial_learning_rate = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=1000,
#     decay_rate=0.96,
#     staircase=True)


# In[32]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.KLDivergence()])
model.summary()


# In[33]:


# Some garbage collection
import gc
gc.collect()


# In[34]:


train_generator = Landmark2020_DataGenerator(pathsTrain, labelsTrain, batch_size, shape, use_cache=False, augment = False, shuffle = False)
val_generator = Landmark2020_DataGenerator(pathsVal, labelsVal, batch_size, shape, use_cache=False, shuffle = False)


# In[35]:


# setup check_generator
check_gen = Landmark2020_DataGenerator(pathsTrain, labelsTrain, 10, shape, use_cache=False, augment = False, shuffle = True)
batch_img,batch_label = check_gen.__getitem__(0)


# In[36]:


#Import the required libaries
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import math
get_ipython().run_line_magic('matplotlib', 'inline')

def show_grid(image_list,nrows,ncols,label_list=None,show_labels=False,savename=None,figsize=(10,10),showaxis='off'):
    if type(image_list) is not list:
        if(image_list.shape[-1]==1):
            image_list = [image_list[i,:,:,0] for i in range(image_list.shape[0])]
        elif(image_list.shape[-1]==3):
            image_list = [image_list[i,:,:,:] for i in range(image_list.shape[0])]
    fig = plt.figure(None, figsize,frameon=False)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     share_all=True,
                     )
    for i in range(nrows*ncols):
        ax = grid[i]
        ax.imshow(image_list[i],cmap='Greys_r')  # The AxesGrid object work as a list of axes.
        ax.axis('off')
        if show_labels:
            ax.set_title(class_mapping[label_list[i]])
            
    if savename != None:
        plt.savefig(savename,bbox_inches='tight')


# In[37]:


# generate class mapping for check data labels
class_mapping = {i:label_encoder.classes_[i] for i in range(label_encoder.classes_.shape[0])}

#Get class int vale from one hot encoded labels
y_int = np.argmax(batch_label,axis=-1)


# In[38]:


show_grid(batch_img,2,5,label_list=y_int,show_labels=True,figsize=(20,20))


# In[39]:


show_grid(image_list=[skimage.io.imread(x) for x in train['path'][train['landmark_id']==113209]],nrows=4,ncols=5)    


# In[40]:


# epochs = 10
use_multiprocessing = True 
#workers = 1 
callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=3,restore_best_weights=True)


# In[41]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=labelsTrain.shape[0]/batch_size,
                                validation_data=val_generator,
                                validation_steps=labelsVal.shape[0]/batch_size,
#                                 class_weight = class_weights,
                                epochs=epochs,
                                callbacks = [callback],
#                                 use_multiprocessing=use_multiprocessing,
                                #workers=workers,
                                verbose=1)


# In[42]:


plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[43]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[44]:


gc.collect()


# In[45]:


TRAINABLE_LAYERS_BASE = 5


# In[46]:


for layer in base_model.layers[-TRAINABLE_LAYERS_BASE:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.KLDivergence()])
model.summary()


# In[47]:


# callback = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  write_graph=True, write_images=True)


# In[48]:


model.fit_generator(
    train_generator,
    steps_per_epoch=labelsTrain.shape[0]/batch_size,
    validation_data=val_generator,
    validation_steps=labelsVal.shape[0]/batch_size,
#     class_weight = class_weights,
    epochs=epochs,
    callbacks = [callback],
#     use_multiprocessing=use_multiprocessing,
    #workers=workers,
    verbose=1)


# In[49]:


from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3,     "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)


# In[50]:


gc.collect()


# In[51]:


test_df = pd.DataFrame()
test_df["paths"] = get_paths('test')


# In[52]:


test_df['id'] = test_df['paths'].apply(lambda path: os.path.split(path)[1].split('.')[0])


# In[53]:


test_df['landmarks'] = ""


# In[54]:


THRESH = 0.1


# In[55]:


def get_test_image(img_id):    
    chars = [char for char in img_id]
    dir_1, dir_2, dir_3 = chars[0], chars[1], chars[2]
    
    test_img_path = '../input/landmark-recognition-2020/test/' + dir_1 + '/' + dir_2 + '/' + dir_3 + '/' + img_id + '.jpg'
    im = cv2_imread(test_img_path)
    im = cv2_resize(im,(IMAGE_SIZE,IMAGE_SIZE))
    im = cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255.0
    
    return im


# In[56]:


def get_class_per_image(probs,classes):
    
    conf = np.max(probs)
#     class_labels = np.argmax(probs)
    class_labels= classes[np.argmax(probs)]
    
    return class_labels,conf


# In[57]:


def get_class_per_batch(batch_probs,classes):
    
    temp_prediction = []
    for i in range(batch_probs.shape[0]):
        labl,conf = get_class_per_image(batch_probs[i,:],classes)
        if conf < THRESH:
            labl = ""
            conf = ""
        temp_prediction.append((str(labl) + ' ' + str(conf)))
    
    return temp_prediction    


# In[58]:


# Some garbage collection
gc.collect()


# In[59]:


len(model.predict(tf.convert_to_tensor([get_test_image(x) for x in test_df['id'][0:10]])))


# In[60]:


import math
test_batch_size  = 10
num_test_images = test_df.shape[0]
num_batches = math.ceil(num_test_images/(test_batch_size))
for i in tqdm(range(0,num_batches)):
    idx_begin = test_batch_size * i
    idx_end = min(num_test_images,test_batch_size * (i+1))
    batch_idx = list(range(idx_begin,idx_end))
#     print(batch_idx)
    batch_probs = model.predict(tf.convert_to_tensor([get_test_image(x) for x in test_df['id'][batch_idx]]))
#     print(batch_probs.shape)
    test_df['landmarks'][batch_idx] = get_class_per_batch(batch_probs,label_encoder.classes_)


# In[61]:


test_df.head()


# In[62]:


# test_df.drop(columns=['paths'],inplace=True)
test_df[['id','landmarks']].to_csv('submission.csv',index=False)


# In[63]:


pd.read_csv('submission.csv').describe()


# In[64]:


# ## We have removed the sensitive portions of this script, and included those
# ## that show you how we:
# ## 1. Load your model
# ## 2. Create embeddings
# ## 3. Compare and score those embeddings.
# ##
# ## Note that this means this code will NOT run as-is.

# import os
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from PIL import Image
# import time
# from scipy.spatial import distance

# import solution
# import metrics

# REQUIRED_SIGNATURE = 'serving_default'
# REQUIRED_OUTPUT = 'global_descriptor'

# DATASET_DIR = '' # path to internal dataset

# SAVED_MODELS_DIR = os.path.join('kaggle', 'input')
# QUERY_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
# INDEX_IMAGE_DIR = os.path.join(DATASET_DIR, 'index')
# SOLUTION_PATH = ''

# def to_hex(image_id: int) -> str:
#     return '{0:0{1}x}'.format(image_id, 16)


# def show_elapsed_time(start):
#     hours, rem = divmod(time.time() - start, 3600)
#     minutes, seconds = divmod(rem, 60)
#     parts = []

#     if hours > 0:
#         parts.append('{:>02}h'.format(hours))

#     if minutes > 0:
#         parts.append('{:>02}m'.format(minutes))

#     parts.append('{:>05.2f}s'.format(seconds))

#     print('Elapsed Time: {}'.format(' '.join(parts)))


# def get_distance(scored_prediction):
#     return scored_prediction[1]

# embedding_fn = None

# def get_embedding(image_path: Path) -> np.ndarray:
#     image_data = np.array(Image.open(str(image_path)).convert('RGB'))
#     image_tensor = tf.convert_to_tensor(image_data)
#     return embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()


# class Submission:
#     def __init__(self, name, model):
#         self.name = name
#         self.model = model
#         public_solution, private_solution, ignored_ids = solution.load(SOLUTION_PATH, 
#                                                          solution.RETRIEVAL_TASK_ID)
#         predictions = self.get_predictions()
        
#         self.private_score = self.get_metrics(predictions, private_solution)
#         self.public_score = self.get_metrics(predictions, public_solution)

#     def load(self, saved_model_proto_filename):
#         saved_model_path = Path(saved_model_proto_filename).parent
        
#         print (saved_model_path, saved_model_proto_filename)
        
#         name = saved_model_path.relative_to(SAVED_MODELS_DIR)
        
#         model = tf.saved_model.load(str(saved_model_path))
        
#         found_signatures = list(model.signatures.keys())
        
#         if REQUIRED_SIGNATURE not in found_signatures:
#             return None
        
#         outputs = model.signatures[REQUIRED_SIGNATURE].structured_outputs
#         if REQUIRED_OUTPUT not in outputs:
#             return None
        
#         global embedding_fn
#         embedding_fn = model.signatures[REQUIRED_SIGNATURE]

#         return Submission(name, model)
    

#     def get_id(self, image_path: Path):
#         return int(image_path.name.split('.')[0], 16)


#     def get_embeddings(self, image_root_dir: str):
#         image_paths = [p for p in Path(image_root_dir).rglob('*.jpg')]
        
#         embeddings = [get_embedding(image_path) 
#                       for i, image_path in enumerate(image_paths)]
#         ids = [self.get_id(image_path) for image_path in image_paths]

#         return ids, embeddings
    
#     def get_predictions(self):
#         print('Embedding queries...')
#         start = time.time()
#         query_ids, query_embeddings = self.get_embeddings(QUERY_IMAGE_DIR)
#         show_elapsed_time(start)

#         print('Embedding index...')
#         start = time.time()
#         index_ids, index_embeddings = self.get_embeddings(INDEX_IMAGE_DIR)
#         show_elapsed_time(start)

#         print('Computing distances...', end='\t')
#         start = time.time()
#         distances = distance.cdist(np.array(query_embeddings), 
#                                    np.array(index_embeddings), 'euclidean')
#         show_elapsed_time(start)

#         print('Finding NN indices...', end='\t')
#         start = time.time()
#         predicted_positions = np.argpartition(distances, K, axis=1)[:, :K]
#         show_elapsed_time(start)

#         print('Converting to dict...', end='\t')
#         predictions = {}
#         for i, query_id in enumerate(query_ids):
#             nearest = [(index_ids[j], distances[i, j]) 
#                        for j in predicted_positions[i]]
#             nearest.sort(key=lambda x: x[1])
#             prediction = [to_hex(index_id) for index_id, d in nearest]
#             predictions[to_hex(query_id)] = prediction
#         show_elapsed_time(start)

#         return predictions
    
#     def get_metrics(self, predictions, solution):
#         relevant_predictions = {}

#         for key in solution.keys():
#             if key in predictions:
#                 relevant_predictions[key] = predictions[key]

#         # Mean average precision.
#         mean_average_precision = metrics.MeanAveragePrecision(
#             relevant_predictions, solution, max_predictions=K)
#         print('Mean Average Precision (mAP): {:.4f}'.format(mean_average_precision))

#         return mean_average_precision
    
# ## after unpacking your zipped submission to /kaggle/working, the saved_model.pb
# ## file and attendant directory structure are passed to the the Submission object
# ## for loading.

# submission_object = Submission.load("/kaggle/working/saved_model.pb")


# In[ ]:




