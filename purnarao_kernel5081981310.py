#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


import tensorflow as tf


# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import tensorflow as tf
import pydicom

from os import listdir

from skimage.transform import resize
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

import keras
from keras.applications import ResNet50, VGG16, ResNet101
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
#from keras.applications.resnet101 import preprocess_input as preprocess_resnet_101
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout
from keras.initializers import glorot_normal
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


from tensorflow.nn import sigmoid_cross_entropy_with_logits


# In[4]:


from keras import applications


# In[5]:


applications.ResNet101(include_top=False, weights="imagenet")


# In[6]:


train_brute_force = True
train_anysubtype_network = True


# In[7]:


#MODELOUTPUT_PATH = ""
#brute_force_model_input = "../input/rsna-ih-detection-baseline-models/_bruteforce_best_model.hdf5"
#brute_force_losses_path = "../input/rsna-ih-detection-baseline-models/brute_force_losses.csv"


# In[8]:


import os
os.listdir('../input/')


# In[9]:


def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    rescaled_image[rescaled_image < -1024] = -1024
    return rescaled_image

def set_manual_window(hu_image, custom_center, custom_width):
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    hu_image[hu_image < min_value] = min_value
    hu_image[hu_image > max_value] = max_value
    return hu_image

class Preprocessor:    
    
    def __init__(self, path, backbone, ct_level, ct_width, augment=False):
        self.path = path
        self.backbone = backbone
        self.nn_input_shape = backbone["nn_input_shape"]
        self.ct_level = ct_level
        self.ct_width = ct_width
        self.augment = augment
        
    # 1. We need to load the dicom dataset
    def load_dicom_dataset(self, filename):
        dataset = pydicom.dcmread(self.path + filename)
        return dataset
    
    # 2. We need to rescale the pixelarray to Hounsfield units
    #    and we need to focus on our custom window:
    def get_hounsfield_window(self, dataset, level, width):
        hu_image = rescale_pixelarray(dataset)
        windowed_image = set_manual_window(hu_image, level, width)
        return windowed_image
        
    
    # 3. Resize the image to the input shape of our CNN
    def resize(self, image):
        image = resize(image, self.nn_input_shape)
        return image
    
    # 4. If we like to augment our image, let's do it:
    def augment_img(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                #iaa.Affine(rotate=(-4, 4)),
                iaa.Fliplr(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def fill_channels(self, image):
        filled_image = np.stack((image,)*3, axis=-1)
        return filled_image
    
    def preprocess(self, identifier):
        filename = identifier +  ".dcm"
        dataset = self.load_dicom_dataset(filename)
        level,width = self.get_windowing(dataset)
        windowed_image = self.get_hounsfield_window(dataset, level, width)
        image = self.resize(windowed_image)
        if self.augment:
            image = self.augment_img(image)
        image = self.fill_channels(image)
        return image
    
    def _normalize(self, image):
        x_max = image.max()
        x_min = image.min()
        if x_max != x_min:
            image = (image - x_min) / (x_max - x_min)
            return image
        return np.zeros(image.shape)
    
    def get_windowing(self,data):
        dicom_fields = [data[('0028','1050')].value,  #window center
                        data[('0028','1051')].value] #window width
#                         data[('0028','1052')].value, #intercept
#                         data[('0028','1053')].value] #slope
        return [self.get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    
    def get_first_of_dicom_field_as_int(self,x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)
    
#     def normalize(self, image):
#         image = 2*(image/255) - 1
#         return image


# In[10]:


class DataLoader(Sequence):
    
    def __init__(self, dataframe,
                 preprocessor,
                 batch_size,
                 num_classes=6,
                 shuffle=False,
                 steps=None):
        self.preprocessor = preprocessor
        self.data_ids = dataframe.index.values
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = self.preprocessor.backbone["nn_input_shape"]
        self.preprocess_fun = self.preprocessor.backbone["preprocess_fun"]
        self.num_classes = num_classes
        self.current_epoch=0
        self.steps=steps
        
    # defines the number of steps per epoch
    def __len__(self):
        if self.steps is None:
            return np.int(np.floor(len(self.data_ids) / np.float(self.batch_size)))
        else:
            return np.int(self.steps)
    
    # at the end of an epoch we may like to shuffle the data_ids:
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.RandomState(self.current_epoch).shuffle(self.data_ids)
            self.current_epoch += 1
    
    # should return a batch of images
    def __getitem__(self, step):
        # select the ids of the current batch
        current_ids = self.data_ids[step*self.batch_size:(step+1)*self.batch_size]
        X, y = self.__generate_batch(current_ids)
        return X, y
    
    # collect the preprocessed images and targets of one batch
    def __generate_batch(self, current_ids):
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y = np.empty((self.batch_size, self.num_classes))
        for idx, ident in enumerate(current_ids):
            # Store sample
            image = self.preprocessor.preprocess(ident)
            X[idx] = self.preprocessor._normalize(image)
            # Store class
            y[idx] = self.__get_target(ident)
        return X, y
    
    # extract the targets of one image id:
    def __get_target(self, ident):
        targets = self.dataframe.loc[ident].values
        return targets


# In[11]:


INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"
train_dir = INPUT_PATH + "stage_1_train_images/"
test_dir = INPUT_PATH + "stage_1_test_images/"


# In[12]:


submission = pd.read_csv(INPUT_PATH+"stage_1_sample_submission.csv")
submission.head(7)


# In[13]:


traindf = pd.read_csv(INPUT_PATH + "stage_1_train.csv")
traindf.head()


# In[14]:


label = traindf.Label.values
traindf = traindf.ID.str.rsplit("_", n=1, expand=True)
traindf.loc[:, "label"] = label
traindf = traindf.rename({0: "id", 1: "subtype"}, axis=1)
traindf.head()


# In[15]:


testdf = submission.ID.str.rsplit("_", n=1, expand=True)
testdf = testdf.rename({0: "id", 1: "subtype"}, axis=1)
testdf.loc[:, "label"] = 0
testdf.head()


# In[16]:


traindf = pd.pivot_table(traindf, index="id", columns="subtype", values="label")
traindf.head()


# In[17]:


testdf = pd.pivot_table(testdf, index="id", columns="subtype", values="label")
testdf.head(1)


# In[18]:


#pretrained_models_path = "../input/keras-pretrained-models/"
#listdir("../input/keras-pretrained-models/")


# In[19]:


pretrained_models = {
    "resnet_50": {"weights": "imagenet",
                  "nn_input_shape": (224,224),
                  "preprocess_fun": preprocess_resnet_50},
    "vgg_16": {"weights": "imagenet",
              "nn_input_shape": (224,224),
              "preprocess_fun": preprocess_vgg_16},
    "resnet_101": {"weights": "imagenet",
          "nn_input_shape": (224,224),
          "preprocess_fun": preprocess_resnet_50}
}


# In[20]:


def resnet_50():
    #weights_path = pretrained_models_path + pretrained_models["resnet_50"]["weights"]
    net = ResNet50(include_top=False, weights="imagenet")
    for layer in net.layers:
        layer.trainable = False
    return net

def vgg_16():
    #weights_path = pretrained_models_path + pretrained_models["vgg_16"]["weights"]
    net = VGG16(include_top=False, weights="imagenet")
    for layer in net.layers:
        layer.trainable = False
    return net

def resnet_101():
    net = ResNet101(include_top=False, weights = "imagenet")
    for layer in net.layers:
        layer.trainable = False
    return net


# In[21]:


MODELOUTPUT_PATH = ""
class MyNetwork:
    
    def __init__(self,
                 model_fun,
                 loss_fun,
                 metrics_list,
                 train_generator,
                 dev_generator,
                 epochs,
                 num_classes=6,
                 checkpoint_path=MODELOUTPUT_PATH):
        self.model_fun = model_fun
        self.loss_fun = loss_fun
        self.metrics_list = metrics_list
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epochs = epochs
        self.num_classes = num_classes
        self.checkpoint_path = "_bruteforce_best_model_2.hdf5"
        self.checkpoint = ModelCheckpoint(filepath=self.checkpoint_path,
                                          mode="min",
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          period=1)
        
    def build_model(self):
        base_model = self.model_fun()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        logits = Dense(self.num_classes,
                     kernel_initializer=glorot_normal(seed=11))(x)
                     #kernel_regularizer=l2(0.2),
                     #bias_regularizer=l2(0.2)
        self.model = Model(inputs=base_model.input, outputs=logits)
    
    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False),
                           loss=self.loss_fun, 
                           metrics=self.metrics_list)
    
    def learn(self):
        return self.model.fit_generator(generator=self.train_generator,
                    validation_data=self.dev_generator,
                    epochs=self.epochs,
                    callbacks=[self.checkpoint],
                    #use_multiprocessing=False,
                    workers=8)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, test_generator):
        logits = self.model.predict_generator(test_generator, 
                                              workers=8)
        predictions = 1./(1+np.exp(-logits))
        return predictions


# In[22]:


undersample_seed=0

num_ill_patients = traindf[traindf["any"]==1].shape[0]
num_ill_patients


# In[23]:


healthy_patients = traindf[traindf["any"]==0].index.values
healthy_patients_selection = np.random.RandomState(undersample_seed).choice(
    healthy_patients, size=num_ill_patients, replace=False
)
len(healthy_patients_selection)


# In[24]:


sick_patients = traindf[traindf["any"]==1].index.values
selected_patients = list(set(healthy_patients_selection).union(set(sick_patients)))
len(selected_patients)/2


# In[25]:


remove_list = ['ID_c6bbec638',
 'ID_2fd4dda7c',
 'ID_2ac7f01ed',
 'ID_b055aafa9',
 'ID_cec3997fa',
 'ID_ff9674e53',
 'ID_d6435f3bf',
 'ID_ea0ddbaf9',
 'ID_cbbb50e6d',
 'ID_038f966b9',
 'ID_7607dbd07',
 'ID_6fbc30b5d',
 'ID_445a92ac2',
 'ID_def2a0e9f',
 'ID_9b68c3f5f',
 'ID_ab474037b',
 'ID_f22730d7b',
 'ID_9cdc7295b',
 'ID_91b9ce430',
 'ID_4e61fb0b2',
 'ID_fd5c41761',
 'ID_11c4f9f91',
 'ID_b76de950b',
 'ID_191369dca',
 'ID_b19f52c76',
 'ID_b1cea5abb',
 'ID_767c42624',
 'ID_80a2dbc4a',
 'ID_b494c2115',
 'ID_72dce7784',
 'ID_567a36143',
 'ID_ba7080372',
 'ID_d1b2d9ad0',
 'ID_12e3b6923',
 'ID_75e3f7e5a',
 'ID_b76b13444',
 'ID_b12bb2b16',
 'ID_61c646098',
 'ID_b4adf8739',
 'ID_21053fe7e',
 'ID_fdbfb2c17',
 'ID_d7777de78',
 'ID_e4b636907',
 'ID_28c4609b3',
 'ID_d1afb9750',
 'ID_55f7bbbf2',
 'ID_97cd49666',
 'ID_6a939bc17',
 'ID_b8665a653',
 'ID_3d5d23058',
 'ID_9da128021',
 'ID_403b4fc67',
 'ID_c4575f13b',
 'ID_b966185b8',
 'ID_f698edc00',
 'ID_3e60e696d',
 'ID_ac47ba810',
 'ID_49ecc6164',
 'ID_94463e98f',
 'ID_56ecdf5c1',
 'ID_61d2718d2',
 'ID_1e633cf27',
 'ID_f145c3cf4',
 'ID_d7229490a',
 'ID_5ab140176',
 'ID_155249efa',
 'ID_75cbdae68',
 'ID_0b0e59911',
 'ID_dd3b5bf4e',
 'ID_845f922f4',
 'ID_9bc2b62cc',
 'ID_73dee8958',
 'ID_dabc2a818',
 'ID_807b56a94',
 'ID_18aac96c0',
 'ID_c51cbe76b',
 'ID_17103c79e',
 'ID_950a06268',
 'ID_8fde47d9f',
 'ID_d3fd5220e',
 'ID_a2f9ba4bf',
 'ID_ff012ee5b',
 'ID_2b3671dd9',
 'ID_75d691728',
 'ID_f03370d7c',
 'ID_21d4bd6f3',
 'ID_23d0b13b7',
 'ID_7c08b7fb7',
 'ID_ac1d14c29',
 'ID_3f422852d',
 'ID_10fe2031e',
 'ID_7a02fdbea',
 'ID_b77ba3355',
 'ID_a40f9b2de',
 'ID_a1bb9bc26',
 'ID_bb2a4a01c',
 'ID_155b9c546',
 'ID_830f46cad',
 'ID_d3b76ef6e',
 'ID_af129aa8e',
 'ID_27757c171',
 'ID_15b3ba199',
 'ID_3274f5977',
 'ID_64b44f180',
 'ID_451f60160',
 'ID_0603b315e',
 'ID_a880e377e',
 'ID_ca4a832a1',
 'ID_c1ff9eb46',
 'ID_09aeb0bbd',
 'ID_3ba8a116c',
 'ID_04280250b',
 'ID_081f4d071',
 'ID_a9e98ab5e',
 'ID_a7e689932',
 'ID_85900eb84',
 'ID_1690a6499',
 'ID_676b0cb59',
 'ID_3eb407dd8',
 'ID_ae691dd29',
 'ID_8a35660d5',
 'ID_798d956d0',
 'ID_25de55880',
 'ID_942e2f95b',
 'ID_9a3bba619',
 'ID_66131f4c9',
 'ID_8fd6d5047',
 'ID_3cb1b59bc',
 'ID_9b297fa83',
 'ID_0f8aa5749',
 'ID_c1a3f037f',
 'ID_d4ea87a35',
 'ID_079945c27',
 'ID_6cb797177',
 'ID_5dbe845c1',
 'ID_9ece1bb21',
 'ID_6dcedd2e1',
 'ID_176e4f16d',
 'ID_362423b57',
 'ID_db48a633d',
 'ID_ced5fabca',
 'ID_c60e34466',
 'ID_317330708',
 'ID_291edd834',
 'ID_4e0bdd2ba',
 'ID_ca9462f49',
 'ID_a432727fd',
 'ID_898ff55b6',
 'ID_5bf2ca43f',
 'ID_19306ecc5',
 'ID_3e31d57d0',
 'ID_f242fed92',
 'ID_c11582dc9',
 'ID_6cc19ac41',
 'ID_d1a1c9a6c',
 'ID_6f92e4481',
 'ID_de10fdac2',
 'ID_2e690fe7c',
 'ID_28d6a694f',
 'ID_66accd2e4',
 'ID_6b1a86148',
 'ID_394ffb5fd',
 'ID_a2e178cc7',
 'ID_9dad2eb09',
 'ID_c964e4096',
 'ID_91c508c7a',
 'ID_7e756c43b',
 'ID_7940bb7d0',
 'ID_53c71fb9d',
 'ID_0e9ac1c5f',
 'ID_97e5a203e',
 'ID_ac39010dc',
 'ID_69974dd3e',
 'ID_cade293be',
 'ID_36ab2e72a',
 'ID_8756b0c04',
 'ID_d9840380c',
 'ID_c64131283',
 'ID_d77fa1286',
 'ID_82ec3b736',
 'ID_046ba342c',
 'ID_4e14d0fe8',
 'ID_3bc141392',
 'ID_abcd58e88',
 'ID_1291d1943',
 'ID_6431af929',
 'ID_f4c2157d8',
 'ID_c45659d3d',
 'ID_8c5fc9e44',
 'ID_c2738e8b1',
 'ID_f0d55b727',
 'ID_bc97a5f4f',
 'ID_1bb3b44c7',
 'ID_b9938c32c',
 'ID_8caa68ebd',
 'ID_cf4d76860',
 'ID_7917d368d',
 'ID_dfa4e344a',
 'ID_3dcbd1b5e',
 'ID_be3fb6c17',
 'ID_c6463f07d',
 'ID_cb970c6dc',
 'ID_3d7a23dbb',
 'ID_19f266244',
 'ID_1bc5771a7',
 'ID_8fc348d44',
 'ID_9a36e4b0e',
 'ID_57d6a6455',
 'ID_53f460f86',
 'ID_985fb5e49',
 'ID_f1fe5334e',
 'ID_ae1689e1b',
 'ID_4f0317d23',
 'ID_5ffae2e26',
 'ID_142f85eb8',
 'ID_8144c7120',
 'ID_842e85173',
 'ID_bd4f3f06f',
 'ID_184c541fa',
 'ID_fe7327fab',
 'ID_88b0d8b4f',
 'ID_cef2af72d',
 'ID_8f5d4b696',
 'ID_ea2861e9a',
 'ID_631f0b556',
 'ID_0b2ec2d3f',
 'ID_37c495912',
 'ID_a356248db',
 'ID_4c9fb82af',
 'ID_7714ead69',
 'ID_c07d2cb73',
 'ID_10f34fb10',
 'ID_c6f2d84be',
 'ID_dfaa49f5c',
 'ID_a3feeadf4',
 'ID_6c19c9f7b',
 'ID_0de0ab1d8',
 'ID_60a1f0e24',
 'ID_0144e4030',
 'ID_6508563e0',
 'ID_f188940f9',
 'ID_22069463a',
 'ID_ae7020fd1',
 'ID_a9ab8569f',
 'ID_7e870621c',
 'ID_dd083e12a',
 'ID_d0c52575a',
 'ID_c037d5727',
 'ID_6d7a27643',
 'ID_0e1861e6d',
 'ID_a3128aa77',
 'ID_680b2194c',
 'ID_882cd57de',
 'ID_44d57858e',
 'ID_b194d2a23',
 'ID_ae7b11865',
 'ID_8dc299456',
 'ID_76f88846f',
 'ID_6b15a7649',
 'ID_f23f8e617',
 'ID_aef6c6df9',
 'ID_c35d5c858',
 'ID_8d0ca7742',
 'ID_0c4987103',
 'ID_3c8b72361',
 'ID_a23a8193f',
 'ID_c65ca5466',
 'ID_12a0d6d34',
 'ID_f4891876d',
 'ID_68e45bca7']


# In[26]:


traindf.drop(remove_list,axis = 0,inplace = True)


# In[27]:


#new_traindf = traindf.loc[selected_patients].copy()
traindf["any"].value_counts()
new_traindf = traindf.copy()


# In[28]:


split_seed = 1
train_data, dev_data = train_test_split(new_traindf,
                                        test_size=0.2,
                                        stratify=new_traindf.values,
                                        random_state=split_seed)
print(train_data.shape)
print(dev_data.shape)


# In[29]:


pos_perc_train = train_data.sum() / train_data.shape[0] * 100
pos_perc_dev = dev_data.sum() / dev_data.shape[0] * 100

fig, ax = plt.subplots(2,1,figsize=(20,14))
sns.barplot(x=pos_perc_train.index, y=pos_perc_train.values, palette="Set2", ax=ax[0]);
ax[0].set_title("Target distribution used for training data")
sns.barplot(x=pos_perc_dev.index, y=pos_perc_dev.values, palette="Set2", ax=ax[1]);
ax[1].set_title("Target distribution used for dev data");


# In[30]:


# multilabel loss (optional weighted)
def multilabel_loss(class_weights=None):
    def multilabel_loss_inner(y_true, logits):
        logits = tf.cast(logits, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # compute single class cross entropies:
        contributions = tf.maximum(logits, 0) - tf.multiply(logits, y_true) + tf.log(1.+tf.exp(-tf.abs(logits)))

        # contributions have shape (n_samples, n_classes), we need to reduce with mean over samples to obtain single class xentropies:
        single_class_cross_entropies = tf.reduce_mean(contributions, axis=0)

        # if None, weight equally:
        if class_weights is None:
            loss = tf.reduce_mean(single_class_cross_entropies)
        else:
            weights = tf.constant(class_weights, dtype=tf.float32)
            loss = tf.reduce_sum(tf.multiply(weights, single_class_cross_entropies))
        return loss
    return multilabel_loss_inner


# In[31]:


def multilabel_focal_loss(class_weights=None, alpha=1, gamma=2):
    def mutlilabel_focal_loss_inner(y_true, logits):
        logits = tf.cast(logits, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # compute predictions for each class in logits: 
        y = tf.sigmoid(logits)
        # compute pred_t:
        y_t = tf.where(tf.equal(y_true,1), y, 1.-y)

        # compute single class cross entropies:
        contributions = tf.maximum(logits, 0) - tf.multiply(logits, y_true) + tf.log(1.+tf.exp(-tf.abs(logits)))

        # compute focal loss contributions
        focal_loss_contributions = alpha * tf.multiply(tf.pow(1-y_t, gamma), contributions)

        # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:
        focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)

        # compute the overall loss if class weights are None (equally weighted):
        if class_weights is None:
            focal_loss = tf.reduce_mean(focal_loss_per_class)
        else:
            # weight the single class losses and compute the overall loss
            weights = tf.constant(class_weights, dtype=tf.float32)
            focal_loss = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))

        return focal_loss
    return mutlilabel_focal_loss_inner


# In[32]:


BACKBONE = "resnet_101"
BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
CT_LEVEL = 40
CT_WIDTH = 150

LR = 0.005


# In[33]:


train_preprocessor = Preprocessor(path=train_dir,
                                  backbone=pretrained_models[BACKBONE],
                                  ct_level=CT_LEVEL,
                                  ct_width=CT_WIDTH,
                                  augment=False)

dev_preprocessor = Preprocessor(path=train_dir,
                                backbone=pretrained_models[BACKBONE],
                                ct_level=CT_LEVEL,
                                ct_width=CT_WIDTH,
                                augment=False)

test_preprocessor = Preprocessor(path=test_dir,
                                backbone=pretrained_models[BACKBONE],
                                ct_level=CT_LEVEL,
                                ct_width=CT_WIDTH,
                                augment=False)


# In[34]:


fig, ax = plt.subplots(1,4,figsize=(20,20))


for m in range(4):
    example = train_data.index.values[m]
    title = [col for col in train_data.loc[example,:].index if train_data.loc[example, col]==1]
    if len(title) == 0:
        title="Healthy"
    preprocess_example = train_preprocessor.preprocess(example)
    ax[m].imshow(preprocess_example[:,:,0], cmap="Spectral")
    ax[m].grid(False)
    ax[m].set_title(title);


# In[35]:


train_dataloader = DataLoader(train_data,
                              train_preprocessor,
                              BATCH_SIZE,
                              shuffle=True,
                              steps = 8000)

dev_dataloader = DataLoader(dev_data, 
                            dev_preprocessor,
                            BATCH_SIZE,
                            shuffle=False)

test_dataloader = DataLoader(testdf, 
                             test_preprocessor,
                             TEST_BATCH_SIZE,
                             shuffle=False)


# In[36]:


dev_dataloader.__len__()


# In[37]:


train_dataloader.__len__()


# In[38]:


test_dataloader.__len__()


# In[39]:


my_class_weights = [0.2, 0.16, 0.16, 0.16, 0.16, 0.16]


# In[40]:


if train_brute_force:
    model = MyNetwork(model_fun=vgg_16,
                      loss_fun=multilabel_focal_loss(class_weights=my_class_weights, alpha=0.25, gamma=2),
                      metrics_list=[multilabel_loss(class_weights=my_class_weights),
                                    multilabel_focal_loss(class_weights=my_class_weights, alpha=0.25, gamma=2)],
                      train_generator=train_dataloader,
                      dev_generator=dev_dataloader,
                      epochs=5,
                      num_classes=6)
    model.build_model()
    model.compile_model()
    history = model.learn()
    
    print(history.history.keys())
    losses_df = pd.DataFrame(history.history["loss"], columns=["multi_wfocal_loss_train"])
    losses_df.loc[:, "multi_wfocal_loss_val"] = history.history["val_loss"]
    losses_df.loc[:, "multi_w_loss_train"] = history.history["multilabel_loss_inner"]
    losses_df.loc[:, "multi_w_loss_val"] = history.history["val_multilabel_loss_inner"]
    losses_df.to_csv("brute_force_losses.csv", index=False)
    
    plt.figure(figsize=(20,5))
    plt.plot(history.history["loss"], 'o-')
    plt.plot(history.history["val_loss"], 'o-')
    
    test_pred = model.predict(test_dataloader)
else:
    losses_df = pd.read_csv(brute_force_losses_path)
    model = MyNetwork(model_fun=resnet_50,
                      loss_fun=multilabel_focal_loss(class_weights=my_class_weights,
                                                     alpha=0.25,keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
                                                     gamma=2),
                      metrics_list=[keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
                          multilabel_loss(class_weights=my_class_weights),
                          multilabel_focal_loss(class_weights=my_class_weights,
                                                alpha=0.25,
                                                gamma=2)],
                      train_generator=train_dataloader,
                      dev_generator=dev_dataloader,
                      epochs=5,
                      num_classes=6)
    model.build_model()
    test_pred = model.predict(test_dataloader)


# In[41]:


testdf.shape


# In[42]:


test_pred = model.predict(test_dataloader)
test_pred.shape


# In[43]:


def turn_to_submission(test_data, pred, submission):
    df = pd.DataFrame(pred, columns=test_data.columns, index=test_data.index)
    df = df.stack().reset_index()
    df.loc[:, "ID"] = df.id.str.cat(df.subtype, sep="_")
    df = df.drop(["id", "subtype"], axis=1)
    df = df.set_index("ID")
    df = df.rename({0: "Label"}, axis=1)
    submission.Label = submission.ID.map(df.Label)
    return submission


# In[44]:


bruteforce_submission = turn_to_submission(testdf, test_pred, submission)
bruteforce_submission.head()


# In[ ]:





# In[45]:


bruteforce_submission.to_csv("bruteforce_submission.csv", index=False)


# In[46]:


dev_pred = model.predict(dev_dataloader)


# In[47]:


class AnySubtypeNetwork(MyNetwork):
    
    def __init__(self,
                 model_fun,
                 loss_fun,
                 metrics_list,
                 train_generator,
                 dev_generator,
                 epochs,
                 num_subtype_classes=5):
        MyNetwork.__init__(self, 
                           model_fun=model_fun,
                           loss_fun=loss_fun,
                           metrics_list=metrics_list,
                           train_generator=train_generator,
                           dev_generator=dev_generator,
                           epochs=epochs,
                           num_classes=num_subtype_classes)
    
    def build_model(self):
        base_model = self.model_fun()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        any_logits = Dense(1, kernel_initializer=glorot_normal(seed=11),
                     kernel_regularizer=l2(0.2),
                     bias_regularizer=l2(0.2))(x)
        any_pred = Activation("sigmoid", name="any_predictions")(any_logits)
        x = concatenate([any_pred, x])
        sub_logits = Dense(self.num_classes,
                           name="subtype_logits",
                           kernel_initializer=glorot_normal(seed=12),
                           kernel_regularizer=l2(0.2),
                           bias_regularizer=l2(0.2))(x) 
        self.model = Model(inputs=base_model.input, outputs=[any_pred, sub_logits])
    
    def compile_model(self):
        self.model.compile(optimizer=Adam(LR),
                           loss=['binary_crossentropy', multilabel_focal_loss(alpha=0.25, gamma=2)], 
                           metrics=self.metrics_list)


# In[48]:


class AnySubtypeDataLoader(DataLoader):
    
    def __init__(self, dataframe,
                 preprocessor,
                 batch_size,
                 num_classes=5,
                 shuffle=False,
                 steps=None):
        DataLoader.__init__(self, 
                            dataframe=dataframe,
                            preprocessor=preprocessor,
                            batch_size=batch_size,
                            num_classes=num_classes,
                            shuffle=shuffle,
                            steps=steps)
    
    # should return a batch of images
    def __getitem__(self, step):
        # select the ids of the current batch
        current_ids = self.data_ids[step*self.batch_size:(step+1)*self.batch_size]
        X, y_any, y_subtype = self.__generate_batch(current_ids)
        return X, [y_any, y_subtype]
    
    # collect the preprocessed images and targets of one batch
    def __generate_batch(self, current_ids):
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y_subtype = np.empty((self.batch_size, self.num_classes))
        y_any = np.empty((self.batch_size, 1))
        for idx, ident in enumerate(current_ids):
            # Store sample
            image = self.preprocessor.preprocess(ident)
            X[idx] = self.preprocess_fun(image)
            # Store class
            y_any[idx], y_subtype[idx] = self.__get_target(ident)
        return X, y_any, y_subtype
    
    # extract the targets of one image id:
    def __get_target(self, ident):
        y_any = self.dataframe.loc[ident, "any"]
        y_subtype = self.dataframe.drop("any", axis=1).loc[ident].values
        return y_any, y_subtype


# In[49]:


train_dataloader = AnySubtypeDataLoader(train_data,
                                        train_preprocessor,
                                        BATCH_SIZE,
                                        shuffle=True,
                                        steps=100)
dev_dataloader = AnySubtypeDataLoader(dev_data, 
                                      dev_preprocessor,
                                      BATCH_SIZE,
                                      shuffle=False,
                                      steps=100)
test_dataloader = AnySubtypeDataLoader(testdf, 
                                       test_preprocessor,
                                       TEST_BATCH_SIZE,
                                       shuffle=False)


# In[50]:


if train_anysubtype_network:
    model = AnySubtypeNetwork(model_fun=resnet_50,
                              loss_fun=None,
                              metrics_list=None,
                              train_generator=train_dataloader,
                              dev_generator=dev_dataloader,
                              epochs=5) 
    model.build_model()
    model.compile_model()
    history = model.learn()
    
    print(history.history.keys())
    
    plt.figure(figsize=(20,5))
    plt.plot(history.history["loss"], 'o-')
    plt.plot(history.history["val_loss"], 'o-')


# In[51]:


train_dir


# In[52]:


f = os.listdir(train_dir)


# In[53]:


len(f)

