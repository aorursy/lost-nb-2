#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

get_ipython().system('ls -l ../input/trainandtest')

#mode = 'train' # internet on
mode = 'test' # internet off
#small = True # small data set
small = False

path = '../input/severstal-steel-defect-detection/'
path_self = '../input/trainandtest/'
os.environ['path_self'] = path_self

train_rate = 0.9
prob_threshold = 0.5

#!cat ../input/trainandtest/submission.csv
#!cat ../input/severstal-steel-defect-detection/train.csv
#import numpy as np, pandas as pd
#submission = pd.read_csv("../input/trainandtest/submission.csv")
#type(submission)


# In[2]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt, time
from PIL import Image 
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv(path + 'train.csv')

# RESTRUCTURE TRAIN DATAFRAME
#train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('_')[0]) # 追加
train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna('',inplace=True); 
train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

files = os.listdir(path+"train_images/")
print("train_images == train.csv:",set(files) == set(train2['ImageId']))

if small == True: train2 = train2[:256]

im = Image.open(path+"train_images/"+train2['ImageId'].iloc[0])
print(im.format,im.size,im.mode)

idx = int(train_rate*len(train2)); print(len(train2),idx)
train_set = train2.iloc[:idx]
val_set = train2.iloc[idx:]

# separate no defect, e1, e2, e3, e4 data
train2_no = train2[train2['count']==0]
train2_no.reset_index(inplace=True,drop=True)
train2_e1 = train2[train2['e1']!='']
train2_e1.reset_index(inplace=True,drop=True)
train2_e2 = train2[train2['e2']!='']
train2_e2.reset_index(inplace=True,drop=True)
train2_e3 = train2[train2['e3']!='']
train2_e3.reset_index(inplace=True,drop=True)
train2_e4 = train2[train2['e4']!='']
train2_e4.reset_index(inplace=True,drop=True)

idx = int(train_rate*len(train2_no))
train_no_set = train2_no.iloc[:idx]
val_no_set = train2_no.iloc[idx:]
idx = int(train_rate*len(train2_e1))
train_e1_set = train2_e1.iloc[:idx]
val_e1_set = train2_e1.iloc[idx:]
idx = int(train_rate*len(train2_e2))
train_e2_set = train2_e2.iloc[:idx]
val_e2_set = train2_e2.iloc[idx:]
idx = int(train_rate*len(train2_e3))
train_e3_set = train2_e3.iloc[:idx]
val_e3_set = train2_e3.iloc[idx:]
idx = int(train_rate*len(train2_e4))
train_e4_set = train2_e4.iloc[:idx]
val_e4_set = train2_e4.iloc[idx:]

print('train_no_set {}\n'.format(len(train_no_set)), train_no_set.head())
print('val_no_set {}\n'.format(len(val_no_set)), val_no_set.head())
print('train_e1_set {}\n'.format(len(train_e1_set)), train_e1_set.head())
print('val_e1_set {}\n'.format(len(val_e1_set)), val_e1_set.head())
print('train_e2_set {}\n'.format(len(train_e2_set)), train_e2_set.head())
print('val_e2_set {}\n'.format(len(val_e2_set)), val_e2_set.head())
print('train_e3_set {}\n'.format(len(val_e3_set)), train_e3_set.head())
print('val_e3_set {}\n'.format(len(val_e3_set)), val_e3_set.head())
print('train_e4_set {}\n'.format(len(train_e4_set)), train_e4_set.head())
print('val_e4_set {}\n'.format(len(val_e4_set)), val_e4_set.head())

# TEST DATAFRAME
test = pd.read_csv(path + 'sample_submission.csv')
test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])
test2 = pd.DataFrame({'ImageId':test['ImageId'][::4]})
test2.reset_index(inplace=True,drop=True)

files = os.listdir(path+"test_images/")
print("test_images == sample_submission.csv:",set(files) == set(test2['ImageId']))

if small == True: test2 = test2[:64]


# In[3]:


test2.head()


# In[4]:


def rle2mask(rle):
    height= 256
    width = 1600

    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((height,width) ,dtype=np.uint8)   

    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1 # 0 origin every 2
    lengths = array[1::2]  # 1 origin every 2    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    
    return mask.reshape( (height,width), order='F' ) # column order
'''
def mask2rle(mask):
    if np.sum(mask) == 0: return ''
    ar = mask.flatten(order='F')
    EncodedPixel = ''
    l = 0
    for i in range(len(ar)):
        if ar[i] == 0:
            if l > 0:
                if EncodedPixel != '': EncodedPixel += ' '
                EncodedPixel += str(st+1)+' '+str(l)
                l = 0
        else: # == 1
            if l == 0: st = i
            l += 1
    return EncodedPixel
'''
def mask2rle(mask):
    if np.sum(mask) == 0: return ''
    pixels = mask.flatten(order='F')
    pixels = np.concatenate(([0],pixels,[0]),axis=None) #　add 0 at both ends
    runs = np.where(pixels[1:] != pixels[:-1])[0]+1 # get change index as array([...],) 
    runs[1::2] -= runs[::2] # get length
    return ' '.join(str(x) for x in runs) # blank is inserted between str(x)

def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 

def mask2pad(mask, pad=2):
    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT
    w = mask.shape[1]
    h = mask.shape[0]
    
    # MASK UP
    for k in range(1,pad,2):
        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK DOWN
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK LEFT
    for k in range(1,pad,2):
        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)
        mask = np.logical_or(mask,temp)
    # MASK RIGHT
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)
        mask = np.logical_or(mask,temp)
    
    return mask 

rle = train2.iloc[0]['e1']
print(rle)
mask = rle2mask(rle)
print(mask.shape)

#mask = np.array([[1,1,1],
#               [1,0,0]])
rle1 = mask2rle(mask)
print(rle == rle1)


# In[5]:


"""
df('e1','e2','e3','e4') added ('pixels_e1','pixels_e2',...)
"""
def defects_rate(df):
    min_pixels = []
    total = len(df)
    print('Total:', total)
    for j in range(4):
        pixels_e = 'pixels_e'+str(j+1)
        defects_e = 'e'+str(j+1)
        df[pixels_e] = [np.sum(rle2mask(x)) for x in df[defects_e]]
        defects = sum(df[pixels_e] != 0)
        if defects > 0:
            defects_min = min(df[df[pixels_e] > 0][pixels_e])
        else:
            defects_min = 0
        min_pixels.append(defects_min)
        defects_max = max(df[pixels_e])
        print(defects_e,':',defects, '({:.3%})'.format(defects/total),
              'max:',defects_max,'min:',defects_min)
    return min_pixels
if mode == 'train' or mode == 'test':
    min_pixels = defects_rate(train2)


# In[6]:


import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size = 32, subset="train", shuffle=False, 
                 preprocess=None, rate=0.0, flipH=False, flipV=False, divide=1):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.IDs = {}
        self.rate = rate
        self.flipH = flipH
        self.flipV = flipV
        self.length = int(len(self.df)*divide)
        self.batches = int(np.ceil(self.length/ self.batch_size)) # round up
#        self.res = len(self.df) // self.batch_size
        self.divide = int(divide)
        self.H = 256
        self.W = int(1600/divide)
        
        if self.subset == "train" or self.subset == "classify":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
        self.on_epoch_end()

    def __len__(self):
        return self.batches
#        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
#        self.indexes = np.arange(len(self.df))
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        st = self.batch_size*index
        en = self.batch_size*(index+1)
        if en > self.length: en = self.length
        X = np.empty((en-st,self.H,self.W,3),dtype=np.float32)
        if self.subset == 'train':
            y = np.empty((en-st,self.H,self.W,4),dtype=np.uint8)
        elif self.subset == 'classify':
            c = np.zeros((en-st),dtype=np.uint8)

        indexes = self.indexes[st:en]
#        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
        for i, ind in enumerate(indexes):
            f_ind = ind // self.divide
            f_div = ind % self.divide
#            print(f_ind,f_div)
            f = self.df['ImageId'].iloc[f_ind]
            if self.divide ==1:
#                self.IDs[index*self.batch_size+i]=f
                self.IDs[i]=f
            else:
#                self.IDs[index*self.batch_size+i]=f+'_'+str(f_div)
                self.IDs[i]=f+'_'+str(f_div)
            img = Image.open(self.data_path + f)
#            print(img.size)
#            assert img.size == (1600,256) # (W,H)
            x = img.resize((1600,256))
            x = np.array(x,dtype=np.float32) # (W,H)->(H,W)
#            print(x.shape)
            x = x[:,f_div*self.W:(f_div+1)*self.W]
#            print(x.shape)
            if self.rate > 0.0:
                if np.random.rand() <= self.rate and self.flipV == True: flipV = True
                else: flipV = False
                    
                if np.random.rand() <= self.rate and self.flipH == True: flipH = True
                else: flipH = False
            else:
                flipV = self.flipV
                flipH = self.flipH
                    
            if flipV == True: x = x[:,::-1]
            if flipH == True: x = x[::-1,:]
            X[i,] = x
#            print("H:",flipH, "V:",flipV)
            if self.subset == 'train': 
                for j in range(4):
#                    m = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
                    m = rle2mask(self.df['e'+str(j+1)].iloc[f_ind])
                    m = m[:,f_div*self.W:(f_div+1)*self.W]
                    if flipV == True: m = m[:,::-1]
                    if flipH == True: m = m[::-1,:]
                    y[i,:,:,j] = m
            elif self.subset == 'classify':
                for j in range(4):
                    m = rle2mask(self.df['e'+str(j+1)].iloc[f_ind])
                    m = m[:,f_div*self.W:(f_div+1)*self.W]
                    if np.sum(m) != 0:
                        c[i]= 1
                        break
              
        if self.preprocess!=None: X = self.preprocess(X)
        if self.subset == 'train': return X,y
        elif self.subset == 'classify':return X,c
        else: return X

#        if self.subset == 'train': return X.reshape((-1,256,400,3)),y.reshape((-1,256,400,4))
#        else: return X.reshape((-1,256,400,3)) #? flip every part 


# In[7]:


#
#  Generator check
#
def plot_imageMask_predMask(X, y=None, preds=None, IDs=None):
    print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')
    
    def make_image_masked(img, mask, extra):
        for j in range(4):
            msk = mask[:,:,j]
            msk = mask2pad(msk,pad=3)
            msk = mask2contour(msk,width=3)
            if np.sum(msk)!=0: extra += ' '+str(j+1)+'('+str(np.sum(msk))+')'
            if j==0: # yellow
                img[msk==1,0] = 235 
                img[msk==1,1] = 235
            elif j==1: img[msk==1,1] = 210 # green
            elif j==2: img[msk==1,2] = 255 # blue
            elif j==3: # magenta
                img[msk==1,0] = 255
                img[msk==1,2] = 255
        return img, extra        

    batch_size = X.shape[0]
    # DISPLAY IMAGES WITH DEFECTS
#    plt.figure(figsize=(14,50)) #20,18
    if batch_size != 1:
        if preds is None:
            fig, ax = plt.subplots(batch_size, 1, figsize=(14,3*batch_size))#(14,50))
        else:
            if preds.dtype != 'uint8':
                fig, ax = plt.subplots(batch_size, 5, figsize=(18,1*batch_size))#(20,36))
            else:
                fig, ax = plt.subplots(batch_size, 2, figsize=(14,2*batch_size))#(20,36))
    else:
        plt.figure(figsize=(14,50)) #20,18
        
    for k in range(batch_size):
#        plt.subplot(batch_size,1,k+1)
        img = X[k,]
        img = Image.fromarray(img.astype('uint8'))
        img = np.array(img)
        extra = ''
        if y is not None:
            if np.sum(y[k]) != 0: extra = '  has defect'
            img_y = np.copy(img)
            img_y, extra_y = make_image_masked(img_y,y[k],extra)
        else:
            img_y = np.copy(img)
            extra_y = ''
                    
        if batch_size != 1:
            if preds is None:
                if IDs is not None:ax[k].set_title(IDs[k] + extra_y)
                ax[k].axis('off')
                ax[k].imshow(img_y)
            else:
                if IDs is not None:ax[k,0].set_title(IDs[k] + extra_y)
                ax[k,0].axis('off')
                ax[k,0].imshow(img_y)
        else:
            if IDs is not None:plt.title(IDs[k]+extra_y)
            plt.axis('off') 
            plt.imshow(img_y)           
            
        # prediction
        if preds is not None:
            if preds.dtype != 'uint8':
                for j in range(4):
                    ax[k,j+1].set_title(" max pixel({:.3f})"
                                    .format(np.max(preds[k,:,:,j])))
                    ax[k,j+1].axis('off')
                    ax[k,j+1].imshow(preds[k,:,:,j])
            else:
                img_p = img.copy()
                img_p, extra_p = make_image_masked(img_p,preds[k],'')
                if IDs is not None:ax[k,1].set_title(IDs[k] + extra_p)
                ax[k,1].axis('off')
                ax[k,1].imshow(img_p)
            
    plt.show()

print("Classify Data")
classify_check = DataGenerator(train2[:2],subset='classify')
(X,c) = classify_check[0]
print(c)
print("Classify Data-2")
classify_check = DataGenerator(train2[:2],subset='classify',divide=2)
(X,c) = classify_check[0]
print(c)

print("Test Data")
test_check = DataGenerator(test2[:2],subset='test')
X = test_check[0]
print(X.shape[0])
print(test_check.indexes)
print(test_check.IDs)
plot_imageMask_predMask(X[:2], IDs=test_check.IDs)
print("Test Data-2")
test_check = DataGenerator(test2[:2],subset='test',divide=2)
X = test_check[0]
print(X.shape[0])
print(test_check.indexes)
print(test_check.IDs)
plot_imageMask_predMask(X[:4], IDs=test_check.IDs)

print("Train Data")
data_check = DataGenerator(train2[:2])
(X,y) = data_check[0]
print(X.shape[0])
print(data_check.indexes)
print(data_check.IDs)
#print(train2['ImageId'].iloc[data_check.indexes])

for i, (X, y) in enumerate(data_check):
    print(i, X.shape[0])
print(data_check.IDs)
    
plot_imageMask_predMask(X[:2], y, IDs=data_check.IDs)

print("Train Data-2")
data_check = DataGenerator(train2[:2],divide=2)
(X,y) = data_check[0]
print(data_check.IDs)
    
plot_imageMask_predMask(X[:4], y, IDs=data_check.IDs)


print("flipH")
data_check = DataGenerator(train2[:2], rate=0.5, flipH=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:2], y, IDs=data_check.IDs)

print("flipH-2")
data_check = DataGenerator(train2[:2], divide=2, rate=0.5, flipH=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:4], y, IDs=data_check.IDs)


print("flipV")
data_check = DataGenerator(train2[:2], rate=0.5, flipV=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:2], y, IDs=data_check.IDs)
print("flipV-2")
data_check = DataGenerator(train2[:2], divide=2, rate=0.5, flipV=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:4], y, IDs=data_check.IDs)


print("flipH,flipV")
data_check = DataGenerator(train2[:2],rate=0.5, flipH=True, flipV=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:2], y, IDs=data_check.IDs)
print("flipH,flipV-2")
data_check = DataGenerator(train2[:2], divide=2, rate=0.5, flipH=True, flipV=True)
(X,y) = data_check[0]
plot_imageMask_predMask(X[:4], y, IDs=data_check.IDs)


print("y,preds")
plot_imageMask_predMask(X[:8], y, y, IDs=data_check.IDs)


# In[8]:


from keras import backend as K

from keras import losses

# COMPETITION METRIC
# Variable
def dice_coef(y_true, y_pred, smooth=1):
#    print("dice_coef:",y_true.shape,y_pred.shape,y_true.dtype,y_pred.dtype)
#    raise Exception
#    assert y_true.shape[0] == y_pred.shape[0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

# ndarray
def dice_coef_nd(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.*intersection, (np.sum(y_true_f)+np.sum(y_pred_f))

def dice_loss(y_true,y_pred):
    return K.constant(1.0) - dice_coef(y_true,y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# In[9]:


from keras.applications.vgg16 import VGG16,preprocess_input
#from keras.applications.densenet import DenseNet121,preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import dill

def make_classify_model():
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)
#    base_model = DenseNet121(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    # and a logistic layer -- let's say we have 2-values
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    classify_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)

    classify_model.summary()
    
    return classify_model, preprocess_input

def train_classify_model(model,train_set,val_set,preprocess=None,epochs=30):
    train_batches = DataGenerator(train_set,shuffle=True,preprocess=preprocess,
                                  rate=0.5, flipH=True, flipV=True, subset='classify')
    valid_batches = DataGenerator(val_set,preprocess=preprocess,subset='classify')
    
    f = open("classify_preprocess.dill","wb")
    dill.dump(preprocess,f)
    f.close

    history = model.fit_generator(train_batches,
                              validation_data = valid_batches, 
                              epochs = epochs, 
                              verbose=1,
                              callbacks=[EarlyStopping(patience=4,min_delta=1.e-6),
                                        ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                                         ModelCheckpoint("classify_model.h5", 
                                                         monitor='val_acc', 
                                                         verbose=1, 
                                                         save_best_only=True, 
                                                         save_weights_only=False, 
                                                         mode='auto', period=1)])
    plt.figure(figsize=(15,5))
    plt.plot(range(history.epoch[-1]+1),history.history['val_acc'],label='val_acc')
    plt.plot(range(history.epoch[-1]+1),history.history['acc'],label='acc')
    plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy');plt.legend(); 
    plt.show()

if mode == 'train':
    (classify_model, classify_preprocess) = make_classify_model()
    classify_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
    classify_history = train_classify_model(classify_model,train_set,val_set,
                                        preprocess=classify_preprocess)


# In[10]:


if mode == 'train':
    #!pip install segmentation-models
    #!pip install git+https://github.com/qubvel/segmentation_models
    get_ipython().system('pip download --no-binary :all: segmentation-models==0.2.1')
    get_ipython().system('pip download --no-deps --no-binary :all: image-classifiers==0.2.0')
    get_ipython().system('pip install segmentation_models-0.2.1.tar.gz')
    get_ipython().system('pip install image_classifiers-0.2.0.tar.gz')
else:
    get_ipython().system('pip install $path_self/segmentation_models-0.2.1.tar.gz')
    get_ipython().system('pip install $path_self/image_classifiers-0.2.0.tar.gz')

get_ipython().system('ls -l ../input')


# In[11]:


from segmentation_models import Unet, PSPNet
from segmentation_models.backbones import get_preprocessing
#from segmentation_models.losses import  DiceLoss
from segmentation_models.losses import bce_jaccard_loss#, bce_dice_loss
from segmentation_models.metrics import iou_score

import dill

backbone = 'resnet34' #'vgg19' #'resnet50'#'vgg16' #'resnet34' 
MODEL = Unet

# LOAD UNET WITH PRETRAINING FROM IMAGENET
def make_model(MODEL, backbone):
#preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0
    preprocess = get_preprocessing(backbone)
    model = MODEL(backbone, freeze_encoder=True,encoder_weights='imagenet',  
             input_shape=(256, 1600, 3), 
                  classes=4, activation='sigmoid')
#    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    #trainable = False
    #for i, layer in enumerate(model.layers):
    #    if 'stage4' in layer.name: trainable = True
    #    layer.trainable = trainable
    #    print(layer.name, layer.trainable)
    model.summary()
    

    
    return model, preprocess

if mode == 'train':
    model, preprocess = make_model(MODEL, backbone)
    print(preprocess)


# In[12]:


#from keras.utils import plot_model
#plot_model(model, to_file='./model.png')
#!ls -l


#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(model).create(prog='dot', format='svg')) 


# In[13]:


# TRAIN AND VALIDATE MODEL
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import dill

def train(model,preprocess,train_set,val_set,epochs=30):
    train_batches = DataGenerator(train_set,shuffle=True,preprocess=preprocess,
                                  rate=0.5, flipH=True, flipV=True,batch_size=10)
    valid_batches = DataGenerator(val_set,preprocess=preprocess,batch_size=10)
    
    f = open("preprocess.dill","wb")
    dill.dump(preprocess,f)
    f.close


    history = model.fit_generator(train_batches,
                              validation_data = valid_batches, 
                              epochs = epochs, 
                              verbose=1,
                              callbacks=[EarlyStopping(patience=5,min_delta=1.e-6),
                                        ReduceLROnPlateau(monitor='val_dice_coef', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                                        ModelCheckpoint("UNET.h5", 
                                                        monitor='val_dice_coef', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        save_weights_only=False, 
                                                        mode='max', period=1)])
    plt.figure(figsize=(15,5))
    plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='val_dice_coef')
    plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='trn_dice_coef')
    plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Dice_coef');plt.legend(); 
    plt.show()
if mode == 'train':
    
#    print("e2")
#    train(model,train_e2_set,val_e2_set, epochs=5)
    
#    print("e2+e4")
#    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
#    train(model,
#          pd.concat([train_e2_set,train_e4_set]),
#          pd.concat([val_e2_set,val_e4_set]),
#          epochs=5)

#    print("e2+e4+e1")
#    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
#    train(model,
#          pd.concat([train_e2_set,train_e4_set,train_e1_set]),
#          pd.concat([val_e2_set,val_e4_set,val_e1_set]),
#          epochs=5)

#    print("e2+e4+e1+e3")
#    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
#    train(model,
#          pd.concat([train_e2_set,train_e4_set,train_e1_set,train_e3_set]),
#          pd.concat([val_e2_set,val_e4_set,val_e1_set,val_e3_set]),
#          epochs=5)
    
#    print("Total")
#    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])    
#    train(model,train_set,val_set)

    print("e2+e4+e1+e3")
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
    train(model,preprocess,
          pd.concat([train_e2_set,train_e4_set,train_e1_set,train_e3_set]),
          pd.concat([val_e2_set,val_e4_set,val_e1_set,val_e3_set]) )

#    print(K.eval(model.optimizer.lr))
#print(model.optimizer.get_config())
#pd.concat([val_e2_set, val_e4_set])


# In[14]:


# LOAD MODEL
from keras.models import load_model
import dill

def load_objects(load_path): 

    model = load_model(load_path+'UNET.h5',
                custom_objects={'dice_coef':dice_coef,
                                'bce_dice_loss':bce_dice_loss})
    f = open(load_path+"preprocess.dill","rb")
    preprocess = dill.load(f)
    f.close
    print(preprocess)
        
    classify_model = load_model(load_path +'classify_model.h5')
    f = open(load_path+"classify_preprocess.dill","rb")
    classify_preprocess = dill.load(f)
    f.close
    print(classify_preprocess)

    return classify_model, classify_preprocess, model, preprocess

# LOAD model, preprocess
if mode == 'train':
    classify_model, classify_preprocess, model, preprocess = load_objects('./')
else:
    classify_model, classify_preprocess, model, preprocess = load_objects(path_self)


# In[15]:



def prob2mask(prob,threshold):
    mask = np.zeros(prob.shape,dtype='uint8')
    mask[prob>=threshold] = 1
    return mask

def filter_defects(val_set,classify_preprocess, classify_model):
    ind =[]
    cls = []
    B = DataGenerator(val_set,batch_size=32,
                      preprocess=classify_preprocess,subset='classify')
    for X, c in B:
        f = classify_model.predict(X,batch_size=X.shape[0])
        f = f.reshape(-1)
        ind.extend(f>0.5)
        cls.extend(c)
    return np.array(ind), np.array(cls)

#defects, cls = filter_defects(val_set,classify_preprocess, classify_model)
#for i in range(10):
#    print( i, " ", defects[i], " ", cls[i])
#pred_real = defects.astype(np.int) - cls.astype(np.int)
#print("Total:", len(pred_real),
#      "pred>real(defect<-no)",np.sum(pred_real>0),
#      "pred == real",np.sum(pred_real==0),
#      "pred<real(no<-defect)",np.sum(pred_real<0))

def evaluate_batch(val_set, preprocess, M, threshold, divide=1):
    
    B = DataGenerator(val_set,batch_size=64,preprocess=preprocess,divide=divide)

    total = 0
    defects = np.zeros(4,dtype='uint')
    pixel_min = np.zeros(4,dtype='uint') 
    pixel_max = np.zeros(4,dtype='uint')
    fact = 0.0
    denom = 0.0
    for X, y in B:
        if defects is None: defects = np.zeros((X.shape[1],X.shape[2]))
        total += X.shape[0]
        p = M.predict(X,batch_size=X.shape[0])
#        print(p.shape,p[0,10,10,:])
#        plot_imageMask_predMask(X,y,p,IDs=B.IDs)
        
#        XH = X[:,:,::-1]
#        pH = M.predict(XH,batch_size=XH.shape[0])
#        plot_imageMask_predMask(XH,y[:,:,::-1],pH,IDs=B.IDs)
        
#        XV = X[:,::-1,:]
#        pV = M.predict(XV,batch_size=XV.shape[0])
#        plot_imageMask_predMask(XV,y[:,::-1,:],pH,IDs=B.IDs)
        
        XHV = X[:,::-1,::-1]
        pHV = M.predict(XHV,batch_size=XHV.shape[0])
#        plot_imageMask_predMask(XHV,y[:,::-1,::-1],pHV,IDs=B.IDs)
#        break
        
#        pH = pH[:,:,::-1]
#        pV = pV[:,::-1,:]
        pHV= pHV[:,::-1,::-1]
        
#        preds = (p + pH + pV + pHV)/4.0
        preds = (p+pHV)/2.0

        mask = prob2mask(preds,threshold)

        for i in range(X.shape[0]):
            for j in range(4):
                pixel = np.sum(mask[i,:,:,j],dtype='uint')
                if pixel < min_pixels[j]:
                    mask[i,:,:,j] = 0
                    pixel = 0 
                if pixel > 0:
                    if pixel_min[j] == 0: pixel_min[j] = pixel
                    elif pixel < pixel_min[j]: pixel_min[j] = pixel
                
                if pixel > pixel_max[j]: pixel_max[j] = pixel
                if pixel > 0:defects[j] += 1
#                if pixel > 0:print(i,j,defects[j],pixel_min[j],pixel_max[j])
        
        (f, d) = dice_coef_nd(y,mask)
        fact += f
        denom += d
    return fact, denom,total,defects,pixel_min,pixel_max


# PREDICT FROM VALIDATION SET (USE ALL)
if mode == 'train':
    defects = sum(val_set['count']!=0)
    print("real defect rate:{:.3}={}/{}".format(defects/len(val_set), defects, len(val_set)))

    filter_defects, cls = filter_defects(val_set,classify_preprocess, classify_model)
    pred_real = filter_defects.astype(np.int) - cls.astype(np.int)
    filter_total = np.sum(filter_defects==False)
    filter_mistake_defect = np.sum(pred_real>0)
    filter_mistake_no = np.sum(pred_real<0)
    defect_set = val_set[filter_defects]
    # defect_set.index.values
    fact, denom,total,defects,pixel_min,pixel_max = evaluate_batch(defect_set, #val_set,
                                                                    preprocess,
                                                                    model,
                                                                    prob_threshold)
    dice_coef_ = (fact+1.0)/(denom+1.0)   
    print("filtered dice_coef:{:.3} total:{}".format(dice_coef_, total))
    for j in range(4):
        print("e{}:{}({:.3%}) max:{} min:{}".format(
                j+1, defects[j], defects[j]/total, pixel_max[j], pixel_min[j]))

# Weighted average is some indication
    weighted_dice = (dice_coef_*(len(val_set)-filter_total)+filter_total)/len(val_set)
    print("total dice_coef:{:.3} total:{}".format(weighted_dice, len(val_set)))


# In[16]:


import time
def predict_batch(test2, preprocess, model, threshold, mini_pixels,
                     classify_preprocess, classify_model ):
    
    B = DataGenerator(test2,batch_size=32,preprocess=None,subset='test')
    
    ImageId_ClassIds = []
    EncodedPixels = []

    for X in B:
        #
        # classify no-defect or defect
        #
        Y = classify_preprocess(X)
        d = classify_model.predict(Y,batch_size=Y.shape[0])
        
        YH = Y[:,:,::-1]
        dH = classify_model.predict(YH,batch_size=YH.shape[0])
                
        YV = Y[:,::-1,:]
        dV = classify_model.predict(YV,batch_size=YV.shape[0])
 
        YHV = Y[:,::-1,::-1]
        dHV = classify_model.predict(YHV,batch_size=YHV.shape[0])

        defects = (d+dH+dV+dHV)/4.0
        
        defects = defects.reshape(-1)
        ind = defects > threshold

        #
        # classify and segment defect
        #
        Y = X[ind]
        Y = preprocess(Y)
        p = model.predict(Y,batch_size=Y.shape[0])
        
        YH = Y[:,:,::-1]
        pH = model.predict(YH,batch_size=YH.shape[0])
        
        YV = Y[:,::-1,:]
        pV = model.predict(YV,batch_size=YV.shape[0])
        
        YHV = Y[:,::-1,::-1]
        pHV = model.predict(YHV,batch_size=YHV.shape[0])
        
        pH = pH[:,:,::-1]
        pV = pV[:,::-1,:]
        pHV= pHV[:,::-1,::-1]
        
        preds = (p + pH + pV + pHV)/4.0

        mask = prob2mask(preds,threshold)
        ii = 0
        for i, defect in enumerate(ind):
            ImageId = B.IDs[i]
            for j in range(4):
                Id = ImageId + '_'+str(j+1)
                if defect == False:
                    EncodedPixel = ''
                else:
                    if np.sum(mask[ii,:,:,j])<min_pixels[j]:
                        EncodedPixel = ''
                    else:
                        EncodedPixel = mask2rle(mask[ii,:,:,j])
                ImageId_ClassIds.append(Id)
                EncodedPixels.append(EncodedPixel)
            if defect == True: ii += 1
                
    return ImageId_ClassIds, EncodedPixels

if mode == 'test' or mode == 'train':
    start = time.time()
    Ids, Encodeds = predict_batch(test2,
                                  preprocess, model, 0.5, min_pixels,
                                  classify_preprocess, classify_model)
    end = time.time()
    print("Time:", end-start)
    
    
    submission = pd.DataFrame({'ImageId_ClassId':Ids,
                     'EncodedPixels':Encodeds})

    submission.to_csv('submission.csv',index=False)
#    submission[submission['EncodedPixels']!='']


# In[17]:


# SAVE MODEL(for the sake of test again)
model.save('UNET.h5', include_optimizer=False)
f = open("preprocess.dill","wb")
dill.dump(preprocess,f)
f.close
classify_model.save('classify_model.h5', include_optimizer=False)
f = open("classify_preprocess.dill","wb")
dill.dump(classify_preprocess,f)
f.close
# UNET
if mode != 'train':
    get_ipython().system('cp $path_self/segmentation_models-0.2.1.tar.gz .')
    get_ipython().system('cp $path_self/trainandtest/image_classifiers-0.2.0.tar.gz .')

get_ipython().system('ls -l .')

