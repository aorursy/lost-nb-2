#!/usr/bin/env python
# coding: utf-8

# In[1]:


#model version with cwt transformation has best score so far
#convertingg time series to image
#https://www.kaggle.com/tigurius/recuplots-and-cnns-for-time-series-classification

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import os

import numpy as np
import pandas as pd

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import pywt

from scipy.signal import find_peaks
from scipy import optimize
from numpy.fft import rfft,rfftfreq, irfft

from keras.layers import *
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam

from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter

import warnings

#os.chdir('D:\\Kaggle\\VSBPowerLineFalutDetection\\all\\code')
#os.getcwd()
print("The version cwt version run after 5th fails")
os.listdir('../input')
dftrain = pd.read_csv('../input/metadata_train.csv') 
#dftrain.head()


# In[2]:


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# In[3]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[4]:


# 800,000 data points taken over 20 ms
# Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
n_samples = 800000

# Sample duration is 20 miliseconds
sample_duration = 0.02

# Sample rate is the number of samples in one second
# Sample rate will be 40mhz
sample_rate = n_samples * (1 / sample_duration)

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=sample_rate):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

        
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, Wn=[norm_low_cutoff*20], btype='highpass', output='sos') 
    filtered_sig = signal.sosfilt(sos, x)
    
    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )
    #sigma = (1/0.25) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )

#very high peacks (above height) are removed alonge with lower following peacks till width
def removeHighPeacks(x_data,width,height) :
    mask = (x_data > height)|(x_data < -height)
    maskCopy = mask.copy()
    for j in np.arange(1,len(mask)-width):
        if(mask[j] ==  True):
            maskCopy[max(j-width,0):j+width] = 1
    x_data[maskCopy] = 0
    return x_data

#peacks are retained minimum instance are present in width range. 
def removeIsolatedPeack(x_data,width, height, min_instances):
    for j in np.arange(0,len(x_data)-width,width):
        #print(j,":",np.sum((x_dn[j:j+width] > 5) | (x_dn[j:j+width] < -5)))
        if(np.sum((x_data[j:j+width] > height) | (x_data[j:j+width] < -height)) < min_instances):
            x_data[j:j+width] = 0
    return (x_data)


# In[5]:


#n_dim=256 divides signal in complete chunk and is power of 2. if want to change
#following function need to handle last partial signal chunk
#train_length = 3
#we are extracting only global features. May need to get features in chunks.
# While taking the chunks rather than fix size try to capture the brust.
#May need padding to make same sizes
def feature_extraction(x_data, low_cutoff=10000,signal_len=800000,n_dim=256):  
    # Apply high pass filter with low cutoff of 10kHz, this will remove the low frequency 50Hz sinusoidal motion in the signal
    #print("high_pass_filter")
    x_hp = high_pass_filter(x_data, low_cutoff=10000, sample_rate=sample_rate)
    
    # Apply denoising
    #print("denoise_signal")
    x_dn = denoise_signal(x_hp, wavelet='haar', level=1)
    # Remove high peacks
    #print("removeHighPeacks")
    x_rh = removeHighPeacks(x_dn,250,35)
    #Remove isolated peacks
    #print("removeIsolatedPeack")
    
    x_clean=removeIsolatedPeack(x_rh,width=1024,height=5, min_instances=4)
    #x_clean=removeIsolatedPeack(x_rh,1024,height=(np.max(x_rh)-np.min(x_rh))/5, min_instances=4)
    #print("greater than 5")
    x_clean[(x_clean<5)&(x_clean>-5)]=0
    
    indexes = np.nonzero(x_clean)[0].ravel()
    
    #print("Collecting faulty data, #of data points : ", len(indexes) )
    faultSignal=[]
    maxLen = 0
    for ind in indexes:
        faultSignal = np.append(faultSignal,np.asarray(x_clean[ind]))
        length = len(faultSignal)
        #print("Length :",length )
        #if (length > maxLen):
        #    maxLen = length
        #    print("********Maxium Length is :", maxLen)
    length = len(faultSignal)
    if (length > 256):
        faultSignal = faultSignal[0:256]
    else:
        faultSignal=np.pad(faultSignal,(0,256-length),'constant',constant_values=0)    
    return (np.asarray(faultSignal))
    #print("Returning Features")
    #plt.plot(faultSignal)
    #new_signal = []
    #bucket_size = int(signal_len / n_dim)
    #for i in range(0, signal_len, bucket_size):
    #    signal_range = faultSignal[i:i + bucket_size]
    #sumRange = np.sum(np.abs(faultSignal))
    #count = len(faultSignal)
    #std = np.std(faultSignal)
    #tweenthyPercentile = np.percentile(x_data, 20)
    #eighthPercentile = np.percentile(x_data, 80)
    #new_signal.append(np.asarray([sumRange,count,std,tweenthyPercentile,eighthPercentile]))
    #return np.asarray([sumRange,count,std,tweenthyPercentile,eighthPercentile])


# In[6]:


def prep_data(start, end):
    #praq_train = pq.read_pandas('../input/train.parquet').to_pandas()
    print(start," : " , end)
    praq_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    print(praq_train.shape)
    X = []
    y = []
    
    for i in tqdm(range(start,end)):
        y.append(dftrain.loc[dftrain.signal_id==i, 'target'].values)
        feature=feature_extraction(praq_train[str(i)])
        X.append(np.asarray([feature]))
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y  
        
    #for id_measurement in df_train.index.levels[0].unique()[int(start/3):int(end/3)]:
        #X_signal = []
        #for phase in [0,1,2]:
   #         signal_id, target = df_train.loc[id_measurement].loc[phase]
   #         if phase == 0:
   #             y.append(target)
            #if (signal_id %100 == 0):
  #          print(str(signal_id))
            
  #          b = praq_train[str(signal_id)]
  #          a=feature_extraction(b)
  #          X_signal.append(a)
        #X_signal = np.concatenate(X_signal, axis=1)
  #      X_signal = np.concatenate(X_signal)
  #      X.append(X_signal)
  #  X = np.asarray(X)
  #  y = np.asarray(y)
  #  return X, y


# In[7]:


df_train = pd.read_csv('../input/metadata_train.csv')
df_train = df_train.set_index(['id_measurement', 'phase'])
df_train.head()
len(df_train)


# In[8]:



X = []
y = []
def load_all():    
    total_size = len(df_train)
    #total_size = 900 # must be multiple of 3
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        print(ini,end)
        X_temp, y_temp = prep_data(ini, end)
        X.append(X_temp)
        y.append(y_temp)
load_all()
X = np.concatenate(X)
y = np.concatenate(y)
print("Loaded data", X.shape)
print("Loaded Target", y.shape)
#Normalise X
X = np.nan_to_num(X)
maxCoefficient=np.amax(X,axis=0, keepdims = True)
maxCoefficient=maxCoefficient+K.epsilon()
X = X/maxCoefficient


# In[9]:


print(X.shape)
print(X[0])
print(y[0])


# In[10]:


print(X.shape)
print(y.shape)
print(maxCoefficient.shape)
#z = X.copy()
#X = X/maxCoefficient
print(np.max(np.ravel(X)))


# In[11]:


#TODO get balanced split.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)


# In[12]:


plt.plot(y_train)
plt.show()

plt.plot(y_valid)
plt.show()

plt.plot(X_valid[:,:,0])
plt.show()
plt.plot(X_valid[:,:,1])
plt.show()
plt.plot(X_valid[:,:,2])
plt.show()


# In[13]:


#from sklearn import tree, ensemble

#gboost = ensemble.GradientBoostingClassifier(max_depth=4, min_samples_leaf=2, n_estimators = 3)
#sample_weight1  = np.ones(y_train.shape[0])
#mask = y_train==1
#sample_weight1[mask.ravel()] = ((y_train.shape[0]+ np.sum(y_train==1))/np.sum(y_train==1))*0.75
#gboost_fit = gboost.fit(X_train, y_train.ravel(), sample_weight=sample_weight1.ravel())
#fit = gboost_fit


# In[14]:


#train_y_pred = fit.predict(X_train)
#train_y_pred=train_y_pred.reshape(len(train_y_pred),1)

#matthews_corrcoef1=matthews_correlation(y_train, train_y_pred)
#print(matthews_corrcoef1)


# In[15]:


#test_y_pred = fit.predict(X_test)
#test_y_pred=test_y_pred.reshape(len(test_y_pred),1)

#matthews_corrcoef2=matthews_corrcoef(y_test, test_y_pred)
#print(matthews_corrcoef2)


# In[16]:





# In[16]:





# In[16]:


#from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
def model_lstm(input_shape,dropout=0.3):
    inp = Input(shape=(input_shape[1],input_shape[2],))
    
    #x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    #x=DropoutWrapper(output_keep_prob=dropout)(x)
    #x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    #x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    #x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LSTM(64, return_sequences=True)(inp)
    x = LSTM(32, return_sequences=True)(x)
    x = Attention(input_shape[1])(x)
    x = Dense(2, activation="relu")(x)
    x  = Dropout(dropout)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    adams= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer= adams, metrics=[matthews_correlation])
    
    return model


# In[17]:


model= model_lstm(X_train.shape)
print(model.metrics_names)
model.summary()


# In[18]:


ckp = ModelCheckpoint('weights.h5', save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
history=model.fit(X_train, y_train, batch_size=100, epochs=250, validation_data=[X_valid, y_valid] , callbacks=[ckp])


# In[19]:


print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for matthews_correlation
plt.plot(history.history['matthews_correlation'])
plt.plot(history.history['val_matthews_correlation'])
plt.title('model matthews_correlation')
plt.ylabel('matthews_correlation')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[20]:


get_ipython().run_cell_magic('time', '', "# 25ms in Kernel\nmeta_test = pd.read_csv('../input/metadata_test.csv')")


# In[21]:


meta_test = meta_test.set_index(['signal_id'])
meta_test.head()


# In[22]:


get_ipython().run_cell_magic('time', '', 'first_sig = meta_test.index[0]\nn_parts = 10\nmax_line = len(meta_test)\n#max_line = 90\npart_size = int(max_line / n_parts)\nlast_part = max_line % n_parts\nprint(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)\nstart_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]\nstart_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]\n\nprint("Change max_line = len(meta_test)")\nX_test = []\nfor start, end in start_end:\n    subset_test = pq.read_pandas(\'../input/test.parquet\', columns=[str(i) for i in range(start, end)]).to_pandas()\n    for i in tqdm(subset_test.columns):\n        id_measurement, phase = meta_test.loc[int(i)]\n        subset_test_col = subset_test[i]\n        subset_trans = extract_features(subset_test_col)\n        X_test.append([i, id_measurement, phase, subset_trans])')


# In[23]:


X_test_input = np.asarray([np.concatenate([X_test[i][3],X_test[i+1][3], X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])
X_test_input.shape


# In[24]:


submission = pd.read_csv('../input/sample_submission.csv')
print(len(submission))
submission.head()


# In[25]:


model.load_weights('weights.h5')


# In[26]:


pred = model.predict(X_test_input, batch_size=300)
plt.plot(pred])
plt.title('Predications Probabilities')
plt.show()


# In[27]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.15))
submission['target'] = pred_3
submission.to_csv('submission15.csv', index=False)
print("submission15 # predicted true",np.sum(submission['target']))


# In[28]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.2))
submission['target'] = pred_3
submission.to_csv('submission2.csv', index=False)
print("submission2 # predicted true",np.sum(submission['target']))


# In[29]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.25))
submission['target'] = pred_3
submission.to_csv('submission25.csv', index=False)
print("submission25 # predicted true",np.sum(submission['target']))


# In[30]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.3))
submission['target'] = pred_3
submission.to_csv('submission3.csv', index=False)
print("submission3 # predicted true",np.sum(submission['target']))


# In[31]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.35))
submission['target'] = pred_3
submission.to_csv('submission35.csv', index=False)
print("submission35 # predicted true",np.sum(submission['target']))


# In[32]:


pred_3 = []
for pred_scalar in pred:
    for i in range(3):
        pred_3.append(int(pred_scalar > 0.40))
submission['target'] = pred_3
submission.to_csv('submission4.csv', index=False)
print("submission4 # predicted true",np.sum(submission['target']))


# In[33]:


f=(1,2)
np.asarray(f)


# In[34]:




