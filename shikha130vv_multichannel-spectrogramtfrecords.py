#!/usr/bin/env python
# coding: utf-8

# In[1]:


#References
#https://www.groundai.com/project/environment-sound-classification-using-multiple-feature-channels-and-deep-convolutional-neural-networks/1
#https://keunwoochoi.wordpress.com/2019/09/28/log-melspectrogram-layer-using-tensorflow-keras/


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os     
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import os
import librosa
import librosa.display
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
#!pip install python_speech_features
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import glob
import glob
import librosa
from librosa import feature
import numpy as np
from pathlib import Path
import cv2
AUTO = tf.data.experimental.AUTOTUNE
from kaggle_datasets import KaggleDatasets
import scipy
import pickle
from sklearn.model_selection import train_test_split
import time


# In[3]:


# Detect hardware, return appropriate distribution strategy
def get_strategy():
    gpu = ""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())     
    except ValueError:
        tpu = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu = tf.config.list_physical_devices("GPU")
        if len(gpu) == 1:
            print('Running on GPU ', gpu)
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        GCS_PATH = KaggleDatasets().get_gcs_path('birdsong-recognition')
    elif len(gpu) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision":True})
        GCS_PATH = "/kaggle/input/birdsong-recognition/"
    else:
        strategy = tf.distribute.get_strategy()
        GCS_PATH = "/kaggle/input/birdsong-recognition/"

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    base_dir = "../input/birdsong-recognition/"
    print(base_dir)
    return strategy, GCS_PATH, base_dir

strategy,GCS_PATH, base_dir = get_strategy()
sns.set_palette("pastel")
palette = sns.color_palette()
CACHE = {}


# In[4]:


num_train_data_per_class = 1
n_fft1 = int(0.0025 * 22050)
hop_length1 = int(0.001 * 22050)

n_fft2 = int(0.005 * 22050)
hop_length2 = int(0.0025 * 22050)

n_fft3 = int(0.01 * 22050)
hop_length3 = int(0.005 * 22050)
n_mels = 128
fmin=150
fmax=15000
img_sz1 = 64
img_sz2 = 512

def load_test_clip(path, start_time, duration=5):
    try:
        data, sr = librosa.load(path, offset=start_time, duration=duration, sr=48000, mono=True)
        return data, sr
    except Exception as e:
        print("Exception:", e)
        return None, 0

    
def buildBandpassFilter(rate, fmin, fmax, order=4):

    global CACHE

    fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')
        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]

def applyBandpassFilter(sig, rate, fmin, fmax):
    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate, fmin, fmax)

    return scipy.signal.sosfiltfilt(filter_sos, sig)

def get_mel_filterbanks(num_banks, fmin, fmax, f_vec, dtype=np.float32):
    '''
    An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
    an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
    flooring down the bin indices.
    '''

    global CACHE

    # Filterbank already in cache?
    fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        
        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

        # Compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # Convert mels to Hz
        banks_ends = (f_break * (10 ** (melpoints / A) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) /                 (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) /                 (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]

def get_spectrogram(sig, rate, shape=(img_sz1, img_sz2), win_len=512, fmin=150, fmax=15000, magnitude_scale='nonlinear', bandpass=True, decompose=False):

    # Compute overlap
    hop_len = int(len(sig) / (shape[1] - 1)) 
    win_overlap = win_len - hop_len + 2
    #print 'WIN_LEN:', win_len, 'HOP_LEN:', hop_len, 'OVERLAP:', win_overlap

    
    n_fft = win_len
    

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig, rate, fmin, fmax)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(win_len),
                                          nperseg=win_len,
                                          noverlap=win_overlap,
                                          nfft=n_fft,
                                          detrend=False,
                                          mode='magnitude')

    # Scale frequency?
   

    # Determine the indices of where to clip the spec
    valid_f_idx_start = f.searchsorted(fmin, side='left')
    valid_f_idx_end = f.searchsorted(fmax, side='right') - 1

    # Get mel filter banks
    mel_filterbank, mel_f = get_mel_filterbanks(shape[0], fmin, fmax, f, dtype=spec.dtype)

    # Clip to non-zero range so that unnecessary multiplications can be avoided
    mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

    # Clip the spec representation and apply the mel filterbank.
    # Due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
    spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
    spec = np.dot(spec, mel_filterbank)
    spec = np.transpose(spec, [1, 0])        

    # Magnitude transformation
    if magnitude_scale == 'pcen':
        
        # Convert scale using per-channel energy normalization as proposed by Wang et al., 2017
        # We adjust the parameters for bird voice recognition based on Lostanlen, 2019
        spec = pcen(spec, rate, hop_len)
        
    elif magnitude_scale == 'log':
        
        # Convert power spec to dB scale (compute dB relative to peak power)
        spec = spec ** 2
        spec = 10.0 * np.log10(np.maximum(1e-10, spec) / np.max(spec))
        spec = np.maximum(spec, spec.max() - 100) # top_db = 100

    elif magnitude_scale == 'nonlinear':

        # Convert magnitudes using nonlinearity as proposed by Schlüter, 2018
        a = -1.2 # Higher values yield better noise suppression
        s = 1.0 / (1.0 + np.exp(-a))
        spec = spec ** s

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()
    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)
    spec = (spec * 255).astype(np.int64)

    return spec



# In[5]:


def get_ebird_filename_dic():
    ebird_code_list = all_train_data["ebird_code"].unique()
    if os.path.exists("../input/birdcall-spectrogram-tfrecords/dic_ebird.pkl"):
        with open("../input/birdcall-spectrogram-tfrecords/dic_ebird.pkl","rb") as f:
            dic_ebird_code = pickle.load(f)
    else:
        dic_ebird_code = {k:v for v,k in enumerate(ebird_code_list)}
    dic_ebird_code_rev = [v for v,k in dic_ebird_code.items()]
    all_train_data["int_ebird_code"] = all_train_data["ebird_code"].map(dic_ebird_code)

    filename_list = all_train_data["filename"].unique()
    if os.path.exists("../input/birdcall-spectrogram-tfrecords/dic_filename.pkl"):
        with open("../input/birdcall-spectrogram-tfrecords/dic_filename.pkl","rb") as f:
            dic_filename = pickle.load(f)
    else:
        dic_filename = {k:v for v,k in enumerate(filename_list)}
    dic_filename_rev = [v for v,k in dic_filename.items()]
    all_train_data["int_filename"] = all_train_data["filename"].map(dic_filename)

    with open("dic_ebird.pkl","wb") as f:
        pickle.dump(dic_ebird_code, f)

    with open("dic_filename.pkl","wb") as f:
        pickle.dump(dic_filename, f)
        
    return dic_ebird_code_rev, dic_filename_rev


# In[6]:


def get_img_file():
    list_img_data = []
    max_duration = all_train_data["duration"].max()
    img_data = all_train_data[["int_ebird_code","int_filename","duration"]]
    duration = 0
    while duration < max_duration:
        img_data = img_data[img_data["duration"] >= duration+5]
        img_data["start"] = duration
        list_img_data.append(img_data)
        duration = duration + 5

    img_data = pd.concat(list_img_data)
    print(img_data.shape)
    print(all_train_data.shape)
    return img_data


def split_train_data():
    pickle_filename = "../input/birdcall-spectrogram-tfrecords/train_dic.pkl"
    if not os.path.exists(pickle_filename):
        print("Pickle file not found!")
        pickle_filename = "train_dic.pkl"
    if not os.path.exists(pickle_filename):
        print("Pickle file not found!")
        img_data = get_img_file()
        size = img_data.shape[0]//64
        arr_data = []
        split_data = img_data
        for i in range(63):
            split_data, test_data = train_test_split(split_data, test_size=size, stratify= split_data["int_ebird_code"])
            arr_data.append(test_data)
        arr_data.append(split_data)
        for df in arr_data[-3:]:
            print(df.shape)

        i  = 0
        arr_dic = []
        for df in arr_data:
            tfrec_path = "train_" + str(i)
            arr_dic.append({"tfrec_path":tfrec_path, "df":df})
            i = i + 1
        with open(pickle_filename, 'wb') as file:
            pickle.dump(arr_dic, file)
        
    else:
        print("Pickle file found!")
        with open(pickle_filename, 'rb') as file:
            arr_dic = pickle.load(file)
        pickle_filename = "train_dic.pkl"
        with open(pickle_filename, 'wb') as file:
            pickle.dump(arr_dic, file)
    return arr_dic


# In[7]:


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def write_all_tfrec(tfrec_path, train_data, bln_crop=False):
    try:
        t0 = time.process_time()
        tfrec_full_path = "../input/birdcall-spectrogram-tfrecords/" + tfrec_path
        is_valid_file = False
        if not os.path.exists(tfrec_full_path):
            print(tfrec_path, ": Valid file not found!")
            with tf.io.TFRecordWriter(tfrec_path) as out_file:
                for idx,row in train_data.iterrows():
                    int_ebird_code = row["int_ebird_code"]
                    int_filename = row["int_filename"]
                    filepath = base_dir + "train_audio/" + dic_ebird_code_rev[int_ebird_code] + "/" + dic_filename_rev[int_filename]
                    start_time=row["start"]
                    clip, sr = load_test_clip(filepath, start_time)
                    if sr > 0:
                        img = get_spectrogram(clip, sr)
                        #img = mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1])

                    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                    feature = {
                        "img": _bytestring_feature([img]),
                    }
                    if "train" in tfrec_path:
                        feature["int_ebird_code"] = _int_feature([row["int_ebird_code"]])
                    tf_record = tf.train.Example(features=tf.train.Features(feature=feature))
                    out_file.write(tf_record.SerializeToString())
        else:
            print("File exists!")
            os.popen('cp ' + tfrec_full_path + ' ' + tfrec_path)
        t1 = time.process_time()
        print("Process time:", t1-t0)
    except Exception as e:
        print("Error:", e, filepath)
    


# In[8]:


all_train_data = pd.read_csv(base_dir + "train.csv")
all_train_data = all_train_data[all_train_data["duration"]>=5]
all_train_data = all_train_data[all_train_data["filename"]!="XC195038.mp3"]


# In[9]:


dic_ebird_code_rev, dic_filename_rev = get_ebird_filename_dic()


# In[10]:


arr_dic = split_train_data()


# In[11]:


for param in arr_dic:
    write_all_tfrec(param["tfrec_path"], param["df"])


# In[ ]:





# In[ ]:




