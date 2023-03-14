#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


## This is the Execution Script of my model (see two cells ahead!) Trained  offline using quite a large % of varied sampled data from the whole 500 GB dataset provided.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
#from keras.models import Sequential

from keras.models import Sequential
from keras.layers import Input,LSTM,Reshape,TimeDistributed,Concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam
import scipy
from keras.applications  import inception_v3
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#from keras.sequences import LSTM
import sklearn
from scipy.io import loadmat,savemat    
import sys,json
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../Documents/Downloads/')
from sklearn.metrics import average_precision_score,classification_report
#from tensorflow import set_random_seed
#set_random_seed(711)
import os,sys,re,time
from collections import defaultdict
import imageio 
from PIL import Image
import cv2
from sklearn import utils
from sklearn.model_selection import train_test_split
from collections import OrderedDict

import psutil
process = psutil.Process(os.getpid())


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

get_ipython().system("pip install '/kaggle/input/dlibpkg/dlib-19.19.0'")
get_ipython().system("pip install '/kaggle/input/face-recognition/face_recognition_models-0.3.0/face_recognition_models-0.3.0'")
get_ipython().system("pip install '/kaggle/input/facerecognition-123/face_recognition-1.2.3-py2.py3-none-any.whl'")
get_ipython().system("pip install '/kaggle/input/imageio-ffmpeg/imageio_ffmpeg-0.3.0-py3-none-manylinux2010_x86_64.whl'")

import scipy.misc
import PIL
import dlib
import face_recognition
import time

dlib.DLIB_USE_CUDA=True

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:



class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length


# In[3]:


## ML Model , Inception(LSTM) - i.e a spatio - temporal model. see -  https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf

seq_len=9
img_dimens=299


import h5py    
import numpy as np    
f1 = h5py.File('/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5','r') 

def predict_fakes(Xtest):

    img_input = Input(shape=(seq_len,img_dimens,img_dimens,3))
    # num_classes=1

    batch_size=2
    sequence_len = 9
    num_lstm_units = num_features_per_Spatialframe_model = 2048
    num_classes=2
    img_width=299
    img_height=299

    ## I implemented an adapted architecture from th epaper - Delp & Guerra
    # It combines the frasmewise spatial fetures using the famous inception_v3 model. Then it feeds this into an LSTM Recurrant NNet to learn the 
    ## temporal behaviour "on top" of the spatial. Or more accurately it combines each spatially learned model per frame across a sequence of frame to represent the video
    
    feature_out_1 = TimeDistributed(inception_v3.InceptionV3(include_top=False,pooling='max',weights='imagenet'))(img_input)
    z = (LSTM(num_lstm_units,input_shape=(sequence_len,num_features_per_Spatialframe_model),return_sequences=False, recurrent_dropout = 0.5, dropout = 0.5))(feature_out_1)
    x = (Dense(512,activation='relu'))(z)
    x = (Dropout(0.5))(x)
    out = (Dense(num_classes,activation='softmax'))(x)    

    Xtest=Xtest.reshape(-1, seq_len,img_dimens,img_dimens,3)
    #Xtest=Xtest.astype('float') / 255.0
    Xtest = Xtest / 255.0
    print('Xtest shape =' + str(Xtest.shape))

    model = Model(img_input,out)
 hdf5")
    model.load_weights("/kaggle/input/inceptv3-lstmworkinglocallyontestset/ExtUtubDtaDelpGEnd2end_inceptV3_numfrmsPerSeq_9lstmUntsBalncd600SmplesNegnPos_With255NormlzePreProcRedoNum_clsses_1_2048_learn_rate_1e-05_dropot_usd_1_weight_usd_0_batch_size_2_epoch_num_12_.hdf5")

    
    te_pr = model.predict(Xtest, batch_size=batch_size, verbose=1)
    print('in model code and te_pr = ' + str(te_pr))
    return te_pr


# In[5]:


import os
from keras.applications import inception_v3

paths = []
preds = []
margin = 0.2
#!pip install '/kaggle/input/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
filenames_init = os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos/')
filenames = filenames_init
#print('no. of test videos = ' + str(len(filenames)))
#print('all video names = ' + str(filenames))

real_eg_count = 0

bad_imageCunt = 0
vid_counter = 0
frame_cutoff = 9 # (== seq_len)

num_chunks = 50
#np.set_printoptions(threshold  = float("inf"))
for i in range(0,int(num_chunks/2)):
    
    step_size = int(len(filenames)/num_chunks)
    
    #print('step size = '+ str(step_size))

    featurPerVideoDict = {}
    allVidsFeatures    = {}
    #print('doing chunk -- ' + str(i*step_size) + ' : ' + str((i+1)*step_size))
    for filename in filenames[i*step_size:(i+1)*step_size]:
        #if filename[-4:] != '.mp4':
        #    continue
        #print(os.path.join(dirname, filename))    
        full_path = os.path.join('/kaggle/input/deepfake-detection-challenge/test_videos/', filename)
        paths.append(filename)

        #print('Processing video = ' + str(filename))

        #try:

        video_capture = Video(os.path.join(full_path)) # + '/', str(vid)))
        #global countVid
        #print('processing  Video : ' + str(filename)  + ' which is no. ' + str(vid_counter)) # has frame rate ' + str(video_capture.fps)  + ' and label = ' + str(json1_data[name]['label']))
        vid_counter+=1

        # Initialize variables
        face_locations = []
        y_img=[]
        #os.mkdir(cropedTrainAllImgDir)
        #except Exception as inst:
        keys = filename[:-4]
        #print("Error occured at doing marker stuff of image file  ...")
        #print(inst)    
        size = (299,299)
        frame_counter = 0
        save_interval = 1
        featureMat = []
        last_key = "pdufsewrec"
        for i in range(100,video_capture.__len__()-40,save_interval):

            #print('back in for at iteration -- ' + str(i))

            #print('just into try   ...  frame_counter = ' + str(frame_counter))
            if frame_counter > 12 :
                break

            frame_counter += 1
            #print(' processing  frame no ' + str(counter)  + '... :)')
            # Grab a single frame of video
            frame = video_capture.get(i)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_positions = face_recognition.face_locations(rgb_frame)
            if face_positions is not None:

                #print('got face_position , so in outer if here ..  face_positions = ' + str(face_positions))
                #face_positions = list(face_positions)
                # Display the results
                for face_position in face_positions:
                    # Draw a box around the face
                    #print('in face_positions for loop  ...  margin = ' + str(margin))
                    #print('in a face of face_position -- face_position[2] =  ' + str(face_position[2]))
                    offset = round(margin * (face_position[2] - face_position[0]))
                    y0 = max(face_position[0] - offset, 0)
                    x1 = min(face_position[1] + offset, rgb_frame.shape[1])
                    y1 = min(face_position[2] + offset, rgb_frame.shape[0])
                    x0 = max(face_position[3] - offset, 0)
                    face = rgb_frame[y0:y1,x0:x1]

                    #inp = cv2.resize(face,(size,size))
                    #IMAGES.append(np.expand_dims(inp,axis=0))   
                    #if face :

                    #print('in a face of face_position ...  face = ' + str(face))

                    im = Image.fromarray(face) 
                    #print('read image from array of face = ')
                    input_img = np.array(im.resize(size, Image.BICUBIC))
                    #print('RESIZED image from array of face = ')
                    ## Change pre-processing to subtract the 3 channel means instead, prior to inputting to Inception 


                    ## usually a preprocessing tchnique prior to running inception but not in Delp's paper
                    #input_img = inception_v3.preprocess_input(input_img.astype(np.float32))
                    input_img = input_img.astype(np.float32)
                    #print('CONVERTEd RESIZED image to a float')
                    featureMat.append(np.expand_dims(np.array(input_img),axis=0))
                    #print('Done appended to featureMat...t')

        #print('size of featureMat  = ' + str(len(featureMat)))
        #print('about to do appendoing of all its feature data into dict @ key = ' + str(keys) + ' ...')
        if featureMat:
            featurPerVideoDict[keys] = np.array(featureMat)
            allImgsPerVidMat = np.concatenate(featureMat,axis=0)
            #last_key = keys

        else:
            #print('Encountered an empty sequnce of images for video => ' + str(keys))
            featurPerVideoDict[keys] = np.random.random_sample((1,frame_cutoff,299,299,3))
            allImgsPerVidMat = np.concatenate(featurPerVideoDict[keys],axis=0)
            #preds.append(0.5)
            bad_imageCunt += 1
            #print('no. videos could not find a face = ' + str(bad_imageCunt))

        #print('size of allImgsPerVidMat  = ' + str(allImgsPerVidMat.shape))        
        allVidsFeatures[keys] = np.concatenate(np.expand_dims(allImgsPerVidMat,axis=0) ,axis=0) 
                            #imageio.imwrite(cropedTrainAllImgDir + '/' + name[:-4] + '_' + str(counter) + '.jpg', face)
            

    print('Total number of videos  = ' + str(len(allVidsFeatures)))

    regurlySpacedVidsONlyDict = []
    final_data=[]
    properConcatdLSTMFetaurDataDict = allVidsFeatures
    key_counter = 0
    for indx in allVidsFeatures.keys():

        #print(' in padding code video = ' + str(indx) + ' i.e video number => ' + str(key_counter))

        if properConcatdLSTMFetaurDataDict[indx].shape[0] != 0 and properConcatdLSTMFetaurDataDict[indx].shape[0] < frame_cutoff :
            firstBlock = properConcatdLSTMFetaurDataDict[indx][:properConcatdLSTMFetaurDataDict[indx].shape[0],:]  
            arr_new = np.expand_dims(properConcatdLSTMFetaurDataDict[indx][properConcatdLSTMFetaurDataDict[indx].shape[0]-1],axis=0) 
            tiled_arr = np.repeat(arr_new,(frame_cutoff - properConcatdLSTMFetaurDataDict[indx].shape[0]),axis=0)
            padded_arr = np.concatenate((firstBlock ,tiled_arr),axis=0)
            regurlySpacedVidsONlyDict.append(padded_arr)

        elif properConcatdLSTMFetaurDataDict[indx].shape[0] == 0 :
            regurlySpacedVidsONlyDict.append(np.zeros((frame_cutoff,2048)))

        else:    
            regurlySpacedVidsONlyDict.append(properConcatdLSTMFetaurDataDict[indx][:frame_cutoff,:])

        
        #final_data.append(np.array(allVidsFeatures))
        final_data.append(np.expand_dims(np.array(regurlySpacedVidsONlyDict[key_counter]),axis=0))  
        #key_counter+=1 
        #print('Processed video no.' + str(key_counter))

    final_data=np.concatenate(final_data,axis=0)
    
    #final_data_np = np.concatenate(final_data,axis=0)             
    probs = predict_fakes(np.array(final_data)) 
    
    for prob in probs:
        preds.append(prob[1])
        
        if prob[1] <= 0.5:
            real_eg_count += 1
            
    print('memory usage = ' + str(process.memory_info().rss))        
            
    print('real_eg_count =  ' + str(real_eg_count))        

    
for i in range(int(num_chunks/2)+1,num_chunks):
    
    step_size = int(len(filenames)/num_chunks)
 
    featurPerVideoDict = {}
    allVidsFeatures    = {}
    #print('doing chunk -- ' + str(i*step_size) + ' : ' + str((i+1)*step_size))
    for filename in filenames[i*step_size:(i+1)*step_size]:
  
        full_path = os.path.join('/kaggle/input/deepfake-detection-challenge/test_videos/', filename)
        paths.append(filename)

        video_capture = Video(os.path.join(full_path)) # + '/', str(vid)))
        #global countVid
        #print('processing  Video : ' + str(filename)  + ' which is no. ' + str(vid_counter)) # has frame rate ' + str(video_capture.fps)  + ' and label = ' + str(json1_data[name]['label']))
        vid_counter+=1

        # Initialize variables
        face_locations = []
        y_img=[]
        #os.mkdir(cropedTrainAllImgDir)
        #except Exception as inst:
        keys = filename[:-4]
        #print("Error occured at doing marker stuff of image file  ...")
        #print(inst)    
        size = (299,299)
        frame_counter = 0
        save_interval = 1
        featureMat = []
        last_key = "pdufsewrec"
        for i in range(100,video_capture.__len__()-40,save_interval):

            #print('back in for at iteration -- ' + str(i))

            #print('just into try   ...  frame_counter = ' + str(frame_counter))
            if frame_counter > 12 :
                break

            frame_counter += 1
            #print(' processing  frame no ' + str(counter)  + '... :)')
            # Grab a single frame of video
            frame = video_capture.get(i)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_positions = face_recognition.face_locations(rgb_frame)
            if face_positions is not None:

                #print('got face_position , so in outer if here ..  face_positions = ' + str(face_positions))
                #face_positions = list(face_positions)
                # Display the results
                for face_position in face_positions:
                    # Draw a box around the face
                    #print('in face_positions for loop  ...  margin = ' + str(margin))
                    #print('in a face of face_position -- face_position[2] =  ' + str(face_position[2]))
                    offset = round(margin * (face_position[2] - face_position[0]))
                    y0 = max(face_position[0] - offset, 0)
                    x1 = min(face_position[1] + offset, rgb_frame.shape[1])
                    y1 = min(face_position[2] + offset, rgb_frame.shape[0])
                    x0 = max(face_position[3] - offset, 0)
                    face = rgb_frame[y0:y1,x0:x1]

                    #inp = cv2.resize(face,(size,size))
                    #IMAGES.append(np.expand_dims(inp,axis=0))   
                    #if face :

                    #print('in a face of face_position ...  face = ' + str(face))

                    im = Image.fromarray(face) 
                    #print('read image from array of face = ')
                    input_img = np.array(im.resize(size, Image.BICUBIC))
                    #print('RESIZED image from array of face = ')
                    ## Change pre-processing to subtract the 3 channel means instead, prior to inputting to Inception 

                    ## usually a preprocessing tchnique prior to running inception but not in Delp's paper
                    #input_img = inception_v3.preprocess_input(input_img.astype(np.float32))
                    input_img = input_img.astype(np.float32)
                    #print('CONVERTEd RESIZED image to a float')
                    featureMat.append(np.expand_dims(np.array(input_img),axis=0))
                    #print('Done appended to featureMat...t')

        #print('size of featureMat  = ' + str(len(featureMat)))
        #print('about to do appendoing of all its feature data into dict @ key = ' + str(keys) + ' ...')
        if featureMat:
            featurPerVideoDict[keys] = np.array(featureMat)
            allImgsPerVidMat = np.concatenate(featureMat,axis=0)
            #last_key = keys

        else:
            #print('Encountered an empty sequnce of images for video => ' + str(keys))
            featurPerVideoDict[keys] = np.random.random_sample((1,frame_cutoff,299,299,3))
            allImgsPerVidMat = np.concatenate(featurPerVideoDict[keys],axis=0)
            #preds.append(0.5)
            bad_imageCunt += 1
            #print('no. videos could not find a face = ' + str(bad_imageCunt))

        #print('size of allImgsPerVidMat  = ' + str(allImgsPerVidMat.shape))        
        allVidsFeatures[keys] = np.concatenate(np.expand_dims(allImgsPerVidMat,axis=0) ,axis=0) 
                            #imageio.imwrite(cropedTrainAllImgDir + '/' + name[:-4] + '_' + str(counter) + '.jpg', face)
            

    print('Total number of videos  = ' + str(len(allVidsFeatures)))

    regurlySpacedVidsONlyDict = []
    final_data=[]
    properConcatdLSTMFetaurDataDict = allVidsFeatures
    key_counter = 0
     #sequence_len
    #unused parameter for now
    #vid_frame_offset = 120
    for indx in allVidsFeatures.keys():

        #print(' in padding code video = ' + str(indx) + ' i.e video number => ' + str(key_counter))

        if properConcatdLSTMFetaurDataDict[indx].shape[0] != 0 and properConcatdLSTMFetaurDataDict[indx].shape[0] < frame_cutoff :
            firstBlock = properConcatdLSTMFetaurDataDict[indx][:properConcatdLSTMFetaurDataDict[indx].shape[0],:]  
            arr_new = np.expand_dims(properConcatdLSTMFetaurDataDict[indx][properConcatdLSTMFetaurDataDict[indx].shape[0]-1],axis=0) 
            tiled_arr = np.repeat(arr_new,(frame_cutoff - properConcatdLSTMFetaurDataDict[indx].shape[0]),axis=0)
            padded_arr = np.concatenate((firstBlock ,tiled_arr),axis=0)
            regurlySpacedVidsONlyDict.append(padded_arr)

        elif properConcatdLSTMFetaurDataDict[indx].shape[0] == 0 :
            regurlySpacedVidsONlyDict.append(np.zeros((frame_cutoff,2048)))

        else:    
            regurlySpacedVidsONlyDict.append(properConcatdLSTMFetaurDataDict[indx][:frame_cutoff,:])

        
        #final_data.append(np.array(allVidsFeatures))
        final_data.append(np.expand_dims(np.array(regurlySpacedVidsONlyDict[key_counter]),axis=0))  
        #key_counter+=1 
        #print('Processed video no.' + str(key_counter))

    final_data=np.concatenate(final_data,axis=0)
    
    #final_data_np = np.concatenate(final_data,axis=0)             
    probs = predict_fakes(np.array(final_data)) 
    
    for prob in probs:
        preds.append(prob[1])
        
        if prob[1] <= 0.5:
            real_eg_count += 1
            
    print('In second batch of chunks loop  -- memory usage = ' + str(process.memory_info().rss))        
            
    print('real_eg_count =  ' + str(real_eg_count))        
    
    
    
    
    
    
# #print('video: ' + str(indx) + ' probability: ' + str(prob))
res = pd.DataFrame({
'filename': paths,
'label': preds,
})    

print('paths =' + str(paths))
print('preds =' + str(preds))




res.sort_values(by='filename', ascending=True, inplace=True)

print(res)
#print(res.filename)
res.to_csv('submission.csv', index=False)                


# In[ ]:





