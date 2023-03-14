#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip","install",package])
install("../input/fastremap/fastremap-1.10.2-cp37-cp37m-manylinux1_x86_64.whl")
install("../input/fillvoids/fill_voids-2.0.0-cp37-cp37m-manylinux1_x86_64.whl")
install("../input/finalmask")
install("pydicom")


# In[2]:


#importing libraries for the analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.notebook import tqdm
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as kb
import pydicom
from pydicom.data import get_testdata_files

import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder 
from sklearn import preprocessing 
from tensorflow import keras
import keras.backend as kb
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
sns.set_context("paper")
from lungmask import mask
import SimpleITK as sitk
import math
import time
from skimage.transform import resize
from skimage import data
from skimage.util import pad


# In[3]:


import os
import random
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


# In[4]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
train=pd.read_csv(f"{ROOT}/train.csv")

train['Patient_Week']=train['Patient']+'_'+train['Weeks'].astype(str)


sample_submission=pd.read_csv(f"{ROOT}/sample_submission.csv")


test=pd.read_csv(f"{ROOT}/test.csv")
test['Patient_Week']=test['Patient']+'_'+test['Weeks'].astype(str)


# In[5]:


def bounding_box(img3d):
        mid_img = img3d
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row and (mid_img!=0).any():
            return True
        else:
            return False


def crop_bounding(input):
  x=0
  dataset=pydicom.dcmread(input)
  image1=dataset.pixel_array
  while x==0:
    bounding=bounding_box(image1)
    if (image1[0, :] == image1[0, 0]).all()==True:
      image1=image1[1:-1,:]
    elif (image1[:, 0] == image1[0, 0]).all()==True:
      image1=image1[:,1:-1]
    else:
      x=x+1
  return image1

def masking(input):
  input_image = sitk.ReadImage(input)
  dataset = pydicom.dcmread(input)
  if bounding_box(dataset.pixel_array)==False:
    segmentation = mask.apply(input_image)
    dataset = pydicom.dcmread(input)
    segment_image=dataset.pixel_array
    segment_image=np.where(segmentation[0]==0,dataset.pixel_array.min(),segment_image)
  else:
    array_ct = sitk.GetArrayFromImage(input_image)
    column_dif=int((array_ct.shape[1]-crop_bounding(input).shape[0])/2)
    row_dif=int((array_ct.shape[2]-crop_bounding(input).shape[1])/2)
    input_image=input_image[row_dif:-row_dif,column_dif:-column_dif]
    segmentation = mask.apply(input_image)
    dataset = pydicom.dcmread(input)
    segment_image=sitk.GetArrayFromImage(input_image)[0]
    segment_image=np.where(segmentation[0]==0,dataset.pixel_array.min(),segment_image)
  return(segment_image)

def pixscale(input):
  dataset = pydicom.dcmread(input)
  scaled_image=dataset.pixel_array*dataset.RescaleSlope + dataset.RescaleIntercept
  return(scaled_image)


# In[6]:


patients=test['Patient'].unique().tolist()

### find length of lung:
def lunglength(patient):
  directory=f"{ROOT}/test/"+patient+"/"
  images=glob.glob(directory+'*')
  images.sort(key = lambda x: int(x.split('/')[5].split('.dcm')[0]))
  first_image=pydicom.dcmread(images[0])
  last_image=pydicom.dcmread(images[-1])
  lung_length=-(last_image.ImagePositionPatient[2]-first_image.ImagePositionPatient[2])
  return(lung_length)


# In[7]:


#patient='ID00007637202177411956430'
no_of_images=[]
slice_thickness=[]
rescaletype=[]
Rescale_slope=[]
Rescale_intercept=[]
Pixel_spacing=[]
rows=[]
columns=[]
padding=[]
image_position=[]
for patient in patients:
  directory=f"{ROOT}/test/"+patient+"/"
  no_of_images.append(len(glob.glob(directory+'*')))
  dataset=pydicom.dcmread(glob.glob(directory+'*')[0])
  slice_thickness.append(dataset.SliceThickness)
  #rescale_type.append(dataset.RescaleType)
  Rescale_slope.append(dataset.RescaleIntercept)
  Rescale_intercept.append(dataset.RescaleSlope)
  Pixel_spacing.append(dataset.PixelSpacing)
  rows.append(dataset.Rows)
  columns.append(dataset.Columns)
  #image_position.append(dataset.ImagePositionPatient)
  #padding_value.append(dataset.PixelPaddingValue)
for patient in patients:
  directory=f"{ROOT}/test/"+patient+"/"
  dataset=pydicom.dcmread(glob.glob(directory+'*')[0])
  try:
    padding.append(dataset.PixelPaddingValue)
  except:
    padding.append(np.nan)
  rescaletype=[]
for patient in patients:
  directory=f"{ROOT}/test/"+patient+"/"
  dataset=pydicom.dcmread(glob.glob(directory+'*')[0])
  try:
    rescaletype.append(dataset.RescaleType)
  except:
    rescaletype.append(np.nan)
for patient in patients:
  directory=f"{ROOT}/test/"+patient+"/"
  dataset=pydicom.dcmread(glob.glob(directory+'*')[0])
  try:
    image_position.append(dataset.ImagePositionPatient)
  except:
    image_position.append(np.nan)
lung_lengths=[]
for patient in patients:
  try:
   lung_lengths.append(abs(lunglength(patient)))
  except:
    lung_lengths.append(np.nan)


# In[8]:


metadata=pd.DataFrame({'patient_ID':test['Patient'].unique(),'no_of_images':no_of_images,
'slice_thickness':slice_thickness,
'rescale_type':rescaletype,
'Rescale_slope':Rescale_slope,
'Rescale_intercept':Rescale_intercept,
'Pixel_spacing':Pixel_spacing,
'rows':rows,
"columns":columns,
"padding_value":padding,
"image_position":image_position,
"lung_lengths":lung_lengths})
metadata['lung_lengths'].fillna(metadata['lung_lengths'].mean(),inplace=True)
extra=[]
for i in range(len(test['Patient'].unique())):
    extra.append((((no_of_images[i]-1)*430/metadata['lung_lengths'].to_list()[i])-no_of_images[i])+1)
metadata['extra']=extra

to_be_extracted=20*metadata['no_of_images']/(metadata['extra']+metadata['no_of_images'])
to_be_extracted=[math.ceil(value) for value in to_be_extracted]
metadata['to_be_extracted']=to_be_extracted

metadata.to_csv('metadata.csv',index=False)


# In[9]:


def image_nn_convert(patient):
  directory=f"{ROOT}/test/"+patient+"/"
  images=glob.glob(directory+'*')
  images.sort(key = lambda x: int(x.split('/')[5].split('.dcm')[0]))
  image_number=[]
  for value in np.linspace(1,len(images),metadata['to_be_extracted'].loc[metadata['patient_ID']==patient].to_list()[0],endpoint=True):
    image_number.append(int(round(value)))
  revised_images=[]
  for i in image_number:
    revised_images.append(images[i-1])
  normalized=[]
  for input in tqdm(revised_images):
    dataset = pydicom.dcmread(input)
    slope=dataset.RescaleSlope
    intercept=dataset.RescaleIntercept
    masked_image=masking(input)
    masked_image=(masked_image*slope) + intercept
    masked_image=np.where((masked_image>=1000)|(masked_image<=-1000),-1000,masked_image)
    new_shape_row=round(float(dataset.PixelSpacing[0])*masked_image.shape[0])
    new_shape_column=round(float(dataset.PixelSpacing[1])*masked_image.shape[1])
    resized=resize(masked_image, (new_shape_row, new_shape_column),preserve_range=True)
    resized=pad(resized,((math.ceil((512-resized.shape[0])/2),math.floor((512-resized.shape[0])/2)),((math.ceil((512-resized.shape[1])/2),math.floor((512-resized.shape[1])/2)))),mode='edge')
    resized=resize(resized,(128,128),preserve_range=True)
    normalized.append(resized)
    empty_images=np.ones([128,128])*-1000
  while np.array(normalized).shape[0]<20:
    normalized=np.insert(normalized,0,empty_images,0)
    if np.array(normalized).shape[0]<20:
      normalized=np.insert(np.array(normalized),np.array(normalized).shape[0],empty_images,0)
    else:
      pass
  return(normalized)


# In[10]:


new_num_arr=[]
#patient="ID00009637202177434476278"
for patient in tqdm(patients):
  start = time.time()
  compy=image_nn_convert(patient)
  new_num_arr.append(compy)
  end = time.time()
  print(end - start)
new_num_arr=np.array(new_num_arr)

fig=plt.figure(figsize=(50, 24))
for i in range(len(compy)):
    img = compy[i]
    fig.add_subplot(4, 5,i+1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.title(i, fontsize = 9)
    plt.axis('off');


# In[11]:


train_num_arr = np.load('../input/image-to-np/data.npy')
train_num_arr.shape


# In[12]:


def image_to_ann(x,ann_convert):
    x=np.repeat(x,ann_convert,0)
    return(x)


# In[13]:


new_num_arr=new_num_arr.astype("float16")


# In[14]:


new_num_arr.shape


# In[15]:


counts=train.groupby(('Patient')).count()['Patient_Week'].to_numpy()
train_num_arr=train_num_arr.astype("float16")
train_num_arr=image_to_ann(train_num_arr,counts)
counts_test=test.groupby(('Patient')).count()['Patient_Week'].to_numpy()
new_num_arr=new_num_arr.astype("float16")
new_num_arr=image_to_ann(new_num_arr,counts_test)


# In[16]:


train_num_arr=np.concatenate((train_num_arr,new_num_arr))
train_num_arr=train_num_arr.astype("float16")


# In[17]:


def score(y_true, y_pred):
   tf.dtypes.cast(y_true, tf.float32)
   tf.dtypes.cast(y_pred, tf.float32)
   sigma = abs(y_pred[:,2] - y_pred[:,0])
   fvc_pred = y_pred[:,1]
   
   #sigma_clip = sigma + C1
   sigma_clip = tf.maximum(sigma, 70)
   delta = tf.abs(y_true[:, 0] - fvc_pred)
   delta = tf.minimum(delta, 1000)
   sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32))
   metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
   return metric
def mloss(_lambda):
   def loss(y_true, y_pred):
       return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
   return loss
def qloss(y_true, y_pred):
   # Pinball loss for multiple quantiles
   qs = [0.8,0.5,0.2]
   q = tf.constant(np.array([qs]), dtype=tf.float32)
   e = y_true - y_pred
   v = tf.maximum(q*e, (q-1)*e)
   return kb.mean(v)


# In[18]:


total=pd.concat([train,test],axis=0)
total=total.reset_index(drop=True)
total


# In[19]:


X1=total[['Weeks','Age','Sex','SmokingStatus']].copy()
y=total.FVC.copy()
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X1[['Sex','SmokingStatus']])
encoded=pd.DataFrame(enc.transform(X1[['Sex','SmokingStatus']]).toarray())
X1=X1.join(encoded)
X1.drop(['SmokingStatus','Sex'],axis=1,inplace=True)
scaler=preprocessing.MinMaxScaler().fit(X1)
X1=pd.DataFrame(scaler.transform(X1))
X1=X1.astype("float16")


# In[20]:


encoder_input = keras.Input(shape=(20,128,128,1), name="input")
x = layers.Conv3D(16, kernel_size=(3, 3, 3),padding='same')(encoder_input)
x=layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
x = layers.Conv3D(32, 3,padding='same')(x)
x=layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x=layers.Conv3D(8, kernel_size=(3, 3, 3),padding='same')(x)

x=layers.Flatten()(x)
#x=layers.Dense(500, activation='relu')(x)
encoder_output=layers.Dense(3)(x)
#encoder_output=layers.LeakyReLU()(x)
#encoder_output=tf.keras.layers.LayerNormalization(axis=1)(x)
input2 = keras.Input(shape=[7])
new_input=tf.keras.layers.Concatenate(axis=1)([encoder_output, input2])

encoder = keras.Model([encoder_input,input2], encoder_output, name="encoder")
#encoder.summary()

dense = layers.Dense(150, activation="relu")(new_input)
x = layers.Dense(100, activation="relu")(dense)
x = layers.Dense(100, activation="relu")(x)
output = layers.Dense(10,activation='linear')(x)
output_encoder=keras.Model([encoder_input,input2], output)
output1 = layers.Dense(1,activation='linear')(output)
full_encoder = keras.Model([encoder_input,input2], output1, name="full_encoder")
full_encoder.summary()


# In[21]:


full_encoder.compile(loss='mae',optimizer='adam')
full_encoder.fit([train_num_arr,X1],y,epochs=250,batch_size=5)


# In[22]:


encoded_train=encoder.predict([train_num_arr,X1])


# In[23]:


encoded_train


# In[24]:


train=pd.concat([train,pd.DataFrame(encoded_train)[:len(train)]],axis=1)
test=pd.read_csv(f"{ROOT}/test.csv")
test=pd.concat([test,pd.DataFrame(encoded_train)[len(train):].reset_index(drop=True)],axis=1)


# In[25]:


add=train.copy()
add.rename(columns={'Weeks':'base_weeks','FVC':'base_fvc'},inplace=True)
final=train.merge(add,on='Patient')
final.drop(['Patient_Week_x','Age_y','Sex_y','SmokingStatus_y','Percent_y'],axis=1,inplace=True)
final.rename(columns={'Weeks':'base_week','FVC':'base_fvc','base_fvc':'FVC','Percent_x':'base_percent','Patient_Week_y':'Patient_Week','Age_x':'Age','Sex_x':'sex','SmokingStatus_x':'smokingstatus','base_weeks':'predict_week'},inplace=True)
final['weeks_passed']=final['predict_week']-final['base_week']
cols=['Patient','Patient_Week', 'base_week', 'base_fvc', 'base_percent', 'Age', 'sex','smokingstatus','predict_week','weeks_passed','0_x','1_x','2_x','FVC']
final=final[cols]
final=final.loc[final['weeks_passed']!=0]
final.reset_index(drop=True,inplace=True)


# In[26]:


final


# In[27]:


test.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'},inplace=True)
Week=sample_submission['Patient_Week'].apply(lambda x : x.split('_')[1]).unique()
Week=np.tile(Week, len(test['Patient']))
test=test.loc[test.index.repeat(146)].reset_index(drop=True)
test['predict_week']=Week

test['Patient_Week']=test['Patient']+'_'+test['predict_week']


test['weeks_passed']=test['predict_week'].astype(int)-test['base_Week'].astype(int)

test.rename(columns={'base_Week':'base_week','base_FVC':'base_fvc','base_Percent':'base_percent','base_Age':'Age','Sex':'sex','SmokingStatus':'smokingstatus',0:'0_x',1:'1_x',2:'2_x'},inplace=True)

cols=['Patient','Patient_Week','base_week','base_fvc','base_percent','Age','sex','smokingstatus','predict_week','weeks_passed','0_x','1_x','2_x']

test=test[cols]


# In[28]:


test


# In[29]:


X1=final[['base_fvc','base_percent','Age','sex','smokingstatus','weeks_passed','base_week','0_x','1_x','2_x']].copy()
y1=final.FVC.copy()
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X1[['sex','smokingstatus']])
encoded=pd.DataFrame(enc.transform(X1[['sex','smokingstatus']]).toarray())
X1=X1.join(encoded)
X1.drop(['smokingstatus','sex'],axis=1,inplace=True)
scaler=preprocessing.MinMaxScaler().fit(X1)
X1=pd.DataFrame(scaler.transform(X1))


# In[30]:


X_test=test[['base_fvc','base_percent','Age','sex','smokingstatus','weeks_passed','base_week','0_x','1_x','2_x']].copy()
encoded=pd.DataFrame(enc.transform(X_test[['sex','smokingstatus']]).toarray())
X_test=X_test.join(encoded)
X_test.drop(['smokingstatus','sex'],axis=1,inplace=True)
X_test=pd.DataFrame(scaler.transform(X_test))


# In[31]:


X1=X1.astype(np.float32)
y1=y1.astype(np.float32)

inputs= keras.Input(shape=[13])
dense = layers.Dense(150, activation="relu")
x = dense(inputs)
x = layers.Dense(100, activation="relu")(x)
x = layers.Dense(100, activation="relu")(x)
output1 = layers.Dense(3,activation='linear')(x)
model = keras.Model(inputs=inputs, outputs=output1)
model.compile(loss=mloss(0.8),optimizer='adam',metrics=score)


# In[32]:


model.compile(loss=mloss(0.8),optimizer='adam',metrics=score)
model.fit(X1,y1,batch_size=512,epochs=200)


# In[33]:


preds_high=model.predict(X_test)[:,0]
preds_low=model.predict(X_test)[:,2]
preds=model.predict(X_test)[:,1]


# In[34]:


preds_set=pd.DataFrame({'preds_high':preds_high})
preds_set['preds']=preds
preds_set['preds_low']=preds_low
preds_set['sigma_pred']=abs(preds_set['preds_high']-preds_set['preds_low'])
preds_set.reset_index(inplace=True,drop=True)


# In[35]:


preds_set


# In[36]:


submission=pd.DataFrame({'Patient_Week':test['Patient_Week'],'FVC': preds_set['preds'],'Confidence':preds_set['sigma_pred']})
submission['FVC']=submission['FVC'].apply(lambda x: round(x, 4))
submission['Confidence']=submission['Confidence'].apply(lambda x: round(x, 4))
                         


# In[37]:


submission.to_csv('submission.csv',index=False)


# In[38]:


submission.tail(130)

