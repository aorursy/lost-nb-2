#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index')
get_ipython().system('pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index')


# In[17]:


conda install -c conda-forge -y gdcm


# In[2]:


get_ipython().system('pip install segmentation-models-pytorch')


# In[4]:


import os
import cv2
import pydicom
import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
import seaborn as sns
from PIL import Image
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from scipy.stats import kurtosis
from scipy.stats import skew

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


# In[5]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# In[12]:


import sys
import glob
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch-master/')
sys.path.append('../input/pretrainedmodels/pretrainedmodels-0.7.4/')
sys.path.append('../input/segmentation-models-pytorch/')
import segmentation_models_pytorch as smp


# In[ ]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 


# In[ ]:


def get_tab(df):
    print(df)
    vector = [(df.Age.values[0] - 30) / 30] 
    
    if df.Sex.values[0] == 'male':
       vector.append(0)
    else:
       vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    return np.array(vector) 


# In[ ]:


A = {} 
TAB = {} 
P = [] 
for i, p in tqdm(enumerate(train.Patient.unique())):
    sub = train.loc[train.Patient == p, :] 
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc)[0]
    
    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)


# In[ ]:


def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(d.pixel_array / 2**11, (512, 512))


# In[ ]:


from tensorflow.keras.utils import Sequence

class IGenerator(Sequence):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size
        
        self.train_data = {}
        for p in train.Patient.values:
            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        x = []
        a, tab = [], [] 
        keys = np.random.choice(self.keys, size = self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')
                x.append(img)
                a.append(self.a[k])
                tab.append(self.tab[k])
            except:
                print(k, i)
       
        x,a,tab = np.array(x), np.array(a), np.array(tab)
        x = np.expand_dims(x, axis=-1)
        return [x, tab] , a


# In[ ]:


from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
import efficientnet.tfkeras as efn

def get_efficientnet(model, shape):
    models_dict = {
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    }
    return models_dict[model]

def build_model(shape=(512, 512, 1), model_class=None):
    inp = Input(shape=shape)
    base = get_efficientnet(model_class, shape)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    inp2 = Input(shape=(4,))
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
    x = Concatenate()([x, x2]) 
    x = Dropout(0.35)(x) 
    x = Dense(1)(x)
    model = Model([inp, inp2] , x)
    
    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]
    model.load_weights('../input/osic-model-weights/' + weights)
    return model

model_classes = ['b5'] #['b0','b1','b2','b3',b4','b5','b6','b7']
models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]
print('Number of models: ' + str(len(models)))


# In[ ]:


from sklearn.model_selection import train_test_split 

tr_p, vl_p = train_test_split(P, 
                              shuffle=True, 
                              train_size= 0.8) 


# In[ ]:


sns.distplot(list(A.values()));


# In[ ]:


def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)


# In[ ]:


subs = []
# mid_df = pd.DataFrame([['patient_id', 'actual_fvc', 'pred_fvc', 'score']])
for model in models:
    metric = []
    for layer in model.layers:
        if len(layer.weights) > 0:
            print(layer.name, layer.weights[0].shape, layer.weights[0])
    for q in tqdm(range(1, 10)):
        m = []
        for p in vl_p:
            x, y = [] , []
            tab = [] 

            if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:
                continue

            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')
            for i in ldir:
#                 print(len(ldir), int(i[:-4]), i, p)
                if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:
                    x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 
                    y.append(i)
                    tab.append(get_tab(train.loc[train.Patient == p, :])) 
            if len(x) < 1:
                continue
            tab = np.array(tab)
#             print(tab)

            x = np.expand_dims(x, axis=-1) 
            _a = model.predict([x, tab]) 
            a = np.quantile(_a, q / 10)

            percent_true = train.Percent.values[train.Patient == p]
            fvc_true = train.FVC.values[train.Patient == p]
            weeks_true = train.Weeks.values[train.Patient == p]

            fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]
            percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])
            m.append(score(fvc_true, fvc, percent))
#             df2 = pd.DataFrame({"patient_id": p,  "image_num": y , 'actual_fvc': fvc_true , 'pred_fvc': fvc ,'percent_true': percent_true, 'percent_pred': percent, 'weeks_true': weeks_true, 'score': score })
#             df2 = pd.DataFrame({"patient_id": p, 'actual_fvc': fvc_true , 'pred_fvc': fvc, 'score': score(fvc_true, fvc, percent) })
#             print(df2)
#             mid_df.append(df2, ignore_index = True)
        print(np.mean(m))
        metric.append(np.mean(m))

    q = (np.argmin(metric) + 1)/ 10

    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 
    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 
    A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 
    STD, WEEK = {}, {} 
    for p in test.Patient.unique():
        x = [] 
        tab = [] 
        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')
        for i in ldir:
            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:
                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 
                tab.append(get_tab(test.loc[test.Patient == p, :])) 
        if len(x) <= 1:
            continue
        tab = np.array(tab) 

        x = np.expand_dims(x, axis=-1) 
        _a = model.predict([x, tab]) 
        a = np.quantile(_a, q)
        A_test[p] = a
        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]
        P_test[p] = test.Percent.values[test.Patient == p] 
        WEEK[p] = test.Weeks.values[test.Patient == p]

    for k in sub.Patient_Week.values:
        p, w = k.split('_')
        w = int(w) 

        fvc = A_test[p] * w + B_test[p]
        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc
        sub.loc[sub.Patient_Week == k, 'Confidence'] = (
            P_test[p] - A_test[p] * abs(WEEK[p] - w) 
    ) 

    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()
    subs.append(_sub)


# In[ ]:


mid_df


# In[ ]:


N = len(subs)
sub = subs[0].copy() # ref
sub["FVC"] = 0
sub["Confidence"] = 0
for i in range(N):
    sub["FVC"] += subs[0]["FVC"] * (1/N)
    sub["Confidence"] += subs[0]["Confidence"] * (1/N)


# In[ ]:


sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)


# In[ ]:


img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()


# In[6]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE=128

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


# In[8]:


dicom_root_path = '../input/osic-pulmonary-fibrosis-progression/train/'
Patients_id = os.listdir(dicom_root_path)
n_dicom_dict = {"Patient":[],"n_dicom":[],"list_dicom":[]}

for Patient_id in Patients_id:
    dicom_id_path = glob.glob(dicom_root_path + Patient_id + "/*")
    n_dicom_dict["n_dicom"].append(len(dicom_id_path))
    n_dicom_dict["Patient"].append(Patient_id)
    list_dicom_id = sorted([int(i.split("/")[-1][:-4]) for i in dicom_id_path])
    n_dicom_dict["list_dicom"].append(list_dicom_id)

dicom_pd = pd.DataFrame(n_dicom_dict)
dicom_pd.head()


# In[ ]:


print(f"min dicom number is {min(dicom_pd['n_dicom'])}\nmax dicom number is {max(dicom_pd['n_dicom'])}")

plt.hist(dicom_pd['n_dicom'], bins=20)
plt.title('Number of dicom per patient');


# In[ ]:


dicom_pd['height'],dicom_pd['width'], dicom_pd['kvp'] = -1,-1, -1
for Patient_id in Patients_id:
    dicom_id_path = glob.glob(dicom_root_path + Patient_id + "/*")
    for patient_dicom_id_path in dicom_id_path:
        dicom = pydicom.dcmread(patient_dicom_id_path)
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'height'] = dicom.Rows
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'width'] = dicom.Columns
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'kvp'] = dicom.KVP
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'PatientPosition'] = dicom.PatientPosition
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'Manufacture'] = dicom.Manufacturer
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImageType0'] = dicom.ImageType[0] if len(dicom.ImageType) >= 1 else np.nan
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImageType1'] = dicom.ImageType[1] if len(dicom.ImageType) >= 2 else np.nan
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImageType2'] = dicom.ImageType[2] if len(dicom.ImageType) >=3 else np.nan
        dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImageType3'] = dicom.ImageType[3] if len(dicom.ImageType) >=4 else np.nan
#         print(dicom.ImageType)
#         dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImagePositionPatientX'] = dicom.PatientPosition[0] if type(dicom.PatientPosition) is tuple else np.nan
#         dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImagePositionPatientY'] = dicom.PatientPosition[1] if type(dicom.PatientPosition) is tuple else np.nan
#         dicom_pd.loc[dicom_pd.Patient==Patient_id,'ImagePositionPatientZ'] = dicom.PatientPosition[2] if type(dicom.PatientPosition) is tuple else np.nan
        break


# In[ ]:


dicom_pd.head()


# In[ ]:


plt.hist(dicom_pd['PatientPosition'])


# In[ ]:


reshape_dicom_pd = dicom_pd.loc[(dicom_pd.height!=512) | (dicom_pd.width!=512),:]
reshape_dicom_pd = reshape_dicom_pd.reset_index(drop=True)
reshape_dicom_pd.head()


# In[ ]:


f, ax = plt.subplots(len(reshape_dicom_pd.head()),2, figsize=(15, 18))
for idx,patient_id in enumerate(reshape_dicom_pd.head()['Patient']):
    paths = random.sample(glob.glob(dicom_root_path + patient_id + "/*"),2)
    dicom1 = pydicom.dcmread(paths[0])
    dicom2 = pydicom.dcmread(paths[1])
    ax[idx,0].set_title(f"{patient_id}-{paths[0].split('/')[-1][:-4]}-{reshape_dicom_pd.loc[idx,'height']}-{reshape_dicom_pd.loc[idx,'width']}")
    ax[idx,0].imshow(dicom1.pixel_array, cmap=plt.cm.bone)
    ax[idx,1].set_title(f"patient id is {patient_id}-{paths[1].split('/')[-1][:-4]}-{reshape_dicom_pd.loc[idx,'height']}-{reshape_dicom_pd.loc[idx,'width']}")
    ax[idx,1].imshow(dicom2.pixel_array, cmap=plt.cm.bone)
plt.show()


# In[ ]:


crop_id = ['ID00240637202264138860065','ID00122637202216437668965','ID00086637202203494931510',
            'ID00419637202311204720264','ID00014637202177757139317','ID00094637202205333947361',
            'ID00067637202189903532242',]
reshape_dicom_pd['resize_type'] = 'resize'
reshape_dicom_pd.loc[reshape_dicom_pd.Patient.isin(crop_id),'resize_type'] = 'crop'


# In[ ]:


dicom_pd['resize_type'] = 'no'
for idx,i in enumerate(reshape_dicom_pd['Patient']):
    dicom_pd.loc[dicom_pd.Patient==i,'resize_type'] = reshape_dicom_pd.loc[idx,'resize_type']
dicom_pd.head()


# In[ ]:


train_pd = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
temp_pd = pd.DataFrame(columns=train_pd.columns)
for i in range(len(dicom_pd)):
    patient_pd = train_pd[train_pd.Patient==dicom_pd.iloc[i].Patient]
    zeroweek = patient_pd['Weeks'].min()
    #if sum(patient_pd.Weeks==zeroweek)>1:
    #    print(pd.unique(patient_pd.Patient))
    temp_pd = temp_pd.append(patient_pd[patient_pd.Weeks==zeroweek].iloc[0])
dicom_pd = pd.merge(dicom_pd, temp_pd, on=['Patient'])
dicom_pd.head()


# In[ ]:


dicom_pd[dicom_pd.resize_type!='no'].head()


# In[ ]:


dicom_pd['ImageType0'] = dicom_pd['ImageType0'].apply(lambda x: 'ORIGINAL' if x not in ['ORIGINAL', 'DERIVED'] else x)
dicom_pd['ImageType1'] = dicom_pd['ImageType1'].apply(lambda x: 'PRIMARY' if x not in ['PRIMARY', 'SECONDARY'] else x)
dicom_pd['ImageType2'] = dicom_pd['ImageType2'].apply(lambda x: 'AXIAL' if x not in ['AXIAL', 'REFORMATTED', 'OTHER'] else x)
dicom_pd[dicom_pd['Patient']=='ID00421637202311550012437']


# In[ ]:


import seaborn as sns
dicom_pd.isna().sum()
# dicom_pd.shape
sns.countplot(x ='Manufacture', hue = 'ImageType3', data = dicom_pd)


# In[ ]:


d = dicom_pd[dicom_pd['Manufacture'].isin(['TOSHIBA', 'GE MEDICAL SYSTEMS'])]
sns.countplot(x ='Manufacture', hue = 'ImageType3', data = d)


# In[ ]:


dicom_pd_mf = dicom_pd[dicom_pd['ImageType3'].isna()][['Manufacture', 'ImageType3']]


# In[ ]:


# dicom_pd.shape
sns.countplot(x ='Manufacture',  data = dicom_pd_mf)


# In[ ]:


# dicom_pd_mf.loc[(dicom_pd_mf['Manufacture']=='SIEMENS') & (dicom_pd['ImageType3'].isna()==True), 'ImageType3'] = 'CT_SOM5_SPI'
# dicom_pd_mf.loc[(dicom_pd_mf['Manufacture']=='Philips') & (dicom_pd['ImageType3'].isna()==True), 'ImageType3'] = 'HELIX'
# dicom_pd_mf.loc[(dicom_pd_mf['Manufacture']=='GE MEDICAL SYSTEMS') & (dicom_pd['ImageType3'].isna()==True), 'ImageType3'] = 'AVERAGE'

dicom_pd_mf.loc[dicom_pd_mf['Manufacture']=='GE MEDICAL SYSTEMS']


# In[ ]:


dicom_pd


# In[7]:


def load_scan(path,resize_type='no'):
    """
    Loads scans from a folder and into a list.
    
    Parameters: path (Folder path)
    
    Returns: slices (List of slices)
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    
    try:
        slice_thickness = abs(slices[-1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])/(len(slices))
    except:
        try:
            slice_thickness = abs(slices[-1].SliceLocation - slices[0].SliceLocation)/(len(slices))
        except:
            slice_thickness = slices[0].SliceThickness
        
    for s in slices:
        s.SliceThickness = slice_thickness
        if resize_type == 'resize':
            s.PixelSpacing = s.PixelSpacing*(s.Rows/512)  
    return slices


# In[8]:


def transform_to_hu(slices):
    """
    transform dicom.pixel_array to Hounsfield.
    Parameters: list dicoms
    Returns:numpy Hounsfield
    """
    
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    # convert ouside pixel-values to air:
    # I'm using <= -1000 to be sure that other defaults are captured as well
    #images[images <= -1000] = 0
    
    # convert to HU
    for n in range(len(slices)):
        
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
            
        images[n] += np.int16(intercept)
    
    return np.array(images, dtype=np.int16)


# In[9]:


def crop_image(img: np.ndarray):
    edge_pixel_value = img[0, 0]
    mask = img != edge_pixel_value
    return img[np.ix_(mask.any(1),mask.any(0))]

def resize_image(img: np.ndarray,reshape=(512,512)):
    img = cv2.resize(img,(512,512))
    return img

def preprocess_img(img,resize_type):
    if resize_type == 'resize':
        img = [resize_image(im) for im in img]
    if resize_type == 'crop':
        img = [crop_image(im) for im in img]
        
    return np.array(img, dtype=np.int64)


# In[10]:


class Test_Generate(Dataset):
    def __init__(self,imgs_dicom,resize_type='no'):
        self.imgs_dicom = imgs_dicom
        self.resize_type = resize_type
    def __getitem__(self,index):
        
        slice_img = self.imgs_dicom[index].pixel_array
        slice_img = (slice_img-slice_img.min())/(slice_img.max()-slice_img.min())
        slice_img = (slice_img*255).astype(np.uint8)
        if self.resize_type == 'crop':
            slice_img = crop_image(slice_img)
        elif self.resize_type == 'resize':
            slice_img = cv2.resize(slice_img,(512,512))
            
        slice_img = slice_img[None,:,:]
        slice_img = (slice_img/255).astype(np.float32)
        return slice_img
        
    def __len__(self):
        return len(self.imgs_dicom)


# In[13]:


device =  torch.device('cuda:0')
model = smp.Unet('densenet121', classes=1, in_channels=1,activation='sigmoid',encoder_weights=None).to(device)
model.load_state_dict(torch.load('../input/2020osic/best_lung_Unet_densenet121.pth'))
batch = 8

def Unet_mask(model: nn.Module,input_data: DataLoader):
    model.eval()
    outs = []
    for idx, sample in enumerate(test_loader):
        image = sample
        image = image.to(device)
        with torch.no_grad():
            out = model(image)
        out = out.cpu().data.numpy()
        out = np.where(out>0.5,1,0)
        out = np.squeeze(out,axis=1)
        outs.append(out)

    outs = np.concatenate(outs)
    return outs


# In[ ]:


f, ax = plt.subplots(4,2, figsize=(14, 14))

for i in range(4):
    path = os.path.join(dicom_root_path,dicom_pd.iloc[i].Patient)
    patient_scans = load_scan(path)
    
    test_db = Test_Generate(patient_scans,dicom_pd.iloc[i].resize_type)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=4)
    
    masks = Unet_mask(model,test_loader)
    
    #patient_images = transform_to_hu(patient_scans)
    
    #patient_images = preprocess_img(patient_images,dicom_pd.iloc[i])
    
    num_slices = len(masks)
    patient_image = test_db[num_slices//2][0]
    patient_mask = masks[num_slices//2]
    
    #Mask = generate_internal_mask(patient_image)
    
    ax[i,0].set_title(f"{dicom_pd.iloc[i].Patient}-{dicom_pd.iloc[i].FVC}")
    ax[i,0].imshow(patient_image,cmap='gray')
    ax[i,1].imshow(patient_mask)
    
plt.show()
plt.close()


# In[ ]:


thresh = [-1000,0]
f, ax = plt.subplots(2,2, figsize=(18, 18))
sampler = random.sample(range(len(dicom_pd)),4)
for i in range(4):
    path = os.path.join(dicom_root_path,dicom_pd.iloc[sampler[i]].Patient)
    patient_scans = load_scan(path)
    
    test_db = Test_Generate(patient_scans,dicom_pd.iloc[sampler[i]].resize_type)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=4)
    
    masks = Unet_mask(model,test_loader)
    
    patient_images = transform_to_hu(patient_scans)
    patient_images = preprocess_img(patient_images,dicom_pd.loc[sampler[i],'resize_type'])
    
    num_slices = len(patient_images)
    #patient_images = patient_images[int(num_slices*0.1):int(num_slices*0.9)]
    #patient_masks = masks[int(num_slices*0.1):int(num_slices*0.9)]
    
    #patient_images = patient_images[num_slices//2]
    #patient_masks = pool.map(generate_internal_mask,patient_images)
    #patient_masks = patient_masks[int(num_slices*0.1):int(num_slices*0.9)]
    patient_images = masks*patient_images
    patient_images_nonzero = patient_images[np.nonzero(patient_images)]
    #patient_images_mean = np.mean(patient_images,0)
    
    s_pixel = patient_images_nonzero.flatten()
    s_pixel = s_pixel[np.where((s_pixel>thresh[0])&(s_pixel<thresh[1]))]
    
    ax[i//2,i%2].set_title(f"{dicom_pd.iloc[i].Patient}-{dicom_pd.iloc[i].FVC}")
    ax[i//2,i%2].hist(s_pixel, bins=20)

plt.show()


# In[14]:


#def func_volume(patient_scan,patient_mask):
    

def caculate_lung_volume(patient_scans,patient_masks):
    """
    caculate volume of lung from mask
    Parameters: list dicom scans,list patient CT Mask
    Returns: volume cm³　(float)
    """
    lung_volume = 0
    for i in range(len(patient_masks)):
        
        pixel_spacing = patient_scans[i].PixelSpacing
        slice_thickness = patient_scans[i].SliceThickness
        lung_volume += np.count_nonzero(patient_masks[i])*pixel_spacing[0]*pixel_spacing[1]*slice_thickness
        
    return lung_volume*0.001


# In[15]:


def caculate_histgram_statistical(patient_images,patient_masks,thresh = [-600,0]):
    """
    caculate hisgram kurthosis of lung hounsfield
    Parameters: list patient CT image 512*512,thresh divide lung
    Returns: histgram statistical characteristic(Mean,Skew,Kurthosis)
    """
    statistical_characteristic = dict(Mean=0,Skew=0,Kurthosis=0)
    num_slices = len(patient_images)
    
    #patient_images = patient_images[int(num_slices*0.1):int(num_slices*0.9)]
    #patient_masks = patient_masks[int(num_slices*0.1):int(num_slices*0.9)]
    patient_images = patient_masks*patient_images
    patient_images_nonzero = patient_images[np.nonzero(patient_images)]
    
    s_pixel = patient_images_nonzero.flatten()
    s_pixel = s_pixel[np.where((s_pixel>thresh[0])&(s_pixel<thresh[1]))]
    
    statistical_characteristic['Mean'] = np.mean(s_pixel)
    statistical_characteristic['Skew'] = skew(s_pixel)
    statistical_characteristic['Kurthosis'] = kurtosis(s_pixel)
    
    return statistical_characteristic


# In[18]:


import gdcm


# In[ ]:


lung_stat_pd = pd.DataFrame(columns=['Patient','Volume','Mean','Skew','Kurthosis'])

for i in tqdm(range(len(dicom_pd))):
    path = os.path.join(dicom_root_path,dicom_pd.iloc[i].Patient)
    lung_stat_pd.loc[i,'Patient'] = dicom_pd.iloc[i].Patient
    patient_scans = load_scan(path)
    
    test_db = Test_Generate(patient_scans,dicom_pd.iloc[i].resize_type)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=4)
    masks = Unet_mask(model,test_loader)
    
    
    patient_images = transform_to_hu(patient_scans)
    patient_images = preprocess_img(patient_images,dicom_pd.loc[i,'resize_type'])
    
    lung_stat_pd.loc[i,'Volume'] = caculate_lung_volume(patient_scans,masks)                           
    #patient_images = resize_image(patient_images) if dicom_pd.iloc[i].resize_type=='resize' else patient_images
    #patient_images = resize_image(patient_masks) if dicom_pd.iloc[i].resize_type=='resize' else patient_images
    
    statistical_characteristic = caculate_histgram_statistical(patient_images,masks,thresh)
    lung_stat_pd.loc[i,'Mean'] = statistical_characteristic['Mean']
    lung_stat_pd.loc[i,'Skew'] = statistical_characteristic['Skew']
    lung_stat_pd.loc[i,'Kurthosis'] = statistical_characteristic['Kurthosis']
    
lung_stat_pd.head()


# In[ ]:


dicom_feature = pd.merge(dicom_pd, lung_stat_pd, on=['Patient'])
dicom_feature.head()


# In[ ]:


dicom_feature = dicom_feature.drop(['list_dicom', 'height','width','resize_type','n_dicom'], axis=1)
dicom_feature = dicom_feature.drop(['ImageType3'], axis=1)
dicom_feature.head(20)


# In[ ]:


dicom_feature


# In[ ]:


dicom_feature.to_csv('./CT_feature.csv',index=False)


# In[19]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
device = torch.device('cuda')

test_df = pd.read_csv(f"{ROOT}/test.csv")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(test_df.drop('Weeks', axis=1), on="Patient")


# In[20]:


class Inference_Generate(Dataset):
    def __init__(self,imgs_dicom):
        self.imgs_dicom = imgs_dicom
        
    def __getitem__(self,index):
        metainf = self.imgs_dicom[index]
        slice_img = metainf.pixel_array
        slice_img = (slice_img-slice_img.min())/(slice_img.max()-slice_img.min())
        slice_img = (slice_img*255).astype(np.uint8)
        if metainf.Rows!=512 or metainf.Columns!=512:
            slice_img = cv2.resize(slice_img,(512,512))
            
        slice_img = slice_img[None,:,:]
        slice_img = (slice_img/255).astype(np.float32)
        return slice_img
        
    def __len__(self):
        return len(self.imgs_dicom)


# In[ ]:


thresh = [-1000,0]
ct_root_path = '../input/osic-pulmonary-fibrosis-progression/test/'
lung_test_feature = pd.DataFrame(columns=['Patient','Volume','Mean','Skew','Kurthosis', 'kvp', 'PatientPosition','Manufacture', 'ImageType0',
                                          'ImageType1', 'ImageType2'])
for idx,i in enumerate(pd.unique(test_df['Patient'])):
    lung_test_feature.loc[idx,'Patient'] = i
    patient_scans = load_scan(ct_root_path + i)
    dicom_id_path = glob.glob(ct_root_path + i + "/*")
    for patient_dicom_id_path in dicom_id_path:
        sc = pydicom.dcmread(patient_dicom_id_path)
        lung_test_feature.loc[lung_test_feature.Patient==i,'kvp'] = sc.KVP
        lung_test_feature.loc[lung_test_feature.Patient==i,'PatientPosition'] = sc.PatientPosition
        lung_test_feature.loc[lung_test_feature.Patient==i,'Manufacture'] = sc.Manufacturer
        lung_test_feature.loc[lung_test_feature.Patient==i,'ImageType0'] = sc.ImageType[0] if len(sc.ImageType) >= 1 else np.nan
        lung_test_feature.loc[lung_test_feature.Patient==i,'ImageType1'] = sc.ImageType[1] if len(sc.ImageType) >= 2 else np.nan
        lung_test_feature.loc[lung_test_feature.Patient==i,'ImageType2'] = sc.ImageType[2] if len(sc.ImageType) >=3 else np.nan
    test_db = Inference_Generate(patient_scans)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=4)
    masks = Unet_mask(model,test_loader)
    
    patient_images = transform_to_hu(patient_scans)
  
    if patient_images[0].shape!=(512,512):
        patient_images = preprocess_img(patient_images,'resize')
  
    lung_test_feature.loc[idx,'Volume'] = caculate_lung_volume(patient_scans,masks)     
    statistical_characteristic = caculate_histgram_statistical(patient_images,masks,thresh)
    lung_test_feature.loc[idx,'Mean'] = statistical_characteristic['Mean']
    lung_test_feature.loc[idx,'Skew'] = statistical_characteristic['Skew']
    lung_test_feature.loc[idx,'Kurthosis'] = statistical_characteristic['Kurthosis']
lung_test_feature.head()


# In[ ]:


lung_test_feature['ImageType0'] = lung_test_feature['ImageType0'].apply(lambda x: 'ORIGINAL' if x not in ['ORIGINAL', 'DERIVED'] else x)
lung_test_feature['ImageType1'] = lung_test_feature['ImageType1'].apply(lambda x: 'PRIMARY' if x not in ['PRIMARY', 'SECONDARY'] else x)
lung_test_feature['ImageType2'] = lung_test_feature['ImageType2'].apply(lambda x: 'AXIAL' if x not in ['AXIAL', 'REFORMATTED', 'OTHER'] else x)


# In[ ]:


test_df


# In[ ]:


test_df = test_df.merge(lung_test_feature, on='Patient')
sub = sub.merge(lung_test_feature, on='Patient')

train_df = pd.read_csv(f"{ROOT}/train.csv")
#train_df = train_df.merge(lung_stat_pd, on='Patient')
feature_ct = pd.read_csv('CT_feature.csv',usecols=['Patient', 'kvp', 'PatientPosition', 'Manufacture', 'ImageType0',
       'ImageType1', 'ImageType2', 'Volume', 'Mean', 'Skew', 'Kurthosis'])
train_df = train_df.merge(feature_ct, on='Patient')

train_df['WHERE'] = 'train'
test_df['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = train_df.append([test_df, sub])

data.head()


# In[21]:


data = pd.read_csv('../input/final-data/final_data.csv')


# In[22]:


list(data.columns)


# In[ ]:


data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

base = (data
    .loc[data.Weeks == data.min_week][['Patient','FVC']]
    .rename({'FVC': 'min_FVC'}, axis=1)
    .groupby('Patient')
    .first()
    .reset_index())

data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base

FE = list(data.Sex.unique()) + list(data.SmokingStatus.unique()) + list(data.PatientPosition.unique()) + list(data.Manufacture.unique()) + list(data.ImageType0.unique()) + list(data.ImageType1.unique()) + list(data.ImageType2.unique())
data = pd.concat([
    data,
    pd.get_dummies(data.Sex),
    pd.get_dummies(data.SmokingStatus),
    pd.get_dummies(data.PatientPosition),
    pd.get_dummies(data.Manufacture),
    pd.get_dummies(data.ImageType0),
    pd.get_dummies(data.ImageType1),
    pd.get_dummies(data.ImageType2),
    pd.get_dummies(data.kvp),
], axis=1)

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min())
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min())

data['volume'] = (data['Volume'] - data['Volume'].min() ) / ( data['Volume'].max() - data['Volume'].min())
data['mean'] = (data['Mean'] - data['Mean'].min()) / (data['Mean'].max() - data['Mean'].min())
data['skew'] = (data['Skew'] - data['Skew'].min())/(data['Skew'].max() - data['Skew'].min())
data['kurthosis'] = (data['Kurthosis'] - data['Kurthosis'].min())/(data['Kurthosis'].max() - data['Kurthosis'].min())

FE += ['age','percent','week', 'BASE','volume','mean','skew','kurthosis']


# In[ ]:


data.rename(columns={100: 'kvp_100', 110: 'kvp_110', 120: 'kvp_120', 130: 'kvp_130', 135: 'kvp_135', 140: 'kvp_140'}, inplace=True)


# In[23]:


train_df = data.loc[data.WHERE=='train']
test_df = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']


# In[24]:


FE = ['Male',
 'Female',
 'Ex-smoker',
 'Never smoked',
 'Currently smokes',
#  'FFS',
#  'HFS',
#  'FFP',
#  'HFP',
#  'GE MEDICAL SYSTEMS',
#  'SIEMENS',
#  'TOSHIBA',
#  'Philips',
#  'PACSMATT',
#  'Hitachi Medical Corporation',
#  'PACSGEAR',
#  'ORIGINAL',
#  'DERIVED',
#  'PRIMARY',
#  'SECONDARY',
#  'AXIAL',
#  'REFORMATTED',
#  'OTHER',
 'age',
 'percent',
 'week',
 'BASE',
 'volume',
 'mean',
 'skew',
 'kurthosis',
#  'kvp_100',
#  'kvp_110',
#  'kvp_120',
#  'kvp_130',
#  'kvp_135',
#  'kvp_140'
     ]


# In[ ]:


data.to_csv('./final_data.csv',index=False)


# In[ ]:


# data = pd.read_csv('../input/final-data/final_data.csv')


# In[25]:


tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
# del data


# In[26]:


tr.shape, chunk.shape, sub.shape


# In[27]:


C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
#     print("fvc_pred", fvc_pred, 'fvc_true',y_true[:, 0], y_true, y_pred)
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

# def make_model(nh):
#     z = L.Input((nh,), name="Patient")
#     x = L.Dense(100, activation="relu", name="d1")(z)
#     x = L.Dense(100, activation="relu", name="d2")(x)
#     p1 = L.Dense(3, activation="linear", name="p1")(x)
#     p2 = L.Dense(3, activation="relu", name="p2")(x)
#     preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
#                      name="preds")([p1, p2])
    
#     model = M.Model(z, preds, name="CNN")
#     model.compile(loss=mloss(0.65), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
#     return model

def make_model(nh):
    z = L.Input((nh,), name="Patient")
    y = L.LSTM(25, return_sequences=True)(tf.expand_dims(z, axis=-1))
    y = L.LSTM(25, return_sequences=True)(y)
    y = L.Reshape([25*nh])(y)
    x = L.Dense(100, activation="relu", name="d1")(y)
    x = L.Dense(100, activation="relu", name="d2")(x)
    x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(0.65), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model


# In[28]:


y = tr['FVC'].values
z=tr[FE].values
ze=sub[FE].values
nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))


# In[29]:


net = make_model(nh)
print(net.summary())
print(net.count_params())


# In[30]:


NFOLD = 15 # originally 5
kf = KFold(n_splits=NFOLD)


# In[31]:


get_ipython().run_cell_magic('time', '', 'cnt = 0\nEPOCHS = 800\nBATCH_SIZE = 128\nfor tr_idx, val_idx in kf.split(z):\n    cnt += 1\n    print(f"FOLD {cnt}")\n    net = make_model(nh)\n    print("working")\n    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, \n            validation_data=(z[val_idx], y[val_idx]), verbose=0) #\n    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("predict val...")\n    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)\n    print(y[val_idx], pred[val_idx])\n    print("predict test...")\n    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD')


# In[32]:


sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)


# In[33]:


idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()


# In[34]:


print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())


# In[35]:


plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


# In[36]:


sub.head()


# In[37]:


# PREDICTION
sub['FVC1'] = 1.*pe[:, 1]
sub['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)


# In[38]:


subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


# In[39]:


subm.head()


# In[40]:


subm.describe().T


# In[41]:


otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1


# In[42]:


subm[["Patient_Week","FVC","Confidence"]].to_csv("submission_regression_0.65_only_pos.csv", index=False)


# In[43]:


reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()


# In[44]:


img_sub = pd.read_csv('../input/img-sub/Submitted_op.csv')
df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)


# In[47]:


df = df1[['Patient_Week']].copy()
df['FVC'] = 0.25*df1['FVC'] + 0.75*df2['FVC']
df['Confidence'] = 0.26*df1['Confidence'] + 0.74*df2['Confidence']
df.head()


# In[46]:


df = pd.read_csv('../input/output/submission.csv')


# In[48]:


df.to_csv('./sample_submission.csv', index=False)

