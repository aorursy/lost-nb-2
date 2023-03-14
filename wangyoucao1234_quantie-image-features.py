#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %% [markdown]
# # OSIC: keras model inference on images and tabular data

# %% [markdown]
# This is an inference part for a notebook with model training [OSIC keras images and tabular data model](https://www.kaggle.com/vgarshin/osic-keras-images-and-tabular-data-model).

# %% [code]
# %%time
# !pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
# !pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index

# %% [code]
import warnings

import scipy
from scipy.ndimage import zoom
from skimage import measure
from tensorflow.python.keras.models import Model

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import pydicom
print('tensorflow version:', tf.__version__)
t_count=0
# %% [code]
def seed_all(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_all(2020)
DATA_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression'
BATCH_SIZE_PRED = 1
FEATURES = True
ADV_FEATURES = True
C_SIGMA, C_DELTA = tf.constant(70, dtype='float32'), tf.constant(1000, dtype='float32')
QS = [.15, .50, .85]
IMG_SIZE = 224
RESIZE = 224
SEQ_LEN = 112
QUANTIE_IMAGE_FEATURE_LEN=32

LAMBDA = .8
MDL_VERSION = 'v3'
MODELS_PATH = '.'


resize_dims = (112, 224, 224)


# %% [code]
train = pd.read_csv(f'{DATA_PATH}/train.csv')
train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
test = pd.read_csv(f'{DATA_PATH}/test.csv')
subm = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
subm['Patient'] = subm['Patient_Week'].apply(lambda x: x.split('_')[0])
subm['Weeks'] = subm['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
subm =  subm[['Patient','Weeks','Confidence','Patient_Week']]
subm = subm.merge(test.drop('Weeks', axis=1), on='Patient')
train['SPLIT'] = 'train'
test['SPLIT'] = 'val'
subm['SPLIT'] = 'test'
data = train.append([test, subm])
print('train:',  train.shape, 'unique Pats:', train.Patient.nunique(),
      '\ntest:', test.shape,  'unique Pats:', test.Patient.nunique(),
      '\nsubm:', subm.shape,  'unique Pats:', subm.Patient.nunique(),
      '\ndata',  data.shape,  'unique Pats:', data.Patient.nunique())
data['min_week'] = data['Weeks']
data.loc[data.SPLIT == 'test', 'min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

# %% [code]
data = pd.concat([data, pd.get_dummies(data.Sex), pd.get_dummies(data.SmokingStatus)], axis=1)
if FEATURES:
    base = data.loc[data.Weeks == data.min_week]
    base = base[['Patient', 'FVC']].copy()
    base.columns = ['Patient', 'min_week_FVC']
    base['nb'] = 1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    base = base[base.nb == 1]
    base.drop('nb', axis=1, inplace=True)
    data = data.merge(base, on='Patient', how='left')
    data['relative_week'] = data['Weeks'] - data['min_week']
    del base
if ADV_FEATURES:
    target_cols = [
        'FVC',
        'Percent',
        'min_week_FVC'
    ]
    enc_cols =  [
        'Female',
        'Male',
        'Currently smokes',
        'Ex-smoker',
        'Never smoked'
    ]
    for t_col in target_cols:
        for col in enc_cols:
            col_name = f'_{col}_{t_col}_'
            data[f'enc{col_name}mean'] = data.groupby(col)[t_col].transform('mean')
            data[f'enc{col_name}std'] = data.groupby(col)[t_col].transform('std')
    data['TC'] = 0
    data.loc[data['Weeks'] == 0, 'TC'] = 1
print(data.shape)
print(data.columns)

# %% [code]
feat_cols = [
    'Female', 'Male',
    'Currently smokes', 'Ex-smoker', 'Never smoked'
]
scale_cols = [
    'Percent',
    'Age',
    'relative_week',
    'min_week_FVC'
]
scale_cols.extend([x for x in data.columns if 'FVC_mean' in x])
scaler = MinMaxScaler()
data[scale_cols] = scaler.fit_transform(data[scale_cols])
feat_cols.extend(scale_cols)
print('all data columns:', data.columns)

# %% [code]
train = data.loc[data.SPLIT == 'train']
test = data.loc[data.SPLIT == 'val']
subm = data.loc[data.SPLIT == 'test']
del data
subm.head()

class CropBoundingBox:
    @staticmethod
    def bounding_box(img3d: np.array):
        mid_img = img3d[int(img3d.shape[0] / 2)]
        same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
        same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
        if same_first_col and same_first_row:
            return True
        else:
            return False

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        if not self.bounding_box(image):
            return sample

        mid_img = image[int(image.shape[0] / 2)]
        r_min, r_max = None, None
        c_min, c_max = None, None
        for row in range(mid_img.shape[0]):
            if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
                r_min = row
            if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None                     and r_min is not None:
                r_max = row
                break

        for col in range(mid_img.shape[1]):
            if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
                c_min = col
            if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None                     and c_min is not None:
                c_max = col
                break

        image = image[:, r_min:r_max, c_min:c_max]
        return {'image': image, 'metadata': data}


# %% [markdown]
# ### 3.2.2. convert_to_hu.py
# Credits to [Guido Zuidhof's tutorial](https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial).

# %% [code]
class ConvertToHU:
    # def __call__(self, sample):
    #     image, data = sample['image'], sample['metadata']
    #
    #     img_type = data.ImageType
    #     is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')
    #     if not is_hu:
    #         warnings.warn(f'Patient {data.PatientID} CT Scan not cannot be'
    #                       f'converted to Hounsfield Units (HU).')
    #
    #     intercept = data.RescaleIntercept
    #     slope = data.RescaleSlope
    #     image = (image * slope + intercept).astype(np.int16)
    #     return {
    #         'features': sample['features'],
    #         'image': image,
    #         'metadata': data,
    #         'target': sample['target']
    #     }
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = data.RescaleIntercept
        slope = data.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return {

            'image': image,
            'metadata': data,

        }

    def get_pixels_hu(self, scans):
            image = np.stack([s.pixel_array.astype(float) for s in scans])
            image = image.astype(np.int16)
            image[image == -2000] = 0
            intercept = scans[0].RescaleIntercept
            slope = scans[0].RescaleSlope
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)
            image += np.int16(intercept)
            return np.array(image, dtype=np.int16)

# %% [markdown]
# ### 3.2.3. resize.py

# %% [code]
class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        resize_factor = np.array(self.output_size) / np.array(image.shape)
        image = zoom(image, resize_factor, mode='nearest')
        # plot_3d(image, 0)
        return {'image': image, 'metadata': data}


class ReSample:

    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']

        image = self.resample(image, data)
        return {'image': image, 'metadata': data}

    def resample(self,image_data, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing

        # spacing = map(float, ([scan.SliceThickness] + scan.PixelSpacing))
        # spacing = np.array(list(spacing))
        spacing = np.array([scan.SliceThickness] + list(scan.PixelSpacing), dtype=np.float32)
#         print("old space\t",spacing)
#         print("new space\t",new_spacing)
        resize_factor = spacing / new_spacing
        new_real_shape = image_data.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image_data.shape
        new_spacing = spacing / real_resize_factor
#         print("Shape before resampling\t", image_data.shape)
        image_data = scipy.ndimage.interpolation.zoom(image_data, real_resize_factor, mode='nearest')
#         print("Shape after resampling\t", image_data.shape)
        return image_data
# %% [code]
class DataGenOsic(Sequence):
    def __init__(self, df, image_features,tab_cols,
                 batch_size=8, mode='fit', shuffle=False,
                 aug=None, resize=None, seq_len=12, img_size=224):
        self.df = df
        self.shuffle = shuffle
        self.mode = mode
        self.aug = aug
        self.resize = resize
        self.batch_size = batch_size
        self.img_size = img_size
        self.seq_len = seq_len
        self.tab_cols = tab_cols
        self.on_epoch_end()
        self.image_features = image_features
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self, index):
        batch_size = min(self.batch_size, len(self.df) - index * self.batch_size)
        X_img = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        X_tab = self.df[index * self.batch_size : (index + 1) * self.batch_size][self.tab_cols].values
        pats_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['Patient'].values
        for i, pat_id in enumerate(pats_batch):


            imgs_seq = self.get_imgs_feature(pat_id)
            # load the pretrained model

            X_img[i, ] = imgs_seq
        if self.mode == 'fit':
            y = np.array(
                self.df[index * self.batch_size : (index + 1) * self.batch_size]['FVC'].values,
                dtype=np.float32
            )
            return (X_img, X_tab), y
        elif self.mode == 'predict':
            y = np.zeros(batch_size, dtype=np.float32)
            return (X_img, X_tab), y
        else:
            raise AttributeError('mode parameter error')


    # def get_imgs_seq(self, pat_id):
    #     seq_imgs = []
    #     slices = self.load_scan(pat_id)
    #     scans = self.get_pixels_hu(slices)
    #     for img_idx in range(self.seq_len):
    #         img = scans[img_idx]
    #         if self.resize:
    #             img = cv2.resize(img, (self.resize, self.resize))
    #         img = img.astype(np.float32)
    #         img = (img - (-1000)) / 1200
    #         img = np.repeat(img[..., np.newaxis], 1, -1)
    #         seq_imgs.append(img)
    #     return np.array(seq_imgs).astype(np.float32)
    def get_imgs_seq(self, pat_id):



        path = '/media/feng/data/osic-pulmonary-fibrosis-progression/osic-image-np-preprocess-cached-dataset-size-112'
        image = np.load(path + '/' + pat_id+'.npy')
        # image = image[:32, :, :]
        image = image.astype(np.float)
        # Normalize the image data
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.

        # Zero center the data
        PIXEL_MEAN = 0.02865046213070556
        image = image - PIXEL_MEAN
        # seq_imgs = []
        # slices = self.load_scan(pat_id)
        # scans = self.get_pixels_hu(slices)
        # for img_idx in range(self.seq_len):
        #     img = scans[img_idx]
        #     if self.resize:
        #         img = cv2.resize(img, (self.resize, self.resize))
        #     img = img.astype(np.float32)
        #     img = (img - (-1000)) / 1200
        image = np.repeat(image[..., np.newaxis], 1, -1)
        #     seq_imgs.append(img)
        return image
    def get_imgs_feature(self, pat_id):
        imgs_feature = self.image_features.values
        feature = imgs_feature[imgs_feature[:,0]==pat_id,1:]
                # imgs_feature.ix[[0, 2, 4, 5, 7], ['Name', 'Height', 'Weight']].head()

        return feature


    def load_scan(self,pat_id):
        global t_count
        if self.mode == 'fit':
            path = f'{DATA_PATH}/train/{pat_id}'
        elif self.mode == 'predict':
            path = f'{DATA_PATH}/test/{pat_id}'
        else:
            raise AttributeError('mode parameter error')
        try:
            # slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]
            # print(slices[0])
            # if slices[0].ManufacturerModelName == 'OsiriX':
            #
            #     image = np.stack([s.pixel_array.astype(float) for s in slices])
            #     return image, slices[0]
            # path = "/media/feng/data/osic-pulmonary-fibrosis-progression/train/ID00426637202313170790466"

            file_names = sorted(os.listdir(path), key=lambda x: int(os.path.splitext(x)[0]))
#             print(file_names)
            slices = [pydicom.read_file(str(path) + "/" + file_names[i]) for i in range(len(file_names))]

            # show([s.pixel_array.astype(float) for s in slices][:10])
#             print("files count: {}".format(len(slices)))
            # print(slices[0])
            # skip files with no SliceLocation (eg scout views)
            # slices = []

            # for f in files:
            #     if hasattr(f,'SliceThickness'):
            #         print("slices[0].SliceThickness=".format(slices[0].SliceThickness))
            #         slices.append(f)
            #     elif hasattr(f, 'SliceLocation'):
            #         slices.append(f)
            #     else:
            #         skipcount = skipcount + 1
            #
            # print("skipped, no SliceLocation: {}".format(skipcount))

            if hasattr(slices[0], 'ImagePositionPatient'):
#                 print("change the sorting with ImagePositionPatient ")
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                slice_thickness = np.abs(
                    float(slices[2].ImagePositionPatient[2]) - float(slices[3].ImagePositionPatient[2]))
                if slice_thickness == 0.0:
                    slice_thickness = slices[0].SliceThickness
#                 print("Slice Thickness: %f" % slices[0].SliceThickness)
                # print("slices[0].SpacingBetweenSlices=: %f" % slice_thickness)
                print("ImagePositionPatient slice_thickness={}".format(slice_thickness))
                for s in slices:
                    s.SliceThickness = slice_thickness
            else:
                t_count = t_count + 1

        except:

            # slice_thickness = np.abs(slices[2].SliceLocation - slices[3].SliceLocation)
            #  # print("SliceLocation slice_thickness={}".format(slice_thickness))
            #  print()
            # # except:
            # #     print("No location value")
            # #     print(slices[0])
            # #     slice_thickness = slices[0].SliceThickness

            warnings.warn(f'Patient {slices[0].PatientID} CT scan does not '
                          f'in the right scan order.')

        # for s in slices:
        #     s.SliceThickness = slice_thickness
#         print("No_ImagePositionPatient_count:" + str(t_count))
        image = np.stack([s.pixel_array.astype(float) for s in slices])

        # show([image])
        return image, slices[0]

class Image_DataGenOsic(Sequence):
    def __init__(self, df, transform,tab_cols,
                 batch_size=8, mode='fit', shuffle=False,
                 aug=None, resize=None, seq_len=12, img_size=224):
        self.df = df
        self.shuffle = shuffle
        self.mode = mode
        self.aug = aug
        self.resize = resize
        self.batch_size = batch_size
        self.img_size = img_size
        self.seq_len = seq_len
        self.tab_cols = tab_cols
        self.on_epoch_end()
        self.transform = transform
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self, index):
        batch_size = min(self.batch_size, len(self.df) - index * self.batch_size)
        X_img = np.zeros((batch_size, self.seq_len, self.img_size, self.img_size, 1), dtype=np.float32)
        pats_batch = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        for i, pat_id in enumerate(pats_batch):
            image, metadata = self.load_scan(pat_id)
            sample = {'image': image, 'metadata': metadata}
            # preprocess data
            if self.transform:
                for t in self.transform:
                    sample = t(sample)
            image, data = sample['image'], sample['metadata']

            image = image.astype(np.float)
            # Normalize the image data
            MIN_BOUND = -1000.0
            MAX_BOUND = 400.0
            image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
            image[image > 1] = 1.
            image[image < 0] = 0.

            # Zero center the data
            PIXEL_MEAN = 0.02865046213070556
            image = image - PIXEL_MEAN

            image = np.repeat(image[..., np.newaxis], 1, -1)

            # imgs_seq = self.get_imgs_feature(pat_id)

            X_img[i, ] = image

            return (X_img, X_img),
        else:
            raise AttributeError('mode parameter error')


    # def get_imgs_seq(self, pat_id):
    #     seq_imgs = []
    #     slices = self.load_scan(pat_id)
    #     scans = self.get_pixels_hu(slices)
    #     for img_idx in range(self.seq_len):
    #         img = scans[img_idx]
    #         if self.resize:
    #             img = cv2.resize(img, (self.resize, self.resize))
    #         img = img.astype(np.float32)
    #         img = (img - (-1000)) / 1200
    #         img = np.repeat(img[..., np.newaxis], 1, -1)
    #         seq_imgs.append(img)
    #     return np.array(seq_imgs).astype(np.float32)
    def get_imgs_seq(self, pat_id):



        path = '/media/feng/data/osic-pulmonary-fibrosis-progression/osic-image-np-preprocess-cached-dataset-size-112'
        image = np.load(path + '/' + pat_id+'.npy')
        # image = image[:32, :, :]
        image = image.astype(np.float)
        # Normalize the image data
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.

        # Zero center the data
        PIXEL_MEAN = 0.02865046213070556
        image = image - PIXEL_MEAN
        # seq_imgs = []
        # slices = self.load_scan(pat_id)
        # scans = self.get_pixels_hu(slices)
        # for img_idx in range(self.seq_len):
        #     img = scans[img_idx]
        #     if self.resize:
        #         img = cv2.resize(img, (self.resize, self.resize))
        #     img = img.astype(np.float32)
        #     img = (img - (-1000)) / 1200
        image = np.repeat(image[..., np.newaxis], 1, -1)
        #     seq_imgs.append(img)
        return image
    def get_imgs_feature(self, pat_id):
        imgs_feature = pd.read_pickle('df_predit_image_features.pkl').values
        feature = imgs_feature[imgs_feature[:,0]==pat_id+'.npy',1:]
                # imgs_feature.ix[[0, 2, 4, 5, 7], ['Name', 'Height', 'Weight']].head()
        return feature


    def load_scan(self,pat_id):
        global t_count
        if self.mode == 'fit':
            path = f'{DATA_PATH}/train/{pat_id}'
        elif self.mode == 'predict':
            path = f'{DATA_PATH}/test/{pat_id}'
        else:
            raise AttributeError('mode parameter error')
        try:
            # slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]
            # print(slices[0])
            # if slices[0].ManufacturerModelName == 'OsiriX':
            #
            #     image = np.stack([s.pixel_array.astype(float) for s in slices])
            #     return image, slices[0]
            # path = "/media/feng/data/osic-pulmonary-fibrosis-progression/train/ID00426637202313170790466"

            file_names = sorted(os.listdir(path), key=lambda x: int(os.path.splitext(x)[0]))
#             print(file_names)
            slices = [pydicom.read_file(str(path) + "/" + file_names[i]) for i in range(len(file_names))]

            # show([s.pixel_array.astype(float) for s in slices][:10])
#             print("files count: {}".format(len(slices)))
            # print(slices[0])
            # skip files with no SliceLocation (eg scout views)
            # slices = []

            # for f in files:
            #     if hasattr(f,'SliceThickness'):
            #         print("slices[0].SliceThickness=".format(slices[0].SliceThickness))
            #         slices.append(f)
            #     elif hasattr(f, 'SliceLocation'):
            #         slices.append(f)
            #     else:
            #         skipcount = skipcount + 1
            #
            # print("skipped, no SliceLocation: {}".format(skipcount))

            if hasattr(slices[0], 'ImagePositionPatient'):
#                 print("change the sorting with ImagePositionPatient ")
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                slice_thickness = np.abs(
                    float(slices[2].ImagePositionPatient[2]) - float(slices[3].ImagePositionPatient[2]))
                if slice_thickness == 0.0:
                    slice_thickness = slices[0].SliceThickness
#                 print("Slice Thickness: %f" % slices[0].SliceThickness)
                # print("slices[0].SpacingBetweenSlices=: %f" % slice_thickness)
#                 print("ImagePositionPatient slice_thickness={}".format(slice_thickness))
                for s in slices:
                    s.SliceThickness = slice_thickness
            else:
                t_count = t_count + 1

        except:

            # slice_thickness = np.abs(slices[2].SliceLocation - slices[3].SliceLocation)
            #  # print("SliceLocation slice_thickness={}".format(slice_thickness))
            #  print()
            # # except:
            # #     print("No location value")
            # #     print(slices[0])
            # #     slice_thickness = slices[0].SliceThickness

            warnings.warn(f'Patient {slices[0].PatientID} CT scan does not '
                          f'in the right scan order.')

        # for s in slices:
        #     s.SliceThickness = slice_thickness
#         print("No_ImagePositionPatient_count:" + str(t_count))
        image = np.stack([s.pixel_array.astype(float) for s in slices])

        # show([image])
        return image, slices[0]


class Segenment_lung:
    def __call__(self, sample):
        image, data = sample['image'], sample['metadata']
        image = self.segment_lung_mask(image, fill_lung_structures=True)
        # plot_3d(image, 0)
        return {'image': image, 'metadata': data}



    def largest_label_volume(self,im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def segment_lung_mask(self,image, fill_lung_structures=True):

        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        binary_image = np.array(image > -320, dtype=np.int8)+1
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0,0,0]

        #Fill the air around the person
        binary_image[background_label == labels] = 2


        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1


        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image
# %% [code]
def metric(y_true, y_pred, pred_std):
    clip_std = np.clip(pred_std, 70, 9e9)
    delta = np.clip(np.abs(y_true - y_pred), 0 , 1000)
    return np.mean(-1 * (np.sqrt(2) * delta / clip_std) - np.log(np.sqrt(2) * clip_std))
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C_SIGMA)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C_DELTA)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = sq2 * (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    q = tf.constant(np.array([QS]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)
def mloss(lmbd):
    def loss(y_true, y_pred):
        return lmbd * qloss(y_true, y_pred) + (1 - lmbd) * score(y_true, y_pred)
    return loss

# %% [code]
# %%time
model_file = '/kaggle/input/model-file/model_v3.h5'
image_model_file='/kaggle/input/image-model/model_3d_v0.h5'


test_image_path = f'{DATA_PATH}/test'
test_data = sorted(os.listdir(test_image_path), key=lambda x: os.path.splitext(x)[0])

test_dataset = Image_DataGenOsic(
    df=test_data,
    transform=[
        CropBoundingBox(),
        ConvertToHU(),
        ReSample(),
        # Clip(bounds=clip_bounds),
        Resize(resize_dims),
        Segenment_lung()],

    tab_cols=feat_cols,
    batch_size=BATCH_SIZE_PRED,
    mode='predict',
    shuffle=False,
    aug=None,
    resize=RESIZE,
    seq_len=SEQ_LEN,
    img_size=IMG_SIZE


)

# load the pretrained model
image_model = load_model(image_model_file)
mean_model = Model(inputs=image_model.inputs, outputs=image_model.get_layer('Dec_VAE_VDraw_Mean').output)

predit_image_features =mean_model.predict(test_dataset,verbose=1)
test_data = pd.DataFrame(test_data)
df_predit_image_features = pd.DataFrame(predit_image_features)

df_predit_image_features = pd.concat([test_data,df_predit_image_features],axis=1)

print(df_predit_image_features)
# %% [code]
subm_datagen = DataGenOsic(
    df=subm,
    image_features=df_predit_image_features,
    tab_cols=feat_cols,
    batch_size=BATCH_SIZE_PRED,
    mode='predict',
    shuffle=False,
    aug=None,
    resize=RESIZE,
    seq_len=QUANTIE_IMAGE_FEATURE_LEN,
    img_size=IMG_SIZE
)

model = load_model(model_file, custom_objects={'qloss': qloss, 'loss': mloss(LAMBDA), 'score': score})
print('model loaded:', model_file)
print(model.summary())

# %% [code]
(Xt_img, Xt_tab), _ = subm_datagen.__getitem__(0)
print('val X img: ', Xt_img.shape)
print('val X tab: ', Xt_tab.shape)
# fig, axes = plt.subplots(figsize=(10, 4), nrows=BATCH_SIZE_PRED, ncols=SEQ_LEN)
# for j in range(BATCH_SIZE_PRED):
#     for i in range(SEQ_LEN):
#         if BATCH_SIZE_PRED > 1:
#             axes[j, i].imshow(Xt_img[j][i])
#             axes[j, i].axis('off')
#             axes[j, i].set_title(f'{j + 1} of {BATCH_SIZE_PRED}')
#         else:
#             axes[i].imshow(Xt_img[j][i])
#             axes[i].axis('off')
#             axes[i].set_title(f'{j + 1} of {BATCH_SIZE_PRED}')
# plt.show()

# %% [code]
# %%time
preds_subm = model.predict(subm_datagen, verbose=1)
print('predictions shape:', preds_subm.shape)
print('predictions sample:', preds_subm[0])

# # %% [code]
# subm['FVC'] = preds_subm[:, 1]
# subm['Confidence'] = preds_subm[:, 2] - preds_subm[:, 0]
# subm[['Patient_Week','FVC','Confidence']].to_csv('submission.csv', index=False)
# subm[['Patient_Week','FVC','Confidence']].describe().T

# %% [code]
subm['FVC1'] = preds_subm[:, 1]
subm['Confidence1'] = preds_subm[:,2] - preds_subm[:,0]

# get rid of unused data and show some non-empty data
submission = subm[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
submission.loc[~submission.FVC1.isnull()].head(10)

# %% [code]
submission.loc[~submission.FVC1.isnull(),'FVC'] = submission.loc[~submission.FVC1.isnull(),'FVC1']


submission.loc[~submission.FVC1.isnull(),'Confidence'] = submission.loc[~submission.FVC1.isnull(),'Confidence1']

org_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(org_test)):
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]
    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70

submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)


# In[ ]:




