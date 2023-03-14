#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\ndiv.h2 {\n    background-color: steelblue; \n    color: white; \n    padding: 8px; \n    padding-right: 300px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 50px;\n}\ndiv.h3 {\n    color: steelblue; \n    font-size: 14px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\ndiv.h4 {\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\nspan.captiona {\n    font-size: 5; \n    color: dimgray; \n    font-style: italic;\n    margin-left: 130px;\n    vertical-align: top;\n}\nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n    text-align: center;\n} \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n}\ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \ntable.rules tr.best\n{\n    color: green;\n}\n\n</style>')


# In[2]:



import os
import glob

import numpy as np
import scipy as sp
import pandas as pd

# skimage
from skimage.io import imshow, imread, imsave
from skimage.transform import rotate, AffineTransform, warp,rescale, resize, downscale_local_mean
from skimage import color,data
from skimage.exposure import adjust_gamma
from skimage.util import random_noise

#OpenCV-Python
import cv2

# imgaug
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

# Albumentations
import albumentations as A

# Augmentor
get_ipython().system('pip install augmentor')
import Augmentor 

# Keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 

# SOLT
get_ipython().system('pip install solt')
import solt
import solt.transforms as slt

#visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import HTML, Image

#source: https://www.kaggle.com/jpmiller/nfl-punt-analytics/edit
# set additional display options for report
pd.set_option("display.max_columns", 100)
th_props = [('font-size', '13px'), ('background-color', 'white'), 
            ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", 
            props=th_props)]

#warnings
import warnings
warnings.filterwarnings("ignore")


#Helper function to display the images in a grid
# Source: https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy which was pointed by
# this excellent article: https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9
def gallery(array, ncols=3):
    '''
    Function to arange images into a grid.
    INPUT:
        array - numpy array containing images
        ncols - number of columns in resulting imahe grid
    OUTPUT:
        result - reshaped array into a grid with given number of columns
    '''
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


# In[3]:


# Defining data path
Image_Data_Path = "../input/plant-pathology-2020-fgvc7/images/"

train_data = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test_data = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

# Loading the training images #refer: https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models
def load_image(image_id):
    file_path = image_id + ".jpg"
    image = imread(Image_Data_Path + file_path)
    return image

train_images = train_data["image_id"][:50].apply(load_image)


# In[4]:



# image titles
#image_titles = ['Frame1', 'Frame2', 'Frame3']

# plotting multiple images using subplots
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(30,16))
for col in range(3):
    for row in range(2):
        ax[row,col].imshow(train_images.loc[train_images.index[row*3+col]])
        #ax[row,col].set_title(image_titles[i])    
        ax[row,col].set_xticks([])
        ax[row,col].set_yticks([])


# In[5]:


image = train_images[15]
imshow(image)
print(image.shape)


# In[6]:


# red filter [R,G,B]
red_filter = [1,0,0]
# blue filter
blue_filter = [0,0,1]
# green filter
green_filter = [0,1,0]


# matplotlib code to display
fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,16))
ax[0].imshow(image*red_filter)
ax[0].set_title("Red Filter",fontweight="bold", size=30)
ax[1].imshow(image*blue_filter)
ax[1].set_title("BLue Filter",fontweight="bold", size=30)
ax[2].imshow(image*green_filter)
ax[2].set_title("Green Filter",fontweight="bold", size=30);


# In[7]:


# import color sub-module
from skimage import color

# converting image to grayscale
grayscale_image = color.rgb2gray(image)
grayscale_image.shape
imshow(grayscale_image)


# In[8]:


#Horizontally flipped
hflipped_image= np.fliplr(image) #fliplr reverse the order of columns of pixels in matrix

#Vertically flipped
vflipped_image= np.flipud(image) #flipud reverse the order of rows of pixels in matrix

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=30)
ax[1].imshow(hflipped_image)
ax[1].set_title("Horizontally flipped", size=30)
ax[2].imshow(vflipped_image)
ax[2].set_title("Vertically flipped", size=30);


# In[9]:


# clockwise rotation
rot_clockwise_image = rotate(image, angle=45) 
# Anticlockwise rotation
rot_anticlockwise_image = rotate(image, angle=-45)


# In[10]:



fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=30)
ax[1].imshow(rot_clockwise_image)
ax[1].set_title("+45 degree Rotation", size=30)
ax[2].imshow(rot_anticlockwise_image)
ax[2].set_title("-45 degree rotation", size=30);


# In[11]:


# source: https://www.kaggle.com/safavieh/image-augmentation-using-skimage
import random
import pylab as pl 
def randRange(a, b):
    '''
    a utility function to generate random float values in desired range
    '''
    return pl.rand() * (b - a) + a
def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1/3.5
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1-margin), im.shape[0])), 
           int(randRange(im.shape[1] * (1-margin), im.shape[1]))]
    cropped_image = (im[start[0]:end[0], start[1]:end[1]])
    return cropped_image


# In[12]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,12))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(randomCrop(image))
ax[1].set_title("Cropped", size=20)


# In[13]:



image_bright = adjust_gamma(image, gamma=0.5,gain=1)
image_dark = adjust_gamma(image, gamma=2,gain=1)


# In[14]:


fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,12))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(image_bright)
ax[1].set_title("Brightened Image", size=20)
ax[2].imshow(image_dark)
ax[2].set_title("Darkened Image", size=20)


# In[15]:



image_resized = resize(image, (image.shape[0] // 2, image.shape[1] // 2),
                       anti_aliasing=True)
#image_downscaled = downscale_local_mean(image, (4, 3))


# In[16]:


fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(image_resized)
ax[1].set_title("Resized image",size=20)


# In[17]:


noisy_image= random_noise(image)


# In[18]:



fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(noisy_image)
ax[1].set_title("Image after adding noise",size=20)


# In[19]:


# selecting a sample image
image13 = train_images[13]
imshow(image13)
print(image13.shape)
plt.axis('off')


# In[20]:


#vertical flip
img_flip_ud = cv2.flip(image13, 0)
plt.imshow(img_flip_ud)

#horizontal flip
img_flip_lr = cv2.flip(image13, 1)
plt.imshow(img_flip_lr)


# In[21]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,12))
ax[0].imshow(img_flip_ud)
ax[0].set_title("vertical flip", size=20)
ax[1].imshow(img_flip_lr)
ax[1].set_title("horizontal flip", size=20)


# In[22]:


img_rotate_90_clockwise = cv2.rotate(image13, cv2.ROTATE_90_CLOCKWISE)
img_rotate_90_counterclockwise = cv2.rotate(image13, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_rotate_180 = cv2.rotate(image13, cv2.ROTATE_180)


# In[23]:


fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,12))
ax[0].imshow(img_rotate_90_clockwise)
ax[0].set_title("90 degrees clockwise", size=20)
ax[1].imshow(img_rotate_90_counterclockwise)
ax[1].set_title("90 degrees anticlockwise", size=20)
ax[2].imshow(img_rotate_180)
ax[2].set_title("180 degree rotation", size=20)


# In[24]:


#RESIZE
def resize_image(image,w,h):
    resized_image = image=cv2.resize(image,(w,h))
    return resized_image

imshow(resize_image(image13, 500,500))


# In[25]:


def add_light(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    return image

imshow(add_light(image13,2))


# In[26]:


#crop
def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    return image
imshow(crop_image(image13,200,800,250,1500))#(y1,y2,x1,x2)(bottom,top,left,right)


# In[27]:


def gaussian_blur(image,blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    return image

imshow(gaussian_blur(image13,0))


# In[28]:


# selecting a sample image
image2 = train_images[25]
imshow(image2)
print(image2.shape)
plt.axis('off')


# In[29]:



#Horizontally flipped
hflip= iaa.Fliplr(p=1.0)
hflipped_image2= hflip.augment_image(image2)

#Vertically flipped
vflip= iaa.Flipud(p=1.0) 
vflipped_image2= vflip.augment_image(image2)


# In[30]:


image=image2
fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=30)
ax[1].imshow(hflipped_image2)
ax[1].set_title("Horizontally flipped", size=30)
ax[2].imshow(vflipped_image2)
ax[2].set_title("Vertically flipped", size=30);


# In[31]:


# clockwise rotation
rot = iaa.Affine(rotate=(-25,25))
rot_clockwise_image2 = rot.augment_image(image2)


# In[32]:


image=image2
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(30,16))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=30)
ax[1].imshow(rot_clockwise_image2)
ax[1].set_title("Rotated Image", size=30)


# In[33]:



image=image2
crop = iaa.Crop(percent=(0, 0.2)) # crop image
corp_image=crop.augment_image(image)


# In[34]:



fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,12))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(randomCrop(corp_image))
ax[1].set_title("Cropped", size=20)


# In[35]:


# 4. Brightness

image = image2
# bright
#contrast1=iaa.GammaContrast(gamma=0.5)
#brightened_image = contrast1.augment_image(image)

#dark
#contrast2=iaa.GammaContrast(gamma=2)
#darkened_image = contrast2.augment_image(image)

#Somehow, this line of code gave me an error in the Kaggle notebook but worked fine in other IDEs. 
#So I have inserted the desired result.ULet me know if the above code works for you?


# In[36]:



#fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,12))
#ax[0].imshow(image)
#ax[0].set_title("Original Image", size=20)
#ax[1].imshow(brightened_image)
#ax[1].set_title("Brightened Image", size=20)
#ax[2].imshow(darkened_image)
#ax[2].set_title("darkened_image", size=20)


# In[37]:


image= image2
scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (0.5, 1.0)})
scale_image =scale_im.augment_image(image)


# In[38]:



fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,12))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(scale_image)
ax[1].set_title("Scaled", size=20)


# In[39]:


image= image2
gaussian_noise=iaa.AdditiveGaussianNoise(15,20)
noise_image=gaussian_noise.augment_image(image)


# In[40]:



fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,12))
ax[0].imshow(image)
ax[0].set_title("Original Image", size=20)
ax[1].imshow(noise_image)
ax[1].set_title("Gaussian Noise added", size=20)


# In[41]:


# Defining a pipeline.
# The example has been taken from the documentation
aug_pipeline = iaa.Sequential([
    iaa.SomeOf((0,3),[
        iaa.Fliplr(1.0), # horizontally flip
        iaa.Flipud(1.0),# Vertical flip
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        iaa.Crop(percent=(0, 0.4)),
        iaa.Sometimes(0.5, iaa.Affine(rotate=5)),
        iaa.Sometimes( 0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    ])
], 
random_order=True # apply the augmentations in random order
)

# apply augmentation pipeline to sample image
images_aug = np.array([aug_pipeline.augment_image(image2) for _ in range(16)])

# visualize the augmented images
plt.figure(figsize=(30,10))
plt.axis('off')
plt.imshow(gallery(images_aug, ncols = 4))
plt.title('Augmentation  examples')


# In[42]:


# initialize augmentations
horizontal_flip = A.HorizontalFlip(p=1)
rotate = A.ShiftScaleRotate(p=1)
gaus_noise = A.GaussNoise() # gaussian noise
bright_contrast = A.RandomBrightnessContrast(p=1) # random brightness and contrast
gamma = A.RandomGamma(p=1) # random gamma
blur = A.Blur()

# apply augmentations to images
img_flip = horizontal_flip(image = image2)
img_gaus = gaus_noise(image = image2)
img_rotate = rotate(image = image2)
img_bc = bright_contrast(image = image2)
img_gamma = gamma(image = image2)
img_blur = blur(image = image2)

# access the augmented image by 'image' key
img_list = [img_flip['image'],img_gaus['image'], img_rotate['image'], img_bc['image'], img_gamma['image'], img_blur['image']]

# visualize the augmented images
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(gallery(np.array(img_list), ncols = 3))
plt.title('Augmentation examples')


# In[43]:


# Passing the path of the image directory 
p = Augmentor.Pipeline(source_directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
                      output_directory="/kaggle/output")
  
# Defining augmentation parameters and generating 10 samples 
p.flip_left_right(probability=0.4) 
p.flip_top_bottom(probability=0.8)
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=10)
p.skew(0.4, 0.5) 
p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
p.sample(10)


# In[44]:


# selecting a sample image
image5 = train_images[15]
imshow(image5)
print(image5.shape)
plt.axis('off')


# In[45]:


# To delete any previously created directory
#import shutil
#shutil.rmtree("../output/keras_augmentations")


# Creating a new directory for placing augmented images
import os
os.mkdir("../output/keras_augmentations")


# In[46]:


# Augmentation process
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 

img_arr = img_to_array(image5)
img_arr = img_arr.reshape((1,) + img_arr.shape)

i = 0
for batch in datagen.flow(
    img_arr,
    batch_size=1,
    save_to_dir='../output/keras_augmentations',
    save_prefix='Augmented_image',
    save_format='jpeg'):
    i += 1
    if i > 20: # create 20 augmented images
        break  # otherwise the generator would loop indefinitely


# In[47]:


images = os.listdir("../output/keras_augmentations/")
images


# In[48]:


# Let's look at the augmented images
aug_images = []
for img_path in glob.glob("../output/keras_augmentations/*.jpeg"):
    aug_images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(aug_images):
    plt.subplot(len(aug_images) / columns + 1, columns, i + 1)
    plt.imshow(image)


# In[49]:


# selecting a sample image
image5 = train_images[25]
imshow(image5)
print(image5.shape)
plt.axis('off')


# In[50]:


h,w,c = image5.shape
img = image5[:w]


# In[51]:


stream = solt.Stream([
    slt.Rotate(angle_range=(-90, 90), p=1, padding='r'),
    slt.Flip(axis=1, p=0.5),
    slt.Flip(axis=0, p=0.5),
    slt.Shear(range_x=0.3, range_y=0.8, p=0.5, padding='r'),
    slt.Scale(range_x=(0.8, 1.3), padding='r', range_y=(0.8, 1.3), same=False, p=0.5),
    slt.Pad((w, h), 'r'),
    slt.Crop((w, w), 'r'),
    slt.CvtColor('rgb2gs', keep_dim=True, p=0.2),
    slt.HSV((0, 10), (0, 10), (0, 10)),
    slt.Blur(k_size=7, blur_type='m'),
    solt.SelectiveStream([
        slt.CutOut(40, p=1),
        slt.CutOut(50, p=1),
        slt.CutOut(10, p=1),
        solt.Stream(),
        solt.Stream(),
    ], n=3),
], ignore_fast_mode=True)


# In[52]:


fig = plt.figure(figsize=(16,16))
n_augs = 6


random.seed(42)
for i in range(n_augs):
    img_aug = stream({'image': img}, return_torch=False, ).data[0].squeeze()

    ax = fig.add_subplot(1,n_augs,i+1)
    if i == 0:
        ax.imshow(img)
    else:
        ax.imshow(img_aug)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


# In[ ]:




