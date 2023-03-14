#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '\n<style type="text/css">\n     \n \ndiv.h2 {\n    background-color: #159957;\n    background-image: linear-gradient(120deg, #155799, #159957);\n    text-align: left;\n    color: white;              \n    padding:9px;\n    padding-right: 100px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 40px; \n}\n                                     \n                                      \nbody {\n  font-size: 11px;\n}    \n     \n                                    \n                                      \ndiv.h3 {\n    color: #159957; \n    font-size: 18px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\n   \n                                      \ndiv.h4 {\n    color: #159957;\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\n   \n                                      \nspan.note {\n    font-size: 7; \n    color: gray; \n    font-style: italic;\n}\n  \n                                      \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}   \n    \n                                      \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\n    \n                                      \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    text-align: center;\n} \n   \n            \n                                      \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;\n}\n       \n                                      \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \n                                       \n                                      \ntable.rules tr.best\n{\n    color: green;\n}    \n                             \n.output { \n    align-items: left; \n}\n        \n                                      \n.output_png {\n    display: table-cell;\n    text-align: left;\n    margin:auto;\n}                                          \n                                                                    \n                                      \n                                      \n</style> \n                                     \n                                      ')


# In[2]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reference: 
#      - I really liked the way JohnM's punt kaggle submission had the headers, extremely aesthetically pleasing
#        and aids viewing - borrowing his div.h header concept (so much nicer looking than using conventional
#        ## headers etc), and adding a 'cayman' color theme to it, as a nod to R ...  
#        Isn't it nice looking ?  ->  https://jasonlong.github.io/cayman-theme/
#      - I would strongly suggest we follow JohnM's push into professoinal looking css-based headers, we can't 
#        keep using old-fashioned markdown for headers, its so limited... just my personal opinion
#
# -%%HTML
# <style type="text/css">
#
# div.h2 {
#     background-color: steelblue; 
#     color: white; 
#     padding: 8px; 
#     padding-right: 300px; 
#     font-size: 20px; 
#     max-width: 1500px; 
#     margin: auto; 
#     margin-top: 50px;
# }
# etc
# etc
# --- end reference ---

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT ALL OF THIS OUT:
# abc
# def
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib. pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import sparklines
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from sklearn import preprocessing
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
# from sklearn.feature_selection import SelectFromModel
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import Video
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2 as cv  # or import cv2 as cv
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import json
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from tqdm import tqdm_notebook
from tqdm import tqdm
#import gc, pickle, tqdm, os, datetime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage.measure import compare_ssim
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# In[3]:


train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.head(20)


# In[4]:


pd.DataFrame(train_sample_metadata['label'].value_counts(normalize=True))


# In[5]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#train_sample_metadata.groupby('label')['label'].count()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
train_video_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.mp4')]
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#df_train.head()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#df_train.shape 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# In[6]:



# FREEZE-FRAME:
import cv2 as cv
import matplotlib.pyplot as plt
dp1 = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/drcyabprvt.mp4'
#dp2 = 'dzieklokdr.mp4'    
fig, ax = plt.subplots(1,1, 
                       figsize=(8,8))
# fake:  cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4') 
# cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dzieklokdr.mp4')
mycap = cv.VideoCapture(dp1); mycap.set(1,2)
ret, image = mycap.read()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
raw_image = image
#print(raw_image.shape)
mycap.release() 
cv.destroyAllWindows()
ax.set_xticks([]); ax.set_yticks([]); ax.imshow(image);


# In[7]:



fig, ax = plt.subplots(1,1, figsize=(8,8))
cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4') 
cap.set(1,2); ret, image = cap.read()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cap.release()   
cv.destroyAllWindows()
file_name = 'dkrvorliqc.mp4'
ax.title.set_text(file_name)
ax.imshow(image); 


# In[8]:


# overall notes, do not destroy:
#
# ![title](https://www.desipio.com/wp-content/uploads/2019/06/walter-payton-leap-2-ah.jpg)
# <br>&ensp; *Walter Payton (34) and the need for z-coordinate data ...*
#
#
#
#
#
#
#
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
#
#
#
#
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/box2.png" width="400px">
#
#
#
#
#
#  https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py
#
#
#
#
# import numpy as np
# import cv2

# cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# keep:
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


