#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io
import os
import glob
import numpy as np
import random
import pandas as pd
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import shutil
import git
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


if not os.path.exists("wheat_yolo_train_data"):
    os.makedirs("wheat_yolo_train_data")


# In[3]:


for filename in glob.glob(os.path.join('../input/global-wheat-detection/train', '*.*')):
    shutil.copy(filename, 'wheat_yolo_train_data')


# In[4]:


get_ipython().system('ls wheat_yolo_train_data/')


# In[5]:


os.getcwd()


# In[6]:


# path to your dataset
DATASET_PATH = './global-wheat-detection'
wheat_data = ['train', 'test']


# In[7]:


full_path_to_dataset = os.path.join(DATASET_PATH, wheat_data[0], '*')


# In[8]:


full_path_to_dataset = os.path.dirname(os.path.abspath(os.path.join(DATASET_PATH, wheat_data[0],'*')))


# In[9]:


print(full_path_to_dataset)


# In[10]:


#List of categories
#Only one class i.e. wheat
wheat = [0]


# In[11]:


wheat_df = pd.read_csv(os.path.join('../input/global-wheat-detection/train.csv'))


# In[12]:


wheat_df.head()


# In[13]:


wheat_df['bbox'] = wheat_df['bbox'].apply(lambda x: x[1:-1].split(","))


# In[14]:


wheat_df[['x_min','y_min','box_width','box_height']] = pd.DataFrame(wheat_df.bbox.tolist(), index= wheat_df.index)


# In[15]:


wheat_df.drop(['bbox'], axis=1, inplace=True)


# In[16]:


wheat_df.head()


# In[17]:


wheat_df['x_min'] = wheat_df['x_min'].astype(float)
wheat_df['y_min'] = wheat_df['y_min'].astype(float)
wheat_df['box_width'] = wheat_df['box_width'].astype(float)
wheat_df['box_height'] = wheat_df['box_height'].astype(float)


# In[18]:


wheat_df['x_max'] = wheat_df['x_min'] + wheat_df['box_width']
wheat_df['y_max'] = wheat_df['y_min'] + wheat_df['box_height']


# In[19]:


wheat_df = wheat_df[['image_id', 'width', 'height', 'x_min','y_min', 'x_max', 'y_max', 'box_width', 'box_height','source']]


# In[20]:


wheat_df["image_id"] = wheat_df["image_id"].apply(lambda x: str(x) + ".jpg")
wheat_df.head()


# In[21]:


#drop the columns that are not required
wheat_df.drop(['width', 'height', 'box_width', 'box_height', 'source'], axis=1, inplace=True)


# In[22]:


wheat_df.head()


# In[23]:


#Adding new empty columns to DataFrame to save numbers for YOLO formats
wheat_df['CategoryID'] = ''
wheat_df['center x'] = ''
wheat_df['center y'] = ''
wheat_df['width'] = ''
wheat_df['height'] = ''


# In[24]:


#Getting category's ID according to the class's ID
wheat_df['CategoryID'] = 0


# In[25]:


wheat_df.head()


# In[26]:


# Calculating bounding box's center in x and y for all rows
# Saving results to appropriate columns
wheat_df['center x'] = (wheat_df['x_max'] + wheat_df['x_min']) / 2
wheat_df['center y'] = (wheat_df['y_max'] + wheat_df['y_min']) / 2


# In[27]:


# Calculating bounding box's width and height for all rows
# Saving results to appropriate columns
wheat_df['width'] = wheat_df['x_max'] - wheat_df['x_min']
wheat_df['height'] = wheat_df['y_max'] - wheat_df['y_min']


# In[28]:


wheat_img_data = wheat_df.loc[:, ['image_id',
                'CategoryID',
                'center x',
                'center y',
                'width',
                'height']].copy()


# In[29]:


wheat_img_data.rename(columns={"image_id": "ImageID"},inplace=True)


# In[30]:


wheat_img_data.head()


# In[31]:


os.chdir('wheat_yolo_train_data')


# In[32]:


print(os.getcwd())


# In[33]:


for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.ppm'
        if f.endswith('.jpg'):
            # Reading image and getting its real width and height
            image_jpg = cv2.imread(f)

            # Slicing from tuple only first two elements
            h, w = image_jpg.shape[:2]

            # Slicing only name of the file without extension
            image_name = f[:-4]

            
            sub_wheat_img = wheat_img_data.loc[wheat_img_data['ImageID'] == f].copy()

            # Normalizing calculated bounding boxes' coordinates
            # according to the real image width and height
            sub_wheat_img['center x'] = sub_wheat_img['center x'] / w
            sub_wheat_img['center y'] = sub_wheat_img['center y'] / h
            sub_wheat_img['width'] = sub_wheat_img['width'] / w
            sub_wheat_img['height'] = sub_wheat_img['height'] / h

            resulted_frame = sub_wheat_img.loc[:, ['CategoryID',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            # Checking if there is no any annotations for current image
            if resulted_frame.isnull().values.all():
                # Skipping this image
                continue

            
            path_to_save = image_name + '.txt'

            # Saving resulted Pandas dataFrame into txt file
            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

           
            path_to_save = image_name + '.jpg'

            # Saving image in jpg format by OpenCV function
            # that uses extension to choose format to save with
            cv2.imwrite(path_to_save, image_jpg)


# In[34]:


len(glob.glob('*'))


# In[35]:


# Defining list to write paths in
p = []


for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.jpg'
        if f.endswith('.jpg'):
            
            path_to_save_into_txt_files = f

            p.append(path_to_save_into_txt_files + '\n')


# Slicing first 15% of elements from the list
# to write into the test.txt file
p_test = p[:int(len(p) * 0.15)]

# Deleting from initial list first 15% of elements
p = p[int(len(p) * 0.15):]


# In[36]:


# Creating file train.txt and writing 85% of lines in it
with open('train.txt', 'w') as train_txt:
    # Going through all elements of the list
    for e in p:
        # Writing current path at the end of the file
        train_txt.write(e)

# Creating file test.txt and writing 15% of lines in it
with open('test.txt', 'w') as test_txt:
    # Going through all elements of the list
    for e in p_test:
        # Writing current path at the end of the file
        test_txt.write(e)


# In[37]:


with open("classes.txt", "w") as file:
    file.write("wheat")


# In[38]:


# Defining counter for classes
c = 0

with open('classes.names', 'w') as names,      open('classes.txt', 'r') as txt:

    # Going through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)  # Copying all info from file txt to names

        # Increasing counter
        c += 1


# In[39]:


with open('wheat_data.data', 'w') as data:
    # Writing needed 5 lines
    # Number of classes
    # By using '\n' we move to the next line
    data.write('classes = ' + str(c) + '\n')

    # Location of the train.txt file
    data.write('train = ' + full_path_to_dataset + '/' + 'train.txt' + '\n')

    # Location of the test.txt file
    data.write('valid = ' + full_path_to_dataset + '/' + 'test.txt' + '\n')

    # Location of the classes.names file
    data.write('names = ' + full_path_to_dataset + '/' + 'classes.names' + '\n')

    # Location where to save weights
    data.write('backup = backup')


# In[40]:


with open('train.txt','r') as fp:
    Lines = fp.readlines() 
    for line in Lines: 
        print("{}".format(line.strip())) 


# In[41]:


os.chdir("..")


# In[42]:


os.getcwd()


# In[43]:


get_ipython().system('git clone https://github.com/AlexeyAB/darknet.git')


# In[44]:


get_ipython().system('cd darknet/')


# In[ ]:




