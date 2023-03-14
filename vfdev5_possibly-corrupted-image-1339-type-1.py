#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob

import numpy as np
from PIL import Image

TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])

TEST_DATA = "../input/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])
print(len(test_ids))
print(test_ids[:10])

ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])

print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_2_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def _get_image_data_pil(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    try:
        img = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)
    return np.asarray(img)


# In[2]:


#type_ids = (type_1_ids, type_2_ids, type_3_ids, test_ids)
#image_types = ["Type_1", "Type_2", "Type_3", "Test"]

#corrupted_image_id_type_list = []
#for ids, image_type in zip(type_ids, image_types):
#    for image_id in ids:
#        img = _get_image_data_pil(image_id, image_type)
#        if img.dtype.kind is not 'u':
#            corrupted_image_id_type_list.append((image_id, image_type))
# print(len(corrupted_image_id_type_list), corrupted_image_id_type_list[:10])


# In[3]:


corrupted_image_id_type_list = [('1339', 'Type_1')]


# In[4]:


import cv2
import matplotlib.pyplot as plt

def _get_image_data_opencv(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[5]:


n = 5
for i in range(len(corrupted_image_id_type_list)):
    image_id, image_type = corrupted_image_id_type_list[i]
    img = _get_image_data_opencv(image_id, image_type)
    if i % n == 0:
        plt.figure(figsize=(20,10))
        plt.suptitle("Possibly corrupted images")
    plt.subplot(1,n,i % n + 1)
    plt.imshow(img)
    plt.title("%s / %s" % (image_id, image_type))   


# In[6]:




