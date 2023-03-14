#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('html', '', '<span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at \'In [25]\'.</span>')


# In[1]:


import numpy as np 
import pandas as pd
import os
import sys
import json
import cv2
import time
import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# In[2]:


df = pd.read_csv('../input/global-wheat-detection/train.csv')
df.head()


# In[3]:


df['bbox'] = df['bbox'].apply(lambda x: x[1:-1].split(","))
df['x'] = df['bbox'].apply(lambda x: x[0]).astype('float32')
df['y'] = df['bbox'].apply(lambda x: x[1]).astype('float32')
df['w'] = df['bbox'].apply(lambda x: x[2]).astype('float32')
df['h'] = df['bbox'].apply(lambda x: x[3]).astype('float32')
df = df[['image_id','x', 'y', 'w', 'h']]

df.head()


# In[4]:


image_ids = df['image_id'].unique()
image_dict = dict(zip(image_ids, range(len(image_ids))))
len(image_dict)


# In[5]:


json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
json_dict


# In[6]:


for image_id in image_ids:
    image = {'file_name': image_id + '.jpg', 
             'height': 1024, 
             'width': 1024, 
             'id': image_dict[image_id]}
    json_dict['images'].append(image)
    
# json_dict


# In[7]:


categories = {'supercategory': 'wh', 'id': 1, 'name': 'wh'}
json_dict['categories'].append(categories)


# In[8]:


[i for i in os.listdir('../input/global-wheat-detection/train') if 'jpg' not in i]


# In[9]:


df.shape


# In[10]:


c = 0

for idx, box_id in df.iterrows(): 
    image_id = image_dict[box_id['image_id']]
    
    ann = {'area': box_id['w'] * box_id['h'], 
           'iscrowd': 0, 
           'image_id': image_id,                        
           'bbox': [box_id['x'], box_id['y'], box_id['w'], box_id['h']],
           'category_id': 1, 
           'id': idx,
           'segmentation': []}
    c += 1

    json_dict['annotations'].append(ann)


# In[11]:


c


# In[12]:


# json_dict


# In[13]:


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# In[14]:


annFile='instances_Images.json'

json_fp = open(annFile, 'w',encoding='utf-8')
json_str = json.dumps(json_dict,cls=NpEncoder)
json_fp.write(json_str)
json_fp.close()


# In[15]:


get_ipython().system('git clone https://github.com/kamauz/EfficientDet.git')


# In[16]:


os.listdir('../working/EfficientDet')


# In[17]:


cd ../working/EfficientDet


# In[18]:


get_ipython().system('python setup.py build_ext --inplace')


# In[19]:


from model import efficientdet
from losses import smooth_l1, focal
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from generators.common import Generator


# In[20]:


get_ipython().system("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' -q")

from pycocotools.coco import COCO


# In[21]:


def preprocess_image(image):
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std

    return image


# In[22]:


def postprocess_boxes(boxes, height, width):
    c_boxes = boxes.copy()
    c_boxes[:, 0] = np.clip(c_boxes[:, 0], 0, width - 1)
    c_boxes[:, 1] = np.clip(c_boxes[:, 1], 0, height - 1)
    c_boxes[:, 2] = np.clip(c_boxes[:, 2], 0, width - 1)
    c_boxes[:, 3] = np.clip(c_boxes[:, 3], 0, height - 1)
    return c_boxes


# In[23]:


os.listdir('/kaggle/working')


# In[24]:


class CocoGenerator(Generator):
    def __init__(self, data_dir, set_name, **kwargs):                                    
        self.coco = COCO('/kaggle/working/instances_Images.json')                
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

        super(CocoGenerator, self).__init__(**kwargs)

    def load_classes(self): 
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        return len(self.image_ids)

    def num_classes(self):
        return 1

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        print(classes[name])
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):        
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]        
        path = os.path.join('/kaggle/input/global-wheat-detection/train/', image_info['file_name'])        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = preprocess_image(image)
        
        return image

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [a['category_id'] - 1]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)           

        return annotations    
    # (1, 4, 1024, 1024, 3), (1, 4, 196416, 2)


# In[25]:


CocoGenerator(data_dir=None, set_name=None, batch_size = 4, phi = phi).name_to_label


# In[26]:


phi = 4
score_threshold=0.4


# In[27]:


train_generator = CocoGenerator(data_dir=None, set_name=None, batch_size = 4, phi = phi)


# In[28]:


df.shape


# In[29]:


196416/4


# In[30]:


# for first_idx, second_idx in train_generator:
#     print(np.array(first_idx))
# #     print(np.array(first_idx).shape)
# #     print(second_idx[0].shape)
#     raise RuntimeError
    
# # first_idx : (1, 4, 1024, 1024, 3) --> (Batch, xywh, img_size, img_size, channel)
# # second_idx : (1, 4, 196416, 2)


# In[31]:


model, prediction_model = efficientdet(phi,
                                       num_classes=1,
                                       weighted_bifpn=True,
                                       freeze_bn=True,
                                       score_threshold=score_threshold
                                       )


# In[32]:


model_name = 'efficientnet-b{}'.format(phi)
file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
file_hash = WEIGHTS_HASHES[model_name][1]
weights_path = tf.keras.utils.get_file(file_name,
                                    BASE_WEIGHTS_PATH + file_name,
                                    cache_subdir='models',
                                    file_hash=file_hash)
model.load_weights(weights_path, by_name=True)


# In[33]:


model.summary()


# In[34]:


for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
    model.layers[i].trainable = False


# In[35]:


model.compile(optimizer=Adam(lr=1e-3), loss={
    'regression': smooth_l1(),
    'classification': focal()
}, )


# In[36]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit_generator(\n        generator=train_generator,\n        epochs=1\n    )')


# In[37]:


cd /kaggle/working/


# In[38]:


model.save('model.h5')


# In[39]:


prediction_model.load_weights('/kaggle/working/model.h5', by_name=True)


# In[40]:


score_threshold = 0.7
result_data = []
for image_path in glob.glob('/kaggle/input/global-wheat-detection/test/53f253011.jpg'):
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    src_image = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]
    h, w = image.shape[:2]

    image = preprocess_image(image)               
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)    
    boxes = postprocess_boxes(boxes=boxes, height=h, width=w)

    indices = np.where(scores[:] > score_threshold)[0]
    boxes = boxes[indices]    
    row = [image_name.replace('.jpg','')]
    r_boxes = ""
    for s,b in zip(scores, boxes):
        if r_boxes != "":
            r_boxes += " "
        r_boxes += f"{round(float(s),2)} {int(b[0])} {int(b[1])} {int(b[2]-b[0])} {int(b[3]-b[1])}"
    
    row.append(r_boxes)
    result_data.append(row)
test_df = pd.DataFrame(result_data, columns=['image_id','PredictionString'])
test_df.head()


# In[41]:


from matplotlib import pyplot as plt

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

name = '53f253011'        
        
test_df['PredictionString'] = test_df['PredictionString'].apply(lambda a: a.split(' ')).apply(lambda myList: [x for i, x in enumerate(myList) if i%5 !=0])
lst1 = test_df[test_df['image_id'] == name]['PredictionString'].values[0]
lst1 = list(map(int, lst1))     
lst1_n = list(chunks(lst1, 4))

sample = plt.imread('/kaggle/input/global-wheat-detection/test/' + name + '.jpg')

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for box in lst1_n:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2] + box[0], box[3] + box[1]),
                  (0, 0, 100), 2)
  
ax.set_axis_off()
ax.imshow(sample)


# In[ ]:




