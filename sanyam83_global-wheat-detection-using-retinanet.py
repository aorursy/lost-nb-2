#!/usr/bin/env python
# coding: utf-8

# In[1]:


#installing keras
get_ipython().system('git clone https://github.com/fizyr/keras-retinanet.git')
get_ipython().run_line_magic('cd', 'keras-retinanet/')
get_ipython().system('pip install .')
get_ipython().system('python setup.py build_ext --inplace')
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import urllib
from tqdm.notebook import tqdm
import os
from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from ast import literal_eval
import cv2
from PIL import Image, ImageDraw

#importing dataset
get_ipython().run_line_magic('cd', '..')
train = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')
train_img = '/kaggle/input/global-wheat-detection/train'
test_img = '/kaggle/input/global-wheat-detection/test'

#visualising images
def show_images(images, num = 5):
    
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_img, image_id + ".jpg")
        image = Image.open(image_path)

        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train[train['image_id'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:    
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

        plt.figure(figsize = (15,15))
        plt.imshow(image)
        plt.show()

unique_images = train['image_id'].unique()
show_images(unique_images)   

#preprocessing data
bboxs=[ bbox[1:-1].split(', ') for bbox in train['bbox']]
bboxs=[ f"{float(bbox[0])},{float(bbox[1])},{(float(bbox[0]))+(float(bbox[2]))},{(float(bbox[1])) + (float(bbox[3]))},wheat" for bbox in bboxs]
train['bbox_']=bboxs
train.head()

train_df=train[['image_id','bbox_']]
train_df.head()
train_df=train_df.sample(frac=1).reset_index(drop=True)
train_df.head()

#creating required csv files for training
with open("annotations.csv","w") as file:
    for idx in range(len(train_df)):
        file.write(train_img+"/"+train_df.iloc[idx,0]+".jpg"+","+train_df.iloc[idx,1]+"\n")
        
with open("classes.csv","w") as file:
    file.write("wheat,0")
    
get_ipython().system('head annotations.csv')

if not os.path.exists('snapshots'):
  os.mkdir('snapshots')

#loading pretrained model
PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)
print('Downloaded pretrained model to ' + PRETRAINED_MODEL)

#training model
get_ipython().system('keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone   --random-transform   --weights {PRETRAINED_MODEL}   --batch-size 8   --steps 500   --epochs 15   csv annotations.csv classes.csv')

get_ipython().system('ls snapshots')
model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)
li=os.listdir(test_img)
li[:5]

def predict(image):
    image = preprocess_image(image.copy())
    #image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(image, axis=0)
  )

    #boxes /= scale

    return boxes, scores, labels

def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)
        

preds=[]
imgid=[]
for img in tqdm(li,total=len(li)):
    img_path = test_img+'/'+img
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict(image)
    boxes=boxes[0]
    scores=scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx]>THRES_SCORE:
            box,score=boxes[idx],scores[idx]
            imgid.append(img.split(".")[0])
            preds.append("{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))

sub={"image_id":imgid, "PredictionString":preds}
sub=pd.DataFrame(sub)
sub_=sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

for idx,imgid in enumerate(samsub['image_id']):
    samsub.iloc[idx,1]=sub_[sub_['image_id']==imgid].values[0,1]

samsub.to_csv('/kaggle/working/submission.csv',index=False)

