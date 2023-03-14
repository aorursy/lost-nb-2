#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')
get_ipython().system('mv yolov5/* ./')


# In[36]:


get_ipython().system('python -m pip install --upgrade pip')
get_ipython().system('pip install -r requirements.txt')


# In[38]:


import numpy as np 
import pandas as pd 
import os
from tqdm.auto import tqdm
import shutil as sh


# In[39]:


df = pd.read_csv('../input/global-wheat-detection/train.csv')
bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    df[column] = bboxs[:,i]
df.drop(columns=['bbox'], inplace=True)
df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2
df['classes'] = 0

df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
df.head()


# In[42]:


index = list(set(df.image_id))
source = 'train'
if True:
    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                row = row/1024
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy("../input/global-wheat-detection/{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))


# In[44]:


get_ipython().system('python train.py --img 1024 --batch 2 --epochs 10                  --data ../input/wheat-detection-yolov5-utils/wheat0.yaml                  --cfg models/yolov5x.yaml                  --weights yolov5x.pt')


# In[50]:


# copy saved model to weights folder
get_ipython().system('cp runs/exp4/weights/best.pt weights')


# In[46]:


# remove convertor of training data
get_ipython().system('rm -rf convertor')


# In[51]:


# Detect test images
get_ipython().system("python detect.py --source '../input/global-wheat-detection/test/' --weight weights/best.pt --output 'inference/output' ")


# In[52]:


get_ipython().system('ls -l inference/output')


# In[53]:


from IPython.display import Image, clear_output  # to display images
Image(filename='inference/output/2fd875eaa.jpg', width=600)


# In[54]:


Image(filename='inference/output/348a992bb.jpg', width=600)

