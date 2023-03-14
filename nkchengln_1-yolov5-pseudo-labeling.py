#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh


# In[2]:


fold = 1
index = pd.read_csv("../input/../input/wheat-image-id/image_id.csv")["image_id"]
index = list(index)
index[:5]


# In[3]:


get_ipython().system('cp -r ../input/yolov5train/* .')


# In[4]:


get_ipython().system('ls')


# In[5]:


def convertTrainLabel(fold):
    df = pd.read_csv('../input/global-wheat-detection/train.csv')
    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]
    df.drop(columns=['bbox'], inplace=True)
    df['x_center'] = df['x'] + df['w']/2
    df['y_center'] = df['y'] + df['h']/2
    df['classes'] = 0
    from tqdm.auto import tqdm
    import shutil as sh
    df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
    
#     index = list(set(df.image_id))
    
    source = 'train'
    if True:
#         for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val/'
            else:
                path2save = 'train/'
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


# In[6]:


convertTrainLabel(fold)


# In[7]:


get_ipython().system('ls')


# In[8]:


get_ipython().system('python train.py --img 1024 --batch 2 --epochs 1 --data ../input/yolov5-for-wheat/wheat_fold1.yaml --cfg ../input/yolov5-for-wheat/yolov5s.yaml --weights ../input/yolov5-for-wheat/yolov5s.pt --name yolov5s_fold1')


# In[9]:


get_ipython().system('rm -rf convertor')

