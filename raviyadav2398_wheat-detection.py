#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import re
from tqdm.notebook import tqdm
from PIL import Image,ImageDraw
import hashlib
import ast
from ast import literal_eval
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import seaborn as sns
sns.set()

DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN_IMAGES = f'{DIR_INPUT}/train'
DIR_TEST_IMAGES = f'{DIR_INPUT}/test'


# In[2]:


train= pd.read_csv(f'{DIR_INPUT}/train.csv')
train.shape


# In[3]:


sub=pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')
sub.shape


# In[4]:


train.head()


# In[5]:


train['image_id'].nunique()


# In[6]:


train['height'].value_counts()


# In[7]:


train['width'].value_counts()


# In[8]:


train['image_id'].value_counts()


# In[9]:


train['width'].unique() == train['height'].unique() == [1024]


# In[10]:


def get_bbox_area(bbox):
    bbox = literal_eval(bbox)
    return bbox[2]*bbox[3]


# In[11]:


train['bbox_area']=train['bbox'].apply(get_bbox_area)


# In[12]:


train['bbox_area'].value_counts().hist(bins=10)


# In[13]:


unique_images = train['image_id'].unique()
len(unique_images)


# In[14]:


def calculate_hash(im):
    md5 = hashlib.md5()
    md5.update(np.array(im).tostring())
    
    return md5.hexdigest()
    
def get_image_meta(image_id, image_src, dataset='train'):
    im = Image.open(image_src)
    extrema = im.getextrema()

    meta = {
        'image_id': image_id,
        'dataset': dataset,
        'hash': calculate_hash(im),
        'r_min': extrema[0][0],
        'r_max': extrema[0][1],
        'g_min': extrema[1][0],
        'g_max': extrema[1][1],
        'b_min': extrema[2][0],
        'b_max': extrema[2][1],
        'height': im.size[0],
        'width': im.size[1],
        'format': im.format,
        'mode': im.mode
    }
    return meta
data = []


# In[15]:



for i, image_id in enumerate(tqdm(train['image_id'].unique(), total=train['image_id'].unique().shape[0])):
    data.append(get_image_meta(image_id, DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_id)))


# In[16]:


meta_df = pd.DataFrame(data)
meta_df.head()


# In[17]:


duplicates = meta_df.groupby(by='hash')[['image_id']].count().reset_index()
duplicates = duplicates[duplicates['image_id'] > 1]
duplicates.reset_index(drop=True, inplace=True)

duplicates = duplicates.merge(meta_df[['image_id', 'hash']], on='hash')

duplicates.head(20)


# In[18]:


train['x'] = -1
train['y'] = -1
train['w'] = -1
train['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train[['x', 'y', 'w', 'h']] = np.stack(train['bbox'].apply(lambda x: expand_bbox(x)))
train.drop(columns=['bbox'], inplace=True)
train['x'] = train['x'].astype(np.float)
train['y'] = train['y'].astype(np.float)
train['w'] = train['w'].astype(np.float)
train['h'] = train['h'].astype(np.float)


# In[19]:


train


# In[20]:


train.groupby(by='image_id')['source'].count().agg(['min', 'max', 'mean'])


# In[21]:


source = train['source'].value_counts()
source


# In[22]:


plt.hist(train['image_id'].value_counts(), bins=10)
plt.show()


# In[23]:


fig = go.Figure(data=[
    go.Pie(labels=source.index, values=source.values)
])

fig.update_layout(title='Source distribution')
fig.show()


# In[24]:


def show_images(image_ids):
    
    col = 5
    row = min(len(image_ids) // col, 5)
    
    fig, ax = plt.subplots(row, col, figsize=(16, 8))
    ax = ax.flatten()

    for i, image_id in enumerate(image_ids):
        image = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(image_id)
        
def show_image_bb(image_data):
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    image = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_data.iloc[0]['image_id']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, row in image_data.iterrows():
        
        cv2.rectangle(image,
                      (int(row['x']), int(row['y'])),
                      (int(row['x']) + int(row['w']), int(row['y']) + int(row['h'])),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title(image_id)


# In[25]:


show_images(train.sample(n=15)['image_id'].values)


# In[26]:


show_image_bb(train[train['image_id'] == '5e0747034'])


# In[27]:


show_image_bb(train[train['image_id'] == '5b13b8160'])


# In[28]:


show_image_bb(train[train['image_id'] == '1f2b1a759'])


# In[29]:


DIR_RESULTS = '/kaggle/input/global-wheat-detection-public'
# Your OOF predictions
VALID_RESULTS = [
    f"{DIR_RESULTS}/validation_results_fold0_best.csv",
    f"{DIR_RESULTS}/validation_results_fold1_best.csv",
    f"{DIR_RESULTS}/validation_results_fold2_best.csv",
    f"{DIR_RESULTS}/validation_results_fold3_best.csv",
    f"{DIR_RESULTS}/validation_results_fold4_best.csv",
]

WEIGHTS_FILE = f'{DIR_RESULTS}/fasterrcnn_resnet50_fpn_best.pth'

# Below this area the size category of the box is 'small'
AREA_SMALL = 56 * 56

# Below this (and above small) is medium;
# Above this is large.
AREA_MEDIUM = 96 * 96

# If the box is at most this far from either of the borders
# we mark the box as 'is_border = True'
BORDER_SIZE = 2

# In these experiments I used 800px inputs.
# For analysis, we have to scale back to 1024px
# because the GT boxes are in that size.
SCALE = 1024/512

# Analizing at this threshold
THRESHOLD = 0.5


# In[30]:


def decode_prediction_string(pred_str):
    data = list(map(float, pred_str.split(" ")))
    data = np.array(data)
    
    return data.reshape(-1, 5)[:, 1:]

def calculate_iou(gt, pr, form='pascal_voc') -> float:

    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0])

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1])

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0]) * (gt[3] - gt[1]) +
            (pr[2] - pr[0]) * (pr[3] - pr[1]) -
            overlap_area
    )

    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def gen_images(data, filters, output_folder='./output', prefix='', limit=100):
    
    res = 'fp'
    resdata = data.copy()

    for filt in filters:
        resdata = resdata[resdata[filt[0]] == filt[1]]
        
        prefix = f"{prefix}_{filt[1]}"
        
        if filt[0] == 'result':
            res = filt[1]
        
        
    if limit > 0:
        resdata = resdata.sample(n=limit)
        
    image_ids = resdata['image_id'].unique()
    res_images = []
    
    for image_id in image_ids:
        img = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_id))
        
        if res == 'fn':
            boxes = resdata[resdata['image_id'] == image_id][['gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']].values
        elif res == 'fp':
            boxes = resdata[resdata['image_id'] == image_id][['pred_x1', 'pred_y1', 'pred_x2', 'pred_y2']].values
        
        for box in boxes:
            # tp
            color = (0, 220, 0)

            if res == 'fp':
                # Showing GT boxes nearby
                tpfilt = (
                    (data['image_id'] == image_id) &
                    (data['gt_x1'] < box[2] + 16) &
                    (data['gt_x2'] > box[0] - 16) &
                    
                    (data['gt_y1'] < box[3] + 16) &
                    (data['gt_y2'] > box[1] - 16)
                )
            
                tps = data[tpfilt][['gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']].values
                for tpbox in tps:
                    cv2.rectangle(img,
                                  (int(tpbox[0]), int(tpbox[1])),
                                  (int(tpbox[2]), int(tpbox[3])),
                                  color, 3)
            
            if res == 'fn':
                color = (40, 40, 198)
            elif res == 'fp':
                color = (198, 40, 40)

            cv2.rectangle(img,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 3)
                
            
        res_images.append((img, f"{output_folder}/{prefix}_{image_id}.jpg"))
        
    return res_images
    
def save_images(data, filters, output_folder='./output', prefix='', limit=100):
    images = gen_images(data=data, filters=filters, limit=limit)
    
    for image, path in images:
        cv2.imwrite(path, image)
        
def show_images(data, filters, rows=3, cols=2):
    
    images = gen_images(data=data, filters=filters, output_folder='', limit=rows*cols)
    
    fig, ax = plt.subplots(rows, cols, figsize=(16,16))
    ax = ax.flatten()
    
    for i, (image, path) in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(path)
        
        
def show_image_boxes(train, data):
    data = data.to_dict('records')

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    ax = ax.flatten()
    
    image = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(data[0]['image_id']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src_img = image.copy()
    
    boxes = train[train['image_id'] == data[0]['image_id']][['x', 'y', 'x2', 'y2']].values
    
    for box in boxes:
        cv2.rectangle(src_img,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 220, 0), 2)

    ax[0].set_axis_off()
    ax[0].imshow(src_img)
    ax[0].set_title("Image + GT boxes")
        
    # noisy targets
    for box_data in data:
        # fn
        color = (40, 40, 198)
        box = [0, 0, 0, 0]

        if box_data['result'] == 'fn':
            box[0], box[1], box[2], box[3] = box_data['gt_x1'],                                             box_data['gt_y1'],                                             box_data['gt_x2'],                                             box_data['gt_y2']


        elif box_data['result'] == 'fp':
            
            box[0], box[1], box[2], box[3] = box_data['pred_x1'],                                             box_data['pred_y1'],                                             box_data['pred_x2'],                                             box_data['pred_y2']

            color = (198, 40, 40)

        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color, 2)

    ax[1].set_axis_off()
    ax[1].imshow(image)
    ax[1].set_title("Blue: FP (predicted, no GT) | Red: FN (GT, no prediction)")


# In[31]:




train['x2'] = train['x'] + train['w']
train['y2'] = train['y'] + train['h']

# Calculate the area of the boxes.
train['area'] = train['w'] * train['h']

# Is the box at the edge of the image
train['is_border'] = False

border_filt = ((train['x'] < BORDER_SIZE) | (train['y'] < BORDER_SIZE) |
             (train['x2'] > 1024 - BORDER_SIZE) | (train['y2'] > 1024 - BORDER_SIZE))
train.loc[border_filt, 'is_border'] = True

train['size'] = 'large'
train.loc[train['area'] < AREA_MEDIUM, 'size'] = 'medium'
train.loc[train['area'] < AREA_SMALL, 'size'] = 'small'

# These are the ground-truth boxes
train['is_gt'] = True

train['brightness'] = 0.0
train['contrast'] = 0.0
train['overlap_iou'] = 0.0

train.sort_values(by='image_id', inplace=True)


# In[32]:


# - Brightness
# - Contrast
# - Hightest overlap with other GT box

last_src_id = None
src = None

for i, row in tqdm(train.iterrows(), total=train.shape[0]):
    
    if last_src_id != row['image_id']:
        src = cv2.imread(DIR_TRAIN_IMAGES+ '/{}.jpg'.format(row['image_id']))
        last_src_id = row['image_id']

    
    y1 = int(row['y'])
    y2 = int(row['y2'])
    x1 = int(row['x'])
    x2 = int(row['x2'])

    image = src[y1:y2, x1:x2].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    train.loc[i, 'brightness'] = image[:, :, 2].mean()
    
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    train.loc[i, 'contrast'] = image.std()

    


# In[33]:


train.head()


# In[34]:


# Format of the validation dataframes
pd.read_csv(VALID_RESULTS[0], usecols=['image_id', 'PredictionString']).head(5)


# In[35]:


valid= []

# helper.
image = train.groupby(by=['image_id', 'source'])[['is_gt']].nunique().reset_index()[['image_id', 'source']]

for src in VALID_RESULTS:
    valid_df = pd.read_csv(src, usecols=['image_id', 'PredictionString'])
    valid_df = valid_df.merge(image[['image_id', 'source']], how='left', on='image_id')
    valid_df.reset_index(drop=True, inplace=True)

    res = []

    for i, row in valid_df.iterrows():
        boxes = decode_prediction_string(row['PredictionString'])
        for box in boxes:
            valid.append({'image_id': row['image_id'],'width': 1024,'height': 1024,'bbox': '',
'source': row['source'],'x': box[0] * SCALE,'y': box[1] * SCALE,'x2': (box[0] + box[2]) * SCALE,'y2': (box[1] + box[3]) * SCALE,'w': box[2] * SCALE,'h': box[3] * SCALE,'area': (box[2] * box[3]) * SCALE,
'size': 'large','is_border': False,'is_gt': False,'brightness': 0.0,'contrast': 0.0})


# Convert the list to a pd.DataFrame
valid = pd.DataFrame(valid)

border_filt = ((valid['x'] < BORDER_SIZE) | (valid['y'] < BORDER_SIZE) |
             (valid['x2'] > 1024 - BORDER_SIZE) | (valid['y2'] > 1024 - BORDER_SIZE))
valid.loc[border_filt, 'is_border'] = True

valid.loc[valid['area'] < AREA_MEDIUM, 'size'] = 'medium'
valid.loc[valid['area'] < AREA_SMALL, 'size'] = 'small'

valid.sort_values(by='image_id', inplace=True)
# Calculate box infos
# - Brightness


# In[36]:


def calc(gts, preds, threshold=0.5, form='pascal-voc'):
    
    def _get_data(image_id, res, gt, pr):
        return {
                'image_id': image_id,
                'gt_x1': gt[1] if gt is not None else np.nan,
                'gt_y1': gt[2] if gt is not None else np.nan,
                'gt_x2': gt[3] if gt is not None else np.nan,
                'gt_y2': gt[4] if gt is not None else np.nan,
                'gt_w': gt[5] if gt is not None else np.nan,
                'gt_h': gt[6] if gt is not None else np.nan,
                'gt_area': gt[7] if gt is not None else np.nan,
                'gt_is_border': gt[8] if gt is not None else False,
                'gt_brightness': gt[12] if gt is not None else np.nan,
                'gt_contrast': gt[13] if gt is not None else np.nan,
                
                'pred_x1': pr[1] if pr is not None else np.nan,
                'pred_y1': pr[2] if pr is not None else np.nan,
                'pred_x2': pr[3] if pr is not None else np.nan,
                'pred_y2': pr[4] if pr is not None else np.nan,
                'pred_w': pr[5] if pr is not None else np.nan,
                'pred_h': pr[6] if pr is not None else np.nan,
                'pred_area': pr[7] if pr is not None else np.nan,
                'pred_is_border': pr[8] if pr is not None else False,
                'pred_brightness': pr[12] if pr is not None else np.nan,
                'pred_contrast': pr[13] if pr is not None else np.nan,
                
                'size': gt[10] if gt is not None else pr[10],
                'source': gt[11] if gt is not None else pr[11],
            
                'result': res
            }
    
    results = []
    
    # Number of predictions
    n = len(preds)
    
    for pred_idx in range(n):
        pr = preds[pred_idx]
        
        best_match_gt_idx = find_best_match(gts[:, 1:5], pr[1:5], pred_idx, threshold=threshold, form=form)
        
        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            gt = gts[best_match_gt_idx]
            results.append(_get_data(gt[0], 'tp', gt, pr))
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            results.append(_get_data(pr[0], 'fp', None, pr))

    for gt in gts:
        if gt[1] < 0:
            continue
            
        results.append(_get_data(gt[0], 'fn', gt, None))
    
    return results


# In[37]:


cols = ['image_id', 'x', 'y', 'x2', 'y2', 'w', 'h', 'area', 'is_border',
        'is_gt', 'size', 'source', 'brightness', 'contrast']

valid_img_ids = valid['image_id'].unique()

results = []

for img_id in tqdm(valid_img_ids, total=len(valid_img_ids)):
    gt_boxes = train[train['image_id'] == img_id][cols].values
    pred_boxes = valid[valid['image_id'] == img_id][cols].values
    
    results += calc(gt_boxes, pred_boxes, threshold=THRESHOLD, form='pascal-voc')
    
results = pd.DataFrame(results)

results['is_border'] = False
results.loc[(results['gt_is_border'] == True) | (results['pred_is_border'] == True), 'is_border'] = True


# In[38]:


results.head()


# In[39]:


def show_by_group(data, filt, group, idx, cols, title='', names=None, colors=None, order=None):
    
    if filt is not None:
        data = data[filt] 
    
    res = data.groupby(by=group).count()[['image_id']].reset_index().sort_index()
    res = res.pivot(index=idx, columns=cols, values='image_id')
    
    fig = go.Figure()
    
    if order is None:
        order = range(res.shape[0])

    for row_idx in order:
        fig.add_trace(go.Bar(
            x=res.columns,
            y=res.iloc[row_idx].values,
            name=names[row_idx] if names is not None else res.index[row_idx],
            marker_color=colors[row_idx] if colors is not None else None
        ))
        
    fig.update_layout(
        barmode='stack',
        barnorm = 'percent',
        title = {
            'text': title
        }
    )
    
    return res, fig


# In[40]:


res, fig = show_by_group(data=results, filt=None, group=['source', 'result'],
                         idx='result',
                         cols='source',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Results (TP|FP|FN) by sources'
                        )

res


# In[41]:


sources = results[results['result'] == 'fn']['source'].value_counts().sort_index().sort_index()

fig = go.Figure([go.Pie(labels=sources.index, values=sources.values)])
fig.update_layout(
    title = {
        'text': f'False negatives - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[42]:


filters = [
    ('result', 'fn'),
    ('is_border', True),
    ('source', 'rres_1')
]

show_images(results.copy(), filters, rows=3, cols=2)


# In[43]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=results[results['result'] == 'fn']['gt_brightness'],
                           histnorm='probability', name='False negatives', marker={'color': '#c62828'}))
fig.add_trace(go.Histogram(x=results[results['result'] == 'tp']['gt_brightness'],
                           histnorm='probability', name='True positives', marker={'color': '#4caf50'}))

fig.update_layout(barmode='overlay', title={
    'text': 'Brightness'
})
fig.update_traces(opacity=0.75)

fig.show()


# In[44]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=results[results['result'] == 'fn']['gt_contrast'],
                           histnorm='probability', name='False negatives', marker={'color': '#c62828'}))
fig.add_trace(go.Histogram(x=results[results['result'] == 'tp']['gt_contrast'],
                           histnorm='probability', name='True positives', marker={'color': '#4caf50'}))

fig.update_layout(barmode='overlay', title={
    'text': 'Contrast'
})
fig.update_traces(opacity=0.75)

fig.show()


# In[45]:


filters = [
    ('result', 'fp'),
    ('size', 'large'),
    ('source', 'inrae_1')
]

show_images(results.copy(), filters, rows=3, cols=2)


# In[46]:


filters=[
    ('result', 'fp'),
    ('is_border', True),
    ('source', 'inrae_1'),
]

show_images(results.copy(), filters, rows=3, cols=2)


# In[47]:


filters = [
    ('result', 'fp'),
    ('size', 'small'),
    ('is_border', True)
]

show_images(results.copy(), filters, rows=3, cols=2)


# In[48]:


res, fig = show_by_group(data=results, filt=results['is_border'] == False,
                         group=['size', 'result'],
                         idx='result',
                         cols='size',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Normal results by size'
                        )

res


# In[49]:


fig.show()


# In[50]:


filters = [
    ('result', 'fp'),
    ('size', 'small'),
    ('is_border', False)
]

show_images(results.copy(), filters, rows=3, cols=2)


# In[51]:


image_ids = results['image_id'].unique()

results_noisy = []

for image_id in tqdm(image_ids, total=image_ids.shape[0]):
    fps = results[(results['image_id'] == image_id) & (results['result'] == 'fp')]
    fps.reset_index(drop=True, inplace=True)

    fns = results[(results['image_id'] == image_id) & (results['result'] == 'fn')]
    fns.reset_index(drop=True, inplace=True)

    for fpi, fp in fps.iterrows():
        
        for fni, fn in fns.iterrows():
            
            if ((fp['pred_x1'] <= fn['gt_x1']) and
                (fp['pred_y1'] <= fn['gt_y1']) and
                (fp['pred_x2'] >= fn['gt_x2']) and   
                (fp['pred_y2'] >= fn['gt_y2'])):
                
                # GT inside predicted
                results_noisy.append(fp.to_dict())
                results_noisy.append(fn.to_dict())
            
            elif ((fp['pred_x1'] >= fn['gt_x1']) and
                  (fp['pred_y1'] >= fn['gt_y1']) and
                  (fp['pred_x2'] <= fn['gt_x2']) and   
                  (fp['pred_y2'] <= fn['gt_y2'])):
                
                # PREDICTED inside GT
                results_noisy.append(fp.to_dict())
                results_noisy.append(fn.to_dict())


results_noisy = pd.DataFrame(results_noisy)


# In[52]:


results_noisy.head(10)


# In[53]:


noisy_sources = pd.DataFrame(train['source'].value_counts().sort_index())
noisy_sources['noisy'] = (results_noisy['source'].value_counts() // 2).sort_index().values
noisy_sources['p'] = noisy_sources['noisy'] / noisy_sources['source'] * 100

noisy_sources.sort_values(by='p', ascending=True)


# In[54]:


show_image_boxes(train, results_noisy[results_noisy['image_id'] == '4021d47d4'].copy())


# In[55]:


show_image_boxes(train, results_noisy[results_noisy['image_id'] == '7b72ea0fb'].copy())


# In[56]:


class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# In[57]:


# Albumentations
def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


# In[58]:


# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)


# In[59]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)


# In[60]:


def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatTestDataset(sub, DIR_TEST_IMAGES, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)


# In[61]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


# In[62]:


detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        
        results.append(result)


# In[63]:


results[0:2]


# In[64]:


sub = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
sub.head()


# In[65]:


sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= detection_threshold].astype(np.int32)


# In[66]:


fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)


# In[67]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




