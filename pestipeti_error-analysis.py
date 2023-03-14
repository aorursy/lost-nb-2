#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
import os
import random
import ast
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm.notebook import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# =========================================
# Replace these with your data.
# =========================================
#  - I have 1 .csv file per fold
#  - All of these have the OOF predictions
#  - You can use one fold too.
#  - The format should be the same as the
#    submission.csv

# Your private dataset
DIR_RESULTS = '/kaggle/input/global-wheat-detection-public'

# Your OOF predictions
VALID_RESULTS = [
    f"{DIR_RESULTS}/validation_results_fold0_best.csv",
    f"{DIR_RESULTS}/validation_results_fold1_best.csv",
    f"{DIR_RESULTS}/validation_results_fold2_best.csv",
    f"{DIR_RESULTS}/validation_results_fold3_best.csv",
    f"{DIR_RESULTS}/validation_results_fold4_best.csv",
]

# =========================================

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
SCALE = 1024/800

# Analizing at this threshold
THRESHOLD = 0.5


# In[3]:


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'


# In[4]:


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
        img = cv2.imread(DIR_TRAIN + '/{}.jpg'.format(image_id))
        
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
        
def show_images(data, filters, rows=2, cols=2):
    
    images = gen_images(data=data, filters=filters, output_folder='', limit=rows*cols)
    
    fig, ax = plt.subplots(rows, cols, figsize=(16,16))
    ax = ax.flatten()
    
    for i, (image, path) in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(path)
        
        
def show_image_boxes(train_df, data):
    data = data.to_dict('records')

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    ax = ax.flatten()
    
    image = cv2.imread(DIR_TRAIN + '/{}.jpg'.format(data[0]['image_id']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src_img = image.copy()
    
    boxes = train_df[train_df['image_id'] == data[0]['image_id']][['x', 'y', 'x2', 'y2']].values
    
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


# In[5]:


train_df = pd.read_csv(f"{DIR_INPUT}/train.csv")

# From Andrew's kernel
train_df[['x', 'y', 'w', 'h']] = pd.DataFrame(
    np.stack(train_df['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)
train_df.drop(columns=['bbox'], inplace=True)

train_df['x2'] = train_df['x'] + train_df['w']
train_df['y2'] = train_df['y'] + train_df['h']

# Calculate the area of the boxes.
train_df['area'] = train_df['w'] * train_df['h']

# Is the box at the edge of the image
train_df['is_border'] = False

border_filt = ((train_df['x'] < BORDER_SIZE) | (train_df['y'] < BORDER_SIZE) |
             (train_df['x2'] > 1024 - BORDER_SIZE) | (train_df['y2'] > 1024 - BORDER_SIZE))
train_df.loc[border_filt, 'is_border'] = True

train_df['size'] = 'large'
train_df.loc[train_df['area'] < AREA_MEDIUM, 'size'] = 'medium'
train_df.loc[train_df['area'] < AREA_SMALL, 'size'] = 'small'

# These are the ground-truth boxes
train_df['is_gt'] = True

train_df['brightness'] = 0.0
train_df['contrast'] = 0.0
train_df['overlap_iou'] = 0.0

train_df.sort_values(by='image_id', inplace=True)


# In[6]:


# Calculate box infos
# - Brightness
# - Contrast
# - Hightest overlap with other GT box

last_src_id = None
src = None

for i, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    
    if last_src_id != row['image_id']:
        src = cv2.imread(DIR_TRAIN + '/{}.jpg'.format(row['image_id']))
        last_src_id = row['image_id']

    
    y1 = int(row['y'])
    y2 = int(row['y2'])
    x1 = int(row['x'])
    x2 = int(row['x2'])

    image = src[y1:y2, x1:x2].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    train_df.loc[i, 'brightness'] = image[:, :, 2].mean()
    
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    train_df.loc[i, 'contrast'] = image.std()


# In[7]:


train_df.head()


# In[8]:


# Format of the validation dataframes
pd.read_csv(VALID_RESULTS[0], usecols=['image_id', 'PredictionString']).head(3)


# In[9]:


valid_df = []

# helper.
image_df = train_df.groupby(by=['image_id', 'source'])[['is_gt']].nunique().reset_index()[['image_id', 'source']]

for src in VALID_RESULTS:
    valid = pd.read_csv(src, usecols=['image_id', 'PredictionString'])
    valid = valid.merge(image_df[['image_id', 'source']], how='left', on='image_id')
    valid.reset_index(drop=True, inplace=True)

    res = []

    for i, row in valid.iterrows():

        boxes = decode_prediction_string(row['PredictionString'])

        for box in boxes:

            valid_df.append({
                'image_id': row['image_id'],
                'width': 1024,
                'height': 1024,
                'bbox': '',
                'source': row['source'],
                'x': box[0] * SCALE,
                'y': box[1] * SCALE,
                'x2': (box[0] + box[2]) * SCALE,
                'y2': (box[1] + box[3]) * SCALE,
                'w': box[2] * SCALE,
                'h': box[3] * SCALE,
                'area': (box[2] * box[3]) * SCALE,
                'size': 'large',
                'is_border': False,
                'is_gt': False,
                'brightness': 0.0,
                'contrast': 0.0

            })


# Convert the list to a pd.DataFrame
valid_df = pd.DataFrame(valid_df)

border_filt = ((valid_df['x'] < BORDER_SIZE) | (valid_df['y'] < BORDER_SIZE) |
             (valid_df['x2'] > 1024 - BORDER_SIZE) | (valid_df['y2'] > 1024 - BORDER_SIZE))
valid_df.loc[border_filt, 'is_border'] = True

valid_df.loc[valid_df['area'] < AREA_MEDIUM, 'size'] = 'medium'
valid_df.loc[valid_df['area'] < AREA_SMALL, 'size'] = 'small'

valid_df.sort_values(by='image_id', inplace=True)


# In[10]:


# Calculate box infos
# - Brightness
# - Contrast
# - Hightest overlap with other GT box

last_src_id = None
src = None

for i, row in tqdm(valid_df.iterrows(), total=valid_df.shape[0]):
    
    if last_src_id != row['image_id']:
        src = cv2.imread(DIR_TRAIN + '/{}.jpg'.format(row['image_id']))
        last_src_id = row['image_id']

    
    y1 = int(row['y'])
    y2 = int(row['y2'])
    x1 = int(row['x'])
    x2 = int(row['x2'])

    image = src[y1:y2, x1:x2].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    valid_df.loc[i, 'brightness'] = image[:, :, 2].mean()
    
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    valid_df.loc[i, 'contrast'] = image.std()


# In[11]:


valid_df.head()


# In[12]:


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


# In[13]:


cols = ['image_id', 'x', 'y', 'x2', 'y2', 'w', 'h', 'area', 'is_border',
        'is_gt', 'size', 'source', 'brightness', 'contrast']

valid_img_ids = valid_df['image_id'].unique()

results = []

for img_id in tqdm(valid_img_ids, total=len(valid_img_ids)):
    gt_boxes = train_df[train_df['image_id'] == img_id][cols].values
    pred_boxes = valid_df[valid_df['image_id'] == img_id][cols].values
    
    results += calc(gt_boxes, pred_boxes, threshold=THRESHOLD, form='pascal-voc')
    
results = pd.DataFrame(results)

results['is_border'] = False
results.loc[(results['gt_is_border'] == True) | (results['pred_is_border'] == True), 'is_border'] = True


# In[14]:


results.head()


# In[15]:


labels = ['True Positive', 'False Positive', 'False Negative']
values = results['result'].value_counts().sort_index(ascending=False).values

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])
fig.add_trace(go.Bar(x=labels, y=values, showlegend=False), row=1, col=1)
fig.add_trace(go.Pie(labels=labels, values=values), row=1, col=2)
fig.update_layout(
    title={
        'text': f'Number and ratio of TP/FP/FN at threshold {THRESHOLD}'
    }
)

fig.show()


# In[16]:


sources = results['source'].value_counts().sort_index()

fig = go.Figure([go.Bar(x=sources.index, y=sources.values)])
fig.update_layout(
    title = {
        'text': f'Number of boxes: GT (TP or FN) + FP - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[17]:


sources = results['size'].value_counts().sort_index()

fig = go.Figure([go.Bar(x=sources.index, y=sources.values)])
fig.update_layout(
    title={
        'text': f'Number of boxes by size: GT (TP or FN) + FP - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[18]:


sources = results['is_border'].value_counts().sort_index().sort_index()

fig = go.Figure([go.Bar(x=['Normal', 'Border'], y=sources.values)])
fig.update_layout(
    title={
        'text': f'Number of boxes by type: GT (TP or FN) + FP - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[19]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=results['gt_brightness']))
fig.update_layout(title={
    'text': "Brightness of GT boxes"
})

fig.show()


# In[20]:


fig = go.Figure()
fig.add_trace(go.Histogram(x = results['gt_contrast']))
fig.update_layout(title={
    'text': 'Contrast value of GT boxes'
})

fig.show()


# In[21]:


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


# In[22]:


res, fig = show_by_group(data=results, filt=None, group=['source', 'result'],
                         idx='result',
                         cols='source',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Results (TP|FP|FN) by sources'
                        )

res


# In[23]:


fig.show()


# In[24]:


sources = results[results['result'] == 'fn']['source'].value_counts().sort_index().sort_index()

fig = go.Figure([go.Pie(labels=sources.index, values=sources.values)])
fig.update_layout(
    title = {
        'text': f'False negatives - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[25]:


res, fig = show_by_group(data=results, filt=results['result'] == 'fn',
                         group=['source', 'size'],
                         idx='size',
                         cols='source',
                         names=['Large', 'Medium', 'Small'],
                         colors=['#c62828', '#e57373','#ffcdd2'],
                         order=[2, 1, 0],
                         title='False negatives by size'
                        )

res


# In[26]:


fig.show()


# In[27]:


res, fig = show_by_group(data=results, filt=results['result'] == 'fn',
                         group=['source', 'is_border'],
                         idx='is_border',
                         cols='source',
                         names=['Normal', 'Border'],
                         colors=['#ffcdd2', '#c62828'],
                         order=[1,0],
                         title='False negatives normal/border'
                        )

res


# In[28]:


fig.show()


# In[29]:


filters = [
    ('result', 'fn'),
    ('is_border', True),
    ('source', 'rres_1')
]

show_images(results.copy(), filters, rows=2, cols=2)


# In[30]:


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


# In[31]:


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


# In[32]:


sources = results[results['result'] == 'fp']['source'].value_counts().sort_index().sort_index()

fig = go.Figure([go.Pie(labels=sources.index, values=sources.values)])
fig.update_layout(
    title = {
        'text': f'False positives - at threshold {THRESHOLD}'
    }
)
fig.show()


# In[33]:


res, fig = show_by_group(data=results, filt=results['result'] == 'fp',
                         group=['source', 'size'],
                         idx='size',
                         cols='source',
                         names=['Large', 'Medium', 'Small'],
                         colors=['#283593', '#3f51b5','#7986cb'],
                         order=[2, 1, 0],
                         title='False positives by size'
                        )

res


# In[34]:


fig.show()


# In[35]:


filters = [
    ('result', 'fp'),
    ('size', 'large'),
    ('source', 'inrae_1')
]

show_images(results.copy(), filters, rows=2, cols=2)


# In[36]:


res, fig = show_by_group(data=results, filt=results['result'] == 'fp',
                         group=['source', 'is_border'],
                         idx='is_border',
                         cols='source',
                         names=['Normal', 'Border'],
                         colors=['#7986cb', '#283593'],
                         order=[1, 0],
                         title='False positives normal/border'
                        )

res


# In[37]:


fig.show()


# In[38]:


filters = [
    ('result', 'fp'),
    ('is_border', True),
    ('source', 'inrae_1'),
]

show_images(results.copy(), filters, rows=2, cols=2)


# In[39]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=results[results['result'] == 'fp']['pred_brightness'],
                           histnorm='probability', name='False positives', marker={'color':'#3f51b5'}))
fig.add_trace(go.Histogram(x=results[results['result'] == 'tp']['gt_brightness'],
                           histnorm='probability', name='True positives', marker={'color': '#4caf50'}))

fig.update_layout(barmode='overlay', title={
    'text': 'Brightness'
})
fig.update_traces(opacity=0.75)

fig.show()


# In[40]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=results[results['result'] == 'fp']['pred_contrast'],
                           histnorm='probability', name='False positives', marker={'color':'#3f51b5'}))
fig.add_trace(go.Histogram(x=results[results['result'] == 'tp']['gt_contrast'],
                           histnorm='probability', name='True positives', marker={'color': '#4caf50'}))

fig.update_layout(barmode='overlay', title={
    'text': 'Contrast'
})
fig.update_traces(opacity=0.75)

fig.show()


# In[41]:


res, fig = show_by_group(data=results, filt=None, group=['size', 'result'],
                         idx='result',
                         cols='size',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Results by size'
                        )

res


# In[42]:


fig.show()


# In[43]:


res, fig = show_by_group(data=results, filt=None, group=['is_border', 'result'],
                         idx='result',
                         cols='is_border',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Results normal/border'
                        )

res


# In[44]:


fig.show()


# In[45]:


res, fig = show_by_group(data=results, filt=results['is_border'] == True,
                         group=['size', 'result'],
                         idx='result',
                         cols='size',
                         names=['False Negative', 'False Positive', 'True Positive'],
                         colors=['#c62828', '#3f51b5', '#4caf50'],
                         title='Results at the borders by size'
                        )

res


# In[46]:


fig.show()


# In[47]:


filters = [
    ('result', 'fp'),
    ('size', 'small'),
    ('is_border', True)
]

show_images(results.copy(), filters, rows=2, cols=2)


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

show_images(results.copy(), filters, rows=2, cols=2)


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


noisy_sources = pd.DataFrame(train_df['source'].value_counts().sort_index())
noisy_sources['noisy'] = (results_noisy['source'].value_counts() // 2).sort_index().values
noisy_sources['p'] = noisy_sources['noisy'] / noisy_sources['source'] * 100

noisy_sources.sort_values(by='p', ascending=True)


# In[54]:


show_image_boxes(train_df, results_noisy[results_noisy['image_id'] == '4021d47d4'].copy())


# In[55]:


show_image_boxes(train_df, results_noisy[results_noisy['image_id'] == '01397a84c'].copy())


# In[56]:


show_image_boxes(train_df, results_noisy[results_noisy['image_id'] == '7b72ea0fb'].copy())


# In[ ]:




