#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
get_ipython().system('pip install -q lyft-dataset-sdk')

# Any results you write to the current directory are saved as output.


# In[2]:


import os
import math

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud


# In[3]:


INP_DIR = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'
# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
get_ipython().system('ln -s {INP_DIR}/train_images images')
get_ipython().system('ln -s {INP_DIR}/train_maps maps')
get_ipython().system('ln -s {INP_DIR}/train_lidar lidar')


# In[4]:


level5data = LyftDataset(
    data_path='.',
    json_path=os.path.join(INP_DIR + 'train_data'),
    verbose=False
)


# In[5]:


my_scene = level5data.scene[0]
token = my_scene['first_sample_token']
sample = level5data.get('sample', token)
lidar = level5data.get_sample_data(sample['data']['LIDAR_TOP'])
lidar[1][3]


# In[6]:


def get_coords_from_ann_idx(ann_idx, sample):
    return np.array(level5data.get('sample_annotation', sample['anns'][ann_idx])['translation'])


# In[7]:


anns_inds_to_show = [0, 1, 2] # i select several cars near lyft's pod
ann_tokens = []
for ind in anns_inds_to_show:
    my_annotation_token = sample['anns'][ind]
    print(f'{ind}: {my_annotation_token}')
    ann_tokens.append(my_annotation_token)
    level5data.render_annotation(my_annotation_token)
    plt.show()


# In[8]:


next_sample = level5data.get('sample', sample['next'])
anns_inds_to_show = [0, 1, 2] # i select several cars near lyft's pod
ann_tokens = []
for ind in anns_inds_to_show:
    my_annotation_token = next_sample['anns'][ind]
    print(f'{ind}: {my_annotation_token}')
    ann_tokens.append(my_annotation_token)
    level5data.render_annotation(my_annotation_token)
    plt.show()


# In[9]:


ltok = my_scene['last_sample_token']
level5data.get('sample', ltok)['next']


# In[10]:


def get_data_from_sample(sample, channel_to_get):
    return level5data.get('sample_data', sample['data'][channel_to_get])
lidar_data = get_data_from_sample(sample, 'LIDAR_TOP')
lidar_data_r = get_data_from_sample(sample, 'LIDAR_FRONT_RIGHT')
lidar_data_l = get_data_from_sample(sample, 'LIDAR_FRONT_LEFT')


# In[11]:


lidar_data = get_data_from_sample(sample, 'LIDAR_TOP')
pc = LidarPointCloud.from_file(Path(lidar_data['filename']))
print(pc.points.shape)
pc.points[:].max(axis=1), pc.points[:].min(axis=1), pc.points.mean(axis=1), pc.points.std(axis=1)


# In[12]:


level5data.get_boxes(sample['data']['LIDAR_TOP'])[32]


# In[13]:


level5data.get_sample_data(sample['data']['LIDAR_TOP'])[1][32]


# In[14]:


from collections import defaultdict
def sample_points(points, T=35, K=250):
    points = points.T
    points = points[:, :-1]
    bounderies = [-40, -40, -3], [40, 40, 1]
    points_in_volume = points[np.logical_and(np.less_equal(bounderies[0], points), np.less_equal(points, bounderies[1])).all(axis=-1)]
    bounderies = np.array((-100, -100, -15)), np.array((100, 100, 5))
    vd, vw, vh = 0.4, 0.4, 0.2
    voxels = (points/np.array([vd, vw, vh]).reshape(1, 3)).round().astype(np.int)
    voxel_dict = defaultdict(list)
    for i, v in enumerate(voxels):
        voxel_dict[tuple(v)].append(i)
    voxel_dict = {k: points[np.random.choice(v, size=T)] for k, v in voxel_dict.items() 
                  if len(v) >= T and np.logical_and(bounderies[0] <= np.array(k), np.array(k) <= bounderies[1]).all()}
    voxel_coords, voxel_features = zip(*list(voxel_dict.items())[:K])
    voxel_features = np.stack(voxel_features)
    voxel_coords = voxel_coords - bounderies[0]
    pad_len = K - voxel_features.shape[0]
    voxel_coords = np.pad(voxel_coords, [(0, pad_len), (0,0)], 'constant', constant_values=-1)
    voxel_features = np.pad(voxel_features, [(0, pad_len), (0,0), (0,0)], 'constant', constant_values=0)
    voxel_means = voxel_features.mean(axis=1, keepdims=True)
    voxel_features = np.concatenate([voxel_features, voxel_features-voxel_means], axis=-1)
    return voxel_features, voxel_coords
    
f, coords = sample_points(pc.points)


# In[15]:


def gen_samples(n_scenes=1000):
    for scene in level5data.scene[:n_scenes]:
        sample = level5data.get('sample', scene['first_sample_token'])
        while True:
            yield sample
            if sample['next']:
                sample = level5data.get('sample', sample['next'])
            else:
                break
    raise StopIteration
from collections import defaultdict  
class Average:
    def __init__(self, init=0.0):
        self.n, self.sum = 0, init
    
    def __add__(self, v):
        self.n += 1
        self.sum += v
        return self
    
    def __repr__(self):
        return str(self.value)
        
    @property
    def value(self):
        if self.n==0:
            return self.sum
        return self.sum/self.n

box_wlh = defaultdict(lambda: Average(np.zeros(3)))
box_z_center = defaultdict(Average)
#nums = defaultdict(int)
for sample in gen_samples(3000):
    for bbox in level5data.get_sample_data(sample['data']['LIDAR_TOP'])[1]:
        box_wlh[bbox.name] += bbox.wlh
        box_z_center[bbox.name] += bbox.center[-1]
        
keys = box_wlh.keys()
bbox_shapes = np.array([box_wlh[k].value for k in keys])
bbox_z_center = np.array([box_z_center[k].value for k in keys])


# In[16]:


cat2id = {cat['name']: i for i, cat in enumerate(level5data.category)}
id2cat = {i: cat['name'] for i, cat in enumerate(level5data.category)}


# In[17]:


bbox_shapes = np.array([box_wlh[cat['name']].value for cat in level5data.category])
bbox_z_center = np.array([box_z_center[cat['name']].value for cat in level5data.category])
def create_anchors():
    bounderies = np.array((-100, -100, -15)), np.array((100, 100, 5))
    fm_w, fm_d = 96, 96
    x_centers = np.linspace(bounderies[0][0], bounderies[1][0], fm_w, endpoint=False)
    y_centers = np.linspace(bounderies[0][1], bounderies[1][1], fm_d, endpoint=False)
    anchor_centers = np.stack(np.meshgrid(x_centers, y_centers), axis=-1)
    anchor_centers = np.expand_dims(anchor_centers, 2)
    n_bb = len(keys)*2
    centers_xy = np.tile(anchor_centers, (1, 1, n_bb, 1))
    centers_z = np.tile(bbox_z_center[np.newaxis,np.newaxis,:,np.newaxis], (fm_w,fm_d,2,1))
    anchor_wlh = np.tile(bbox_shapes[np.newaxis,np.newaxis,:,:], (fm_w,fm_d,2,1))
    anchor_angs = np.tile(np.array([0, np.pi/2])[np.newaxis,np.newaxis,:,np.newaxis], (fm_w,fm_d,len(keys),1))
    return np.concatenate([centers_xy, centers_z, anchor_wlh, anchor_angs], axis=-1)
anchors = create_anchors()


# In[18]:


boxes = []
for sample in gen_samples(3):
    for bbox in level5data.get_sample_data(sample['data']['LIDAR_TOP'])[1]:
        boxes.append(np.concatenate([bbox.center, bbox.wlh, np.array(bbox.orientation.yaw_pitch_roll[:1])]))
    break
true_boxes = np.array(boxes)
true_boxes.shape


# In[19]:


def get_transform_matrix(x_center, y_center, w, d, theta):
    return np.array([[w/2*np.cos(theta), d/2*np.sin(theta), x_center],
                     [w/2*np.sin(theta), -d/2*np.cos(theta), y_center],
                     [0, 0, 1]])
    
def monte_carlo_overlap_2d(box1, box2 ,N=300):
    x1,y1,z1,w1,d1,l1,r1 = box1
    x2,y2,z2,w2,d2,l2,r2 = box2
    to_box1 = get_transform_matrix(x1,y1,w1,d1,r1)
    from_box2 = np.linalg.inv(get_transform_matrix(x2,y2,w2,d2,r2))
    points = np.random.rand(2, N)*2 - 1 
    points = np.concatenate([points, np.ones((1,N))], axis=0)
    transformed_points = (from_box2@to_box1@points)[:-1]
    in_box2 = np.all(np.logical_and(transformed_points < 1, -1 < transformed_points), axis=0)
    return np.sum(in_box2)/N

def IOU_2d(box1, box2):
    area1 = box1[3]*box1[4]
    area2 = box2[3]*box2[4]
    if False and area1 < area2:
        box1, box2, area1, area2 = box2, box1, area2, area1
    intersection = area1*monte_carlo_overlap_2d(box1, box2)
    union = area1 + area2 - intersection
    return intersection/union


# In[20]:


def bbox_overlap(anchors, gt_boxes):
    anchors = anchors.reshape((-1, 7))
    n, k = anchors.shape[0], gt_boxes.shape[0]
    anchor_radii = np.linalg.norm(anchors[:,3:5], axis=-1)[:,np.newaxis]
    boxes_radii = np.linalg.norm(gt_boxes[:,3:5], axis=-1)[np.newaxis,:]
    sum_radii = anchor_radii + boxes_radii
    center_distances = np.linalg.norm(anchors[:,np.newaxis,:2] - gt_boxes[np.newaxis,:,:2], axis=-1)
    possible_overlaps = (center_distances < sum_radii)
    possible_overlaps1 = np.abs(np.log(anchor_radii) - np.log(boxes_radii)) < np.log(2)
    possible_overlaps = np.logical_and(possible_overlaps, possible_overlaps1)
    iou = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            if possible_overlaps[i,j]:
                iou[i,j] = IOU_2d(anchors[i], gt_boxes[j])
    return iou
    
iou = bbox_overlap(anchors, true_boxes)


# In[21]:


mask = np.max(iou, axis=0)>0.35
np.arange(iou.shape[1])[mask], np.argmax(iou, axis=0)[mask]
np.where(iou > 0.5)


# In[22]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# In[23]:


cat2id = {cat['name']: i+1 for i, cat in enumerate(level5data.category)}
id2cat = {i+1: cat['name'] for i, cat in enumerate(level5data.category)}
bbox_shapes = np.array([box_wlh[cat['name']].value for cat in level5data.category])
bbox_z_center = np.array([box_z_center[cat['name']].value for cat in level5data.category])
class lyft_data(Dataset):
    T = 35 # מספר הנקודות שנדגמות מכל ווקסל
    K = 250 # number of sampled voxels in a sample
    vd, vw, vh = 0.4, 0.4, 0.2 # voxel sizes
    voxel_dims = np.array([vd, vw, vh])
    fm_w, fm_d = 96, 96 # feature map shape
    n_anchors_per_position = 2*len(level5data.category) # 2 rotations (0, 90 degrees) * number of categories
    pos_threshold, neg_threshold = 0.5, 0.35
    bounderies = np.array([[-40, -40, -3],[40, 40, 1]])# 
    def __init__(self, samples):
        self.samples = samples
        self.anchors = create_anchors()
        
    def create_anchors():
        x_centers = np.linspace(bounderies[0][0], bounderies[1][0], fm_w, endpoint=False)
        y_centers = np.linspace(bounderies[0][1], bounderies[1][1], fm_d, endpoint=False)
        anchor_centers = np.stack(np.meshgrid(x_centers, y_centers), axis=-1)
        anchor_centers = np.expand_dims(anchor_centers, 2)
        n_bb = self.n_anchors_per_position
        centers_xy = np.tile(anchor_centers, (1, 1, n_bb, 1))
        centers_z = np.tile(bbox_z_center[np.newaxis,np.newaxis,:,np.newaxis], (fm_w,fm_d,2,1))
        anchor_wlh = np.tile(bbox_shapes[np.newaxis,np.newaxis,:,:], (fm_w,fm_d,2,1))
        anchor_angs = np.tile(np.array([0, np.pi/2])[np.newaxis,np.newaxis,:,np.newaxis], (fm_w,fm_d,n_bb//2,1))
        return np.concatenate([centers_xy, centers_z, anchor_wlh, anchor_angs], axis=-1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        sample = self.samples[i]
        lidar_tok = sample['data']['LIDAR_TOP']
        bboxes = level5data.get_sample_data(lidar_tok)[1]
        lidar_data = level5data.get('sample_data', lidar_tok)
        pc = LidarPointCloud.from_file(Path(lidar_data['filename']))
        voxel_features, voxel_coords = self.sample_points(pc.points) 
        pos, neg_equal_one, targets = self.cal_target(bboxes)
        return voxel_features, voxel_coords, pos, neg_equal_one, targets
        
    def cal_target(self, gt_box3d):
        # Input:
        #   labels: (N,)
        #   feature_map_shape: (w, l)
        #   anchors: (w, l, 2, 7)
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview
        pos_equal_one = np.zeros((self.fm_w, self.fm_d, self.n_anchors_per_position))
        neg_equal_one = np.zeros((self.fm_w, self.fm_d, self.n_anchors_per_position))
        targets = np.zeros((self.fm_w, self.fm_d, 7*self.n_anchors_per_position))
        
        self.anchors = self.anchors.reshape((-1, 7))
        
        gt_xyzwlhr = np.stack([np.concatenate([bbox.center, bbox.wlh, np.array(bbox.orientation.yaw_pitch_roll[:1])])                               for bbox in gt_box3d], axis=0)
        gt_categories = np.array([cat2id[bbox.name] for bbox in gt_box3d], dtype=np.int)
        iou = bbox_overlap(self.anchors, gt_xyzwlhr)
        
        id_highest = np.argmax(iou, axis=0)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.shape[1])
        mask = iou[id_highest, id_highest_gt] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
        
        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold, axis=1) == iou.shape[1])[0]
        
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        category_gt = gt_categories[id_pos_gt]
        id_neg.sort()
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (self.fm_w, self.fm_d, self.n_anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = category_gt
        # ATTENTION: index_z should be np.array
        
        anchors_d = np.sqrt(self.anchors[:, 3]**2 + self.anchors[:, 4]**2)
        targets[index_x, index_y, np.array(index_z) * 7] =             (gt_xyzwlhr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] =             (gt_xyzwlhr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] =             (gt_xyzwlhr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 5]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_xyzwlhr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_xyzwlhr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_xyzwlhr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzwlhr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (self.fm_w, self.fm_d, self.n_anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (self.fm_w, self.fm_d, self.n_anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        return pos_equal_one, neg_equal_one, targets
        
    
    def sample_points(self, points):
        points = points.T
        points = points[:, :-1]
        voxels = (points/self.voxel_dims.reshape(1, 3)).round().astype(np.int)
        voxel_bounderies = self.bounderies/self.voxel_dims.reshape(1, 3)
        voxel_dict = defaultdict(list)
        for i, v in enumerate(voxels):
            voxel_dict[tuple(v)].append(i)
        i = 0
        voxel_dict = {k: points[np.random.choice(v, size=self.T)] for k, v in voxel_dict.items() 
                      if len(v) >= self.T and np.logical_and(voxel_bounderies[0] <= np.array(k), np.array(k) <= voxel_bounderies[1]).all()}
        voxel_coords, voxel_features = zip(*list(voxel_dict.items())[:self.K])
        voxel_features = np.stack(voxel_features)
        voxel_coords = voxel_coords - voxel_bounderies[0]
        pad_len = self.K - voxel_features.shape[0]
        voxel_coords = np.pad(voxel_coords, [(0, pad_len), (0,0)], 'constant', constant_values=-1)
        voxel_features = np.pad(voxel_features, [(0, pad_len), (0,0), (0,0)], 'constant', constant_values=0)
        voxel_means = voxel_features.mean(axis=1, keepdims=True)
        voxel_features = np.concatenate([voxel_features, voxel_features-voxel_means], axis=-1)
        return voxel_features, voxel_coords

        


# In[24]:


data = lyft_data(list(gen_samples(3)))


# In[25]:


data[4]


# In[26]:


pos_equal_one, neg_equal_one, targets = data.cal_target(level5data.get_sample_data(data.samples[20]['data']['LIDAR_TOP'])[1])


# In[27]:


pos_equal_one.shape


# In[28]:


pos = pos_equal_one.reshape((96,96,9,2))
np.where(pos), pos[np.where(pos)]


# In[29]:


len([b.name for b in level5data.get_sample_data(data.samples[0]['data']['LIDAR_TOP'])[1]]), len(pos[np.where(pos)])

