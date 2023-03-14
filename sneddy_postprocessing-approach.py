#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
pd.set_option("display.max_rows", 50)

def calc_logloss(targets, outputs, eps=1e-5):
    logloss_classes = [log_loss(np.floor(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
    }

def build_2d_labels(labels_1d):
    labels = np.zeros((labels_1d.shape[0] // 6, 6))
    for class_id in range(6):
        labels[:, class_id] = labels_1d[class_id::6]
    return labels

def eval_submit(sub, answers_path='../input/hemorrhagehelpers/stage_1_answers.npy'):
    labels_1d = sub.sort_values('ID').Label.values
    labels_2d = build_2d_labels(labels_1d)
    targets = np.load(answers_path)
    return calc_logloss(targets, labels_2d)

def eval_df(df, answers_path='../input/hemorrhagehelpers/stage_1_answers.npy'):
    df.sort_index(inplace=True)
    targets = np.load(answers_path)
    metrics = {}
    for col in tqdm(df.columns):
        labels_1d = df[col].values
        labels_2d = build_2d_labels(labels_1d)
        current_loss = calc_logloss(targets, labels_2d)['logloss']
        current_repr = float('{:.6}'.format(current_loss))
        metrics[col] = current_repr
    
    info = pd.DataFrame.from_dict([metrics]).T
    info.columns = ['leaderboard']
    return info.sort_values('leaderboard')

def add_table(df, fname, colname):
    sub = pd.read_csv(fname)
    sub = sub.sort_values('ID').reset_index(drop=True)
    sub['image_id'] = sub.ID.apply(lambda x: '_'.join(x.split('_')[:-1]))
    df[colname] = sub.set_index('ID').Label


# In[2]:


stage1_names = {    
    'takato_original_seresnext': '../input/hemoorhagesubsfirststage/se_resnext50_32x4d_102105_hflip.csv',
    'takato_original_effb1': '../input/hemoorhagesubsfirststage/efficientnet-b1_102407_hflip.csv',
    'takato_original_effb2': '../input/hemoorhagesubsfirststage/efficientnet-b2_102000_hflip.csv',
    'takato_original_effb3': '../input/hemoorhagesubsfirststage/efficientnet-b3_102112_hflip.csv',
    'takato_original_effb4': '../input/hemoorhagesubsfirststage/efficientnet-b4_102100_hflip.csv',
    
    'agro_b7': '../input/hemoorhagesubsfirststage/agroeffnet_stage1_test_tta8.csv',
    'sneddy_b7': '../input/hemoorhagesubsfirststage/effnet-b7-e1.csv',
    'resnet152': '../input/hemoorhagesubsfirststage/resnet152_stage1.csv',
    'effnet_b5': '../input/hemoorhagesubsfirststage/effnetb5_stage1_test_tta8.csv',
    'balanced_effb4': '../input/hemoorhagesubsfirststage/balansed_effb4_stage1.csv',
    'inceptionv4': '../input/hemoorhagesubsfirststage/inceptionv4_stage1.csv',
    'effnet_b0': '../input/hemoorhagesubsfirststage/effb0_stage1.csv',
    'densenet121': '../input/hemoorhagesubsfirststage/densenet121_stage1.csv',
}

stage1 = pd.read_csv('../input/hemorrhagehelpers/stage_1_sample_submission.csv').sort_values('ID')
stage1['image_id'] = stage1.ID.apply(lambda x: '_'.join(x.split('_')[:-1]))
stage1['group_id'] = stage1.ID.apply(lambda x: x.split('_')[-1])
stage1.drop('Label', 1, inplace=True)
helpers_cols = ['image_id', 'group_id']
stage1.set_index('ID', inplace=True)
for name, fpath in tqdm(stage1_names.items()):
    add_table(stage1, fpath, name)

stage1.head()


# In[3]:


current_leaderboard = eval_df(stage1.drop(helpers_cols, 1))
current_leaderboard


# In[4]:


stage1_meta_df = pd.read_csv('../input/hemoorrhagemetadata/stage_1_test_meta.csv')

stage1_meta_df['Axial'] = stage1_meta_df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))
stage1_meta_df = stage1_meta_df.sort_values(['StudyInstanceUID', 'Axial']).reset_index(drop=True)
stage1_meta_df = stage1_meta_df[['SOPInstanceUID', 'StudyInstanceUID', 'Axial']]

stage1_meta_df['InstancePosition'] = stage1_meta_df.groupby(['StudyInstanceUID']).Axial.cumcount()

stage1_meta_df['PrevPositionUID'] = stage1_meta_df.SOPInstanceUID.shift(1)
stage1_meta_df.loc[stage1_meta_df.InstancePosition==0, 'PrevPositionUID'] = None

stage1_meta_df['NextPosition'] = stage1_meta_df.InstancePosition.shift(-1)
stage1_meta_df['NextPositionUID'] = stage1_meta_df.SOPInstanceUID.shift(-1)
stage1_meta_df.loc[stage1_meta_df.NextPosition==0, 'NextPositionUID'] = None

stage1_meta_df.drop('NextPosition', 1, inplace=True)

stage1_meta_df.columns = ['image_id'] + list(stage1_meta_df.columns[1:])
stage1_meta_df.iloc[28:37]


# In[5]:


def get_channel_postproc(test_df, group_name, w_next, w_prev, levels=1):
    sample = test_df[test_df.group_id==group_name].copy()
    w_center = 1 - w_next - w_prev
    for lvl in range(levels):
        label_mapper = sample.set_index('ID').Label
        next_labels = (sample.NextPositionUID + '_' + sample.group_id).map(
            label_mapper).fillna(0)
        prev_labels = (sample.PrevPositionUID + '_' + sample.group_id).map(
            label_mapper).fillna(0)
        sample.Label = sample.Label * w_center +             next_labels * w_next +             prev_labels * w_prev
    return sample[['image_id', 'Label']].set_index('image_id')

def apply_triplets(df, class_triplets):
    channel_df = pd.DataFrame(index=df.image_id.unique())
    for current_class, current_triplet in class_triplets.items():
        channel_df[current_class] = get_channel_postproc(df, current_class, *current_triplet)
    channel_df.reset_index(inplace=True)
    channel_df = channel_df.melt(id_vars=['index'])
    channel_df['ID'] = channel_df['index'] + '_' + channel_df.variable
    channel_df['Label'] = channel_df['value']
    channel_df['image_id'] = channel_df['index']
    channel_df = channel_df[['ID', 'Label', 'image_id']]
    return channel_df.sort_values('ID').reset_index(drop=True)

def add_custom_postproc(df, col, test_meta_df, used_triplets, postfix='postproc'):
    sub = df[['image_id', 'group_id', col]].reset_index()
    sub.columns = ['ID', 'image_id', 'group_id', 'Label']
    merged = pd.merge(sub, test_meta_df, on='image_id', how='left')
    postprocessed = apply_triplets(merged, used_triplets)
    
    postproc_name = '{}_{}'.format(col, postfix)
    df[postproc_name] = postprocessed.set_index('ID').Label

optimal_triplets = {
    'any': (0.1, 0.1, 3), 
    'epidural': (0.2, 0.2, 3),
    'intraparenchymal': (0.05, 0.05, 4), 
    'intraventricular': (0.05, 0.05, 3),
    'subarachnoid': (0.2, 0.2, 2), 
    'subdural': (0.2, 0.2, 3)
}


# In[6]:


current_leaderboard = eval_df(stage1.drop(helpers_cols, 1))
current_leaderboard


# In[7]:


postproc_cols = [col for col in stage1.columns if col not in helpers_cols]
for col in postproc_cols:
    print('Make postprocessing for {}'.format(col))
#     used_triplets = add_autopostproc(stage1, col, stage1_meta_df, triplet_search_space)
    add_custom_postproc(stage1, col, stage1_meta_df, optimal_triplets)


# In[8]:


updated_leaderboard = eval_df(stage1.drop(helpers_cols, 1))
updated_leaderboard


# In[9]:


stage2_names = {
    'takato_original_seresnext': '../input/hemorrhagesubssecondstage/se_resnext50_32x4d_102105_hflip.csv',
    'takato_original_effb1': '../input/hemorrhagesubssecondstage/efficientnet-b1_102407_hflip.csv',
    'takato_original_effb2': '../input/hemorrhagesubssecondstage/efficientnet-b2_102000_hflip.csv',
    'takato_original_effb3': '../input/hemorrhagesubssecondstage/efficientnet-b3_102112_hflip.csv',
    'takato_original_effb4': '../input/hemorrhagesubssecondstage/efficientnet-b4_102100_hflip.csv',
    
    'agro_b7': '../input/hemorrhagesubssecondstage/agroeffnet_stage2_test_tta10.csv',
    'sneddy_b7': '../input/hemorrhagesubssecondstage/effnet-b7-new_stage2_test_tta10.csv',
    'resnet152': '../input/hemorrhagesubssecondstage/resnet152_stage2.csv',
    'effnet_b5': '../input/hemorrhagesubssecondstage/effnetb5_stage2_test_tta8.csv',
    'balanced_effb4': '../input/hemorrhagesubssecondstage/balansed_effb4_stage2.csv',
    'inceptionv4': '../input/hemorrhagesubssecondstage/inceptionv4_stage2.csv',
    'effnet_b0': '../input/hemorrhagesubssecondstage/effb0_stage2.csv',
    'densenet121': '../input/hemorrhagesubssecondstage/densenet121_stage2.csv',
}

stage2 = pd.read_csv('../input/hemorrhagehelpers/stage_2_sample_submission.csv').sort_values('ID')
stage2['image_id'] = stage2.ID.apply(lambda x: '_'.join(x.split('_')[:-1]))
stage2['group_id'] = stage2.ID.apply(lambda x: x.split('_')[-1])

stage2.drop('Label', 1, inplace=True)
helpers_cols = ['image_id', 'group_id']
stage2.set_index('ID', inplace=True)
for name, fpath in tqdm(stage2_names.items()):
    add_table(stage2, fpath, name)

stage2.head()


# In[10]:


stage2_meta_df = pd.read_csv('../input/hemoorrhagemetadata/stage_2_test_meta.csv')

stage2_meta_df['Axial'] = stage2_meta_df['ImagePositionPatient'].apply(lambda s: float(s.split('\'')[-2]))
stage2_meta_df = stage2_meta_df.sort_values(['StudyInstanceUID', 'Axial']).reset_index(drop=True)
stage2_meta_df = stage2_meta_df[['SOPInstanceUID', 'StudyInstanceUID', 'Axial']]

stage2_meta_df['InstancePosition'] = stage2_meta_df.groupby(['StudyInstanceUID']).Axial.cumcount()

stage2_meta_df['PrevPositionUID'] = stage2_meta_df.SOPInstanceUID.shift(1)
stage2_meta_df.loc[stage2_meta_df.InstancePosition==0, 'PrevPositionUID'] = None

stage2_meta_df['NextPosition'] = stage2_meta_df.InstancePosition.shift(-1)
stage2_meta_df['NextPositionUID'] = stage2_meta_df.SOPInstanceUID.shift(-1)
stage2_meta_df.loc[stage2_meta_df.NextPosition==0, 'NextPositionUID'] = None

stage2_meta_df.drop('NextPosition', 1, inplace=True)

stage2_meta_df.columns = ['image_id'] + list(stage2_meta_df.columns[1:])
stage2_meta_df.iloc[27:37]


# In[11]:


postproc_cols = [col for col in stage2.columns if col not in helpers_cols]
for col in postproc_cols:
    print('Make postprocessing for {}'.format(col))
#     used_triplets = add_autopostproc(stage1, col, stage1_meta_df, triplet_search_space)
    add_custom_postproc(stage2, col, stage2_meta_df, optimal_triplets)


# In[12]:


stage2.to_csv('updated_stage_2.csv', index=False)

