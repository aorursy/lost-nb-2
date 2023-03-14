#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import json
import ast
import seaborn as sns
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split

from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('ggplot')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# In[2]:


get_ipython().system('ls -GFlash --color ../input/stanford-covid-vaccine/')


# In[3]:


get_ipython().system('ls -GFlash --color ../input/stanford-covid-vaccine/bpps/ | head')


# In[4]:


get_ipython().system('du -h ../input/stanford-covid-vaccine/bpps/')


# In[5]:


train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
print(f'Train shape: {train.shape}, test shape: {test.shape}, sample submission shape: {ss.shape}')

print('========= train columns ==========')
print([c for c in train.columns])

print('========= test columns ==========')
print([c for c in test.columns])


# In[6]:


bpps_files = os.listdir('../input/stanford-covid-vaccine/bpps/')
example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_files[0]}')
print('bpps file shape:', example_bpps.shape)


# In[7]:


plt.style.use('default')
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
axs = axs.flatten()
for i, f in enumerate(bpps_files):
    if i == 25:
        break
    example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{f}')
    axs[i].imshow(example_bpps)
    axs[i].set_title(f)
plt.tight_layout()
plt.show()


# In[8]:


ss.head()


# In[9]:


train['reactivity'].head()


# In[10]:


print('===== Example Train Reacivity ======')
print([round(r, 2) for r in train['reactivity'][0]])


# In[11]:


print('===== Example Train deg_Mg_pH10 value ======')
print([round(r, 2) for r in train['deg_Mg_pH10'][0]])


# In[12]:


print('===== Example Train deg_Mg_50C value ======')
print([round(r, 2) for r in train['deg_Mg_50C'][0]])


# In[13]:


plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 3))
ax = sns.distplot(train['signal_to_noise'])
ax.set_title('Signal to Noise feature (train)')
plt.show()


# In[14]:


test['seq_length'].value_counts()     .plot(kind='bar', figsize=(10, 4),
          color=color_pal[4],
         title='Sequence Length in public test set')
plt.show()


# In[15]:


fig, axs = plt.subplots(3, 1,
                        figsize=(10, 6),
                        sharex=True)
axs = axs.flatten()
train['mean_reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))
train['mean_deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))
train['mean_deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))

train['mean_reactivity']     .plot(kind='hist',
          bins=50,
          color=color_pal[0],
          title='Distribution of Mean Reactivity in training set',
         ax=axs[0])
train['mean_deg_Mg_pH10']     .plot(kind='hist',
          bins=50,
          ax=axs[1],
          color=color_pal[4],
          title='Distribution of Mean deg_Mg_pH10 in training set')
train['mean_deg_Mg_50C']     .plot(kind='hist',
          bins=50,
          ax=axs[2],
          color=color_pal[3],
          title='Distribution of Mean deg_Mg_50C in training set')
plt.tight_layout()
plt.show()


# In[16]:


mean_react = train['mean_reactivity'].mean()
mean_deg_Mg_pH10 = train['mean_deg_Mg_pH10'].mean()
mean_deg_Mg_50C = train['mean_deg_Mg_50C'].mean()

ss['reactivity'] = mean_react
ss['deg_Mg_pH10'] = mean_deg_Mg_pH10
ss['deg_Mg_50C'] = mean_deg_Mg_50C

ss.to_csv('submission.csv', index=False)
ss.head()


# In[17]:


# Split the 68 Reactivity values each into it's own column
for n in range(68):
    train[f'reactivity_{n}'] = train['reactivity'].apply(lambda x: x[n])
    
REACTIVITY_COLS = [r for r in train.columns if 'reactivity_' in r and 'error' not in r]

ax = train.set_index('id')[REACTIVITY_COLS]     .T     .plot(color='black',
          alpha=0.01,
          ylim=(-0.5, 5),
          title='reactivity of training set',
          figsize=(15, 5))
ax.get_legend().remove()


# In[18]:


for n in range(68):
    train[f'deg_Mg_pH10_{n}'] = train['deg_Mg_pH10'].apply(lambda x: x[n])
    
DEG_MG_PH10_COLS = [r for r in train.columns if 'deg_Mg_pH10_' in r and 'error' not in r]

ax = train.set_index('id')[DEG_MG_PH10_COLS]     .T     .plot(color='c',
          alpha=0.01,
          ylim=(-0.5, 5),
          title='Deg Mg Ph10 of training set',
          figsize=(15, 5))
ax.get_legend().remove()


# In[19]:


for n in range(68):
    train[f'deg_Mg_50C_{n}'] = train['deg_Mg_50C'].apply(lambda x: x[n])
    
DEG_MG_50C_COLS = [r for r in train.columns if 'deg_Mg_50C_' in r and 'error' not in r]

ax = train.set_index('id')[DEG_MG_50C_COLS]     .T     .plot(color='m',
          alpha=0.2,
          ylim=(-2, 7),
          title='Deg Mg 50C of training set',
          figsize=(15, 5)
         )
ax.get_legend().remove()


# In[20]:


sns.pairplot(data=train,
             vars=['mean_reactivity',
                   'mean_deg_Mg_pH10',
                    'mean_deg_Mg_50C'],
            hue='SN_filter')
plt.show()


# In[21]:


# Expand Sequence Features
for n in range(107):
    train[f'sequence_{n}'] = train['sequence'].apply(lambda x: x[n]).astype('category')
    test[f'sequence_{n}'] = test['sequence'].apply(lambda x: x[n]).astype('category')

SEQUENCE_COLS = [c for c in train.columns if 'sequence_' in c]

for target in ['reactivity','deg_Mg_pH10','deg_Mg_50C']:

    X = train[SEQUENCE_COLS]
    y = train[f'mean_{target}']
    X_test = test[SEQUENCE_COLS]

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    reg = lgb.LGBMRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=(X_val, y_val),
           early_stopping_rounds=100,
           verbose=100)

    test[f'mean_{target}_pred'] = reg.predict(X_test)


# In[22]:


ss['id'] = 'id_' + ss['id_seqpos'].str.split('_', expand=True)[1]

# Merge my predicted average values
ss_new = ss.     drop(['reactivity','deg_Mg_pH10','deg_Mg_50C'], axis=1)     .merge(test[['id',
               'mean_reactivity_pred',
               'mean_deg_Mg_pH10_pred',
               'mean_deg_Mg_50C_pred']] \
               .rename(columns={'mean_reactivity_pred' : 'reactivity',
                                'mean_deg_Mg_pH10_pred': 'deg_Mg_pH10',
                                'mean_deg_Mg_50C_pred' : 'deg_Mg_50C'}
                      ),
         on='id',
        validate='m:1')


# In[23]:


TARGETS = ['reactivity','deg_Mg_pH10','deg_Mg_50C']
for i, t in enumerate(TARGETS):
    ss_new[t].plot(kind='hist',
                              figsize=(10, 3),
                              bins=100,
                              color=color_pal[i*3],
                              title=f'Submission {t}')
    plt.show()


# In[24]:


ss_new.sample(10)


# In[25]:


# Make Submission
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
ss_new[ss.columns].to_csv('submission_lgbm_v1.csv', index=False)


# In[26]:


# Expand Sequence Features
for n in range(107):
    train[f'structure_{n}'] = train['structure'].apply(lambda x: x[n]).astype('category')
    test[f'structure_{n}'] = test['structure'].apply(lambda x: x[n]).astype('category')
    train[f'predicted_loop_type_{n}'] = train['predicted_loop_type'].apply(lambda x: x[n]).astype('category')
    test[f'predicted_loop_type_{n}'] = test['predicted_loop_type'].apply(lambda x: x[n]).astype('category')
    train[f'sequence_{n}'] = train['sequence'].apply(lambda x: x[n]).astype('category')
    test[f'sequence_{n}'] = test['sequence'].apply(lambda x: x[n]).astype('category')

SEQUENCE_COLS = [c for c in train.columns if 'sequence_' in c]
STRUCTURE_COLS = [c for c in train.columns if 'structure_' in c]
PLT_COLS = [c for c in train.columns if 'predicted_loop_type_' in c]

for target in ['reactivity','deg_Mg_pH10','deg_Mg_50C']:

    X = train[SEQUENCE_COLS + STRUCTURE_COLS + PLT_COLS]
    y = train[f'mean_{target}']
    X_test = test[SEQUENCE_COLS + STRUCTURE_COLS + PLT_COLS]

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    reg = lgb.LGBMRegressor(n_estimators=10000,
                            learning_rate=0.001,
                            feature_fraction=0.8)
    reg.fit(X_train, y_train,
            eval_set=(X_val, y_val),
           early_stopping_rounds=100,
           verbose=1000)

    test[f'mean_{target}_pred'] = reg.predict(X_test)
    
ss['id'] = 'id_' + ss['id_seqpos'].str.split('_', expand=True)[1]

# Merge my predicted average values
ss_new = ss.     drop(['reactivity','deg_Mg_pH10','deg_Mg_50C'], axis=1)     .merge(test[['id',
               'mean_reactivity_pred',
               'mean_deg_Mg_pH10_pred',
               'mean_deg_Mg_50C_pred']] \
               .rename(columns={'mean_reactivity_pred' : 'reactivity',
                                'mean_deg_Mg_pH10_pred': 'deg_Mg_pH10',
                                'mean_deg_Mg_50C_pred' : 'deg_Mg_50C'}
                      ),
         on='id',
        validate='m:1')

ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
ss_new[ss.columns].to_csv('submission.csv', index=False)

TARGETS = ['reactivity','deg_Mg_pH10','deg_Mg_50C']
for i, t in enumerate(TARGETS):
    ss_new[t].plot(kind='hist',
                              figsize=(10, 3),
                              bins=100,
                              color=color_pal[i*3],
                              title=f'Submission {t}')
    plt.show()


# In[27]:


fig, axs = plt.subplots(5, 5, figsize=(12, 10))
axs = axs.flatten()
i = 0
for row in train.sample(25, random_state=42).iterrows():
    myid = row[1]['id']
    reactivity_array = row[1][REACTIVITY_COLS].values
    sns.regplot(np.array(range(68)).reshape(-1, 1),
                reactivity_array,
                ax=axs[i],
                color=next(color_cycle))
    axs[i].set_title(myid)
    i += 1
fig.suptitle('Reactivity Array for 25 Train Examples with Regression Line',
             fontsize=18,
             y=1.02)
plt.tight_layout()
plt.show()


# In[28]:


fig, axs = plt.subplots(5, 5,
                        figsize=(12, 10),
                       sharex=True)
axs = axs.flatten()
i = 0
for row in train.sample(25, random_state=42).iterrows():
    myid = row[1]['id']
    reactivity_array = row[1][DEG_MG_50C_COLS].values
    sns.regplot(np.array(range(68)).reshape(-1, 1),
                reactivity_array,
                ax=axs[i],
                color=next(color_cycle))
    axs[i].set_title(myid)
    i += 1
fig.suptitle('"DEG_MG_50C" Array for 25 Train Examples with Regression Line',
             fontsize=18,
             y=1.02)
plt.tight_layout()
plt.show()


# In[29]:


fig, axs = plt.subplots(5, 5, figsize=(12, 10))
axs = axs.flatten()
i = 0
for row in train.sample(25, random_state=42).iterrows():
    myid = row[1]['id']
    reactivity_array = row[1][DEG_MG_PH10_COLS].values
    sns.regplot(np.array(range(68)).reshape(-1, 1),
                reactivity_array,
                ax=axs[i],
                color=next(color_cycle))
    axs[i].set_title(myid)
    i += 1
fig.suptitle('"DEG_MG_PH10" Array for 25 Train Examples with Regression Line',
             fontsize=18,
             y=1.02)
plt.tight_layout()
plt.show()


# In[30]:


import pandas as pd
import numpy as np
import json
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pylab as plt

def expand_columns(df):
    df = df.copy()
    df = df.drop('index', axis=1)
    max_seq_length = df['seq_length'].max()
    SEQUENCE_COLS = []; STRUCTURE_COLS = []; PRED_LOOP_TYPE_COLS = []
    for s in range(130):
        df[f'sequence_{s}'] = df['sequence'].str[s]
        df[f'structure_{s}'] = df['structure'].str[s]
        df[f'predicted_loop_type_{s}'] = df['predicted_loop_type'].str[s]
        SEQUENCE_COLS.append(f'sequence_{s}')
        STRUCTURE_COLS.append(f'structure_{s}')
    return df, SEQUENCE_COLS, STRUCTURE_COLS

def parse_sample_submission(ss):
    ss = ss.copy()
    ss['id'] = ss['id_seqpos'].str.split('_', expand=True)[1]
    ss['seqpos'] = ss['id_seqpos'].str.split('_', expand=True)[2].astype('int')
    return ss


# In[31]:


def get_train_long(train):
    dfs = []

    def pad(feat, tolen):
        padded = np.pad(feat,
                        (0, tolen-len(feat)),
                        mode='constant',
                        constant_values=np.nan)
        return padded

    for d in tqdm(train.itertuples(), total=len(train)):
        sequence = [s for s in d[3]]
        seq_len = len(sequence)
        structure = [s for s in d[4]]
        predicted_loop_type = [s for s in d[5]]
        reactivity_error = pad([s for s in d[10]], seq_len)
        deg_error_Mg_pH10 = pad([s for s in d[11]], seq_len)
        deg_error_pH10 = pad([s for s in d[12]], seq_len)
        deg_error_Mg_50C = pad([s for s in d[13]], seq_len)
        deg_error_50C = pad([s for s in d[14]], seq_len)

        reactivity = pad([s for s in d[15]], seq_len)
        deg_Mg_pH10 = pad([s for s in d[16]], seq_len)
        deg_pH10 = pad([s for s in d[17]], seq_len)
        deg_Mg_50C = pad([s for s in d[18]], seq_len)
        deg_50C = pad([s for s in d[10]], seq_len)
        myid = [d[2]] * len(sequence)
        seqpos = [c for c in range(len(sequence))]
        dfs.append(pd.DataFrame(np.array([myid,
                                          seqpos,
                                          sequence,
                                          structure,
                                          predicted_loop_type,
                                          reactivity_error,
                                          deg_error_Mg_pH10,
                                          deg_error_pH10,
                                          deg_error_Mg_50C,
                                          deg_error_50C,
                                          reactivity,
                                          deg_Mg_pH10,
                                          deg_pH10,
                                          deg_Mg_50C,
                                         ]).T))
    train_long = pd.concat(dfs)

    train_long.columns=['id',
               'seqpos',
               'sequence',
               'structure',
               'predicted_loop_type',
               'reactivity_error',
               'deg_error_Mg_pH10',
               'deg_error_pH10',
               'deg_error_Mg_50C',
               'deg_error_50C',
               'reactivity',
               'deg_Mg_pH10',
               'deg_pH10',
               'deg_Mg_50C']

    return train_long


def get_test_long(test):
    dfs = []

    def pad(feat, tolen):
        padded = np.pad(feat,
                        (0, tolen-len(feat)),
                        mode='constant',
                        constant_values=np.nan)
        return padded

    for d in tqdm(test.itertuples(), total=len(test)):
        sequence = [s for s in d[3]]
        seq_len = len(sequence)
        structure = [s for s in d[4]]
        predicted_loop_type = [s for s in d[5]]
        myid = [d[2]] * len(sequence)
        seqpos = [c for c in range(len(sequence))]
        dfs.append(pd.DataFrame(np.array([myid,
                                          seqpos,
                                          sequence,
                                          structure,
                                          predicted_loop_type,
                                         ]).T))
    test_long = pd.concat(dfs)

    test_long.columns=['id',
               'seqpos',
               'sequence',
               'structure',
               'predicted_loop_type']

    return test_long

def add_long_features(df):
    df = df.copy()
    df['seqpos'] = df['seqpos'].astype('int')
    df = df.merge(df.query('seqpos <= 106')                     .groupby('id')['sequence']                       .value_counts()                       .unstack()                       .reset_index(),
             how='left',
             on=['id'],
             validate='m:1'
            )
    
    df = df.merge(df.query('seqpos <= 106')                   .groupby('id')['structure']                       .value_counts()                       .unstack()                       .reset_index(),
             how='left',
             on=['id'],
             validate='m:1'
            )

    df = df.merge(df.query('seqpos <= 106')                   .groupby('id')['predicted_loop_type']                       .value_counts()                       .unstack()                       .reset_index(),
             how='left',
             on=['id'],
             validate='m:1'
            )
    for shift in [-5, -4, -3, -2 -1, 1, 2, 3, 4, 5]:
        for f in ['sequence','structure','predicted_loop_type']:
            df[f'{f}_shift{shift}'] = df.groupby('id')[f].shift(shift)
    return df


# In[32]:


def make_feature_types(df, features):
    df = df.copy()
    df = df.replace('nan', np.nan)
    for f in features:
        try:
            df[f] = pd.to_numeric(df[f])
        except ValueError:
            df[f] = df[f].astype('category')
    return df


# In[33]:


train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

train_expanded, SEQUENCE_COLS, STRUCTURE_COLS = expand_columns(train)
test_expanded, SEQUENCE_COLS, STRUCTURE_COLS = expand_columns(test)
ss = parse_sample_submission(ss)

train_long = get_train_long(train)
test_long = get_test_long(test)

train_long = add_long_features(train_long)
test_long = add_long_features(test_long)

FEATURES = ['seqpos',
            'sequence',
            'structure',
            'predicted_loop_type',
            'A', 'C', 'G', 'U', '(', ')', '.', 'B', 'E',
            'H', 'I', 'M', 'S', 'X',
            'sequence_shift-5', 'structure_shift-5',
            'predicted_loop_type_shift-5', 'sequence_shift-4', 'structure_shift-4',
            'predicted_loop_type_shift-4', 'sequence_shift-3', 'structure_shift-3',
            'predicted_loop_type_shift-3', 'sequence_shift1', 'structure_shift1',
            'predicted_loop_type_shift1', 'sequence_shift2', 'structure_shift2',
            'predicted_loop_type_shift2', 'sequence_shift3', 'structure_shift3',
            'predicted_loop_type_shift3', 'sequence_shift4', 'structure_shift4',
            'predicted_loop_type_shift4', 'sequence_shift5', 'structure_shift5',
            'predicted_loop_type_shift5']

train_long = make_feature_types(train_long, FEATURES)
test_long = make_feature_types(test_long, FEATURES)

train_ids, val_ids = train_test_split(train['id'].unique())

TARGETS = ['reactivity','deg_Mg_pH10','deg_Mg_50C']
fis = []
for t in TARGETS:
    print(f'==== Running for target {t} ====')
    X_train = train_long.dropna(subset=[t]).loc[train_long['id'].isin(train_ids)][FEATURES].copy()
    y_train = train_long.dropna(subset=[t]).loc[train_long['id'].isin(train_ids)][t].copy()
    X_val = train_long.dropna(subset=[t]).loc[train_long['id'].isin(val_ids)][FEATURES].copy()
    y_val = train_long.dropna(subset=[t]).loc[train_long['id'].isin(val_ids)][t].copy()
    X_test = test_long[FEATURES].copy()
    y_train = pd.to_numeric(y_train)
    y_val = pd.to_numeric(y_val)
    
    reg = lgb.LGBMRegressor(n_estimators=10000,
                            learning_rate=0.01,
                            importance_type='gain')
    reg.fit(X_train, y_train,
            eval_set=(X_val, y_val),
           verbose=1000,
           early_stopping_rounds=500)

    fi_df = pd.DataFrame(index=FEATURES, 
                 data=reg.feature_importances_,
                 columns=[f'importance_{t}'])
    
    fi_df.sort_values(f'importance_{t}')         .plot(kind='barh', figsize=(8, 15), title=t)
    plt.show()
    fis.append(fi_df)
    
    test_long[f'{t}_pred'] = reg.predict(X_test)


# In[34]:


test_long['id_seqpos'] = test_long['id'] + '_' + test_long['seqpos'].astype('str')

test_long['deg_pH10'] = 0
test_long['deg_50C'] = 0
test_long = test_long.rename(columns={'reactivity_pred':'reactivity',
                          'deg_Mg_pH10_pred': 'deg_Mg_pH10',
                          'deg_Mg_50C_pred': 'deg_Mg_50C'})

ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
assert test_long[ss.columns].shape == ss.shape

test_long[ss.columns].to_csv('submission.csv', index=False)


# In[35]:


for t in TARGETS:
    train_long[t].dropna().astype('float').plot(kind='hist', bins=50, figsize=(10, 3), title=t)
    test_long[t].plot(kind='hist', bins=50, figsize=(10, 3), title=t)
    plt.show()


# In[ ]:




