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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


DATA_DIR='/kaggle/input/data-science-bowl-2019'


# In[3]:


train = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))


# In[4]:


# Recreate the train_labels.csv file for episodes in the training data

def extract_accuracy_group(df: pd.DataFrame) -> pd.DataFrame:
    # Regex strings for matching Assessment Types
    assessment_4100 = '|'.join(['Mushroom Sorter',
                                'Chest Sorter',
                                'Cauldron Filler',
                                'Cart Balancer'])
    assessment_4110 = 'Bird Measurer'
    
    # 1. Extract all assessment scoring events
    score_events = df[((df['title'].str.contains(assessment_4110)) & (df['event_code']==4110)) |                      ((df['title'].str.contains(assessment_4100)) & (df['event_code']==4100))]
    
    # 2. Count number of correct vs. attempts
    # 2.a. Create flags for correct vs incorrect
    score_events['num_correct'] = 1
    score_events['num_correct'] = score_events['num_correct'].where(score_events['event_data'].str.contains('"correct":true'),other=0)
    
    score_events['num_incorrect'] = 1
    score_events['num_incorrect'] = score_events['num_incorrect'].where(score_events['event_data'].str.contains('"correct":false'),other=0)
    
    # 2.b. Aggregate by `installation_id`,`game_session`,`title`
    score_events_sum = score_events.groupby(['installation_id','game_session','title'])['num_correct','num_incorrect'].sum()
    
    # 3. Apply heuristic to convert counts into accuracy group
    # 3.a. Define heuristic
    def acc_group(row: pd.Series) -> int:
        if row['num_correct'] == 0:
            return 0
        elif row['num_incorrect'] == 0:
            return 3
        elif row['num_incorrect'] == 1:
            return 2
        else:
            return 1
        
    # 3.b. Apply heuristic to count data
    score_events_sum['accuracy_group'] = score_events_sum.apply(acc_group,axis=1)
    
    return score_events_sum.reset_index()


# In[5]:


test_labels = extract_accuracy_group(test)


# In[6]:


def build_episode_game_sessions(df):
    return pd.DataFrame(index=df['game_session'])

def build_starts_only(df):
#     return df[df['event_code']==2000]
    return df.groupby(['installation_id','game_session']).last()

def build_start_end_times(df, df_labels):
    start_end_times = pd.merge(left=df, right=df_labels,left_on='game_session', right_index=True)        .groupby(['installation_id','game_session'])        .first()['timestamp']
    start_end_times = start_end_times.reset_index().sort_values(by=['installation_id','timestamp'])
    start_end_times.columns = ['installation_id','episode_session','end_time']
    start_end_times['start_time'] = start_end_times.groupby('installation_id')['end_time'].shift(1,fill_value='2018-09-11T18:56:11.918Z')
    return start_end_times

def append_times_to_labels(labels, start_end_times):
    new_labels = pd.merge(left=labels,
                          right=start_end_times,
                          left_on=['game_session','installation_id'],
                          right_on=['episode_session','installation_id'])
    return new_labels.drop('game_session',axis=1)

def add_labels_to_sessions(sessions, labels_with_times):
    outer = pd.merge(left=sessions.reset_index(),
                     right=labels_with_times,
                     left_on='installation_id',
                     right_on='installation_id',
                     suffixes=('','_episode') )
    labeled_sessions = outer[(outer['timestamp']>=outer['start_time']) & (outer['timestamp']<=outer['end_time'])]
    
    labeled_session_ids = pd.DataFrame(index=labeled_sessions['game_session'],
                                       data=np.ones(len(labeled_sessions)),
                                       columns=['has_label'])
    unlabeled_sessions = pd.merge(sessions.reset_index(),labeled_session_ids,how='left',left_on='game_session',right_index=True)
    unlabeled_sessions = unlabeled_sessions[unlabeled_sessions['has_label']!=1].drop('has_label',axis=1)
    return labeled_sessions, unlabeled_sessions

def build_session_labels(events, labels):
    episodes_game_sessions = build_episode_game_sessions(labels)
    starts = build_starts_only(events)
    start_end_times = build_start_end_times(starts, episodes_game_sessions)
    labels_with_times = append_times_to_labels(labels, start_end_times)
    return add_labels_to_sessions(starts, labels_with_times)


# In[7]:


train_bm_starts, train_no_asmt = build_session_labels(train, train_labels)


# In[8]:


test_bm_starts, test_to_predict = build_session_labels(test, test_labels)
bm_starts = pd.concat([train_bm_starts, test_bm_starts], sort=True)


# In[9]:


idx_to_titles = list(set(bm_starts['title'])) + ['<START>','<END>']
titles_to_idx = {title: idx for idx, title in enumerate(idx_to_titles)}
bm_starts['title_idx'] = bm_starts['title'].map(titles_to_idx)


# In[10]:


idx_to_asmt = list(set(bm_starts['title_episode']))
asmt_to_idx = {title: idx for idx, title in enumerate(idx_to_asmt)}
bm_starts['asmt_idx'] = bm_starts['title_episode'].map(asmt_to_idx)


# In[11]:


asmt_to_idx


# In[12]:


bm_session_columns = ['installation_id','episode_session',
        'accuracy_group', 'asmt_idx']
bm_episodes = bm_starts.groupby(bm_session_columns)['title_idx'].aggregate(lambda x: [len(idx_to_titles)-2] + list(x) ).reset_index()


# In[13]:


bm_episodes.head()


# In[14]:


num_asmt, num_scores, num_titles = len(idx_to_asmt), 4, len(idx_to_titles)
transition_matrix = np.ones((num_asmt, num_scores, num_titles, num_titles),dtype=np.float32)

asmt_idx = list(bm_episodes['asmt_idx'])
accuracy_group = list(bm_episodes['accuracy_group'])
paths = list(bm_episodes['title_idx'])

for asmt, acc, path in zip(asmt_idx, accuracy_group, paths):
    for prior, current in zip(path[:-1],path[1:]):
        transition_matrix[asmt, acc, prior, current] += 1
        
transition_matrix = transition_matrix / transition_matrix.sum(axis=-1).reshape(*transition_matrix.shape[:-1],1)

He we calculate a prior probability of a given `accuracy_group` for each assessment type.
# In[15]:


def build_priors(sessions):
    by_asmt_by_acc = sessions.groupby(['asmt_idx','accuracy_group'])['episode_session'].count().unstack(-1).values
    by_asmt = np.sum(by_asmt_by_acc,axis=-1).reshape(-1,1)
    return by_asmt_by_acc/by_asmt

priors = build_priors(bm_episodes)


# In[16]:


from sklearn.metrics import cohen_kappa_score

class PathNaiveBayes:
    def __init__(self, priors_, transitions_, start_, end_):
        self.priors = priors_
        self.transitions = transitions_
        self.start = start_
        self.end = end_
        
    def predict(self, asmt, seq):
        # Check if the provide sequence has a start code
        if seq[0] != self.start:
            seq = [self.start] + seq + [self.end]
            
        # Use sum of log probabilities instead of multiply probabilities directly to help avoid underflow
        
        # Initialize with our prior probability
        log_prob = np.log(self.priors[asmt])
        
        # Calculate the Markov chain probability of the user's path
        for prev, current in zip(seq[:-1],seq[1:]):
            log_prob += np.log(self.transitions[asmt,:,prev,current])
            
        # Our prediction is the accuracy score with the highest posterior log probability
        return {"prediction": np.argmax(log_prob), "probabilities": log_prob }
    
    def evaluate(self, asmts, seqs, scores, verbose=False):
        correct, total = np.zeros((self.priors.shape[0])), np.zeros((self.priors.shape[0]))
        preds = []
        for asmt, seq, score in zip(asmts, seqs, scores):
            pred = self.predict(asmt, seq)['prediction']
            if pred == score:
                correct[asmt] += 1
            total[asmt] += 1
            preds.append(pred)
            if verbose:
                print(f"Asmt:{asmt} True:{score} Pred:{pred}")
        acc_per_asmt = correct/total
        total_acc = np.sum(correct)/np.sum(total)
        kappa = cohen_kappa_score(preds,scores,weights='quadratic')
        
        return {'predictions': preds,
               'accuracy_by_asmt': acc_per_asmt,
               'accuracy': total_acc,
               'kappa': kappa}


# In[17]:


model = PathNaiveBayes(priors,transition_matrix, titles_to_idx['<START>'], titles_to_idx['<END>'] )


# In[18]:


## Results on Training Data


# In[19]:


results = model.evaluate(asmts=bm_episodes['asmt_idx'],
               scores=bm_episodes['accuracy_group'],
              seqs=bm_episodes['title_idx'])
results.pop('predictions')
print(results)


# In[20]:


test.set_index(['installation_id','timestamp'])
last_entries = test.assign(rn=test.sort_values(['timestamp'], ascending=False)            .groupby(['installation_id'])            .cumcount() + 1)            .query('rn == 1')            .sort_values(['installation_id'])
last_entries[['installation_id','title']].to_csv('test_final_title.csv',index=False)


# In[21]:


submission = pd.read_csv('test_final_title.csv')
submission['asmt_idx'] = submission['title'].map(asmt_to_idx)
submission_asmt = pd.DataFrame(index=submission['installation_id'], data=submission['asmt_idx'].values, columns=['asmt_idx'])


# In[22]:


test_to_predict['title_idx'] = test_to_predict['title'].map(titles_to_idx)
test_to_predict = pd.merge(left=test_to_predict,right=submission_asmt,left_on='installation_id',right_index=True)
test_to_predict


# In[23]:


test_session_columns = ['installation_id','asmt_idx']
test_episodes = test_to_predict.groupby(test_session_columns)['title_idx'].aggregate(lambda x: [len(idx_to_titles)-2] + list(x) ).reset_index()
test_episodes


# In[24]:


def predict_row(row):
    return model.predict(row['asmt_idx'],row['title_idx'])['prediction']

test_episodes['accuracy_group'] = test_episodes.apply(predict_row,axis=1)
test_episodes


# In[25]:


test_episodes[['installation_id','accuracy_group']].to_csv('submission.csv',index=False)


# In[26]:


test_episodes.groupby(['asmt_idx','accuracy_group'])['installation_id'].count().unstack(-1)

