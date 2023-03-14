#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastai
from fastai.vision import *


# In[2]:


work_dir = Path('/kaggle/working/')
path = Path('../input')


# In[3]:


train = 'train_images/train_images'
test =  path/'leaderboard_test_data/leaderboard_test_data'
holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'
sample_sub = path/'SampleSubmission.csv'
labels = path/'traininglabels.csv'


# In[4]:


df = pd.read_csv(labels)
df_sample = pd.read_csv(sample_sub)


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df[df['score']<0.75]


# In[8]:


(df.has_oilpalm==1).sum()


# In[9]:


test_names = [f for f in test.iterdir()]
holdout_names = [f for f in holdout.iterdir()]


# In[10]:


src = (ImageItemList.from_df(df, path, folder=train)
      .random_split_by_pct(0.2, seed=2019)
      .label_from_df('has_oilpalm')
      .add_test(test_names+holdout_names))


# In[11]:


data =  (src.transform(get_transforms(), size=128)
         .databunch(bs=64)
         .normalize(imagenet_stats))


# In[12]:


data.show_batch(3, figsize=(10,7))


# In[13]:


#This was working perfectly some minutes ago!
from sklearn.metrics import roc_auc_score
def auc_score(preds,targets):
    return torch.tensor(roc_auc_score(targets,preds[:,1]))


# In[14]:


learn = create_cnn(data, models.resnet18, 
                   metrics=[accuracy], #<---add aoc metric?
                   model_dir='/kaggle/working/models')


# In[15]:


learn.lr_find(); learn.recorder.plot()


# In[16]:


lr = 1e-2


# In[17]:


learn.fit_one_cycle(6, lr)


# In[18]:


Then we unfreeze and train the whole model, with lower lr.


# In[19]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-3))


# In[20]:


learn.save('128')


# In[21]:


p,t = learn.get_preds()
auc_score(p,t)


# In[22]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[23]:


interp.plot_top_losses(9, figsize=(15,11))


# In[24]:


p,t = learn.get_preds(ds_type=DatasetType.Test)


# In[25]:


p = to_np(p); p.shape


# In[26]:


ids = np.array([f.name for f in (test_names+holdout_names)]);ids.shape


# In[27]:


#We only recover the probs of having palmoil (column 1)
sub = pd.DataFrame(np.stack([ids, p[:,1]], axis=1), columns=df_sample.columns)


# In[28]:


sub.to_csv(work_dir/'sub.csv', index=False)


# In[29]:




