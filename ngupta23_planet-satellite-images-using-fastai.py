#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *


# In[2]:


verbose = 1


# In[3]:


import os
if verbose >= 1:
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# In[4]:


input_path = '../input/planet-understanding-the-amazon-from-space/'
path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)
if verbose >= 1:
    print(path)


# In[5]:


# Files already exist here, so not need to untar
if verbose >= 2:
    get_ipython().system(' ls -al {input_path}/train-jpg')


# In[6]:


if verbose >= 2:
    get_ipython().system(' ls -al /tmp/.fastai/data/planet/')


# In[7]:


# Copying Data to a writable path since DataBunch will be created from here and the model will be saved here (hence needs write permisions)
# Will take some time
import time
start_time = time.time()
get_ipython().system('cp -r {input_path}train_v2.csv {path}/.')
get_ipython().system('cp -r {input_path}train-jpg {path}/.')
end_time = time.time()
print("Time Taken: {}".format(end_time - start_time))


# In[8]:


df_train = pd.read_csv(path/'train_v2.csv')
df_train.head()


# In[9]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[10]:


if verbose >= 2:
    doc(ImageList.from_csv)


# In[11]:


np.random.seed(42)
src = (ImageList.from_csv(path=path, csv_name='train_v2.csv', folder='train-jpg', suffix='.jpg')  # Where to find the data? -> in path/'train-jpg' folder
       .split_by_rand_pct(0.2) # How to split in train/valid? -> randomly with the default 20% in valid
       .label_from_df(label_delim=' ')) # How to label? -> use the second column of the csv file and split the tags by ' '


# In[12]:


data = (src.transform(tfms, size=128) # Data augmentation? -> use tfms with a size of 128
        .databunch().normalize(imagenet_stats)) # Finally -> use the defaults for conversion to databunch


# In[13]:


data.show_batch(rows=3, figsize=(12,9))


# In[14]:


arch = models.resnet50


# In[15]:


# Kaggle comes with internet off. So have to copy over model to the location where fastai would have downloaded it.
# https://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/23

get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[16]:


# We are looking for multiple labels here, so we look for anything with prob > thresh (you decide what the threshold to use) 
# Can be achieved by creating a partial function --> Create something like the other function with some arguments fixed to certain values
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2) 
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])


# In[17]:


learn.lr_find()
learn.recorder.plot()


# In[18]:


lr = 0.01


# In[19]:


learn.fit_one_cycle(5, slice(lr))


# In[20]:


learn.recorder.plot_losses()


# In[21]:


learn.save('stage-1-rn50')


# In[22]:


learn.unfreeze()


# In[23]:


learn.lr_find()
learn.recorder.plot()


# In[24]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))  # lr is 0.01, so lr/5 is 0.002


# In[25]:


learn.save('stage-2-rn50')


# In[26]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

# Start with the same learner, just replace the data with the new dataset of 256x256 inputs
learn.data = data
data.train_ds[0][0].shape


# In[27]:


learn.freeze()


# In[28]:


learn.lr_find()
learn.recorder.plot()


# In[29]:


lr=1e-2/2


# In[30]:


learn.fit_one_cycle(5, slice(lr))


# In[31]:


learn.save('stage-1-256-rn50')


# In[32]:


learn.unfreeze()


# In[33]:


learn.lr_find()
learn.recorder.plot()


# In[34]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[35]:


learn.recorder.plot_losses()


# In[36]:


learn.save('stage-2-256-rn50')


# In[37]:


learn.export()


# In[38]:


get_ipython().system(' ls ../input/planet-understanding-the-amazon-from-space/')


# In[39]:


# Will take some time
import time
start_time = time.time()
get_ipython().system('cp -r {input_path}test-jpg-v2 {path}/.')
end_time = time.time()
print("Time Taken: {}".format(end_time - start_time))


# In[40]:


test = ImageList.from_folder(path/'test-jpg-v2') #.add(ImageList.from_folder(path/'test-jpg-additional'))
len(test)


# In[41]:


learn = load_learner(path, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[42]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[43]:


labelled_preds[:5]


# In[44]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[45]:


df = pd.DataFrame({'image_name':fnames,
                   'tags':labelled_preds},
                  columns=['image_name', 'tags'])


# In[46]:


df.to_csv('submission.csv', index=False)


# In[47]:


get_ipython().system(' ls ../working')


# In[ ]:




