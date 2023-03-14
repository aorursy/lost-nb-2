#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
from fastai import *
from fastai.vision import *


# In[2]:


import fastai
fastai.__version__


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)


# In[4]:


PATH = Path('../input/aptos2019-blindness-detection')
df_train = pd.read_csv(PATH/"train.csv")            .assign(filename = lambda df: "train_images/" + df.id_code + ".png")
df_test = pd.read_csv(PATH/"test.csv")           .assign(filename = lambda df: "test_images/" + df.id_code + ".png")


# In[5]:


_ = df_train.hist()


# In[6]:


transforms = get_transforms(
    do_flip = True,
    flip_vert = True,
    max_zoom = 1,
    max_rotate = 180, #default 10
    max_lighting = 0.2, #default 0.2
    max_warp = 0.1 #default 0.1
)


# In[7]:


data = ImageDataBunch.from_df(path = "../input/aptos2019-blindness-detection",
                              df = df_train,
                              fn_col = "filename",
                              label_col = "diagnosis",
                              ds_tfms = transforms,
                             size=224)\
        .normalize(imagenet_stats)


# In[8]:


data.show_batch(rows=3, figsize=(7,6))


# In[9]:


# copy pretrained weights for resnet50 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")


# In[10]:


kappa = KappaScore()
kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet50,
                    metrics=[error_rate, kappa],
                    model_dir="/tmp/model/")


# In[11]:


learn.lr_find(end_lr=0.5)
learn.recorder.plot(suggestion=True)


# In[12]:


lr = 7e-3
learn.fit_one_cycle(10, lr)


# In[13]:


learn.recorder.plot_losses()


# In[14]:


learn.unfreeze()


# In[15]:


lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(10,lrs)


# In[16]:


learn.recorder.plot_losses()


# In[17]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))


# In[18]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# In[19]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[20]:


learn.data.add_test(ImageList.from_df(
    sample_df, PATH,
    folder='test_images',
    suffix='.png'
))


# In[21]:


preds,y = learn.get_preds(DatasetType.Test)


# In[22]:


sample_df.diagnosis = preds.argmax(1)
sample_df.head()


# In[23]:


sample_df.to_csv('submission.csv',index=False)
_ = sample_df.hist()

