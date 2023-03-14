#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###  majic calls
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Fastai's Library
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from fastai.imports import *


# In[ ]:


torch.cuda.set_device(1)


# In[ ]:


PATH = "../input/"
sz=224
arch=resnext101_64
bs=58


# In[ ]:


label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)


# In[ ]:


n


# In[ ]:


len(val_idxs)


# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


label_df = pd.read_csv(label_csv)


# In[ ]:


label_df.head()


# In[ ]:


label_df.info()


# In[ ]:


label_df.pivot_table(index='breed', aggfunc=len).sort_values('id',ascending=False)


# In[ ]:


tfms = tfms_from_model(arch,sz,aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv',test_name='test',
                                     val_idxs=val_idxs, suffix='.jpg', tfms=tfms,bs=bs)


# In[ ]:


fn = PATH+data.trn_ds.fnames[0]; fn


# In[ ]:


Here we need to find out how big the sizes are for the rows and columns of the images


# In[ ]:


fn = PATH+data.trn_ds.fnames[0]; fn


# In[ ]:


img = PIL.Image.open(fn); img


# In[ ]:


img.size


# In[ ]:


size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}


# In[ ]:


row_sz,col_sz = list(zip(*size_d.values()))


# In[ ]:


row_sz=np.array(row_sz); col_sz=np.array(col_sz)
row_sz[:5]


# In[ ]:


plt.hist(row_sz)


# In[ ]:


plt.hist(row_sz[row_sz<1000])


# In[ ]:


len(data.trn_ds), len(data.test_ds)


# In[ ]:


len(data.classes), data.classes[:5]


# In[ ]:





# In[ ]:


def get_data(sz,bs):
    tfms = tfms_from_model(arch,sz,aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv',test_name='test', num_workers=4,
                                        val_idxs=val_idxs, suffix='.jpg', tfms=tfms,bs=bs)
 


# In[ ]:


data = get_data(sz,bs)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True) 


# In[ ]:


learn.fit(1e-2,5)


# In[ ]:


from sklearn import metric


# In[ ]:


data = get_data(sz,bs)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True,ps=0.5) 


# In[ ]:


learn.fit(1e-2,2)


# In[ ]:


learn.precompute=False


# In[ ]:


learn.fit(1e-2,5,cycle_len=1)


# In[ ]:


learn.save('224_pre')


# In[ ]:


learn.load('224_pre')


# In[ ]:


learn.set_data(get_data(299,bs))
learn.freeze()


# In[ ]:


learn.fit(1e-2,3,cycle_len=1)


# In[ ]:


learn.fit(1e-2,3,cycle_len=1,cycle_mult=2)


# In[ ]:


log_preds,y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds,y), metrics.log_loss(y,probs)


# In[ ]:


learn.save('299_pre')


# In[ ]:


learn.load('299_pre')


# In[ ]:


learn.fit(1e-2,3,cycle_len=1)


# In[ ]:


learn.save('299_pre')


# In[ ]:


log_preds,y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds,y), metrics.log_loss(y,probs)


# In[ ]:





# In[ ]:




