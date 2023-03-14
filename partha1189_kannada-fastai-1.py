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


#%reload_ext autoreload
#%autoreload 2
# %matplotlib inline


# In[3]:


from fastai.vision import *


# In[4]:


path = Path('../input/Kannada-MNIST')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
train_other = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


# In[5]:


data,labels = (train.iloc[:,1:],train.iloc[:,0])


# In[6]:


data_other,labels_other = (train_other.iloc[:,1:],train_other.iloc[:,0])


# In[7]:


data_train,labels_train = (pd.concat([data, data_other]),pd.concat([labels, labels_other]))


# In[8]:


from sklearn.model_selection import train_test_split

data_train, data_valid, labels_train, labels_valid = train_test_split(data_train, labels_train, test_size=0.25, random_state=42,stratify=labels_train)


# In[9]:


data_train


# In[10]:


#import pdb
import imageio
def save_img_to_folder(path:Path,data,labels):
    path.mkdir(parents=True,exist_ok=True)
    
    for i in range(len(data)):
        test = path
        #pdb.set_trace()
        temp_path = test/(str(labels[i]))
        if os.path.isdir(temp_path):
            imageio.imwrite(str(temp_path/(str(i)+'.jpg')), data[i])
            #plt.imsave(str( temp_path/(str(i)+'.jpg') ), data[i], cmap='Greys')
        else:
            temp_path.mkdir(parents=True,exist_ok=True)
            #plt.imsave(str( temp_path/(str(i)+'.jpg') ), data[i], cmap='Greys')
            imageio.imwrite(str(temp_path/(str(i)+'.jpg')), data[i])


# In[11]:


data_arr = np.array(data_valid).reshape(-1,28,28)
labels_arr = np.array(labels_valid)


# In[12]:


save_img_to_folder(Path('valid'),data_arr,labels_arr)


# In[13]:


data_arr1 = np.array(data_train).reshape(-1,28,28)
labels_arr1 = np.array(labels_train)


# In[14]:


save_img_to_folder(Path('train'),data_arr1,labels_arr1)


# In[15]:


path = Path('/kaggle/working')
train_path = path/'train'
valid_path = path/'valid'
path.ls()


# In[16]:


# test_data,test_labels = (test.iloc[:,1:],test.iloc[:,0])


# In[17]:


# data_arr2 = np.array(test_data).reshape(-1,28,28)
# labels_arr2 = np.array(test_labels)


# In[18]:


# save_img_to_folder(Path('test'),data_arr2,labels_arr2)


# In[19]:


from fastai.metrics import error_rate
from fastai.vision import *


# In[20]:


np.random.seed(42)
tfms = get_transforms(do_flip=False)


# In[21]:


src = (ImageList.from_folder(path)
        .split_by_folder(train='train', valid='valid')
        .label_from_folder())


# In[22]:


data = (src.transform(tfms,size=64)
        .databunch(bs=64)
        .normalize(mnist_stats))


# In[23]:


# data = (ImageList.from_folder(path)
#         .split_by_folder(train='train', valid='valid')
#         .label_from_folder() 
#         .transform(tfms, size=64)
#         .databunch(bs=64)
#         .normalize(mnist_stats)
#        )


# In[24]:


data.train_ds


# In[25]:


data.classes


# In[26]:


# data = ImageDataBunch.from_folder(path, train="train", valid="valid",test="test",
#         ds_tfms=get_transforms(do_flip=False), size=64,bs=32).normalize(imagenet_stats)


# In[27]:


data.show_batch(rows=4,figsize=(7,8))


# In[28]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')


# In[29]:


get_ipython().system('ls /kaggle/input/resnet152')


# In[30]:


get_ipython().system('cp /kaggle/input/resnet152/resnet152.pth /tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth')


# In[31]:


get_ipython().system('ls /tmp/.cache/torch/checkpoints/')


# In[32]:


learner = cnn_learner(data,models.resnet152, metrics=[error_rate, accuracy])


# In[33]:


learner.fit_one_cycle(12)


# In[34]:


learner.save('kannada-stage1')


# In[35]:


learner.recorder.plot_losses()


# In[36]:


interp = ClassificationInterpretation.from_learner(learner)


# In[37]:


losses,idx = interp.top_losses()


# In[38]:


len(data.valid_ds)==len(losses)==len(idx)


# In[39]:


interp.plot_top_losses(9, figsize=(15,11))


# In[40]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[41]:


learner.lr_find()


# In[42]:


learner.recorder.plot(suggestion=True)


# In[43]:


learner.unfreeze()
learner.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))


# In[44]:


learner.save('kannada-stage2')


# In[45]:


# learner.fit_one_cycle(8,1e-5)


# In[46]:


# learner.save('kannada-stage3')


# In[47]:


# data = (src.transform(tfms, size=64)
#         .databunch().normalize(mnist_stats))

# learner.data = data
# data.train_ds[0][0].shape


# In[48]:


# learner.freeze()


# In[49]:


# learner.lr_find()
# learner.recorder.plot()


# In[50]:


# learner.fit_one_cycle(3, slice(1e-1))


# In[51]:


# learner.save('kannada-stage3')


# In[52]:


# learner.unfreeze()


# In[53]:


# learner.fit_one_cycle(4, max_lr=slice(1e-2, 1e-1/2))


# In[54]:


# learner.recorder.plot_losses()


# In[55]:


# learner.save('kannada-stage4')


# In[56]:


learner.load('kannada-stage2')


# In[57]:


img = learner.data.valid_ds[0][0]


# In[58]:


learner.predict(img)


# In[59]:


learner.data.test_ds


# In[60]:


#sub_df = pd.DataFrame(columns=['id','label'])


# In[61]:


#len(learner.data.test_ds.x)


# In[62]:


#learner.predict(learner.data.test_ds.x[6])[1]


# In[63]:


# for i in range(len(learner.data.test_ds.x)):
    
#     sub_df.loc[i]=[i+1,int(learner.predict(learner.data.test_ds.x[i])[1])]


# In[64]:


test_csv = pd.read_csv('../input/Kannada-MNIST/test.csv')
test_csv.drop('id',axis = 'columns',inplace = True)
sub_df = pd.DataFrame(columns=['id','label'])
test_data = np.array(test_csv)


# In[65]:


def get_img(data):
    t1 = data.reshape(28,28)/255
    t1 = np.stack([t1]*3,axis=0)
    img = Image(FloatTensor(t1))
    return img


# In[66]:


from fastprogress import progress_bar


# In[67]:


mb=progress_bar(range(test_data.shape[0]))
for i in mb:
    timg=test_data[i]
    img = get_img(timg)
    sub_df.loc[i]=[i+1,int(learner.predict(img)[1])]


# In[68]:


def decr(ido):
    return ido-1

sub_df['id'] = sub_df['id'].map(decr)


# In[69]:


sub_df.head()


# In[70]:


sub_df.to_csv("submission.csv", index=False)


# In[71]:


get_ipython().system('rm -r valid train')

