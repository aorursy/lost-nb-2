#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from fastai.vision import *
from fastai.metrics import accuracy, error_rate, FBeta
from fastai.callbacks import *
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve,     average_precision_score, classification_report, f1_score, fbeta_score, precision_score
from tqdm.notebook import tqdm
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value
    
    if plot:
        # plots the gradients of the losses in respect to the learning rate change
        plt.plot(loss_grad)
        plt.plot(len(losses)+l_idx, loss_grad[l_idx],markersize=10,marker='o',color='red')
        plt.ylabel("Loss")
        plt.xlabel("Index of LRs")
        plt.show()

        plt.plot(np.log10(lrs), losses)
        plt.ylabel("Loss")
        plt.xlabel("Log 10 Transform of Learning Rate")
        loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
        plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
        plt.show()
        
    return lr_to_use


# In[3]:


train='../input/i2a2-brasil-pneumonia-classification/train.csv'
train_d = pd.read_csv(train)
train_d


# In[4]:


data = (ImageList.from_df(train_d, '../input/i2a2-brasil-pneumonia-classification/images', cols='fileName')
         .split_by_rand_pct(valid_pct=0.1)
         .label_from_df(cols='pneumonia')
         .transform(get_transforms(), size=299)
         .databunch(bs=64))

# Normalizing to ImageNet 
data.normalize(imagenet_stats)


# In[5]:


#  Kaggle F1 score

f1loss = FBeta()

learn = cnn_learner(data, models.resnet50, metrics=[accuracy, f1loss],model_dir='/tmp/model')

callbacks = [SaveModelCallback(learn, mode='min', monitor='valid_loss', name='best_model'),
             EarlyStoppingCallback(learn, mode='min', monitor='valid_loss', patience=5)]


# In[6]:


learn.unfreeze()
learn.fit_one_cycle(20, 1e-3, callbacks=callbacks)


# In[7]:


learn.freeze()
learn.fit_one_cycle(20, 1e-5, callbacks=callbacks)


# In[8]:


preds, y, losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)


# In[9]:


interp.plot_confusion_matrix()


# In[10]:


# improve f1 score by threshold
pred = preds[:,1].numpy()
ths = np.linspace(0.01, .99, 99)
f1s = []

for th in tqdm(ths):
  # BINARY
  _pred = (pred >= th).astype(int)
  f1s.append(f1_score(y.numpy(), 
                      _pred
                     ))

best_th = ths[len(f1s) - 1 - np.argmax(np.flip(f1s))]

# BINARY
print(f1_score(y.numpy(), (pred >= best_th).astype(int)))
print(confusion_matrix(y.numpy(), (pred >= best_th).astype(int)))
print(classification_report(y.numpy(), (pred >= best_th).astype(int)))

plt.figure(figsize=(8,8))
sns.lineplot(ths, f1s)
plt.plot([best_th, best_th], [0, np.max(f1s)], 
         label=f'best f1 in th: {round(best_th, 3)} ({round(np.max(f1s), 3)})')
plt.legend(loc="lower left")


# In[11]:


# submit
sub = pd.read_csv('../input/i2a2-brasil-pneumonia-classification/sample_submission.csv')
sub


# In[12]:


for i in tqdm(range(len(sub))):
    
    imageT = open_image('/kaggle/input/i2a2-brasil-pneumonia-classification/images/' + sub.loc[i,'fileName'])

    pred = learn.predict(imageT)[2][1].numpy()
    sub.loc[i,'pneumonia'] = (pred >= best_th).astype(int)


# In[13]:


sub.head(20)


# In[14]:


sub.to_csv('sample_submission_fastia.csv', index=False)

