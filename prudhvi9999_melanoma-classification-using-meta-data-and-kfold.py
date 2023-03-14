#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras import *
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from kaggle_datasets import KaggleDatasets
import numpy as np
import pandas as pd
gcs_path=KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
BATCH_SIZE=128


# In[2]:


train_csv=pd.read_csv(gcs_path+'/train.csv')
test_csv=pd.read_csv(gcs_path+'/test.csv')


# In[3]:


train_csv['age_approx']=train_csv['age_approx'].fillna(0)
train_csv['sex']=train_csv['sex'].fillna('na')
train_csv['anatom_site_general_challenge']=train_csv['anatom_site_general_challenge'].fillna('na')


# In[4]:


train_csv.isna().any()


# In[5]:


le=LabelEncoder()
bi=LabelBinarizer()


# In[6]:


train_csv['sex']=bi.fit_transform(train_csv['sex'])
train_csv['anatom_site_general_challenge']=le.fit_transform(train_csv['anatom_site_general_challenge'])


# In[7]:


train_csv.anatom_site_general_challenge.value_counts().plot(kind='barh')


# In[8]:


test_csv['sex']=bi.fit_transform(test_csv['sex'])
test_csv['anatom_site_general_challenge']=test_csv['anatom_site_general_challenge'].fillna('na')
test_csv['anatom_site_general_challenge']=le.fit_transform(test_csv['anatom_site_general_challenge'])


# In[9]:


feat=['age_approx','sex','anatom_site_general_challenge']


# In[10]:


X=train_csv[feat]
y=train_csv['target']


# In[11]:


#X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=5)


# In[12]:


def get_dataset(features,target,shuffle=False):
   X=tf.data.Dataset.from_tensor_slices(tf.stack(features))
   y=tf.data.Dataset.from_tensor_slices(target)
   ds=tf.data.Dataset.zip((X,y))
   ds=ds.repeat()
   ds=ds.batch(BATCH_SIZE)
   if shuffle:
     ds=ds.shuffle(1234,reshuffle_each_iteration=True) #reshuffle_each_iteration=True
   ds=ds.cache()
   return ds


# In[13]:


"""train_X,train_y=X.iloc[train],y[train]
valid_X,valid_y=X.iloc[valid],y[valid]
train_ds=get_dataset(X_train,y_train,shuffle=True)
val_ds=get_dataset(X_val,y_val,shuffle=False)"""


# In[14]:


test_ds=tf.data.Dataset.from_tensor_slices(tf.stack(test_csv[feat]))
test_ds=test_ds.batch(BATCH_SIZE)
test_ds=test_ds.cache()


# In[15]:


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


# In[16]:


def create_model():
  model=Sequential([
                    Dense(256,activation='relu',input_shape=(3,),
                          kernel_regularizer=regularizers.l2(0.001)),
                    Dropout(0.2),
                    BatchNormalization(),
                    Dense(108,activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
                    Dropout(0.2),
                    Dense(182,activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
                    Dropout(0.2),
                    Dense(108,activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
                    Dropout(0.2),
                    Dense(108,activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
                    Dense(1024,activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(1,activation='sigmoid')
  ])
  model.compile(optimizer='sgd',
                      loss=[binary_focal_loss(gamma = 2.2, alpha = 0.82)],
                      metrics=[metrics.BinaryAccuracy(),metrics.AUC()]
                )
  return model


# In[17]:


# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.004
LR_MAX = 0.00005 * 16
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 4
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


# In[18]:


models=[]
oof_predictions=[]
oof_target=[]
kf=KFold(n_splits=15,shuffle=True,random_state=1234)

for folds,(train,valid) in enumerate(kf.split(X,y)):
  print('\n')
  print('-'*50)
  print(f'Training fold {folds + 1}')
  train_X,train_y=X.iloc[train],y[train]
  valid_X,valid_y=X.iloc[valid],y[valid]
  train_ds=get_dataset(train_X,train_y,True)
  valid_ds=get_dataset(valid_X,valid_y,False)
  K.clear_session()
  model=create_model()
  STEPS_PER_EPOCH=len(train_X)//BATCH_SIZE
  VALIDATION_STEPS=len(valid_X)//BATCH_SIZE
  es=tf.keras.callbacks.EarlyStopping(monitor = 'val_auc', mode = 'max', patience = 8, 
                                      verbose = 1, min_delta = 0.0001, restore_best_weights = True)
  cb_schd=tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
  tb=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folds +1}')
  history=model.fit(train_ds,
          epochs=50,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=valid_ds,
          validation_steps=VALIDATION_STEPS,
          callbacks=[es,cb_schd,tb]
          )
  models.append(model)
  probabilities = model.predict(valid_X)
  oof_target.extend(list(valid_y))
  oof_predictions.extend(list(np.concatenate(probabilities)))


# In[19]:


from sklearn.metrics import roc_curve,auc
act,pred,threshold=roc_curve(oof_target,oof_predictions)
print("AUC SCORE : ",auc(act,pred))


# In[20]:


sample_sub=pd.read_csv(gcs_path+'/sample_submission.csv')


# In[21]:


sample_sub.head(5)


# In[22]:


df=sample_sub.copy()


# In[23]:


preds = np.average([np.concatenate(models[i].predict(test_ds)) for i in range(folds)], axis = 0)


# In[24]:


df.target=preds


# In[25]:


df.head(5)


# In[26]:


df.to_csv('sub.csv',index=False)

