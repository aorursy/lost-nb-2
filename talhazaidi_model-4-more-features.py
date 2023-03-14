#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
warnings.filterwarnings('ignore')


# In[2]:


PATH="../input/"
os.listdir(PATH)


# In[3]:


print("There are {} files in test folder".format(len(os.listdir(os.path.join(PATH, 'test' )))))


# In[4]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(os.path.join(PATH,'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})")


# In[5]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))


# In[6]:


rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


# In[7]:


train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
train_X .shape


# In[8]:


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)   
    zc = np.fft.fft(xc)
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_25000'] = xc[:25000].std()
    X.loc[seg_id, 'std_last_25000'] = xc[-25000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()


# In[9]:


# iterate over all segments
for seg_id in tqdm_notebook(range(segments)):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg, train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]


# In[10]:


train_X


# In[11]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)


# In[12]:


for seg_id in tqdm_notebook(test_X.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, test_X)


# In[13]:


print("Train X: {} y: {} Test X: {}".format(train_X.shape, train_y.shape, test_X.shape))


# In[14]:


train_X.head()


# In[15]:


test_X.head()


# In[16]:


def plot_distplot(feature):
    plt.figure(figsize=(16,6))
    plt.title("Distribution of {} values in the train and test set".format(feature))
    sns.distplot(train_X[feature],color="green", kde=True,bins=120, label='train')
    sns.distplot(test_X[feature],color="blue", kde=True,bins=120, label='test')
    plt.legend()
    plt.show()


# In[17]:


def plot_distplot_features(features, nlines=4, colors=['green', 'blue'], df1=train_X, df2=test_X):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(nlines,2,figsize=(16,4*nlines))
    for feature in features:
        i += 1
        plt.subplot(nlines,2,i)
        sns.distplot(df1[feature],color=colors[0], kde=True,bins=40, label='train')
        sns.distplot(df2[feature],color=colors[1], kde=True,bins=40, label='test')
    plt.show()


# In[18]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']

plot_distplot_features(features)


# In[19]:


features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']
plot_distplot_features(features,3)


# In[20]:


features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']
plot_distplot_features(features)


# In[21]:


features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']
plot_distplot_features(features,3)


# In[22]:


scaler = StandardScaler()
scaler.fit(pd.concat([train_X, test_X]))
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)


# In[23]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']
plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[24]:


features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']
plot_distplot_features(features, nlines=3, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[25]:


features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']
plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[26]:


features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']
plot_distplot_features(features, nlines=3, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[27]:


def plot_acc_agg_ttf_data(feature, title="Averaged accoustic data and ttf"):
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title('Averaged accoustic data ({}) and time to failure'.format(feature))
    plt.plot(train_X[feature], color='r')
    ax1.set_xlabel('training samples')
    ax1.set_ylabel('acoustic data ({})'.format(feature), color='r')
    plt.legend(['acoustic data ({})'.format(feature)], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_y, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)


# In[28]:


plot_acc_agg_ttf_data('mean')


# In[29]:


plot_acc_agg_ttf_data('std')


# In[30]:


plot_acc_agg_ttf_data('max')


# In[31]:


plot_acc_agg_ttf_data('min')


# In[32]:


plot_acc_agg_ttf_data('sum')


# In[33]:


plot_acc_agg_ttf_data('mad')


# In[34]:


plot_acc_agg_ttf_data('kurt')


# In[35]:


plot_acc_agg_ttf_data('skew')


# In[36]:


plot_acc_agg_ttf_data('std_first_50000')


# In[37]:


plot_acc_agg_ttf_data('std_last_50000')


# In[38]:


plot_acc_agg_ttf_data('std_first_25000')


# In[39]:


plot_acc_agg_ttf_data('std_last_25000')


# In[40]:


plot_acc_agg_ttf_data('std_first_10000')


# In[41]:


plot_acc_agg_ttf_data('std_last_10000')


# In[42]:


validation_point=351
endpoint=train_X.shape[0]-1
tttt=validation_point-endpoint

X_train=scaled_train_X.values[validation_point:endpoint,].reshape(tttt,28,1)
print('X_train.shape',X_train.shape)
y_train=train_y.values[validation_point:endpoint,]
print('y_train.shape',y_train.shape)

X_validation=scaled_train_X.values[0:validation_point-1,].reshape(validation_point-1,28,1)
print('X_validation.shape',X_validation.shape)
y_validation=train_y.values[0:validation_point-1,]
print('y_validation.shape',y_validation.shape)

X_train.shape[1]


# In[43]:


from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, SimpleRNN, LSTM ,  Dropout, Activation, Flatten, Input, Conv1D, MaxPooling1D, Reshape,  Conv2D, MaxPooling2D, Reshape, Flatten
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.utils import plot_model


# In[44]:


i = (X_train.shape[1],X_train.shape[2])
#i = (X_train.shape[1])
i
model = Sequential ()

model.add(Reshape((14,2,1), input_shape=i))
model.add(Conv2D(16, (2, 2), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
#model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(8, (2, 2), strides = (2,2), kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.2))
#model.add(MaxPooling2D())
model.add(Dropout(0.3))


model.add(Reshape((4,8)))
#model.add(Flatten())

model.add(LSTM(8,  return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4))
model.add(Dropout(0.2))

model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))

model.summary()
plot_model(model, to_file='model.png')


# In[45]:


i = (X_train.shape[1],X_train.shape[2])
model = Sequential ()
model.add(Conv1D(5, 3, activation='relu', input_shape= i))
model.add(MaxPooling1D(2))
model.add(LSTM(50,  return_sequences=True))
model.add(LSTM(10))
model.add(Dense(240))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1))


model.summary()


# In[46]:


i = (X_train.shape[1],X_train.shape[2])
#i = (X_train.shape[1])
i
model = Sequential ()

model.add(Reshape((28,1), input_shape=i))
model.add(Conv1D(16, 2, strides =2, kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(Conv1D(16, 2, strides =2, kernel_initializer='he_normal', padding='same'))
model.add(LeakyReLU(0.1))
model.add(MaxPooling1D())
#model.add(Dropout(0.3))

model.add(Flatten())
model.add(Softmax())
model.add(Reshape((48,1)))



model.add(LSTM(48,  return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(48))
model.add(Dropout(0.2))

model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(1))

model.summary()
plot_model(model, to_file='model.png')


# In[47]:


import keras
from keras.optimizers import RMSprop
opt = keras.optimizers.adam(lr=.005)

model.compile(loss="mae",
              optimizer=opt, metrics=['mean_absolute_error'])
             # metrics=['accuracy'])


batch_size = 64 # mini-batch with 32 examples
epochs = 50
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_validation ,y_validation ))


# In[48]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})


# In[49]:


X_test=scaled_test_X.values.reshape(test_X.shape[0],28,1)
print(X_test.shape)

for i, seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i]= model.predict(np.expand_dims(X_test[i], 0))
    
    


# In[50]:


submission_newfeatures=submission
submission_newfeatures.head()


# In[51]:


submission_newfeatures.to_csv('submission_cnn_lstm_dense2 vgg.csv')

