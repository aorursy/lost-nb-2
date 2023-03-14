#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[2]:


get_ipython().system('conda install -y gdcm -c conda-forge')


# In[3]:


import pydicom
import math
import PIL
from PIL import Image
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from scipy.stats import shapiro
from scipy import stats 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import gc


# In[5]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[6]:


BATCH_SIZE = 30
TRAIN_VAL_RATIO = 0.35
EPOCHS_M1 = 200
EPOCHS_M2 = 400
LR = 0.005
imSize = 224


# In[7]:


train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[8]:


sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split("_", 1)[0])
sub_df['Weeks'] = sub_df['Patient_Week'].apply(lambda x: x.split("_", 1)[1])
sub_df.head()


# In[9]:


plt.hist(train_df['FVC'])


# In[10]:


# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# normality test
stat, p = shapiro(train_df['FVC'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.01
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[11]:


# transform training data & save lambda value 
_, fitted_lambda = stats.boxcox(train_df['FVC']) 
print("solved lambda: ", fitted_lambda)


# In[12]:


def BoxCoxTransform(x, lmbda):
    part1 = x**lmbda
    part2 = part1-1
    result = part2/lmbda
    return result

def ReverseBoxCoxTranform(x, lmbda):
    x = np.where(x<0,0,x)
    part1 = x*lmbda + 1
    result = part1**(1/lmbda)
    return result


# In[13]:


fitted_data = BoxCoxTransform(train_df['FVC'], lmbda=fitted_lambda)
plt.hist(fitted_data)


# In[14]:


# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
# normality test
stat, p = shapiro(fitted_data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.01
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[15]:


# show that we can reverse transformed values back to original values
plt.hist(ReverseBoxCoxTranform(fitted_data, lmbda=fitted_lambda))


# In[16]:


def preprocess_image(image_path, desired_size=imSize):
    im = pydicom.dcmread(image_path).pixel_array
    im = Image.fromarray(im, mode="L")
    im = im.resize((desired_size,desired_size)) 
    im = np.array(im).flatten().astype(np.uint8)
    return im


# In[17]:


def process_patient_images(patient_path, imSize=imSize):
    image_filenames = os.listdir(patient_path)
    final_array = np.zeros((imSize*imSize), dtype=np.uint8)
    total_images = len(image_filenames)
    for image_filename in image_filenames:
        image_path = patient_path + "/" + image_filename
        image_arr = preprocess_image(image_path, desired_size=imSize)
        final_array += image_arr
    final_array = final_array / total_images
    return final_array


# In[18]:


# get the number of training images from the target\id dataset
N = train_df.shape[0]
# create an empty matrix for storing the images
x_train = np.empty((N, imSize*imSize), dtype=np.uint8)
# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, Patient in enumerate(tqdm(train_df['Patient'])):
    x_train[i, :] = process_patient_images(
        f'../input/osic-pulmonary-fibrosis-progression/train/{Patient}'
    )


# In[19]:


# get the number of training images from the target\id dataset
N = test_df.shape[0]
# create an empty matrix for storing the images
x_test = np.empty((N, imSize*imSize), dtype=np.uint8)
# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, Patient in enumerate(tqdm(test_df['Patient'])):
    x_test[i, :] = process_patient_images(
        f'../input/osic-pulmonary-fibrosis-progression/train/{Patient}'
    )


# In[20]:


# get the number of training images from the target\id dataset
N = sub_df.shape[0]
# create an empty matrix for storing the images
x_sub = np.empty((N, imSize*imSize), dtype=np.uint8)
# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, Patient in enumerate(tqdm(sub_df['Patient'])):
    x_sub[i, :] = process_patient_images(
        f'../input/osic-pulmonary-fibrosis-progression/train/{Patient}'
    )


# In[21]:


# one hot encoding
train_df['Sex_Male'] = train_df['Sex'].apply(lambda x: 1 if str(x)=='Male' else 0)
train_df['SmokingStatusEx'] = train_df['SmokingStatus'].apply(lambda x: 1 if str(x)=='Ex-smoker' else 0)
test_df['Sex_Male'] = test_df['Sex'].apply(lambda x: 1 if str(x)=='Male' else 0)
test_df['SmokingStatusEx'] = test_df['SmokingStatus'].apply(lambda x: 1 if str(x)=='Ex-smoker' else 0)


# In[22]:


train_df.head()


# In[23]:


# patient profile
patient_profile = train_df.groupby("Patient", as_index=False)                           .agg({'Percent':'mean', 'Sex_Male':'max', 'SmokingStatusEx':'max', 'Age':'mean', 'Weeks':'min'})
patient_profile = patient_profile.rename(columns={'Percent':'AvgPercent','Age':'AvgAge','Weeks':'BaseWeek'})


# In[24]:


patient_profile.head()


# In[25]:


# merge profile and create more features
train_df = train_df.merge(patient_profile[["Patient","AvgPercent","AvgAge","BaseWeek"]], on="Patient", how='left')
train_df['RelativeWeek'] = train_df['Weeks'].apply(lambda x: int(x)) - train_df['BaseWeek'].apply(lambda x: int(x))
test_df = test_df.merge(patient_profile[["Patient","AvgPercent","AvgAge","BaseWeek"]], on="Patient", how='left')
test_df['RelativeWeek'] = test_df['Weeks'].apply(lambda x: int(x)) - test_df['BaseWeek'].apply(lambda x: int(x))
sub_df = sub_df.merge(patient_profile, on="Patient", how='left')
sub_df['RelativeWeek'] = sub_df['Weeks'].apply(lambda x: int(x)) - sub_df['BaseWeek'].apply(lambda x: int(x))


# In[26]:


train_df.head()


# In[27]:


test_df.head()


# In[28]:


sub_df.head()


# In[29]:


# final dataframes and tabular features
# train
x_train_tabular_features = train_df[['Weeks','AvgPercent','AvgAge','Sex_Male','SmokingStatusEx','BaseWeek','RelativeWeek']].values
# test
x_test_tabular_features = test_df[['Weeks','AvgPercent','AvgAge','Sex_Male','SmokingStatusEx','BaseWeek','RelativeWeek']].values
# submission
x_sub_tabular_features = sub_df[['Weeks','AvgPercent','AvgAge','Sex_Male','SmokingStatusEx','BaseWeek','RelativeWeek']].values


# In[30]:


# dimensionality reduction
pca = PCA(n_components=100)
pca.fit(x_train)


# In[31]:


x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_sub = pca.transform(x_sub)


# In[32]:


# merge
x_train_full = np.concatenate((x_train_tabular_features, x_train), axis=1)
x_test_full = np.concatenate((x_test_tabular_features, x_test), axis=1)
x_sub_full = np.concatenate((x_sub_tabular_features, x_sub), axis=1)


# In[33]:


scaler = StandardScaler()
scaler.fit(x_train_full)
x_train_full = scaler.transform(x_train_full)
x_test_full = scaler.transform(x_test_full)
x_sub_full = scaler.transform(x_sub_full)


# In[34]:


# squared features for some model flexability
x_train_full2 = np.square(x_train_full)
x_test_full2 = np.square(x_test_full)
x_sub_full2 = np.square(x_sub_full)


# In[35]:


# merge
x_train_full = np.concatenate((x_train_full, x_train_full2), axis=1)
x_test_full = np.concatenate((x_test_full, x_test_full2), axis=1)
x_sub_full = np.concatenate((x_sub_full, x_sub_full2), axis=1)


# In[36]:


print(x_train_full.shape)
print(x_test_full.shape)
print(x_sub_full.shape)


# In[37]:


# view data
x_train_full[:2,:]


# In[38]:


# This section is commented out, because it doesn't prove useful based on the way the holdout set is designed

# save the data to disk so we can save as a Kaggle Dataset
# and skip data preprocessing in other scripts
# filename = 'osic_processed_train_data_v1.pkl'
# pickle.dump(x_train_full, open(filename, 'wb'))
# filename = 'osic_processed_test_data_v1.pkl'
# pickle.dump(x_test_full, open(filename, 'wb'))
# filename = 'osic_processed_sub_data_v1.pkl'
# pickle.dump(x_sub_full, open(filename, 'wb'))


# In[39]:


x_train_fvc, x_val_fvc, y_train_fvc, y_val_fvc = train_test_split(
    x_train_full, BoxCoxTransform(train_df['FVC'], lmbda=fitted_lambda),
    test_size=TRAIN_VAL_RATIO, 
    random_state=2020
)


# In[40]:


with strategy.scope():
    # define structure
    xin = tf.keras.layers.Input(shape=(x_train_full.shape[1], ))
    xout = tf.keras.layers.Dense(1, activation='linear')(xin)
    # put it together
    model1 = tf.keras.Model(inputs=xin, outputs=xout)
    # compile
    opt = tf.optimizers.RMSprop(LR)
    model1.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
# print summary
model1.summary()


# In[41]:


### define callbacks
earlystopper = EarlyStopping(
    monitor='val_mean_squared_error', 
    patience=30,
    verbose=1,
    mode='min'
)

lrreducer = ReduceLROnPlateau(
    monitor='val_mean_squared_error',
    factor=.5,
    patience=10,
    verbose=1,
    min_lr=1e-9
)


# In[42]:


print("Fit model on training data")
history = model1.fit(
    x_train_fvc,
    y_train_fvc,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_M1,
    validation_data=(x_val_fvc, y_val_fvc),
    callbacks=[earlystopper,lrreducer]
)


# In[43]:


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['mean_squared_error', 'val_mean_squared_error']].plot()


# In[44]:


train_df['FVC_pred1'] = ReverseBoxCoxTranform(model1.predict(x_train_full), lmbda=fitted_lambda)


# In[45]:


train_df.head()


# In[46]:


stats.describe(model1.predict(x_train_full))


# In[47]:


get_ipython().run_cell_magic('time', '', '\nbest_confidence = np.zeros(len(train_df))\nBestRandSearchScore = -1000000000\nscorehist = []\nprint("running random search...")\nfor i in range(10000):\n    trial_confidence = np.random.randint(70, 1000, size=len(train_df))\n    train_df[\'Confidence\'] = trial_confidence\n    train_df[\'sigma_clipped\'] = train_df[\'Confidence\'].apply(lambda x: max(x, 70))\n    train_df[\'diff\'] = abs(train_df[\'FVC\'] - train_df[\'FVC_pred1\'])\n    train_df[\'delta\'] = train_df[\'diff\'].apply(lambda x: min(x, 1000))\n    train_df[\'score\'] = -math.sqrt(2)*train_df[\'delta\']/train_df[\'sigma_clipped\'] - np.log(math.sqrt(2)*train_df[\'sigma_clipped\'])\n    score = train_df[\'score\'].mean()\n    if score>BestRandSearchScore:\n        BestRandSearchScore = score\n        best_confidence = trial_confidence\n        print("best confidence values found in round {} with best score of {}".format(i,score))\n    scorehist.append(BestRandSearchScore)')


# In[48]:


plt.plot(scorehist)
plt.ylabel('best score')
plt.xlabel('round')
plt.show()


# In[49]:


def md_learning_rate(val):
    if val == 0:
        return 10
    elif val == 1:
        return 8
    else:
        return 5/np.log(val)


# In[50]:


get_ipython().run_cell_magic('time', '', '\nRounds = 100\nbest_md_confidence = np.zeros(len(train_df))\nBestManualDescentScore = -1000000000\nrowScore = -1000000000\ntrain_df[\'Confidence\'] = best_confidence\ntrain_df[\'diff\'] = abs(train_df[\'FVC\'] - train_df[\'FVC_pred1\']) # don\'t need to compute this every time\ntrain_df[\'delta\'] = train_df[\'diff\'].apply(lambda x: min(x, 1000)) # don\'t need to compute this every time\nfor j in range(Rounds):\n    for i in range(len(train_df)):\n        originalValue = train_df[\'Confidence\'].iloc[i]\n        # try moving value up\n        train_df[\'Confidence\'].iloc[i] = originalValue + md_learning_rate(j)\n        train_df[\'sigma_clipped\'] = train_df[\'Confidence\'].apply(lambda x: max(x, 70))\n        train_df[\'score\'] = -math.sqrt(2)*train_df[\'delta\']/train_df[\'sigma_clipped\'] - np.log(math.sqrt(2)*train_df[\'sigma_clipped\'])\n        scoreup = train_df[\'score\'].mean()\n        # try moving value down\n        train_df[\'Confidence\'].iloc[i] = originalValue - md_learning_rate(j)\n        train_df[\'sigma_clipped\'] = train_df[\'Confidence\'].apply(lambda x: max(x, 70))\n        train_df[\'score\'] = -math.sqrt(2)*train_df[\'delta\']/train_df[\'sigma_clipped\'] - np.log(math.sqrt(2)*train_df[\'sigma_clipped\'])\n        scoredown = train_df[\'score\'].mean()\n        if scoreup>scoredown:\n            train_df[\'Confidence\'].iloc[i] = originalValue + md_learning_rate(j)\n            rowScore = scoreup\n        else:\n            train_df[\'Confidence\'].iloc[i] = originalValue - md_learning_rate(j)\n            rowScore = scoredown\n    if rowScore>BestManualDescentScore:\n        BestManualDescentScore = rowScore\n        best_md_confidence = train_df[\'Confidence\'].to_numpy()\n        if j % 10 == 0:\n            print("best confidence values found in round {} with best score of {}".format(j,BestManualDescentScore))')


# In[51]:


if BestManualDescentScore>BestRandSearchScore:
    best_confidence = best_md_confidence
    print("some manual descent improved confidence values")


# In[52]:


# x_train, x_val, y_train1, y_val1, y_train2, y_val2 = train_test_split(
#     x_train_full, 
#     best_confidence,
#     BoxCoxTransform(train_df['FVC'], lmbda=fitted_lambda), 
#     test_size=TRAIN_VAL_RATIO, 
#     random_state=2020
# )

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, 
    best_confidence,
    test_size=TRAIN_VAL_RATIO, 
    random_state=2020
)


# In[53]:


# def Laplace_Log_Likelihood(y_true, y_pred):
#     # get predictions
#     y_pred1 = tf.cast(y_pred[:,0], dtype=tf.float32) # confidence
#     y_pred2 = tf.cast(y_pred[:,1], dtype=tf.float32) # fvc
#     # reverse box cox
#     tfz = tf.cast(tf.constant([0]), dtype=tf.float32) 
#     y_pred2 = tf.where(y_pred2<tfz,tfz,y_pred1)
#     lbda = tf.cast(tf.constant([0.376401998544658]), dtype=tf.float32) 
#     tf1 = tf.cast(tf.constant([1]), dtype=tf.float32)
#     p1 = tf.math.add(tf.math.multiply(y_pred2,lbda), tf1)
#     y_pred2 = tf.pow(p1,tf.math.divide(tf1,lbda)) # fvc reverse box cox
#     # laplace log likelihood                
#     threshold = tf.cast(tf.constant([70]), dtype=tf.float32) 
#     sig_clip = tf.math.maximum(y_pred1, threshold)
#     threshold = tf.cast(tf.constant([1000]), dtype=tf.float32) 
#     delta = tf.math.minimum(tf.math.abs(tf.math.subtract(y_true,y_pred2)),threshold)
#     sqrt2 = tf.cast(tf.constant([1.4142135623730951]), dtype=tf.float32) 
#     numerator = tf.math.multiply(sqrt2,delta)
#     part1 = tf.math.divide(numerator,sig_clip)
#     innerlog = tf.math.multiply(sqrt2,sig_clip)
#     metric = tf.math.subtract(-part1,tf.math.log(innerlog))
#     return tf.math.reduce_mean(metric)


# In[54]:


with strategy.scope():
    # define structure
    xin = tf.keras.layers.Input(shape=(x_train_full.shape[1], ))
    xout = tf.keras.layers.Dense(1, activation='linear')(xin)
    # put it together
    model2 = tf.keras.Model(inputs=xin, outputs=xout)
    # compile
    opt = tf.optimizers.RMSprop(LR)
    model2.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
# print summary
model2.summary()


# In[55]:


### define callbacks
earlystopper = EarlyStopping(
    monitor='val_mean_squared_error', 
    patience=3,
    verbose=1,
    mode='min'
)

lrreducer = ReduceLROnPlateau(
    monitor='val_mean_squared_error',
    factor=.5,
    patience=2,
    verbose=1,
    min_lr=1e-9
)


# In[56]:


print("Fit model on training data")
history = model2.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_M2,
    validation_data=(x_val, y_val),
    callbacks=[earlystopper,lrreducer]
)


# In[57]:


model2.save('model.h5')


# In[58]:


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['mean_squared_error', 'val_mean_squared_error']].plot()
# history_df[['Laplace_Log_Likelihood', 'val_Laplace_Log_Likelihood']].plot()


# In[59]:


model2.predict(x_test_full)


# In[60]:


fvc = model1.predict(x_train_full)
conf = model2.predict(x_train_full)[:,0]
fvc = ReverseBoxCoxTranform(fvc, lmbda=fitted_lambda)
train_df['FVC_pred1'] = fvc
train_df['Confidence'] = conf
train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))
train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred1'])
train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))
train_df['score'] = -math.sqrt(2)*train_df['delta']/train_df['sigma_clipped'] - np.log(math.sqrt(2)*train_df['sigma_clipped'])
score = train_df['score'].mean()
print("train score: ", score)


# In[61]:


train_df.head()


# In[62]:


fvc = model1.predict(x_test_full)
conf = model2.predict(x_test_full)
fvc = ReverseBoxCoxTranform(fvc, lmbda=fitted_lambda)
test_df['FVC_pred1'] = fvc
test_df['Confidence'] = conf
test_df['sigma_clipped'] = test_df['Confidence'].apply(lambda x: max(x, 70))
test_df['diff'] = abs(test_df['FVC'] - test_df['FVC_pred1'])
test_df['delta'] = test_df['diff'].apply(lambda x: min(x, 1000))
test_df['score'] = -math.sqrt(2)*test_df['delta']/test_df['sigma_clipped'] - np.log(math.sqrt(2)*test_df['sigma_clipped'])
score = test_df['score'].mean()
print("test score: ", score)


# In[63]:


test_df.head()


# In[ ]:




