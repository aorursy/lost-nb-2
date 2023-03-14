#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm


# In[2]:


df_train = pd.read_csv('/kaggle/input/weather-postprocessing/pp_train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/weather-postprocessing/pp_test.csv', index_col=0)

X_train = pd.read_csv('/kaggle/input/nb1-linear-regression/X_train.csv', index_col=0)
y_train = pd.read_csv('/kaggle/input/nb1-linear-regression/y_train.csv', index_col=0, squeeze=True)
X_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/X_valid.csv', index_col=0)
y_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/y_valid.csv', index_col=0, squeeze=True)
X_test = pd.read_csv('/kaggle/input/nb1-linear-regression/X_test.csv', index_col=0)


# In[3]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import mean_squared_error


# In[4]:


X_train1 = X_train[['t2m_fc_mean']].values


# In[5]:


X_train1.shape


# In[6]:


model = Sequential([Dense(1, input_shape=(1,))])


# In[7]:


model.summary()


# In[8]:


model.layers[0].weights


# In[9]:


a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
a, b


# In[10]:


plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
x = np.linspace(-15, 20, 2)
plt.plot(x, a*x+b, c='r', lw=2)


# In[11]:


preds = model(X_train1)  # Same as model.predict(X_train1)


# In[12]:


y_train.shape, preds.shape


# In[13]:


mse = mean_squared_error(y_train, preds[:, 0])
mse


# In[14]:


e = preds[:, 0] - y_train
dl_da = np.mean(2 * X_train1[:, 0] * e)
dl_db = np.mean(2 * e)
dl_da, dl_db


# In[15]:


with tf.GradientTape() as g:
    preds = model(X_train1)
    mse = mean_squared_error(y_train, preds[:, 0])


# In[16]:


dloss_dparam = g.gradient(mse, model.trainable_weights)


# In[17]:


dl_da, dl_db = dloss_dparam
dl_da, dl_db


# In[18]:


lr = 1e-3


# In[19]:


a, b = model.get_weights()


# In[20]:


a -= lr * dl_da
b -= lr * dl_db


# In[21]:


model.set_weights([a, b])


# In[22]:


a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
x = np.linspace(-15, 20, 2)
plt.plot(x, a*x+b, c='r', lw=2)


# In[23]:


preds = model(X_train1)
mse = mean_squared_error(y_train, preds[:, 0])
mse


# In[24]:


def gradient_descent_step(model, lr):
    with tf.GradientTape() as g:
        preds = model(X_train1)
        mse = mean_squared_error(y_train, preds[:, 0])
    dl_da, dl_db = g.gradient(mse, model.trainable_weights)
    a, b = model.get_weights()
    a -= lr * dl_da
    b -= lr * dl_db
    model.set_weights([a, b])
    return (a, b), mse.numpy()


# In[25]:


model = Sequential([Dense(1, input_shape=(1,))])


# In[26]:


from IPython.display import clear_output
from time import sleep


# In[27]:


def plot_line():
    a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
    plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
    x = np.linspace(-15, 20, 2)
    plt.plot(x, a*x+b, c='r', lw=2)
    plt.show()


# In[28]:


params = []
loss = []
for _ in range(20):
    p, l = gradient_descent_step(model, 1e-3)
    params.append(p)
    loss.append(l)
    plot_line()
    sleep(0.2)
    clear_output(True)


# In[29]:


loss[-1]


# In[30]:


plt.plot(loss);


# In[31]:


model = Sequential([Dense(1, input_shape=(1,))])


# In[32]:


model.compile(tf.keras.optimizers.SGD(1e-3), 'mse')


# In[33]:


model.fit(
    X_train1, 
    y_train.values, 
    epochs=20,
    batch_size=len(X_train1)
)


# In[34]:


model = Sequential([Dense(1, input_shape=(1,))])
model.compile(tf.keras.optimizers.SGD(1e-3), 'mse')
model.fit(
    X_train1, 
    y_train.values, 
    batch_size=128,
    epochs=2,
)


# In[35]:


X_train.shape, y_train.shape


# In[36]:


model = Sequential([Dense(1, input_shape=(22,))])


# In[37]:


model.compile(tf.keras.optimizers.SGD(1e-4), 'mse')


# In[38]:


model.summary()


# In[39]:


model.fit(
    X_train.values,
    y_train.values,
    validation_data=(X_valid.values, y_valid.values)
)


# In[40]:


get_ipython().set_next_input('Oh crap, something went wrong... What could it be');get_ipython().run_line_magic('pinfo', 'be')


# In[41]:


Oh crap, something went wrong... What could it be


# In[42]:


X_train.std().plot.bar()
plt.yscale('log')


# In[43]:


mean = X_train.mean()
std = X_train.std()
X_train_norm = ((X_train - mean) / std).values
X_valid_norm = ((X_valid - mean) / std).values
X_test_norm = ((X_test - mean) / std).values
y_train = y_train.values
y_valid = y_valid.values


# In[44]:


X_train_norm.std(0)


# In[45]:


model = Sequential([Dense(1, input_shape=(22,))])


# In[46]:


model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')


# In[47]:


model.fit(
    X_train_norm,
    y_train,
    batch_size=128,
    epochs=4,
    validation_data=(X_valid_norm, y_valid)
)


# In[48]:


from sklearn.metrics import r2_score, mean_squared_error
def print_scores(model, X_train=X_train_norm, X_valid=X_valid_norm):
    preds_train = model.predict(X_train, batch_size=10_000)
    preds_valid = model.predict(X_valid, batch_size=10_000)
    r2_train = r2_score(y_train, preds_train)
    r2_valid = r2_score(y_valid, preds_valid)
    mse_train = mean_squared_error(y_train, preds_train)
    mse_valid = mean_squared_error(y_valid, preds_valid)
    print(f'Train R2 = {r2_train}\nValid R2 = {r2_valid}\nTrain MSE = {mse_train}\nValid MSE = {mse_valid}')


# In[49]:


print_scores(model)


# In[50]:


model = Sequential([
    Dense(32, input_shape=(22,), activation='relu'),
    Dense(1, activation='linear')
])


# In[51]:


model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')


# In[52]:


model.fit(X_train_norm, y_train, 1024, epochs=12, validation_data=(X_valid_norm, y_valid))


# In[53]:


print_scores(model)


# In[54]:


model = Sequential([
    Dense(256, input_shape=(22,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])


# In[55]:


model.summary()


# In[56]:


model.compile(tf.keras.optimizers.Adam(1e-4), 'mse')


# In[57]:


model.summary()


# In[58]:


h = model.fit(X_train_norm, y_train, 1024, epochs=30, validation_data=(X_valid_norm, y_valid))


# In[59]:


plt.plot(h.history['loss'][1:])
plt.plot(h.history['val_loss'][1:])


# In[60]:


early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


# In[61]:


model = Sequential([
    Dense(256, input_shape=(22,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])


# In[62]:


model.compile(tf.keras.optimizers.Adam(1e-4), 'mse')


# In[63]:


model.fit(X_train_norm, y_train, 1024, epochs=30, validation_data=(X_valid_norm, y_valid), callbacks=[early_stopping])


# In[64]:


print_scores(model)


# In[65]:


preds = model.predict(X_test, 10_000).squeeze()
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.to_csv('submission1.csv', index=False)


# In[66]:


split_date = '2015-01-01'
df_train = df_train.dropna(subset=['t2m_obs'])
stations = df_train.station
stations_train = df_train.station[df_train.time < split_date]
stations_valid = df_train.station[df_train.time >= split_date]
stations_test = df_test.station


# In[67]:


stations_train.head()


# In[68]:


unique_stations = pd.concat([df_train.station, df_test.station]).unique()


# In[69]:


len(unique_stations)


# In[70]:


stat2id = {s: i for i, s in enumerate(unique_stations)}


# In[71]:


ids = stations.apply(lambda x: stat2id[x])


# In[72]:


ids_train = ids[df_train.time < split_date]
ids_valid = ids[df_train.time >= split_date]
ids_test = stations_test.apply(lambda x: stat2id[x])


# In[73]:


ids_train.head()


# In[74]:


features_in = Input(shape=(22,))
id_in = Input(shape=(1,))


# In[75]:


emb_layer = Embedding(len(unique_stations), 2)
emb = emb_layer(id_in)


# In[76]:


emb_layer.get_weights()[0].shape


# In[77]:


id_in, emb


# In[78]:


emb = Flatten()(emb)
emb


# In[79]:


x = Concatenate()([features_in, emb])
x


# In[80]:


x = Dense(100, activation='relu')(x)
out = Dense(1, activation='linear')(x)


# In[81]:


model = tf.keras.models.Model(inputs=[features_in, id_in], outputs=out)


# In[82]:


model.summary()


# In[83]:


tf.keras.utils.plot_model(model)


# In[84]:


model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')


# In[85]:


model.fit([X_train_norm, ids_train], y_train, 1024, 20, 
          validation_data=([X_valid_norm, ids_valid], y_valid))


# In[86]:


print_scores(model, X_train=[X_train_norm, ids_train], X_valid=[X_valid_norm, ids_valid])


# In[87]:


preds = model.predict([X_test, ids_test], 10_000).squeeze()
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.to_csv('submission2.csv', index=False)


# In[ ]:




