#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape)


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    )

model.fit(x_train, y_train, epochs=10)


# In[ ]:


model.evaluate(x_val, y_val)


# In[ ]:


def build_model(nodes1, nodes2, lr):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,)),
        tf.keras.layers.Dense(nodes1, activation='relu'),
        tf.keras.layers.Dense(nodes2, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
        ])
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
        )
    
    return model


# In[ ]:


epochs_list = [??] # "default" is 10. check if it overfit at 10 - if not, try maybe double (and maybe more)
nodes1_list = [??] # "default" is 64 - try half and double (and maybe more)
nodes2_list = [??] # "default" is 128 - try half and double (and maybe more)
lr_list = [??] # 0.001 is default for adam, try half and double (and maybe more)

results = []

for epochs in epochs_list:
    for nodes1 in nodes1_list:
        for nodes2 in nodes2_list:
            for lr in lr_list:
                model_current = build_model(nodes1, nodes2, lr)
                model_current.fit(x_train, y_train, epochs=epochs, verbose=0)
                loss, acc = model_current.evaluate(x_val, y_val)
                results.append([loss, acc, epochs, nodes1, nodes2, lr])

results = pd.DataFrame(results, columns=['loss', 'acc', 'epochs', 'nodes1', 'nodes2', 'lr'])


# In[ ]:


results


# In[ ]:


results[results['loss'] == results['loss'].min()]


# In[ ]:


results[results['acc'] == results['acc'].max()]


# In[ ]:


model_final = build_model(???)
model_final.fit(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]), epochs=??)


# In[ ]:


y_test_hat = model_final.predict(x_test)
y_test_hat = np.argmax(y_test_hat, axis=1)
y_test_hat_pd = pd.DataFrame({
    'Id': list(range(5000)),
    'Category': y_test_hat,
})


# In[ ]:


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Category'])
assert len(y_test_hat_pd) == 5000

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)

