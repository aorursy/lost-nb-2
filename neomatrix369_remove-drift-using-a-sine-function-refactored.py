#!/usr/bin/env python
# coding: utf-8

# In[1]:


# suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os


# In[3]:


# prettify plots
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")


# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 5.0]


# In[4]:


get_ipython().run_cell_magic('time', '', 'df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")\ndf_test  = pd.read_csv("../input/liverpool-ion-switching/test.csv")')


# In[5]:


training_batch_size = 500_000
training_batch_range = [
    [0, 500000],
    [500000, 1000000],
    [1000000, 1500000],
    [1500000, 2000000],
    [2000000, 2500000],
    [2500000, 3000000],
    [3000000, 3500000],
    [3500000, 4000000],
    [4000000, 4500000],
    [4500000, 5000000]
]

test_batch_size = 100_000
test_batch_range = [
    [0, 100000],
    [100000, 200000],
    [200000, 300000],
    [300000, 400000],
    [400000, 500000],
    [500000, 600000],
    [600000, 700000],
    [700000, 800000],
    [800000, 900000],
    [900000, 1000000],
    [1000000, 1500000], # 11th batch (500_000)
    [1500000, 2000000]  # 12th batches (500_000)
]

def make_batches(df, batch_range):
    batches = []
    for start_batch, end_batch in batch_range:
        print(f"[start_batch: {start_batch}, end_batch: {end_batch}],")
        batches.append(df[start_batch: end_batch])
        
    return batches


# In[6]:


df_train_batched = make_batches(df_train, training_batch_range)


# In[7]:


df_test_batched = make_batches(df_test, test_batch_range)


# In[8]:


def plot_all(name, dataset, sublplot_index, start, end, increment):
    plt.figure(figsize=(25, 5))
    plt.subplot(sublplot_index)
    plt.title(name)
    plt.ylabel("Signal")
    plt.xticks(np.arange(start, end, increment))
    for x in dataset:
        plt.plot(x['time'], x['signal'], linewidth=.1)
    plt.grid()

plot_all("Train Original",df_train_batched, 211, 0, 501, 50)
plot_all("Test Original", df_test_batched, 212, 500, 701, 10)


# In[9]:


# ---- Linear drift 

linear_train_idx = [1]
linear_test_idx = [0, 1, 4, 6, 7, 8]


def poly1(x, a, b):
    return a * (x - b)


def linear_drift_fit(x, y):
    popt, _ = curve_fit(poly1, x, y)
    print("Linear drift, popt:", popt)
    return popt


def linear_drift(x, x0):
    return 0.3 * (x - x0)


def my_sin(x, A, ph, d):
    frequency = 0.01
    omega = 2 * np.pi * frequency
    return A * np.sin(omega * x + ph) + d


def remove_linear_drift(linear_idx, data, time_column_name, signal_column_name, batch_start, batch_end):
    for idx in linear_idx:
        data[idx].loc[data[idx].index[batch_start: batch_end], signal_column_name] =             data[idx][signal_column_name][batch_start: batch_end].values - linear_drift(
            data[idx][time_column_name][batch_start: batch_end].values, data[idx][time_column_name][0:1].values)

    return data


# In[10]:


plt.figure(figsize=(30, 4))
plt.subplot("171")
plt.title("Train 1 (part)")
plt.ylabel("Signal", fontsize=8)
plt.plot(df_train_batched[1]['time'][0:100000], df_train_batched[1]['signal'][0:100000], linewidth=.1)
plt.grid()
plt.ylim([np.min(df_train_batched[1]['signal'][0:100000]), np.min(df_train_batched[1]['signal'][0:100000]) + 15])
for n, idx in enumerate(linear_test_idx):
    plt.subplot("17" + str(n + 2))
    plt.title("Test " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.ylim([np.min(df_test_batched[idx]['signal']), np.min(df_test_batched[idx]['signal']) + 15])
    plt.plot(df_test_batched[idx]['time'], df_test_batched[idx]['signal'], linewidth=.1)
    plt.grid()


# In[11]:


linear_params = []
train_data = df_test_batched[linear_train_idx[0]][0:100000]
linear_params.append(linear_drift_fit(train_data['time'], train_data['signal']))
for idx in linear_test_idx:
    linear_params.append(linear_drift_fit(df_test_batched[idx]['time'], df_test_batched[idx]['signal']))


# In[12]:


plt.figure(figsize=(30, 4))
plt.subplot("171")
plt.title("Train 1 (part)")
plt.ylabel("Signal", fontsize=8)
plt.plot(df_train_batched[1]['time'][0:100000], df_train_batched[1]['signal'][0:100000], linewidth=.1)
plt.plot(df_train_batched[1]['time'][0:100000], poly1(df_train_batched[1]['time'][0:100000], *linear_params[0]), 'y')
plt.grid()
plt.ylim([np.min(df_train_batched[1]['signal'][0:100000]), np.min(df_train_batched[1]['signal'][0:100000]) + 15])
for n, idx in enumerate(linear_test_idx):
    plt.subplot("17" + str(n + 2))
    plt.title("Test " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.ylim([np.min(df_test_batched[idx]['signal']), np.min(df_test_batched[idx]['signal']) + 15])
    plt.plot(df_test_batched[idx]['time'], df_test_batched[idx]['signal'], linewidth=.1)
    plt.plot(df_test_batched[idx]['time'], poly1(df_test_batched[idx]['time'], *linear_params[1 + n]), 'y')
    plt.grid()


# In[13]:


df_train_linear_drift_removed = remove_linear_drift([1], df_train_batched, 'time', 'signal', 0, test_batch_size)
df_test_linear_drift_removed = remove_linear_drift(linear_test_idx, df_test_batched, 'time', 'signal', 0, test_batch_size)


# In[14]:


plot_all("Train - Linear Drift Removed", df_train_linear_drift_removed, "211", 0, 501, 50)
plot_all("Test - Linear Drift Removed", df_test_linear_drift_removed, "212", 500, 701, 10)


# In[15]:


# --- Parabolic drift

parabola_train_idx = [6, 7, 8, 9]
parabola_train_time = [0, 1, 0, 1]
parabola_test_idx = [10]

def parabolic_drift_fit(x, y):
    frequency = 0.01
    omega = 2 * np.pi * frequency
    M = np.array([[np.sin(omega * t), np.cos(omega * t), 1] for t in x])
    y = np.array(y).reshape(len(y), 1)

    (theta, _, _, _) = np.linalg.lstsq(M, y)

    A = np.sqrt(theta[0, 0] ** 2 + theta[1, 0] ** 2)
    ph = math.atan2(theta[1, 0], theta[0, 0])
    d = theta[2, 0]

    popt = [A, ph, d]
    print("Parabolic drift, popt", popt)
    return popt


def parabolic_drift(x, t=0):
    f = 0.01
    omega = 2 * np.pi * f
    return 5 * np.sin(omega * x + t * np.pi)


def remove_parabolic_drift(parabola_idx, parabola_time, data, time_column_name, signal_column_name, batch_start, batch_end):
    for idx, ctr in zip(parabola_idx, range(len(parabola_idx))):
        target_index = data[idx].index[batch_start:batch_end]
        target_values = data[idx][time_column_name][batch_start:batch_end].values
        data[idx].loc[target_index, signal_column_name] =             data[idx][signal_column_name][batch_start:batch_end].values -             parabolic_drift(target_values, parabola_time[ctr])

    return data


# In[16]:


def plot_parabolic_drift(dataframe, name, subplot, parabola_indices):
    for n, idx in enumerate(parabola_indices):
        plt.subplot(str(subplot + n + 1))
        plt.title(name.strip() + " " + str(idx))
        plt.ylabel("Signal", fontsize=8)
        plt.plot(dataframe[idx]['time'], dataframe[idx]['signal'], linewidth=.1)
        plt.grid()
        plt.ylim([np.min(dataframe[idx]['signal']), np.min(dataframe[idx]['signal']) + 18])

plt.figure(figsize=(30, 4))
plot_parabolic_drift(df_train_linear_drift_removed, "Train", 150, parabola_train_idx)
plot_parabolic_drift(df_test_linear_drift_removed, "Test", 154, parabola_test_idx)


# In[17]:


parabola_params = []
for idx in parabola_train_idx:
    data = df_train_linear_drift_removed[idx]
    parabola_params.append(parabolic_drift_fit(data['time'], data['signal']))
data = df_test_linear_drift_removed[parabola_test_idx[0]]
parabola_params.append(parabolic_drift_fit(data['time'], data['signal']))


# In[18]:


plt.figure(figsize=(30, 4))
for n, idx in enumerate(parabola_train_idx):
    plt.subplot("15" + str(n + 1))
    plt.title("Train " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.plot(df_train_linear_drift_removed[idx]['time'], df_train_linear_drift_removed[idx]['signal'], linewidth=.1)
    plt.plot(df_train_linear_drift_removed[idx]['time'], my_sin(df_train_linear_drift_removed[idx]['time'], *parabola_params[n]), 'y')
    plt.grid()
    plt.ylim([np.min(df_train_linear_drift_removed[idx]['signal']), np.min(df_train_linear_drift_removed[idx]['signal']) + 18])
plt.subplot("155")
plt.title("Test 10")
plt.ylabel("Signal", fontsize=8)
plt.ylim([np.min(df_test_linear_drift_removed[10]['signal']), np.min(df_test_linear_drift_removed[10]['signal']) + 18])
plt.plot(df_test_linear_drift_removed[10]['time'], df_test_linear_drift_removed[10]['signal'], linewidth=.1)
plt.plot(df_test_linear_drift_removed[10]['time'], my_sin(df_test_linear_drift_removed[10]['time'], *parabola_params[-1]), 'y')
plt.grid()


# In[19]:


df_train_parabolic_drift_removed = remove_parabolic_drift(parabola_train_idx, parabola_train_time, df_train_linear_drift_removed, 'time', 'signal', 0, training_batch_size)
df_test_parabolic_drift_removed = remove_parabolic_drift([10], [0], df_test_linear_drift_removed, 'time', 'signal', 0, training_batch_size)


# In[20]:


plot_all("Train - Without Drift (linear or parabolic)", df_train_parabolic_drift_removed, "211", 0, 501, 50)
plot_all("Test - Without Drift (linear or parabolic)", df_test_parabolic_drift_removed, "212", 500, 701, 10)


# In[21]:


def plot_dist(data, labels, m):
    plt.title("Signal Distribution Model " + str(m))
    for i, x in enumerate(data):
        x = x['signal']
        sns.distplot(x, label=labels[i], kde=True, bins=np.arange(np.min(x), np.max(x), 0.01))
#         sns.distplot(x, label=labels[i], kde=True)
    plt.xlabel("signal value")
    plt.ylabel("frequency")
    plt.legend(loc="best")    
    

M = [[df_train_parabolic_drift_removed[0], df_train_parabolic_drift_removed[1], df_test_parabolic_drift_removed[0], df_test_parabolic_drift_removed[3], df_test_parabolic_drift_removed[8], df_test_parabolic_drift_removed[10], df_test_parabolic_drift_removed[11]],
     [df_train_parabolic_drift_removed[2], df_train_parabolic_drift_removed[6], df_test_parabolic_drift_removed[4]],
     [df_train_parabolic_drift_removed[3], df_train_parabolic_drift_removed[7], df_test_parabolic_drift_removed[1], df_test_parabolic_drift_removed[9]],
     [df_train_parabolic_drift_removed[4], df_train_parabolic_drift_removed[9], df_test_parabolic_drift_removed[5], df_test_parabolic_drift_removed[7]],
     [df_train_parabolic_drift_removed[5], df_train_parabolic_drift_removed[8], df_test_parabolic_drift_removed[2], df_test_parabolic_drift_removed[6]]]
labels = [["train 0", "train 1 (line)", "test 0 (line)", "test 3", "test 8 (line)", "test 10 (sine)", "test 11"],
          ["train 2", "train 6 (sine)", "test 4 (line)"],
          ["train 3", "train 7 (sine)", "test 1 (line)", "test 9"],
          ["train 4", "train 9 (sine)", "test 5", "test 7 (line)"],
          ["train 5", "train 8 (sine)", "test 2", "test 6 (line)"]]

plt.figure(figsize=(25, 8))
for i in range(5):
    plt.subplot("15" + str(i + 1))
    plot_dist(M[i], labels[i], i)


# In[22]:


def save_dataframe(dataframe: pd.DataFrame,
                   filename_with_path: str,
                   force_overwrite=False):
    print("Shape:", dataframe.shape)
    print("Contents:\n", dataframe)
    print()
    print(f'force_overwrite = {force_overwrite}')
    if force_overwrite or (not os.path.exists(filename_with_path)):
        print(f"Saving dataframe to {filename_with_path}.")
        dataframe.to_csv(filename_with_path, index=False, float_format='%.9f',
                                       chunksize=100000, compression='gzip', encoding='utf-8')
    else:
        print(f"{filename_with_path} already exists, not overwriting. Remove it and try again.")


df_train_clean = df_train_parabolic_drift_removed[0]
df_test_clean = df_test_parabolic_drift_removed[0]
for df in df_train_parabolic_drift_removed[1:]:
    df_train_clean = pd.concat([df_train_clean, df], ignore_index=True)
for df in df_test_parabolic_drift_removed[1:]:
    df_test_clean = pd.concat([df_test_clean, df], ignore_index=True)


# In[23]:


df_train_clean


# In[24]:


df_test_clean


# In[25]:


save_dataframe(df_train_clean, "train_wo_drift_sine.csv.gz")


# In[26]:


save_dataframe(df_test_clean, "test_wo_drift_sine.csv.gz")

