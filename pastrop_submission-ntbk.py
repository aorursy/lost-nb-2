#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub


# In[ ]:


SEQUENCE_LENGTH = 256

DATA_PATH =  "../input/jigsaw-multilingual-toxic-comment-classification" # data location
UPLOADED_DATA = '../input/train-toxic-seqlen256/'


# In[ ]:


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
    tpu_strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


#df = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv',
                #nrows = 200000)
df = pd.read_csv('/kaggle/input/toxic-comments-256/jigsaw-toxic-comment-train-processed-seqlen256.csv')
df.head(3)


# In[ ]:


#simplification of the code for the tf dataset creation - Test Dataset
test = df.filter(['toxic','input_word_ids','input_mask','segment_ids'])
#test = test.rename(columns={"all_segment_id": "segment_ids"})
test.head(3)


# In[ ]:


#test.to_csv('test_processed_128.csv', index = False, mode='w')
test.to_csv('test_processed_128.csv', index = False, mode='w')
#validate.to_csv('validate_processed_256.csv', index = False, mode='w')


# In[ ]:


def get_dataset(file_path = '../working/test_processed_128.csv'):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # Artificially small to make examples easier to show.
      label_name='toxic',
      na_value="?",
      num_epochs=1,
      shuffle=False)
    return dataset


# In[ ]:


train_data = get_dataset()
train_data = train_data.unbatch()
#validate_data = get_dataset('../working/validate_processed_128.csv')
#validate_data = validate_data.unbatch()


# In[ ]:


next(iter(train_data))


# In[ ]:


validation_data = get_dataset('../working/validate_processed_256.csv')
validation_data = validation_data.unbatch()


# In[ ]:


train_bias_set = get_dataset('../input/bias-train-filtered-256/jigsaw-unintended-bias-train-filtered-seqlen256.csv')
train_bias_set = train_bias_set.unbatch()


# In[ ]:


def parse_string_list_into_ints(strlist):
    s = tf.strings.strip(strlist)
    s = tf.strings.substr(s, 1, tf.strings.length(s) - 2)  # Remove parentheses around list
    #s = tf.strings.split(s, ',', maxsplit=128)
    s = tf.strings.split(s, ',', maxsplit=256)
    s = tf.strings.to_number(s, tf.int32)
    #s = tf.reshape(s, [128])  # Force shape here needed for XLA compilation (TPU)
    s = tf.reshape(s, [256])  # Force shape here needed for XLA compilation (TPU)
    return s


# In[ ]:


# prototype function to process the dataset for the Bert layer
def elem_mod(data,label):
    for k,v in data.items():
        data[k] = parse_string_list_into_ints(v)
    return data,label 


# In[ ]:


result = train_data.map(lambda x,y:elem_mod(x,y))


# In[ ]:


next(iter(result))


# In[ ]:


result_val = validation_data.map(lambda x,y:elem_mod(x,y))


# In[ ]:


result_bias = train_bias_set.map(lambda x,y:elem_mod(x,y))


# In[ ]:


def make_dataset_pipeline(dataset, repeat_and_shuffle=True):
    """Set up the pipeline for the given dataset.   
    Caches, repeats, shuffles, and sets the pipeline up to prefetch batches."""
    cached_dataset = dataset.cache()
    if repeat_and_shuffle:
        cached_dataset = cached_dataset.shuffle(2048)
    #cached_dataset = cached_dataset.batch(32 * strategy.num_replicas_in_sync)
    cached_dataset = cached_dataset.batch(32)
    cached_dataset = cached_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return cached_dataset


# In[ ]:


#cached_train_data = make_dataset_pipeline(result)
train_set = make_dataset_pipeline(result)


# In[ ]:


cached_train_bias_data = make_dataset_pipeline(result_bias)


# In[ ]:


next(iter(train_set))


# In[ ]:


#Building the model (reformat as a function...)
max_seq_length = 256  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
# BERT layer from pretrained model
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",trainable=True)
# Dense Layers
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)


# In[ ]:


# Model
model = tf.keras.Model(inputs={'input_word_ids': input_word_ids,
                                  'input_mask': input_mask,
                                  'all_segment_id': segment_ids},
                          outputs=output)


# In[ ]:


#strategy = tf.distribute.MirroredStrategy()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    metrics=[tf.keras.metrics.AUC()])

model.summary()


# In[ ]:


# Model for TPU distribution:
with strategy.scope():
    max_seq_length = 256  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    # BERT layer from pretrained model
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",trainable=True)
    # Dense Layers
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)


# In[ ]:


# Train on English Wikipedia comment data.
history = model.fit(
    # Set steps such that the number of examples per epoch is fixed.
    # This makes training on different accelerators more comparable.
    train_set,steps_per_epoch=4000//256,
    epochs=50, verbose=1)


# In[ ]:


results = model.evaluate(cached_train_data,
                                     steps=100, verbose=0)
print('\nEnglish loss, AUC before training:', results)


# In[ ]:


#TPU based model DON'T RUN WITHOUT TPU
# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = tf.keras.Sequential( … ) # define your model normally
    model.compile( … )

