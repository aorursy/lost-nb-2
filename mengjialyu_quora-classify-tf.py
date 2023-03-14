#!/usr/bin/env python
# coding: utf-8

# In[114]:


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


# In[115]:


get_ipython().system('pip install bert-for-tf2')
get_ipython().system('pip install sentencepiece')


# In[116]:


try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
import bert


# In[117]:


question_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

question_df.isnull().values.any()

question_df.shape


# In[118]:


question_df = question_df[:50000]


# In[119]:


question_df.info


# In[120]:


import re
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

questions = []
sentences = list(question_df['question_text'])
for sen in sentences:
    questions.append(preprocess_text(sen))


# In[121]:


print(question_df.columns.values)


# In[122]:


question_df.drop("qid", axis=1, inplace=True)


# In[123]:


print(question_df.columns.values)


# In[124]:


question_df.target.unique()


# In[125]:


# text tokenization using the BERT tokenizer
BertTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# In[126]:


tokenizer.tokenize("don't be so judgmental")


# In[127]:


tokenizer.convert_tokens_to_ids(tokenizer.tokenize("dont be so judgmental"))


# In[128]:


def tokenize_questions(questions):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(questions))


# In[129]:


tokenized_questions = [tokenize_questions(question) for question in questions]


# In[130]:


target = question_df['target']


# In[131]:


target = np.array(target)


# In[132]:


questions_with_len = [[question, target[i], len(question)]
                 for i, question in enumerate(tokenized_questions)]


# In[133]:


import random
random.shuffle(questions_with_len)


# In[134]:


questions_with_len.sort(key=lambda x: x[2])


# In[135]:


sorted_questions_labels = [(question_lab[0], question_lab[1]) for question_lab in questions_with_len]


# In[136]:


processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_questions_labels, output_types=(tf.int32, tf.int32))


# In[137]:


BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))


# In[138]:


next(iter(batched_dataset))


# In[139]:


import math

TOTAL_BATCHES = math.ceil(len(sorted_questions_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)


# In[140]:


class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output


# In[141]:


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2
#  hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
NB_EPOCHS = 10


# In[142]:


text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)


# In[143]:


if OUTPUT_CLASSES == 2:
    text_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])


# In[144]:


text_model.fit(train_data, epochs=NB_EPOCHS)


# In[145]:


test_loss, test_acc = text_model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[146]:


DROPOUT_RATE = 0.5
text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
text_model.fit(train_data, epochs=NB_EPOCHS)


# In[147]:


test_loss, test_acc = text_model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[148]:


#param_grid = dict(num_filters=[32, 64, 128],
 #                 kernel_size=[3, 5, 7],
  #                vocab_size=[5000], 
   #               embedding_dim=[50],
    #              maxlen=[100])


# In[149]:


#VOCAB_LENGTH = len(tokenizer.vocab)
#EMB_DIM = 200
#CNN_FILTERS = 100
#RNN_UNITS = 256


# In[150]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_LENGTH, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[151]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[152]:


history = model.fit(train_data, epochs=10,
                    validation_data=test_data, 
                    validation_steps=30)


# In[153]:


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[154]:


import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()


# In[155]:


plot_graphs(history, 'accuracy')


# In[156]:


plot_graphs(history, 'loss')


# In[157]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_LENGTH, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


# In[158]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[159]:


history = model.fit(train_data, epochs=10,
                    validation_data=test_data,
                    validation_steps=30)


# In[160]:


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[161]:


plot_graphs(history, 'accuracy')


# In[162]:


plot_graphs(history, 'loss')


# In[163]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_LENGTH, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


# In[164]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[165]:


history = model.fit(train_data, epochs=10,
                    validation_data=test_data,
                    validation_steps=30)


# In[166]:


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[167]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_LENGTH, 64),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[168]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[169]:


history = model.fit(train_data, epochs=10,
                    validation_data=test_data,
                    validation_steps=30)


# In[170]:


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[171]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_LENGTH, 64),
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True, input_shape=(1, 2)),
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(25, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
   
])


# In[172]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[173]:


history = model.fit(train_data, epochs=10,
                    validation_data=test_data,
                    validation_steps=30)


# In[174]:


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[ ]:




