#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import numpy as np
import pandas as pd
from fastText import load_model

window_length = 200 # The amount of words we look at per example. Experiment with this.

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

print('\nLoading data')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')


# In[ ]:


classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model')
ft_model = load_model('ft_model.bin')
n_features = ft_model.get_dimension()

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x


# In[ ]:


x_train = df_to_data(train)
y_train = train[classes].values

x_test = df_to_data(test)
y_test = test[classes].values


# In[ ]:


# Split the dataset:
split_index = round(len(train) * 0.9)
shuffled_train = train.sample(frac=1)
df_train = shuffled_train.iloc[:split_index]
df_val = shuffled_train.iloc[split_index:]

# Convert validation set to fixed array
x_val = df_to_data(df_val)
y_val = df_val[classes].values

def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data
    
    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch
        
        for i, row in df.iterrows():
            comment = row['comment_text']
            
            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')
                
            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0


# In[ ]:


model = build_model()  # TODO: Implement

batch_size = 128
training_steps_per_epoch = round(len(df_train) / batch_size)
training_generator = data_generator(df_train, batch_size)

# Ready to start training:
model.fit_generator(
    training_generator,
    steps_per_epoch=training_steps_per_epoch,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    ...
)

