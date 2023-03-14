#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


PATH = '../input/google-quest-challenge/'


# In[3]:


train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')


# In[4]:


train.head()


# In[5]:


train_x = train.loc[:, 'qa_id':'host']


# In[6]:


train_x.head()


# In[7]:


train_y = train.loc[:, 'question_asker_intent_understanding':'answer_well_written']


# In[8]:


train_y.head()


# In[9]:


train_x = train_x[['question_title','question_body','answer']]
test_x = test[['question_title','question_body','answer']]


# In[10]:


train_x.head()


# In[11]:


test_x.head()


# In[12]:


import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
from tensorflow.keras.models import Model      
import tensorflow.keras.backend as K
import math


# In[13]:


MAX_SEQ_LENGTH = 512
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'


# In[14]:


def create_model(max_seq_length=MAX_SEQ_LENGTH,bert_path=BERT_PATH):
    # BERT needs 3 inputs: ids, masks, segments     
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    # pretrained BERT_base     
    bert_layer = hub.KerasLayer(bert_path,
                                trainable=True)
    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    # Output layer for 30 classes to predict     
    pooling = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    dropout = tf.keras.layers.Dropout(0.2)(pooling)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(dropout)

    return Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)


# In[15]:


def get_masks(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    first_sep = True
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
            
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):
    
    t_len,q_len,a_len = len(title),len(question),len(answer)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len <= t_len:
            t_new_len = t_max_len
        else:
            t_new_len = t_len
            a_max_len = a_max_len + math.floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + math.ceil((t_max_len - t_len)/2)            
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
            
        title,question,answer = title[:t_new_len], question[:q_new_len], answer[:a_new_len]
    
    return title,question,answer

def get_inputs(title, question, answer, tokenizer,max_seq_length):
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    t,q,a = trim_input(t, q, a,max_seq_length)
    stokens = ["[CLS]"] + t + ["[SEP]"] + q + ["[SEP]"] + a + ["[SEP]"]

    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)
    return input_ids,input_masks,input_segments

def compute_input_arays(df, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df.iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids, masks, segments = get_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


# In[16]:


from scipy.stats import spearmanr


# In[17]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )


# In[18]:


def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback


# In[19]:


from sklearn.model_selection import GroupKFold


# In[20]:


gkf = GroupKFold(n_splits=5).split(X=train_x.question_body, groups=train_x.question_body)


# In[21]:


tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt')


# In[22]:


inputs = compute_input_arays(train_x, tokenizer,MAX_SEQ_LENGTH)
test_inputs = compute_input_arays(test_x, tokenizer,MAX_SEQ_LENGTH)
outputs = np.asarray(train_y)


# In[23]:


histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < 3:
        K.clear_session()
        model = create_model()

        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]

        # history contains two lists of valid and test preds respectively:
        #  [valid_predictions_{fold}, test_predictions_{fold}]
        history = train_and_predict(model, 
                          train_data=(train_inputs, train_outputs), 
                          valid_data=(valid_inputs, valid_outputs),
                          test_data=test_inputs, 
                          learning_rate=3e-5, epochs=4, batch_size=8,
                          loss_function='binary_crossentropy', fold=fold)

        histories.append(history)


# In[24]:


test_predictions = [histories[i].test_predictions for i in range(len(histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub = pd.read_csv(PATH + 'sample_submission.csv')

df_sub.iloc[:, 1:] = test_predictions

df_sub.to_csv('submission.csv', index=False)


# In[ ]:




