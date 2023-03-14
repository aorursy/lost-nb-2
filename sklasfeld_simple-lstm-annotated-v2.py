#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# import keras libraries
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
# import pytorch
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# other libraries
import time
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from fastai.basics import *
from fastai.basic_train import Learner
from fastai.callbacks.general_sched import *
import gc


# In[3]:


# universal parameter settings

# identity columns that are featured in the testing data
# according to the data description of the competition
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

# columns that describe the comment
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

# column with text data that will need to be converted for processing
TEXT_COLUMN = 'comment_text'

# column we eventually need to predict
TARGET_COLUMN = 'target'


# In[4]:


# characters that we can ignore when tokenizating the TEXT_COLUMN
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'


# In[5]:


# Rate at which comments are dropped for training
# too high can underfit
# too low can overfit
DROPOUT_RATE = 0.2

# NUMBER OF EPOCHS
# One Epoch is when an entire dataset is passed forward and backward
# through the neural network once.
EPOCHS = 2

# dimensions of the output vectors of each LSTM cell.
# Too high can overfit
# Too low can underfit
# The length of this vector reflects the number of
# Bidirectional CuDNNLSTM layers there will be
LSTM_UNITS = 128


# dimensions of the densely-connected NN layer cells.
# The length of this vector reflects the number of
# Dense layers there will be
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS


# In[6]:


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


# In[7]:


#sample_weights = torch.from_numpy(train_df[TARGET_COLUMN].values[:,np.newaxis])
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    #train_df[column] = np.where(train_df[column] >= 0.5, True, False)
    train_df[column] = np.where(train_df[column] >= 0.5, 1, 0)


# In[8]:


x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values[:,np.newaxis]
#y_aux_train = train_df[AUX_COLUMNS].values
#y_aux_train[:,1:] = np.where(y_aux_train[:,1:] >= .5,1,0)
x_test = test_df[TEXT_COLUMN].astype(str)


# In[9]:


tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))


# In[10]:


x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# In[11]:


batch_size = 1024


# In[12]:


#y_train_torch = torch.from_numpy(y_train).float()
#y_aux_train_torch = torch.from_numpy(y_aux_train).float()
#y_train_torch = torch.cat([y_train_torch.unsqueeze(1),y_aux_train_torch],1)


# In[13]:


y_train_torch = torch.from_numpy(y_train)
y_train_torch = y_train_torch.float()


# In[14]:


train_lengths = torch.from_numpy(np.array([len(x) for x in x_train]))
test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))


# In[15]:


maxlen = train_lengths.max() # length of longest comment


# In[16]:


x_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))
print("x_train_padded size:")
print(x_train_padded.shape)

x_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=maxlen))
print("x_test_padded size:")
print(x_test_padded.shape)
# save the space within RAM
del x_train, x_test


# In[17]:


# The following object calls a `batch` which is a 
# TensorDataset that contains two or three items:
# 1. Mandatory - a torch object with a matrix that has rows containing 
#    word incides for each comment (eg. x_train_padded)
# 2. Mandatory - a torch object that contains a list of the lengths of 
#    each of these comments in the same order as the matrix
#   (eg. train_lengths)
# 3. Optional- a torch object that contins a list of the 
# target values for each comment (eg. y_train_torch)
class SequenceBucketCollator():
    # initalizing features
    # choose_length - function to choose uniform length of each comment
    # sequence_index - index in Tensor Dataset where a torch object with 
    # a matrix that has rows containing word incides for each comment 
    # is located.
    # length_index - index in Tensor Dataset where a list of the lengths of 
    # each of these comments in the same order as the matrix
    # is located.
    # label_index - index in Tensor Dataset where a torch object that contins
    # a list of the target values for each comment is located (Optional)
    def __init__(self, choose_length, sequence_index, length_index, label_index=None, weight_index = None):
        self.choose_length = choose_length 
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.weight_index = weight_index
        self.label_index = label_index
    
    # An example of batch is:
    # data.TensorDataset(x_train_padded, train_lengths, y_train_torch)
    def __call__(self, batch):
        # make a list 
        # eg. [x_train_padded, train_lengths, y_train_torch]
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        # put the padded comment matrix in a variable `sequences`
        sequences = batch[self.sequence_index]
        
        # put list of lengths of the comments in a variable `lengths`
        lengths = batch[self.length_index]
        
        # set uniform length to set all the comments to
        length = self.choose_length(lengths) 
        
        # add 0's to the comments that are shorter than `length`
        mask = torch.arange(start=maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        # reset the batch sequences
        batch[self.sequence_index] = padded_sequences
        
        # if present, add target labels to the batch
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i not in [self.label_index,self.weight_index]],[batch[self.label_index],batch[self.weight_index]]

        return batch


# In[18]:


train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(np.bool8)
train_df[IDENTITY_COLUMNS] = train_df[IDENTITY_COLUMNS].astype(np.bool8)
# first we make all comments equal weights of 1
# this makes the math easier later
sample_weights = np.ones(len(x_train_padded), dtype=np.float32)
# then we add weight to columns with identity labels
# if a comment has more identity labels it gets more
# weight
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1).values
# if the comment is labeled a toxic, then we add
# the amount of identity columns that are not
# labeled
sample_weights += np.abs(train_df[TARGET_COLUMN] *     (~train_df[IDENTITY_COLUMNS]).sum(axis=1).values)
sample_weights += np.abs(train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1).values)

# if the comment is NOT labeled as toxic then we add
# 5 times the amount of identity columns
sample_weights += np.abs(~train_df[TARGET_COLUMN]) *     train_df[IDENTITY_COLUMNS].sum(axis=1).values * 5
# then we normalize the weights by dividing them all by the mean
sample_weights /= sample_weights.mean()
sample_weights = torch.from_numpy(sample_weights.values[:,np.newaxis]).float()


# In[19]:


train_dataset = data.TensorDataset(x_train_padded, train_lengths, y_train_torch, sample_weights)


# In[20]:


test_dataset = data.TensorDataset(x_test_padded, test_lengths)


# In[21]:


valid_dataset = data.Subset(train_dataset, indices=[0,1])


# In[22]:


# initialize the SequenceBucketCollator objects
train_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(), 
                                        sequence_index=0, 
                                        length_index=1, 
                                        label_index=2,
                                        weight_index = 3)
test_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(), sequence_index=0, length_index=1)

# run the SequenceBucketCollator method to uniformly change 
# each of the comments in each batch to be the size of the 
# longest comment in each batch
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator)

databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)


# In[23]:


#fastText_wordEmbedder_f = '../input/fasttextsubword/crawl-300d-2m-subword/crawl-300d-2M-subword.vec'
#fastText_wordEmbedder_f='../input/fasttext-toxic/crawl-300d-2M.vec'
fastText_wordEmbedder_f='../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
glove_wordEmbedder_f='../input/glove840b300dtxt/glove.840B.300d.txt'
glove_twitter = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'


# In[24]:


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float16')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path,dim = 300):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def create_embedding_index(file,header=False):
    f = open(file,'r')
    lines = []
    if header:
        line_vec = f.readlines()[1:]
    else:
        line_vec = f.readlines()
    return dict(get_coefs(*line.strip().split(' ')) for line in line_vec)


# In[25]:


fastText_embedding_index,unknown_words_ft =       build_matrix(tokenizer.word_index, fastText_wordEmbedder_f)
    #create_embedding_index(fastText_wordEmbedder_f,True)


# In[26]:


glove_embedding_index,_ = build_matrix(tokenizer.word_index,glove_wordEmbedder_f,300)


# In[27]:


glove_twitter,_ = build_matrix(tokenizer.word_index,glove_twitter,200)


# In[28]:


embedding_matrix = np.concatenate([fastText_embedding_index, glove_embedding_index,glove_twitter], axis=-1)
embedding_matrix.shape

del glove_embedding_index
del fastText_embedding_index
del glove_twitter
gc.collect()


# In[29]:


word_vec_size=300


# In[30]:


def get_embedding_matrix(word_index, embedding_index, word_vec_size):
    # toxinizer_vocab = all word indexes plus index 0
    toxinizer_vocab = len(word_index) +1
    embedding_matrix = np.zeros((toxinizer_vocab, word_vec_size))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return(embedding_matrix)


# In[31]:


"""
fastText_embedding_matrix = get_embedding_matrix(tokenizer.word_index,fastText_embedding_index,word_vec_size)

glove_embedding_matrix = get_embedding_matrix(tokenizer.word_index,glove_embedding_index,word_vec_size)
#embedding_matrix = np.concatenate([fastText_embedding_matrix, glove_embedding_matrix], axis=-1)

#del fastText_embedding_matrix
#del glove_embedding_matrix
#gc.collect()
"""


# In[32]:


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


# In[33]:


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


# In[34]:


a = torch.rand(10)
for s in range(3):
    if s != 1:
        a = a.unsqueeze(s)


# In[35]:


def weighted_avg(tensor,dim):
    weight_array = np.arange(1,tensor.shape[dim] + 1)
    
    weight_array = weight_array / np.sum(weight_array)
    weight_array = torch.from_numpy(weight_array).float().cuda()
    for a in range(len(tensor.shape)):
        if a != dim:
            weight_array = weight_array.unsqueeze(a)
    weighted_tensor = tensor * weight_array
    return torch.sum(weighted_tensor,dim)
    


# In[36]:


class NeuralNet(nn.Module):
    # initializing parameters:
    ## embedding matrix - 2D matrix containing a unique vectors in each row 
    ## that corresponds to words based on each word indexes in a specific 
    ## vocabulary
    ## num_aux_targets - number of AUX columns in training set
    ## drouput_rate - rate at which input layer drops out comments
    ## lstm_units - dimension of the lstm outputs
    ## dense_hidden_units  - dimension of the dense-layer outputs
    def __init__(self, embedding_matrix, dropout_rate,
                lstm_units, dense_hidden_units):
        super(NeuralNet, self).__init__()
        
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        
        # Create a table using nn.Embedding shaped by the size of the 
        ## vocabulary (vocab_size) by the size of the word vectors 
        ## (embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # set the embedding.weight based on the embedding matrix that
        # was created using the word em
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,
                                                          dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(dropout_rate)
        
        
        #self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        #self.lstm1 = BNLSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        #self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True,dropout=.5)
        #self.lstm2 = BNLSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(embed_size,lstm_units,bidirectional=True,batch_first=True,num_layers=2)
        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)
        #self.linear2 = nn.Sequential(nn.Linear(dense_hidden_units, int(dense_hidden_units / 2)),nn.BatchNorm1d(int(dense_hidden_units / 2)),nn.ReLU(),nn.Linear(int(dense_hidden_units / 2), dense_hidden_units))
        
        self.dropout = nn.Dropout(.2)
        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.bn = nn.BatchNorm1d(dense_hidden_units)
        #self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm2, _ = self.lstm1(h_embedding)
        #h_lstm2, _ = self.lstm2(h_lstm1)
        # see what happens if we add skip connection here
        #h_lstm2 = h_lstm2 + h_lstm1
        # global average pooling
        avg_pool_1 = torch.mean(h_lstm2, 1)
        #avg_pool_2 = weighted_avg(h_lstm2,1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool_1), 1)
        h_conc_linear1  = F.relu(self.bn(self.linear1(h_conc)))
        h_conc_linear2  = self.dropout(F.relu(self.bn(self.linear2(h_conc))))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        #aux_result = self.linear_aux_out(hidden)
        #out = torch.cat([result, aux_result], 1)
        #print(out.dtype)
        return result


# In[37]:


def custom_loss(data, targets, weights):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=weights.float())(data[:,:1].float(),targets[:,:1].float())
    #bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,2:].float(),targets[:,1:].float())
    #return (bce_loss_1 ) + bce_loss_2
    return bce_loss_1


# In[38]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[39]:


def train_model(learn,test,output_dim,lr=0.002,
                batch_size=512, n_epochs=5,
                enable_checkpoint_ensemble=True):
    
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6**(i)))).schedule_hp('wd',1e-2) for i in range(n_epochs)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    for epoch in range(n_epochs):
        learn.fit(3)
        test_preds = np.zeros((len(test), output_dim))    
        for i, x_batch in enumerate(test_loader):
            X = x_batch[0].cuda()
            y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)


    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds


# In[40]:



print('fastText Model')
seed_everything(1234 + 0)
fastText_model = NeuralNet(embedding_matrix,dropout_rate=DROPOUT_RATE,lstm_units=LSTM_UNITS,dense_hidden_units=DENSE_HIDDEN_UNITS)
fastText_learn = Learner(databunch, fastText_model, loss_func=custom_loss)

fastText_test_preds = train_model(fastText_learn,test_dataset,output_dim=1,batch_size = batch_size,n_epochs = EPOCHS)    
#all_test_preds.append(fastText_test_preds)

fastText_model_submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': np.squeeze(fastText_test_preds[:,0])
})
fastText_model_submission.to_csv('submission.csv', index=False)


# In[41]:




'''
all_test_preds = []

print('fastText Model')
seed_everything(1234 + 0)
fastText_model = NeuralNet(fastText_embedding_matrix,dropout_rate=DROPOUT_RATE,lstm_units=LSTM_UNITS,dense_hidden_units=DENSE_HIDDEN_UNITS)
fastText_learn = Learner(databunch, fastText_model, loss_func=custom_loss)

fastText_test_preds = train_model(fastText_learn,test_dataset,output_dim=1,batch_size = batch_size,n_epochs = EPOCHS)    
all_test_preds.append(fastText_test_preds)

fastText_model_submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': np.squeeze(fastText_test_preds[:,0])
})
fastText_model_submission.to_csv('fastText_submission.csv', index=False)
fastText_model_submission.to_csv('submission.csv', index=False)

print('glove Model')
seed_everything(1234 + 0)
glove_model = NeuralNet(glove_embedding_matrix,dropout_rate=DROPOUT_RATE,lstm_units=LSTM_UNITS,dense_hidden_units=DENSE_HIDDEN_UNITS)
glove_learn = Learner(databunch, glove_model, loss_func=custom_loss)
glove_test_preds = train_model(glove_learn,test_dataset,output_dim=1,batch_size = batch_size,n_epochs = EPOCHS)    
all_test_preds.append(glove_test_preds)


glove_model_submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': glove_test_preds
})
glove_model_submission.to_csv('glove_submission.csv', index=False)

'''


# In[42]:


'''
submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': np.mean(all_test_preds, axis=0)[:, 0]
})

submission.to_csv('submission.csv', index=False)
'''

