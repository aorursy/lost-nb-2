#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/')
get_ipython().system('pip install ../input/transformers/transformers-master/')


# In[2]:


import pandas as pd
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
DATA_DIR = '../input/google-quest-challenge'


# In[3]:


get_ipython().system('ls ../input')
get_ipython().system('ls ../input/bertbaseuncased')
get_ipython().system('ls ../input/google-quest-challenge/')


# In[4]:


sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
sub.head()


# In[5]:


target_columns = sub.columns.values[1:].tolist()
target_columns


# In[6]:


size_correlation = pd.read_csv(f"../input/size-corr/size_correlation.csv")
size_correlation.head()


# In[7]:


size_correlation.iloc[:, 1:].shape

#  21 columns question and 9 columns answers
question_corr = size_correlation.iloc[:2, 1:22].sum(axis=0).values.reshape((1, -1))
ans_corr = size_correlation.iloc[:, 22:].sum(axis=0).values.reshape((1,-1))
question_corr.shape, ans_corr.shape


# In[8]:


train = pd.read_csv(f'{DATA_DIR}/train.csv')
train.head()


# In[9]:


test = pd.read_csv(f'{DATA_DIR}/test.csv')
test.head()


# In[10]:


import torch
#import torch.utils.data as data
from torchvision import datasets, models, transforms
from transformers import *
from sklearn.utils import shuffle
import random
from math import floor, ceil
from sklearn.model_selection import GroupKFold

MAX_LEN = 512
#MAX_Q_LEN = 250
#MAX_A_LEN = 259
SEP_TOKEN_ID = 102

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df, train_mode=True, labeled=True):
        self.df = df
        self.train_mode = train_mode
        self.labeled = labeled
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('../input/bertbaseuncased/')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        ids_a, seg_ids_a, ids_q, seg_ids_q, curr_n_a, curr_n_q  = self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return (ids_a, seg_ids_a, ids_q, seg_ids_q, curr_n_a, curr_n_q, labels)
        else:
            return (ids_a, seg_ids_a, ids_q, seg_ids_q, curr_n_a, curr_n_q)

    def __len__(self):
        return len(self.df)

    def select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = random.randint(0, len(tokens)-num_remove-1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num//2] + tokens[-(max_num - max_num//2):]

    def trim_input(self, title, question, answer, max_sequence_length=MAX_LEN, 
                t_max_len=30, q_max_len=239, a_max_len=239):
        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(question)
        a = self.tokenizer.tokenize(answer)

        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len+q_len+a_len+4) > max_sequence_length:

            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len

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

            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]

        return t, q, a
        
    def get_token_ids(self, row):
        t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer)

        tokens_q = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]']
        token_ids_q = self.tokenizer.convert_tokens_to_ids(tokens_q)
        curr_n_q = len(token_ids_q)
        if len(token_ids_q) < MAX_LEN:
            token_ids_q += [0] * (MAX_LEN - len(token_ids_q))
        ids_q = torch.tensor(token_ids_q)
        seg_ids_q = self.get_seg_ids(ids_q)
        
        tokens_a = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        token_ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
        curr_n_a = len(token_ids_a)
        if len(token_ids_a) < MAX_LEN:
            token_ids_a += [0] * (MAX_LEN - len(token_ids_a))
        ids_a = torch.tensor(token_ids_a)
        seg_ids_a = self.get_seg_ids(ids_a)
        return ids_a, seg_ids_a, ids_q, seg_ids_q, curr_n_a, curr_n_q
    
    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == SEP_TOKEN_ID:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == 0)
        seg_ids[pad_idx] = 0

        return seg_ids

    def get_label(self, row):
        #print(row[target_columns].values)
        return torch.tensor(row[target_columns].values.astype(np.float32))

    def collate_fn(self, batch):
        token_ids_a = torch.stack([x[0] for x in batch])
        seg_ids_a = torch.stack([x[1] for x in batch])
        token_ids_q = torch.stack([x[2] for x in batch])
        seg_ids_q = torch.stack([x[3] for x in batch])
        curr_n_a = torch.FloatTensor( [x[4] for x in batch] ) / MAX_LEN
        curr_n_q = torch.FloatTensor( [x[5] for x in batch] ) / MAX_LEN
    
        if self.labeled:
            labels = torch.stack([x[6] for x in batch])
            return token_ids_a, seg_ids_a, token_ids_q, seg_ids_q, curr_n_a, curr_n_q, labels
        else:
            return token_ids_a, seg_ids_a, token_ids_q, seg_ids_q, curr_n_a, curr_n_q

def get_test_loader(batch_size=4):
    df = pd.read_csv(f'{DATA_DIR}/test.csv')
    ds_test = QuestDataset(df, train_mode=False, labeled=False)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)
    
    return loader
        
def get_train_val_loaders(batch_size=4, val_batch_size=4, ifold=0):
    df = pd.read_csv(f'{DATA_DIR}/train.csv')
    df = shuffle(df, random_state=1234)
    #split_index = int(len(df) * (1-val_percent))
    gkf = GroupKFold(n_splits=5).split(X=df.question_body, groups=df.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if fold == ifold:
            df_train = df.iloc[train_idx]
            df_val = df.iloc[valid_idx]
            break

    #print(df_val.head())
    #df_train = df[:split_index]
    #df_val = df[split_index:]

    print(df_train.shape)
    print(df_val.shape)

    ds_train = QuestDataset(df_train)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = QuestDataset(df_val, train_mode=False)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=0, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)
    val_loader.df = df_val

    return train_loader, val_loader

def test_train_loader():
    loader, _ = get_train_val_loaders(4, 4, 1)
    for ids_a, seg_ids_a, ids_q, seg_ids_q, n_a, n_q, labels in loader:
        print(ids_a)
        print(seg_ids_a)
        print(ids_q)
        print(seg_ids_q)
        print(labels)
        print(n_a)
        print(n_q)
        break
def test_test_loader():
    loader = get_test_loader(4)
    for ids_a, seg_ids_a, ids_q, seg_ids_q, n_a, n_q  in loader:
        print(ids_a.numpy().shape)
        print(seg_ids_a.numpy().shape)
        print(ids_q.numpy().shape)
        print(seg_ids_q.numpy().shape)
        print(n_a, n_a.numpy().shape)
        print(n_q, n_q.numpy().shape)
        break


# In[11]:


test_test_loader()
print()
print("   ################### Test Over Train Starting ###################   ")
print()
test_train_loader()


# In[12]:


from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestModel(nn.Module):
    def __init__(self, a_corr, q_corr, n_classes=30):
        super(QuestModel, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = BertModel.from_pretrained('../input/bertbaseuncased/')    
        self.fc = nn.Linear(1536, 30)
        self.q_corr = q_corr
        self.a_corr = a_corr

    def forward(self, ids_a, seg_ids_a, ids_q, seg_ids_q, n_a, n_q):
        attention_mask = (ids_a > 0)
        layers_a, pool_out_a = self.bert_model(input_ids=ids_a, token_type_ids=seg_ids_a, attention_mask=attention_mask)
        attention_mask = (ids_q > 0)
        layers_q, pool_out_q = self.bert_model(input_ids=ids_q, token_type_ids=seg_ids_q, attention_mask=attention_mask)
        #print(layers[-1][0].size())
        #print(pool_out.size())

        #out = F.dropout(layers[-1][:, 0, :], p=0.2, training=self.training)
        pool_out = torch.cat( (pool_out_q, pool_out_a), 1)
        out =  F.dropout(pool_out, p=0.2, training=self.training)
        corr_score = torch.cat( (n_q.matmul( self.q_corr ), n_a.matmul( self.a_corr )), 1)
        logit = self.fc(out)
        logit = logit + corr_score
        return logit
    
    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    
def test_model():
    x = torch.tensor([[1,2,3,4,5, 0, 0], [1,2,3,4,5, 0, 0]])
    seg_ids = torch.tensor([[0,0,0,0,0, 0, 0], [0,0,0,0,0, 0, 0]])
    x1 = torch.tensor([[1,2,3,4,5, 0, 0], [1,2,3,4,5, 0, 0]])
    seg_ids1 = torch.tensor([[0,0,0,0,0, 0, 0], [0,0,0,0,0, 0, 0]])
    n_a = torch.FloatTensor( [4,3] ) / MAX_LEN
    n_q = torch.FloatTensor( [5,9] ) / MAX_LEN
    n_a = n_a.unsqueeze(1)
    n_q = n_q.unsqueeze(1)
    print(n_a.shape, n_q.shape)
    model = QuestModel(torch.FloatTensor( ans_corr ), torch.FloatTensor( question_corr ))
    print('...Model Loaded...')
    y = model(x, seg_ids, x1, seg_ids1, n_a, n_q)
    print(y)
    print(y.shape)


# In[13]:


test_model()


# In[14]:


def create_model(model_file, n_cls=30):
    model = QuestModel(torch.FloatTensor( ans_corr ).to(device), torch.FloatTensor( question_corr ).to(device), n_classes=n_cls)
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    #model = DataParallel(model)
    return model

def create_models():
    models = []
    model=create_model(f'../input/quest-models/best.pth')
    model.eval()
    models.append(model)
    return models


# In[15]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def training(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader):

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            #   for ids, seg_ids, labels in loader:
            b_input_ids_a = batch[0].to(device)
            b_input_mask_a = batch[1].to(device)
            b_input_ids_q = batch[2].to(device)
            b_input_mask_q = batch[3].to(device)
            b_n_a =  batch[4].unsqueeze(1).to(device)
            b_n_q = batch[5].unsqueeze(1).to(device)
            b_labels = batch[6].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids_a, b_input_mask_a, b_input_ids_q, b_input_mask_q, b_n_a, b_n_q)

            logits = outputs
            loss1 = criterion(logits[:,0:9], b_labels[:,0:9])
            loss2 = criterion(logits[:,9:10], b_labels[:,9:10])
            loss3 = criterion(logits[:,10:21], b_labels[:,10:21])
            loss4 = criterion(logits[:,21:26], b_labels[:,21:26])
            loss5 = criterion(logits[:,26:30], b_labels[:,26:30])
            loss = loss1+loss2+loss3+loss4+loss5

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    t1 = time.time()
    # Evaluate data for one epoch
    for step, batch in enumerate( val_dataloader ):

        # Progress update every 40 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t1)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(val_dataloader), elapsed))

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids_a, b_input_mask_a, b_input_ids_q, b_input_mask_q, b_n_a, b_n_q, b_labels = batch
        b_n_a = b_n_a.unsqueeze(1)
        b_n_q = b_n_q.unsqueeze(1)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids_a, b_input_mask_a, b_input_ids_q, b_input_mask_q, b_n_a, b_n_q)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = torch.sigmoid( outputs )

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        for i in range(logits.shape[1]):
            #print(i, spearmanr(label_ids[:,i], logits[:,i]))
            eval_accuracy += np.nan_to_num(spearmanr(label_ids[:, i], logits[:, i]).correlation)
            # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Spearman Coefficient: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
        
    print("  Saving Model...")
    torch.save(model.state_dict(), 'checkpoint_new_{0}_{1}.pth'.format(epoch_i, epochs))
    print("  Model Saved...")
    print("")
    print("Training complete!")
    return model


# In[16]:


from tqdm import tqdm
import torch
def predict(model, test_loader):
    all_scores = []
    with torch.no_grad():
        for ids_a, seg_ids_a, ids_q, seg_ids_q, b_n_a, b_n_q in tqdm(test_loader, total=test_loader.num // test_loader.batch_size):
            ids_q, seg_ids_q = ids_q.cuda(), seg_ids_q.cuda()
            ids_a, seg_ids_a = ids_a.cuda(), seg_ids_a.cuda()
            b_n_a = b_n_a.unsqueeze(1).cuda()
            b_n_q = b_n_q.unsqueeze(1).cuda()
            scores = []
            outputs = torch.sigmoid(model(ids_a, seg_ids_a, ids_q, seg_ids_q, b_n_a, b_n_q)).cpu()
            scores.append(outputs)
            all_scores.append(torch.mean(torch.stack(scores), 0))

    all_scores = torch.cat(all_scores, 0).numpy()
    print(all_scores.shape)
    
    return all_scores


# In[17]:


test_loader = get_test_loader(batch_size=32)
train_loader, val_loader = get_train_val_loaders()


# In[18]:


if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
models = None
models = create_models()


# In[19]:


# Setup model
model = models[0]
#model.freeze_bert()

# Create Optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

criterion = torch.nn.BCEWithLogitsLoss()

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

model = training(model, optimizer, scheduler, criterion, epochs, train_loader, val_loader)


# In[20]:


preds = predict(model, test_loader)


# In[21]:


preds[:1]


# In[22]:


sub[target_columns] = preds


# In[23]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




