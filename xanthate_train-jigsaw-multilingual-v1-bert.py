#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
import os
import time

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# training data paths
TRAIN_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-input-ids.npz'
TRAIN_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-token-type-ids.npz'
TRAIN_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-attention-mask.npz'
TRAIN_TARGETS_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-targets.npz'

# validation data paths
VALID_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-input-ids.npz'
VALID_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-token-type-ids.npz'
VALID_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-attention-mask.npz'
VALID_TARGETS_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-targets.npz'

# test data paths
TEST_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-input-ids.npz'
TEST_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-token-type-ids.npz'
TEST_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-attention-mask.npz'


# In[4]:


# torch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# transformer modules
from transformers import BertModel
from transformers import XLMRobertaModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# tpu-specific modules
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


from sklearn import metrics


# In[5]:


class JigsawDataset(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, targets=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.targets = targets
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        token_type_ids = self.token_type_ids[item]
        attention_mask = self.attention_mask[item]

        if self.targets is None:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)}
        
        else:
            targets = self.targets[item]
        
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.float)}
            


# In[6]:


# loading compressed numpy training data
load_train_input_ids = np.load(TRAIN_INPUT_ID_PATH, mmap_mode='r')
load_train_token_type_ids = np.load(TRAIN_TOKEN_ID_PATH, mmap_mode='r')
load_train_attention_mask = np.load(TRAIN_ATTENTION_PATH, mmap_mode='r')
load_train_targets = np.load(TRAIN_TARGETS_PATH, mmap_mode='r')

# training data
train_input_ids = load_train_input_ids.f.arr_0
train_token_type_ids = load_train_token_type_ids.f.arr_0
train_attention_mask = load_train_attention_mask.f.arr_0
train_targets = load_train_targets.f.arr_0

# loading compressed numpy validation data
load_valid_input_ids = np.load(VALID_INPUT_ID_PATH, mmap_mode='r')
load_valid_token_type_ids = np.load(VALID_TOKEN_ID_PATH, mmap_mode='r')
load_valid_attention_mask = np.load(VALID_ATTENTION_PATH, mmap_mode='r')
load_valid_targets = np.load(VALID_TARGETS_PATH, mmap_mode='r')

# validation data
valid_input_ids = load_valid_input_ids.f.arr_0
valid_token_type_ids = load_valid_token_type_ids.f.arr_0
valid_attention_mask = load_valid_attention_mask.f.arr_0
valid_targets = load_valid_targets.f.arr_0

# loading compressed numpy test data
load_test_input_ids = np.load(TEST_INPUT_ID_PATH, mmap_mode='r')
load_test_token_type_ids = np.load(TEST_TOKEN_ID_PATH, mmap_mode='r')
load_test_attention_mask = np.load(TEST_ATTENTION_PATH, mmap_mode='r')

# test data
test_input_ids = load_test_input_ids.f.arr_0
test_token_type_ids = load_test_token_type_ids.f.arr_0
test_attention_mask = load_test_attention_mask.f.arr_0


# sanity check for the sizes
assert train_input_ids.shape[0] == train_token_type_ids.shape[0]         == train_attention_mask.shape[0] == train_targets.shape[0]

assert valid_input_ids.shape[0] == valid_token_type_ids.shape[0]         == valid_attention_mask.shape[0] == valid_targets.shape[0]

assert test_input_ids.shape[0] == test_token_type_ids.shape[0] == test_attention_mask.shape[0]


# uncomment the lines below to check the sizes of the data rows
print(train_input_ids.shape[0], train_token_type_ids.shape[0], train_attention_mask.shape[0], train_targets.shape[0])
print(valid_input_ids.shape[0], valid_token_type_ids.shape[0], valid_attention_mask.shape[0], valid_targets.shape[0])
print(test_input_ids.shape[0], test_token_type_ids.shape[0], test_attention_mask.shape[0])


# In[7]:


del load_train_input_ids, load_train_token_type_ids, load_train_attention_mask, load_train_targets
del load_valid_input_ids, load_valid_token_type_ids, load_valid_attention_mask, load_valid_targets
del load_test_input_ids, load_test_token_type_ids, load_test_attention_mask
import gc; gc.collect()


# In[8]:


gc.collect()


# In[9]:


class BertBaseUncased(nn.Module):
    def __init__(self, bert_model, dropout):
        super(BertBaseUncased, self).__init__()
        self.bert_model = bert_model
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(True)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, out2 = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        
        bert_out = self.dropout(out2)
        out = self.relu(self.fc1(bert_out))
        out = self.fc2(out)
        return out
    

class XLMRoberta(nn.Module):
    def __init__(self, model, dropout):
        super(XLMRoberta, self).__init__()
        self.roberta = model
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(True)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, out2 = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        
        out2 = self.dropout(out2)
        out = self.relu(self.fc1(out2))
        out = self.fc2(out)
        return out


# In[10]:


def loss_fn(output, target):
    return nn.BCEWithLogitsLoss()(output, target)


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    
    for i, data in enumerate(dataloader):
        input_ids = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        targets = data['targets']
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        output = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(output, targets.unsqueeze(1))
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            xm.master_print(f"iteration: {i}, train loss: {loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
    return train_loss


def valid_fn(dataloader, model, device):
    valid_loss = []
    outputs = []
    targets = []
    
    with torch.no_grad():
      
        for i, data in enumerate(dataloader):
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            target = data['targets']

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            
            output_np = output.cpu().detach().numpy().tolist()
            target_np = target.cpu().detach().numpy().tolist()
            
            outputs.extend(output_np)
            targets.extend(target_np)
            
    return outputs, targets


def run(
    epochs, 
    batch_size, 
    num_workers, 
    learning_rate, 
    warmup_steps,
    pretrained_model,
    dropout):
    
    
    # datasets, samplers and dataloaders
    trainset = JigsawDataset(
        input_ids=train_input_ids[:512000],
        token_type_ids=train_token_type_ids[:512000],
        attention_mask=train_attention_mask[:512000],
        targets=train_targets[:512000])
    
    validset = JigsawDataset(
        input_ids=valid_input_ids,
        token_type_ids=valid_token_type_ids,
        attention_mask=valid_attention_mask,
        targets=valid_targets)

    
    # samplers
    trainsampler = DistributedSampler(
        dataset=trainset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    validsampler = DistributedSampler(
        dataset=validset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    
    
    # dataloaders
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        sampler=trainsampler,
        num_workers=num_workers,
        drop_last=True,)
    
    validloader = DataLoader(
        dataset=validset,
        batch_size=batch_size,
        sampler=validsampler,
        num_workers=num_workers,
        drop_last=True)

    
    xm.master_print(f"Loading datasets....Complete!")
    
    # model
    device = xm.xla_device()
    model = BertBaseUncased(pretrained_model, dropout)
    model = model.to(device)
    xm.master_print(f"Loading model....Complete!")
    
    # training_parameters, optimizers and schedulers
    not_decay = ['LayerNorm.weight', 'LayerNorm.bias', 'bias']
    
    parameters = list(model.named_parameters())
    
    train_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in not_decay)], 
         'weight_decay': 0.001},
        
        {'params': [p for n, p in parameters if any(nd in n for nd in not_decay)], 
         'weight_decay': 0.001 }]
    
    
    num_training_steps = int(len(trainset) / batch_size / xm.xrt_world_size())
    xm.master_print(f"Iterations per epoch: {num_training_steps}, world_size: {xm.xrt_world_size()}")
    
    optimizer = AdamW(train_parameters, lr=learning_rate)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)
    
    AUC_SCORE = []
    
    # training and evaluation
    for epoch in range(epochs):
        gc.collect()
        
        # train
        para_loader = pl.ParallelLoader(trainloader, [device])
        
        start_time = time.time()
        
        train_loss = train_fn(
            model, 
            para_loader.per_device_loader(device), 
            optimizer, 
            device, 
            scheduler=scheduler)
        
        end_time = time.time()
        
        del para_loader
        gc.collect()
        
        time_per_epoch = end_time - start_time
        
        xm.master_print(f"Time taken: {(time_per_epoch/60):.2f} mins")
        
        xm.master_print(f"epoch: {epoch+1}/{epochs}, train loss: {np.mean(train_loss):.4f}")
        
        # eval
        para_loader = pl.ParallelLoader(validloader, [device])
        outputs, targets = valid_fn(
            para_loader.per_device_loader(device),
            model,
            device)
        
        del para_loader
        gc.collect()
        
        auc = metrics.roc_auc_score(np.array(targets) >= 0.5, outputs)
            
        xm.master_print(f"auc_score: {auc:.4f}")
        #xm.master_print(AUC_SCORE)
        
        # save model
        #if epoch > 13  and auc > max(AUC_SCORE):
            #AUC_SCORE.append(auc)
            #xm.master_print(f"Saving model {epoch+1}")
    xm.save(model.state_dict(), f"bert_multilingual_{batch_size}_e{epochs}_auc_{auc:.4f}.bin")
        
        #AUC_SCORE.append(auc)
        


# In[11]:


# hyper parameters
MODEL_PATH = 'bert-base-multilingual-uncased'
BATCH_SIZE = 128
NUM_WORKERS = 8
DROPOUT = 0.3
LR = 1e-5
EPOCHS = 10
WARMUP_STEPS = 0

MODEL = BertModel.from_pretrained(MODEL_PATH)

def _mp_fn(rank, flags):
    
    a = run(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        pretrained_model=MODEL,
        dropout=DROPOUT)
    
    

FLAGS = {}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# In[12]:


"""
def predictions(dataloader, model, device):
    outputs = []

    for i, data in enumerate(dataloader):
        input_ids = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        output = model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask)
        
        output_np = output.cpu().detach().numpy().tolist()
        outputs.extend(output_np)
        xm.master_print(len(outputs))
    return outputs



def pred_run():
    gc.collect()
    # load model
    model = BertBaseUncased(MODEL, DROPOUT)
    device = torch.device('cpu')
    #device = xm.xla_device()
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load("bert_multilingual_32_model.bin"))

    # dataset, sampler and dataloader
    testset = JigsawDataset(
        input_ids=test_input_ids[:4000],
        token_type_ids=test_token_type_ids[:4000],
        attention_mask=test_attention_mask[:4000])
    
    #xm.master_print(f"size of testset: {len(testset)}")
    print(len(testset))
    
    
    testsampler = DistributedSampler(
        dataset=testset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    
    
    print("Sampler Done")
    
    testloader = DataLoader(
        dataset=testset,
        batch_size=32,
        shuffle=False,
        drop_last=True)

    # predictions
    pred = predictions(dataloader=testloader, model=model, device=device)
    del para_loader
    gc.collect()
    return pred

pred = pred_run()

#FLAGS = {}
#pred = xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
"""
print("Finished")

