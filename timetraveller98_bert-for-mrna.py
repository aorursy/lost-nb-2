#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import math
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()    


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN = "../input/stanford-covid-vaccine/train.json"
TEST = "../input/stanford-covid-vaccine/test.json"
SS = "../input/stanford-covid-vaccine/sample_submission.csv"
train = pd.read_json(TRAIN, lines=True)
test = pd.read_json(TEST, lines=True)
sample_sub = pd.read_csv(SS)
print(f"Using {device}")


# In[3]:


inp_seq_cols = ['sequence', 'structure', 'predicted_loop_type']
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
vocab = {
    'sequence': {x:i for i, x in enumerate("A C G U".split())},
    'structure': {x:i for i, x in enumerate("( . )".split())},
    'predicted_loop_type': {x:i for i, x in enumerate("B E H I M S X".split())},
}


# In[4]:


def preprocess_inputs(df, cols=inp_seq_cols):
    
    def f(x):
        return [vocab['sequence'][x] for x in x[0]],                [vocab['structure'][x] for x in x[1]],                [vocab['predicted_loop_type'][x] for x in x[2]],

    return np.array(
            df[cols]
            .apply(f, axis=1)
            .values
            .tolist()
        )


# In[5]:


train_filtered = train.loc[train.SN_filter == 1]
train_inputs = torch.tensor(preprocess_inputs(train_filtered)).to(device)
print("input shape: ", train_inputs.shape)

train_labels = torch.tensor(
    np.array(
        train_filtered[target_cols]
        .values.tolist()
    ).transpose(0, 2, 1)
).float().to(device)
print("output shape: ", train_labels.shape)


# In[6]:


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class Config(NamedTuple):
    "Configuration for BERT model"
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 130
    n_bases: int = 4
    n_structures: int = 3
    n_loop: int = 7

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask=None):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h
    
def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`
    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.
    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.
    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


# In[7]:


class Embeddings(nn.Module):
    """Modified Embeddings for mRNA degradation"""
    def __init__(self, cfg):
        super().__init__()
        
        self.dim = cfg.dim
        self.base_embed = nn.Embedding(cfg.n_bases, cfg.dim)
        self.struct_embed = nn.Embedding(cfg.n_structures, cfg.dim)
        self.loop_embed = nn.Embedding(cfg.n_loop, cfg.dim) 
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        
        
        self.embed = nn.Embedding(84, cfg.dim)

    def forward(self, x, flip=False):
        seq_len = x.size(2)
        base_seq, struct_seq, loop_seq = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        
        if flip:
            base_seq = base_seq.flip(1)
            struct_seq = struct_seq.flip(1)
            loop_seq = loop_seq.flip(1)
        
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(base_seq) # (S,) -> (B, S)
        pe = self.pos_embed(pos)
        
#         pe = generate_regular_PE(seq_len, self.dim).to(x.device)
        
        se = self.struct_embed(struct_seq)
        le = self.loop_embed(loop_seq)
        be = self.base_embed(base_seq)
    
        e =  self.norm1(be + se + le) + self.norm2(pe)
#         print(f"Embed: {e.shape}")
            
        return self.drop(self.norm(e))
  


# In[8]:



  
class Transformer(nn.Module):
  """ The BERT transformer with slight modifications 
      for the use-case of mRNA degradation"""
  def __init__(self, cfg, pred_len=68):
      super().__init__()
      self.embed = Embeddings(cfg)
      self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
      self.pred_len = pred_len
      self.out = nn.Linear(cfg.dim, 5)

  def forward(self, x):
      h = self.embed(x)
      for block in self.blocks:
          h = block(h)
      truncated = h[: , :self.pred_len]
      out = self.out(truncated)
      return out    


# In[9]:


config = """
{
    "dim": 192,
    "dim_ff": 384,
    "n_layers": 6,
    "p_drop_attn": 0.3,
    "n_heads": 6,
    "p_drop_hidden": 0.25,
    "n_bases": 4,
    "n_structures": 3,
    "n_loop": 7,
    "max_len": 130
}
"""
CFILE = "config.json"
with open(CFILE, 'w') as handle:
    handle.write(config)
    
cfg = Config().from_json(CFILE)


# In[10]:


def compute_loss(batch_X, batch_Y, model, optimizer=None, is_train=True, ret_pred=False):
    """custom MCRMSE"""
    model.train(is_train)

    pred_Y = model(batch_X)

    loss = torch.pow(
            torch.pow(
            batch_Y-pred_Y, 
            2).mean(dim=1), 
            .5).mean()

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ret_pred:
        return loss.item(), pred_Y
    return loss.item()


# In[11]:


FOLDS = 3
EPOCHS = 120
PRINT_FREQ = EPOCHS//10
BATCH_SIZE = 32
LR = 1e-4


# In[12]:


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = torch.tensor(preprocess_inputs(public_df)).to(device)
private_inputs = torch.tensor(preprocess_inputs(private_df)).to(device)

public_loader = DataLoader(TensorDataset(public_inputs), shuffle=False, batch_size=BATCH_SIZE)
private_loader = DataLoader(TensorDataset(private_inputs), shuffle=False, batch_size=BATCH_SIZE)

bert_private_preds = np.zeros((private_df.shape[0], 130, 5))
bert_public_preds = np.zeros((public_df.shape[0], 107, 5))

kfold = KFold(FOLDS, shuffle=True, random_state=2020)
bert_histories = []

for k, (train_index, val_index) in enumerate(kfold.split(train_inputs)):
    train_dataset = TensorDataset(train_inputs[train_index], train_labels[train_index])
    val_dataset = TensorDataset(train_inputs[val_index], train_labels[val_index])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

    seed_everything()
    model = Transformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR) #AdamW 

    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(EPOCHS)):
        train_losses_batch = []
        val_losses_batch = []
        for (batch_X, batch_Y) in train_loader:
            model.train()
            train_loss = compute_loss(batch_X, batch_Y, model, optimizer=optimizer, is_train=True)
            train_losses_batch.append(train_loss)
        for (batch_X, batch_Y) in val_loader:
            model.eval()
            val_loss = compute_loss(batch_X, 
                                batch_Y, model, 
                                optimizer=optimizer, 
                                is_train=False)
            val_losses_batch.append(val_loss)
        avg_train_loss = sum(train_losses_batch) / len(train_losses_batch)
        avg_val_loss = sum(val_losses_batch) / len(val_losses_batch)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if not (epoch+1) % PRINT_FREQ:
            print(f"[{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
    model_state = model.state_dict()
    del model
            
    bert_histories.append({'train_loss': train_losses, 'val_loss': val_losses})


    bert_short = Transformer(cfg, pred_len=107).to(device)
    bert_short.load_state_dict(model_state)
    bert_short.eval()
    bert_public_pred = np.ndarray((0, 107, 5))
    for batch in public_loader:
        batch_X = batch[0]
        pred = bert_short(batch_X).detach().cpu().numpy()
        bert_public_pred = np.concatenate([bert_public_pred, pred], axis=0)
    bert_public_preds += bert_public_pred / FOLDS

    bert_long = Transformer(cfg, pred_len=130).to(device)
    bert_long.load_state_dict(model_state)
    bert_long.eval()
    bert_private_pred = np.ndarray((0, 130, 5))
    
    for batch in private_loader:
        batch_X = batch[0]
        pred = bert_long(batch_X).detach().cpu().numpy()
        bert_private_pred = np.concatenate([bert_private_pred, pred], axis=0)
    bert_private_preds += bert_private_pred / FOLDS
    
    del bert_short, bert_long


# In[13]:


print(f" BERT mean fold validation loss: {np.mean([min(history['val_loss']) for history in bert_histories])}")
      
for i, hist in enumerate(bert_histories):
    plt.title(f"Model #{i+1}")
    plt.plot(hist['train_loss'])
    plt.plot(hist['val_loss']) 
    plt.legend(['Train', 'Val'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# In[14]:


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

preds_bert = []

for df, preds in [(public_df, bert_public_preds), (private_df, bert_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_bert.append(single_df)

preds_bert_df = pd.concat(preds_bert)
preds_bert_df.head()


# In[15]:


submission = sample_sub[['id_seqpos']].merge(preds_bert_df, on=['id_seqpos'])
sub_fname = "submission.csv"
submission.to_csv(sub_fname, index=False)
print(f"Submission saved in {sub_fname}")
submission.head()


# In[ ]:




