#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

from fastai import *
from fastai.tabular import *
from sklearn.metrics import cohen_kappa_score


# In[2]:


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# In[3]:


train_csv = pd.read_csv('../input/train/train.csv', low_memory=False)
test_csv = pd.read_csv('../input/test/test.csv', low_memory=False)

def preprocess(csv):
    csv['Description_len'] = [len(str(tt)) for tt in csv['Description']]
    csv['Name_len'] = [len(str(tt)) for tt in csv['Name']]
    return csv

train_csv = preprocess(train_csv)
test_csv = preprocess(test_csv)


# In[4]:


cat_names = ['Type','Breed1','Breed2','Gender','Color1','Color2','State','Color3','FurLength', 'Vaccinated','Dewormed','Sterilized','Health']
cont_names = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'Description_len', 'Name_len']


# In[5]:


bs = len(train_csv)
procs = [FillMissing, Categorify, Normalize]
df = TabularList.from_df(train_csv, path='../input', cat_names=cat_names, cont_names=cont_names, procs=procs)                .no_split()                .label_from_df(cols='AdoptionSpeed')
df_test = TabularList.from_df(test_csv, path='../input', cat_names=cat_names, cont_names=cont_names, processor=df.train.x.processor)
data = df.add_test(df_test).databunch(num_workers=0, bs=bs)


# In[6]:


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in, track_running_stats=False)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class TabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont, track_running_stats=False)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


# In[7]:


unique, counts = np.unique(train_csv['AdoptionSpeed'], return_counts=True)
E_hist = counts / sum(counts)
ww = [sum([E_hist[k]*(k-i)**2 for k in range(5)]) for i in range(5)]

class KappaLoss(nn.Module):
    def forward(self, y1, y2, *args):
        numer = (torch.matmul(y2.float().reshape([-1,1]),torch.tensor(()).new_ones((1, 5)).cuda()) - 
     torch.matmul(torch.tensor(()).new_ones((bs, 1)), torch.tensor(range(5), dtype=torch.float).reshape([1,5])).cuda())**2
        numer = (numer * y1).sum()
        denom = torch.matmul(torch.tensor(()).new_ones((bs, 1)), torch.tensor(ww).reshape([1,5])).cuda()
        denom = (denom * y1).sum()
        loss =  numer / denom
        return loss


# In[8]:


emb_szs = data.get_emb_szs({})
model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=[200,100], ps=None, emb_drop=0., y_range=None, use_bn=True)
learn = Learner(data, model, loss_func=KappaLoss(), path='/tmp')


# In[9]:


learn.fit(10, 5e-2)

pred = learn.get_preds(ds_type=DatasetType.Train)
y_pred = [int(np.argmax(row)) for row in pred[0]]
print('QWK insample', cohen_kappa_score(y_pred, pred[1], weights='quadratic'))


# In[10]:


print('RMSE insample', np.sqrt(np.mean((np.array(y_pred) - np.array(pred[1]))**2)))


# In[11]:


np.unique(y_pred, return_counts=True)


# In[12]:


pred = learn.get_preds(ds_type=DatasetType.Test)
y_pred = [int(np.argmax(row)) for row in pred[0]]

test_csv['AdoptionSpeed'] = y_pred
test_csv[['PetID', 'AdoptionSpeed']].to_csv('submission.csv', index=False)


# In[13]:




