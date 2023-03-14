#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai import *
from fastai.vision import *
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


def get_images_and_labels(csv,label):
    fileraw = pd.read_csv(csv)
    labels = fileraw[label].to_numpy()
    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32)
    data = np.true_divide(data,255.).reshape((fileraw.shape[0],28,28))
    data = np.expand_dims(data, axis=1)
    return data, labels


# In[3]:


train_data, train_labels = get_images_and_labels(DATAPATH/'train.csv','label')
test_data, test_labels = get_images_and_labels(DATAPATH/'test.csv','id')
other_data, other_labels = get_images_and_labels(DATAPATH/'Dig-MNIST.csv','label')

print(f' Train:\tdata shape {train_data.shape}\tlabel shape {train_labels.shape}\n Test:\tdata shape {test_data.shape}\tlabel shape {test_labels.shape}\n Other:\tdata shape {other_data.shape}\tlabel shape {other_labels.shape}')

The resulting data arrays look reasonable, and the size of the labels is the same. Let's display a labelled image:
# In[4]:


plt.title(f'Training Label: {train_labels[4]}')
plt.imshow(train_data[4,0],cmap='gray');


# In[5]:


np.random.seed(42)
ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .1

train_90_labels = train_labels[np.invert(ran_10_pct_idx)]
train_90_data = train_data[np.invert(ran_10_pct_idx)]

valid_10_labels = train_labels[ran_10_pct_idx]
valid_10_data = train_data[ran_10_pct_idx]


# In[6]:


class ArrayDataset(Dataset):
    "Dataset for numpy arrays based on fastai example: "
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = len(np.unique(y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


# In[7]:


train_ds = ArrayDataset(train_90_data,train_90_labels)
valid_ds = ArrayDataset(valid_10_data,valid_10_labels)
other_ds = ArrayDataset(other_data, other_labels)
test_ds = ArrayDataset(test_data, test_labels)


# In[8]:


bs = 128
databunch = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)


# In[9]:


leak = 0.15

best_architecture = nn.Sequential(
    
    conv_layer(1,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.4),
    
    conv_layer(32,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.4),
    
    Flatten(),
    nn.Linear(3136, 128),
    relu(inplace=True),
    nn.BatchNorm1d(128),
    nn.Dropout(0.4),
    nn.Linear(128,10)
)


# In[10]:


learn = Learner(databunch, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )


# In[11]:


learn.fit_one_cycle(20)


# In[12]:


preds, ids = learn.get_preds(DatasetType.Test)
y = torch.argmax(preds, dim=1)


# In[13]:


submission = pd.DataFrame({ 'id': ids,'label': y })
submission.to_csv(path_or_buf ="submission.csv", index=False)

