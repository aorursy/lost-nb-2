#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Python Stuff
import numpy as np
import pandas as pd
import zipfile
import os
import gc
import sys
import string
from collections import defaultdict, Counter
import urllib.request
import os.path

# Visualization Stuff
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 240)

# Statistics Stuff
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats

# NLP Stuff
if 'transformers' not in sys.modules:
  get_ipython().system('pip install transformers')
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
if 'nltk' not in sys.modules:
  get_ipython().system('pip install nltk')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Neural Networks Stuff
import torch
from torch import nn, optim
from torch.utils import data

is_colab = 'google.colab' in sys.modules
if is_colab:
    from google.colab import drive

is_kaggle = 'kaggle' in os.getcwd()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
if torch.cuda.is_available():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))


# In[2]:


if is_colab:
  drive.mount('/content/drive')
  DIR_NAME = '/content/drive/My Drive/Colab Notebooks/Spooky/data/'
elif is_kaggle:
    DIR_NAME = "/kaggle/input/spooky-author-identification/"
    zips = os.listdir(DIR_NAME)
    for name in zips:
        with zipfile.ZipFile(DIR_NAME + name, 'r') as zip_ref:
            zip_ref.extractall(".")
else:
    fnames = [r'./train.csv',r'./test.csv', r'./sample_submission.csv']
    url = 'https://drive.google.com/drive/folders/1tP8T8_-6Xy5BgQa3q2g7reFHBv4Uer_a?usp=sharing'

    if not os.path.exists(fnames[0]):
        for fname in fnames:
            urllib.request.urlretrieve(url, fname)
    print(fname, 'exists:', os.path.exists(fname))
    DIR_NAME = './' 


# In[3]:


if is_colab:
  train = pd.read_csv(DIR_NAME+'train.csv')
  test = pd.read_csv(DIR_NAME+'test.csv')
  sample = pd.read_csv(DIR_NAME+'sample_submission.csv')
else:
  train = pd.read_csv('./train.csv')
  test = pd.read_csv('./test.csv')
  sample = pd.read_csv('./sample_submission.csv')
train.head(3)


# In[4]:


PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'
tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

bert_embeddings = bert_model.get_input_embeddings()


# In[5]:


sample_sentence = "Creative minds are uneven, and the best of fabrics have their dull spots."
tokens = tokenizer.tokenize(sample_sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
encoded = tokenizer.encode(sample_sentence)
print(tokens)
print(token_ids)
print(encoded)


# In[6]:


train['word_ids'] = train['text'].apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
train['word_count'] = train['word_ids'].apply(lambda x: len(x))

eng_stopword_toekns = set([tokenizer.convert_tokens_to_ids(str(x)) for x in stopwords.words("english")])
train['stopword_count'] = train['word_ids'].apply(lambda x: len([token for token in x if token in eng_stopword_toekns]))

train.head(1)


# In[7]:


def count_words(df, author_key):
    df = df[df['author']==author_key]
    word_count = df['text'].str.split(expand=True).stack().value_counts()
    return word_count


# In[8]:


word_count_EAP = count_words(train, 'EAP')
word_count_HPL = count_words(train, 'HPL')
word_count_MWS = count_words(train, 'MWS')
print(f'word_count_EAP size: {len(word_count_EAP)}')
print(f'word_count_HPL size: {len(word_count_HPL)}')
print(f'word_count_MWS size: {len(word_count_MWS)}')


# In[9]:


english_stopwords = stopwords.words("english") + list(string.punctuation)

word_count_EAP = word_count_EAP.drop(labels=english_stopwords, errors='ignore')
word_count_HPL = word_count_HPL.drop(labels=english_stopwords, errors='ignore')
word_count_MWS = word_count_MWS.drop(labels=english_stopwords, errors='ignore')


# In[10]:


PCA_ON_N_WORDS = 1000
COMMON_WORDS_POOL = 400
PLOT_M_WORDS = 100

common_words = pd.Series(list(set(word_count_EAP[:COMMON_WORDS_POOL].index) & set(word_count_HPL[:COMMON_WORDS_POOL].index)  & set(word_count_MWS[:COMMON_WORDS_POOL].index)))

word_count_EAP = word_count_EAP.drop(common_words)
word_count_HPL = word_count_HPL.drop(common_words)
word_count_MWS = word_count_MWS.drop(common_words)


# In[11]:


cols = ['count']
word_count_EAP_top_n_for_pca = pd.DataFrame(word_count_EAP[:PCA_ON_N_WORDS], columns = cols)
word_count_EAP_top_n_for_plot = pd.DataFrame(word_count_EAP[:PLOT_M_WORDS], columns = cols)

word_count_EAP_top_n_for_pca['author'] = 'EAP'
word_count_EAP_top_n_for_plot['author'] = 'EAP'


word_count_HPL_top_n_for_pca = pd.DataFrame(word_count_HPL[:PCA_ON_N_WORDS], columns = cols)
word_count_HPL_top_n_for_plot = pd.DataFrame(word_count_HPL[:PLOT_M_WORDS], columns = cols)

word_count_HPL_top_n_for_pca['author'] = 'HPL'
word_count_HPL_top_n_for_plot['author'] = 'HPL'

word_count_MWS_top_n_for_pca = pd.DataFrame(word_count_MWS[:PCA_ON_N_WORDS], columns = cols)
word_count_MWS_top_n_for_plot = pd.DataFrame(word_count_MWS[:PLOT_M_WORDS], columns = cols)

word_count_MWS_top_n_for_pca['author'] = 'MWS'
word_count_MWS_top_n_for_plot['author'] = 'MWS'

df_for_pca = pd.concat([word_count_EAP_top_n_for_pca, word_count_HPL_top_n_for_pca, word_count_MWS_top_n_for_pca])
df_for_pca = df_for_pca.reset_index()
df_for_pca.columns = ['word', 'count', 'author']

df_for_plot = pd.concat([word_count_EAP_top_n_for_plot, word_count_HPL_top_n_for_plot, word_count_MWS_top_n_for_plot])
df_for_plot = df_for_plot.reset_index()
df_for_plot.columns = ['word', 'count', 'author']
df_for_plot.head(2)


# In[12]:


def word_to_index(word):
    token_id = tokenizer.convert_tokens_to_ids(word)
    return token_id

print(word_to_index('Hello'))

def indices_to_vec(word_ids):
    embeded_tokens = bert_embeddings(torch.Tensor(word_ids).to(torch.long))
    return embeded_tokens.detach().numpy()

print(indices_to_vec([word_to_index('Hello'), word_to_index('Jacob')]).shape)

def index_to_vec(word_id):
    embeded_token = bert_embeddings(torch.Tensor([word_id]).to(torch.long))
    return embeded_token.detach().numpy()
vec = index_to_vec(word_to_index('Hello'))
print(vec.shape)


# In[13]:


df_for_plot['word_id'] = df_for_plot['word'].apply(word_to_index)
df_for_pca['word_id'] = df_for_pca['word'].apply(word_to_index)
vectors = index_to_vec(df_for_pca['word_id'].to_numpy())
pca = PCA(n_components=2)
pca.fit(vectors.squeeze())
df_for_pca['word_vec'] = df_for_pca['word_id'].apply(index_to_vec)

vectors.shape


# In[14]:


def vec_to_2dim(word_vec):
    xy = pca.transform(word_vec)
    return xy[0][0], xy[0][1]
x, y = vec_to_2dim(vec)
print(x, y)

def tuple_x(xy):
    return tuple(xy)[0]
print(tuple_x((1,2)))

def tuple_y(xy):
    return tuple(xy)[1]
print(tuple_y((1,2)))


# In[15]:


df_for_pca['word_xy'] = df_for_pca['word_vec'].apply(vec_to_2dim)
df_for_pca['word_x'] = df_for_pca['word_xy'].apply(tuple_x)
df_for_pca['word_y'] = df_for_pca['word_xy'].apply(tuple_y)


# In[16]:


merged_df = df_for_plot.join(df_for_pca, on='word_id',lsuffix='_l', rsuffix='_r') 
# df_cleaned = merged_df[(merged_df['word_x'] < 0.5 ) & (merged_df['count_x'] > np.median(merged_df['count_x']))]
df_cleaned = merged_df[merged_df['word_x'] < 0.6 ]

# df_cleaned = merged_df
df_cleaned.columns


# In[17]:


df_cleaned['count_l'].min()


# In[18]:


sns.countplot(train['author'])
plt.xlabel('Authors');


# In[19]:


f, axes = plt.subplots(1, 2, figsize=(40,10))

sns.distplot(ax=axes[0], a=train['word_count'])
sns.distplot(ax=axes[1], a=train['word_count'][train['word_count'] < 150]);


# In[20]:


stats.normaltest(train['word_count'])


# In[21]:


f, axes = plt.subplots(1, 2, figsize=(40,10))

sns.distplot(ax=axes[0], a=train['stopword_count'])
sns.distplot(ax=axes[1], a=train['stopword_count'][train['stopword_count'] < 50]);


# In[22]:


stats.normaltest(train['stopword_count'])


# In[23]:


f, axes = plt.subplots(1, 2, figsize=(20,10))

sns.violinplot(x='author', y='word_count', data=train[train['word_count'] < 60], ax=axes[0])
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15);

sns.violinplot(x='author', y='word_count', data=train[train['word_count'] < 90], ax=axes[1])
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15);


# In[24]:


def calculate_one_way_anova(column_name):
    return stats.f_oneway(train[column_name][train['author'] == 'EAP'],
               train[column_name][train['author'] == 'HPL'],
               train[column_name][train['author'] == 'MWS'])


# In[25]:


one_way = calculate_one_way_anova('word_count')
print(f'The F score when comaring all the authors: {one_way.statistic}, which reflect pvalue of: {one_way.pvalue}')


# In[26]:


t_EAP_HPL = stats.ttest_ind(train['word_count'][train['author'] == 'EAP'], train['word_count'][train['author'] == 'HPL'])
t_MWS_HPL = stats.ttest_ind(train['word_count'][train['author'] == 'MWS'], train['word_count'][train['author'] == 'HPL'])
t_MWS_EAP = stats.ttest_ind(train['word_count'][train['author'] == 'MWS'], train['word_count'][train['author'] == 'EAP'])

print(f'The T score when comaring EAP and HPL: {t_EAP_HPL.statistic} which reflect pvalue of: {t_EAP_HPL.pvalue}')
print(f'The T score when comaring MWS and HPL: {t_MWS_HPL.statistic} which reflect pvalue of: {t_MWS_HPL.pvalue}')
print(f'The T score when comaring MWS and EAP: {t_MWS_EAP.statistic} which reflect pvalue of: {t_MWS_EAP.pvalue}')


# In[27]:


f, axes = plt.subplots(1, 2, figsize=(20,10))

sns.violinplot(x='author', y='stopword_count', data=train[train['stopword_count'] < 30], ax=axes[0])
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of stop words in text', fontsize=12)
plt.title("Number of stop words by author", fontsize=15);

sns.violinplot(x='author', y='stopword_count', data=train[train['stopword_count'] < 60], ax=axes[1])
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of stop words in text', fontsize=12)
plt.title("Number of stop words by author", fontsize=15);


# In[28]:


one_way = calculate_one_way_anova('stopword_count')
print(f'The F score when comaring all the authors: {one_way.statistic}, which reflect pvalue of: {one_way.pvalue}')


# In[29]:


t_EAP_HPL = stats.ttest_ind(train['stopword_count'][train['author'] == 'EAP'], train['stopword_count'][train['author'] == 'HPL'])
t_MWS_HPL = stats.ttest_ind(train['stopword_count'][train['author'] == 'MWS'], train['stopword_count'][train['author'] == 'HPL'])
t_MWS_EAP = stats.ttest_ind(train['stopword_count'][train['author'] == 'MWS'], train['stopword_count'][train['author'] == 'EAP'])

print(f'The T score when comaring EAP and HPL: {t_EAP_HPL.statistic} which reflect pvalue of: {t_EAP_HPL.pvalue}')
print(f'The T score when comaring MWS and HPL: {t_MWS_HPL.statistic} which reflect pvalue of: {t_MWS_HPL.pvalue}')
print(f'The T score when comaring MWS and EAP: {t_MWS_EAP.statistic} which reflect pvalue of: {t_MWS_EAP.pvalue}')


# In[30]:


p = sns.relplot(x="word_x", y="word_y", hue="author_l", size="count_l",
             alpha=.5, palette="muted", sizes=(70,450),
            height=20, aspect=1, data=df_cleaned)
ax = p.axes[0,0]

for idx, row in df_cleaned.iterrows():
     ax.text(row['word_x']+ 0.001, row['word_y'], row['word_l'], horizontalalignment='left', size='large', color='black')


# In[31]:


sns.set(style="white", context="talk", font_scale = 4)

# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(90, 60), sharex=False)

# Edgar Allan Poe
x1 = word_count_EAP[:20].index
y1 = word_count_EAP[:20]
sns.barplot(x=x1, y=y1, ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Edgar Allan Poe",fontsize=80)

# Mary Shelley
x2 = word_count_MWS[:20].index
y2 = word_count_MWS[:20]
sns.barplot(x=x2, y=y2, ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("Mary Shelley",fontsize=80)

# "H.P. Lovecraft
x3 = word_count_HPL[:20].index
y3 = word_count_HPL[:20]
sns.barplot(x=x3, y=y3, palette="deep", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("H.P. Lovecraft",fontsize=80)

# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=5)


# In[32]:


def author_to_label(author):
    labels = {'EAP': 0,'HPL': 1,'MWS': 2}
    return labels[author]

def label_to_author(label):
    authors = ['EAP','HPL','MWS']
    return authors[int(label)]


# In[33]:


train['label'] = train['author'].apply(author_to_label)
dummies = pd.get_dummies(train['author'])
train = pd.concat([train,dummies], axis=1)
train.head(1)


# In[34]:


class TrainDataSet(data.Dataset):
    def __init__(self, excerpts, labels, tokenizer, max_len):
        self.excerpts = excerpts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.excerpts)
    
    def __getitem__(self, item):
        excerpt = str(self.excerpts[item])
        
        encoding  = self.tokenizer.encode_plus(
            excerpt,
            max_length = self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        return {
            'excerpt_text': excerpt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.labels[item], dtype=torch.long)
        }


# In[35]:


class TestDataSet(data.Dataset):
    def __init__(self, ids, excerpts, tokenizer, max_len):
        self.ids = ids
        self.excerpts = excerpts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.excerpts)
    
    def __getitem__(self, item):
        excerpt = str(self.excerpts[item])
        excerpt_id = str(self.ids[item])
        
        encoding  = self.tokenizer.encode_plus(
            excerpt,
            max_length = self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        return {
            'excerpt_id': excerpt_id,
            'excerpt_text': excerpt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


# In[36]:


def create_train_data_loader(df, tokenizer, max_len, batch_size):
    excerpts = df['text'].to_numpy(),
    print(f'Excerpts size: {len(excerpts)}')
    labels = df['label'].to_numpy(),
    dataset = TrainDataSet(excerpts=excerpts[0], labels=labels[0], tokenizer=tokenizer, max_len=max_len)
    print(f'Dataset size: {len(dataset)}')
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

def create_test_data_loader(df, tokenizer, max_len, batch_size):
    excerpts = df['text'].to_numpy(),
    ids = df['id'].to_numpy(),
    print(f'Excerpts size: {len(excerpts)}')
    dataset = TestDataSet(ids= ids[0], excerpts=excerpts[0], tokenizer=tokenizer, max_len=max_len)
    print(f'Dataset size: {len(dataset)}')
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=4)


# In[37]:


train_set, val_set = train_test_split(train, test_size=0.2)


# In[38]:


BATCH_SIZE = 16
MAX_LEN = 160


# In[39]:


train_data_loader = create_train_data_loader(train_set, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
val_data_loader = create_train_data_loader(val_set, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)


# In[40]:


sample = next(iter(train_data_loader))
print(sample['input_ids'].shape)


# In[41]:


class DistilBertAuthorClassifier(nn.Module):
    def __init__(self):
        super(DistilBertAuthorClassifier, self).__init__()
        self.num_labels = 3

        self.softmax = nn.Softmax(dim=1)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
        # self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(self.distilbert.config.dim, 3)
        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        # print(f'hidden_state shape: {hidden_state.shape}')                
        # print(f'hidden_state shape[2]: {hidden_state.shape[2]}')                
        pooled_output = hidden_state[:, 0, :]                   
        # pooled_output = self.pre_classifier(pooled_output)   
        # pooled_output = nn.ReLU()(pooled_output)             
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)
        # logits = self.softmax(logits)
        return logits


# In[42]:


gc.collect()
model = DistilBertAuthorClassifier()
model = model.to(device)


# In[43]:


input_ids = sample['input_ids'].to(device)
attention_mask = sample['attention_mask'].to(device)

print(input_ids.shape)
print(attention_mask.shape)
prob, pred = torch.max(model(input_ids=input_ids, attention_mask=attention_mask),dim=1)
print(prob)
print(pred)
print(sample['targets'])


# In[44]:


model.distilbert.config


# In[45]:


EPOCHES = 3
if torch.cuda.is_available():
    optimizer = transformers.AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHES

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


# In[46]:


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        # print(targets.shape)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


# In[47]:


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


# In[48]:


def predict_authors(model, data_loader, submission_df):
    model = model.eval()
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(outputs)
            


# In[49]:


if is_colab and torch.cuda.is_available():
    optimizer = transformers.AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHES

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.4)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)


# In[50]:


filename = 'finalized_model.pt'

if is_colab and torch.cuda.is_available():
    model = model.to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHES):
        print(f'Epoch {epoch + 1}/{EPOCHES}')
        print('-'*10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            len(train_set)   
        )
        print(f'Train loss: {train_loss}, accuracy: {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            len(val_set)   
        )
        print(f'Validation loss: {val_loss}, accuracy: {val_acc}')

    torch.save(model.state_dict(), DIR_NAME+filename)


# In[51]:


if not is_colab:
    model.load_state_dict(torch.load('/kaggle/input/spookydistilbert/finalized_model.pt', map_location=device))
    model.eval()


# In[52]:


test_data_loader = create_test_data_loader(test, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
sample = next(iter(test_data_loader))

input_ids = sample['input_ids'].to(device)
attention_mask = sample['attention_mask'].to(device)

print(input_ids.shape)
print(attention_mask.shape)
prob, pred = torch.max(model(input_ids=input_ids, attention_mask=attention_mask),dim=1)
print(prob)
print(pred)


# In[53]:


# plt.plot(history['train_acc'], label='train accuracy')
# plt.plot(history['val_acc'], label='validation accuracy')

# plt.title('Training history')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
# plt.ylim([0, 1]);


# In[54]:


def test_model(model, data_loader, results_df):
    model = model.eval()
    submission = []

    with torch.no_grad():
        for d in data_loader:
            excerpt_ids = d['excerpt_id'],
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = nn.functional.softmax(outputs,dim=1)
#             print(excerpt_ids[0])
            for i, excerpt_id in enumerate(excerpt_ids[0]):
#                 print(i, excerpt_id)
#                 print(results_df[results_df['id']==excerpt_id])
                results_df.loc[results_df['id'] == excerpt_id, ['EAP','HPL','MWS']] =  outputs[i].tolist()
#                 results_df[results_df['id']==excerpt_id][['EAP','HPL','MWS']] = outputs[i].tolist()
    return results_df


# In[55]:


results_df = pd.read_csv('./sample_submission.csv')
results_df.head()


# In[56]:


results_df = test_model(model, test_data_loader, results_df)
results_df.head()


# In[57]:


results_df.to_csv("submission.csv", index=False)


# In[58]:


class LM:
  def __init__(self, n):
    self.n_gram = n
    
  def train(self, text):
    self.n_counts = defaultdict(Counter)
    for i in range(0, len(text) - self.n_gram + 1):
      t = text[i:i+self.n_gram-1]
      n_char = text[i+self.n_gram-1]
      self.n_counts[t][n_char] += 1
  
  def generate(self, init_text, n):
    text = init_text
    while len(text) < n:
      lookup_text = text[-self.n_gram+1:]
      if lookup_text not in self.n_counts:
        break
      counter = self.n_counts[text[-self.n_gram+1:]]
      keys = list(counter.keys())
      values = list(counter.values())
      probs = [v/sum(values) for v in values]
      cummulative_probs = np.cumsum(probs)
      p = np.random.rand()
      for i in range(len(cummulative_probs)):
        if p <= cummulative_probs[i]:
          text += keys[i]
          break
    return text


# In[59]:


def get_trained_lm(n, excerpts):
    lm = LM(n)
    text = ' '.join(list(excerpts))
    lm.train(text)
    return lm


# In[60]:


n=7
lm_EAP = get_trained_lm(n, train[train['author']=='EAP']['text'])
lm_HPL = get_trained_lm(n, train[train['author']=='HPL']['text'])
lm_MWS = get_trained_lm(n, train[train['author']=='MWS']['text'])
print(f"lm_EAP size: {len(lm_EAP.n_counts)}, lm_HPL size: {len(lm_HPL.n_counts)}, lm_MWS size: {len(lm_MWS.n_counts)}")


# In[61]:


seed_string = "Dark night"
l = 120
models = {'Poe':lm_EAP, 'Lovecraft': lm_HPL, 'Shelley': lm_MWS}
for model_name in models.keys():
    generated = models[model_name].generate(seed_string, l)
    print(f"\"{generated}...\", The bot {model_name}, 2020\n")

