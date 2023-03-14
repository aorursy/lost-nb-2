#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as py
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
import re
from nltk.tokenize import word_tokenize 
from PIL import Image
output_notebook()

import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda,BatchNormalization
from keras.optimizers import Adam,Adadelta
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm_notebook
import keras
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from keras.optimizers import SGD
from gensim.models import Doc2Vec

import torch
# Any results you write to the current directory are saved as output.


# In[2]:


#Distilbert for embeddings
import sys
get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')
sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers


# In[3]:


inputpath='../input/google-quest-challenge'

print("Reading the data")
traindata=pd.read_csv(inputpath+'/train.csv')
testdata=pd.read_csv(inputpath+'/test.csv')
submission=pd.read_csv(inputpath+'/sample_submission.csv')


# In[4]:


nrows = traindata.shape[0]
ncols=traindata.shape[1]

nrows1 = testdata.shape[0]
ncols1=testdata.shape[1]

categories = traindata["category"].nunique()
target_labels=[i for i in list(set(traindata.columns).intersection(submission.columns)) if i!='qa_id']

display(HTML(f"""<br>Number of rows in the training dataset: {nrows:,}</br>
                 <br>Number of rows in the test dataset: {nrows1:,}</br>
                 <br>Number of cols in the training dataset: {ncols:,}</br>
                 <br>Number of cols in the test dataset: {ncols1:,}</br>
                 <br>Number of unique categories in the training dataset: {categories:,}</br>
                  <br>Number of target labels: {len(target_labels):,}</br>
             """))


# In[5]:


traindata.head(2)


# In[6]:


traindata['sourcename']=traindata['host'].apply(lambda x: x.split('.')[0])
testdata['sourcename']=testdata['host'].apply(lambda x: x.split('.')[0])


# In[7]:


cnt_srs = traindata["sourcename"].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Different sources - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="sources")


# In[8]:


cnt_srs = traindata["category"].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Different Categories - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="categories")


# In[9]:


#Reference source: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-ashrae


# In[10]:



def make_plot(title, hist, edges, xlabel):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#1E90FF", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = f'{xlabel}'
    p.yaxis.axis_label = 'Distribution'
    p.grid.grid_line_color="white"
    return p

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_type_reason_explanation"].values, density=True, bins=10)
p1 = make_plot("Answer type reason", hist, edges, "answer")

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_well_written"].values, density=True, bins=10)
p2 = make_plot("Answer well written", hist, edges, 'answer')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_helpful"].values, density=True, bins=10)
p3 = make_plot("Answer helpful", hist, edges, 'answer')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_relevance"].values, density=True, bins=10)
p4 = make_plot("Answer relevant", hist, edges, 'answer')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_level_of_information"].values, density=True, bins=10)
p5 = make_plot("Answer level of information", hist, edges, 'answer')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["answer_plausible"].values, density=True, bins=10)
p6 = make_plot("Answer plausible", hist, edges, 'answer')

show(gridplot([p1,p2,p3,p4,p5,p6], ncols=3, plot_width=400, plot_height=400, toolbar_location=None))

del p1,p2,p3,p4,p5,p6


# In[11]:



def make_plot(title, hist, edges, xlabel):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#1E90FF", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = f'{xlabel}'
    p.yaxis.axis_label = 'Distribution'
    p.grid.grid_line_color="white"
    return p

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_well_written"].values, density=True, bins=10)
p1 = make_plot("Question well written", hist, edges, "Question")

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_type_entity"].values, density=True, bins=10)
p2 = make_plot("Question type entity", hist, edges, 'Question')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_type_choice"].values, density=True, bins=10)
p3 = make_plot("Question type choice", hist, edges, 'Question')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_fact_seeking"].values, density=True, bins=10)
p4 = make_plot("Question fact seeking", hist, edges, 'Question')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_not_really_a_question"].values, density=True, bins=10)
p5 = make_plot("Question?", hist, edges, 'Question')

temp_df = traindata[traindata["category"]=='STACKOVERFLOW']
hist, edges = np.histogram(temp_df["question_multi_intent"].values, density=True, bins=10)
p6 = make_plot("Question multi intent", hist, edges, 'Question')

show(gridplot([p1,p2,p3,p4,p5,p6], ncols=3, plot_width=400, plot_height=400, toolbar_location=None))

del p1,p2,p3,p4,p5,p6


# In[12]:


plt.figure(figsize=(16,12))
corr = traindata[target_labels].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3,
     center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[13]:



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}


def clean_text(x):
    x = str(x).replace("\n","")
    
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    
    stops  = set(STOPWORDS)
    text = [w for w in word_tokenize(x) if w not in stops]    
    text = " ".join(text)
    
    return text


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].apply(lambda x: clean_numbers(x))
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

    return df


# In[14]:


input_columns = ['question_title','question_body','answer']

traindata = clean_data(traindata, input_columns)
testdata = clean_data(testdata, input_columns)


# In[15]:


traindata.head(2)


# In[16]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
d = '../input/masks/masks-wordclouds/'


# In[17]:


comments_text = str(traindata.question_body)
comments_mask = np.array(Image.open(d + 'upvote.png'))
plot_wordcloud(comments_text, comments_mask, max_words=2000, max_font_size=300, 
               title = 'Most common words in all of the questions body', title_size=30)


# In[18]:


comments_text = str(traindata.answer)
comments_mask = np.array(Image.open(d + 'upvote.png'))
plot_wordcloud(comments_text, comments_mask, max_words=2000, max_font_size=300, 
               title = 'Most common words in all of the answers', title_size=30)


# In[19]:


#Credits: https://www.kaggle.com/abazdyrev/use-features-oof

features = ['sourcename', 'category']
merged = pd.concat([traindata[features], testdata[features]])
ohe = OneHotEncoder()
ohe.fit(merged)

features_train = ohe.transform(traindata[features]).toarray()
features_test = ohe.transform(testdata[features]).toarray()


# In[20]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[21]:


def fetch_vectors(string_list, batch_size=64):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in tqdm_notebook(chunks(string_list, batch_size)):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:200])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        
        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


# In[22]:


gc.collect()

train_question_body_dense = fetch_vectors(traindata.question_body.values)
train_answer_dense = fetch_vectors(traindata.answer.values)

test_question_body_dense = fetch_vectors(testdata.question_body.values)
test_answer_dense = fetch_vectors(testdata.answer.values)


# In[23]:


module_url = "../input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)


# In[24]:


embeddings_test = {}
embeddings_train = {}

for text in input_columns:
    print(text)
    train_text = traindata[text].str.replace('?', '.').str.replace('!', '.').tolist()
    test_text = testdata[text].str.replace('?', '.').str.replace('!', '.').tolist()
    
    curr_train_emb = []
    curr_test_emb = []
    batch_size = 4
    ind = 0
    
    while ind*batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1
        
    ind = 0
    while ind*batch_size < len(test_text):
        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1    
        
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)
    
del embed,curr_train_emb,curr_test_emb
K.clear_session()
gc.collect()


# In[25]:


# tfidf = TfidfVectorizer(ngram_range=(1, 3))
# tsvd = TruncatedSVD(n_components = 128, n_iter=5)

# tfquestion_title = tfidf.fit_transform(traindata["question_title"].values)
# tfquestion_title_test = tfidf.transform(testdata["question_title"].values)
# tfquestion_title = tsvd.fit_transform(tfquestion_title)
# tfquestion_title_test = tsvd.transform(tfquestion_title_test)

# tfquestion_body = tfidf.fit_transform(traindata["question_body"].values)
# tfquestion_body_test = tfidf.transform(testdata["question_body"].values)
# tfquestion_body = tsvd.fit_transform(tfquestion_body)
# tfquestion_body_test = tsvd.transform(tfquestion_body_test)

# tfanswer = tfidf.fit_transform(traindata["answer"].values)
# tfanswer_test = tfidf.transform(testdata["answer"].values)
# tfanswer = tsvd.fit_transform(tfanswer)
# tfanswer_test = tsvd.transform(tfanswer_test)

# del tfidf,tsvd


# In[26]:


def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


train_question_body_sentences = constructLabeledSentences(traindata['question_body'])
train_question_title_sentences = constructLabeledSentences(traindata['question_title'])
train_answer_sentences = constructLabeledSentences(traindata['answer'])

test_question_body_sentences = constructLabeledSentences(testdata['question_body'])
test_question_title_sentences = constructLabeledSentences(testdata['question_title'])
test_answer_sentences = constructLabeledSentences(testdata['answer'])


# In[27]:


all_sentences = train_question_body_sentences +                 train_answer_sentences +                 test_question_body_sentences +                 test_answer_sentences

Text_INPUT_DIM=128
text_model = Doc2Vec(min_count=1, window=5, vector_size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, epochs=5,seed=1)
text_model.build_vocab(all_sentences)
text_model.train(all_sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)


# In[28]:


def infer_vec(df, columns: list):
    for col in columns:
        df[col+'_vec'] = df[col].apply(lambda x: np.array(text_model.infer_vector([x])))

    return df

traindata = infer_vec(traindata, input_columns)
testdata = infer_vec(testdata, input_columns)


# In[29]:


l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)
cos_dist = lambda x, y: (x*y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
]).T

dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
]).T

X_train = np.hstack([item for k, item in embeddings_train.items()]+ [features_train, dist_features_train])
X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])
y_train = traindata[target_labels].values


# In[30]:


X_train = np.concatenate((X_train,train_question_body_dense,train_answer_dense,np.array(traindata['question_title_vec'].tolist())
                    ,np.array(traindata['question_body_vec'].tolist()),np.array(traindata['answer_vec'].tolist())),axis=1)
X_test = np.concatenate((X_test,test_question_body_dense,test_answer_dense,np.array(testdata['question_title_vec'].tolist())
                  ,np.array(testdata['question_body_vec'].tolist()),np.array(testdata['answer_vec'].tolist())),axis=1)


# In[31]:


class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
            self.model.save_weights(self.model_name)
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[32]:


def swish(x):
    return K.sigmoid(x) * x

def mish(x):
    return x * keras.backend.tanh(keras.backend.softplus(x))


def create_model():
    inps = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation=swish)(inps)
    x = Dropout(0.2)(x)
    x = Dense(y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=['binary_crossentropy']
    )
    model.summary()
    return model

def create_model2():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation=mish
              )(input1)
    x = Dropout(0.2)(x)
    output = Dense(len(target_labels),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adadelta()
    
    model.compile(
        optimizer=optimizer,
        loss=['binary_crossentropy']
    )
    model.summary()
    return model

def create_model3():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation=swish)(input1)
    x = Dropout(0.2)(x)
    x = Dense(256, activation=swish,
              kernel_regularizer=keras.regularizers.l2(0.01)
             )(x)
    x = Dropout(0.2)(x)
    output = Dense(len(target_labels),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=['binary_crossentropy']
    )
    model.summary()
    return model


# In[33]:


all_predictions = []

kf = MultilabelStratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for ind, (tr, val) in enumerate(kf.split(X_train,y_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=8, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=u'best_model_batch.h5')]
    )
    model.load_weights('best_model_batch.h5')
    all_predictions.append(model.predict(X_test))
    
    os.remove('best_model_batch.h5')
    
model = create_model3()
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))
    
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model2()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=64, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=u'best_model_batch.h5')]
    )
    model.load_weights('best_model_batch.h5')
    all_predictions.append(model.predict(X_test))
    
    os.remove('best_model_batch.h5')
    
    
# model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)
# model.fit(X_train, y_train)
# all_predictions.append(model.predict(X_test))

model = create_model3()
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))

model = create_model3()
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=False)
all_predictions.append(model.predict(X_test))


# In[34]:


test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)
max_val = test_preds.max() + 1
test_preds = test_preds/max_val + 1e-12


# In[35]:


submission = pd.read_csv(path_join(inputpath, 'sample_submission.csv'))
submission[target_labels] = test_preds
submission.to_csv("submission.csv", index = False)
submission.head()


# In[ ]:




