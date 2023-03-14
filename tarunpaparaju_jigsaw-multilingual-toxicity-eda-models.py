#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q pyicu')
get_ipython().system('pip install -q pycld2')
get_ipython().system('pip install -q polyglot')
get_ipython().system('pip install -q textstat')
get_ipython().system('pip install -q googletrans')


# In[2]:


import warnings
warnings.filterwarnings("ignore")

import os
import gc
import re
import folium
import textstat
from scipy import stats
from colorama import Fore, Back, Style, init

import math
import numpy as np
import scipy as sp
import pandas as pd

import random
import networkx as nx
from pandas import Timestamp

from PIL import Image
from IPython.display import SVG
from keras.utils import model_to_dot

import requests
from IPython.display import HTML

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

tqdm.pandas()

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import transformers
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.optimizers import Adam
from tokenizers import BertWordPieceTokenizer
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding
from tensorflow.keras.layers import LSTM, GRU, Conv1D, SpatialDropout1D

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *

from sklearn import metrics
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer,                                            CountVectorizer,                                            HashingVectorizer

from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer  

import nltk
from textblob import TextBlob

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from googletrans import Translator
from nltk import WordNetLemmatizer
from polyglot.detect import Detector
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stopword=set(STOPWORDS)

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

np.random.seed(0)


# In[3]:


DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"
os.listdir(DATA_PATH)


# In[4]:


TEST_PATH = DATA_PATH + "test.csv"
VAL_PATH = DATA_PATH + "validation.csv"
TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"

val_data = pd.read_csv(VAL_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)


# In[5]:


train_data.head()


# In[6]:


val_data.head()


# In[7]:


test_data.head()


# In[8]:


def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in train_data["comment_text"]])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='Common words in comments')


# In[9]:


def get_language(text):
    return Detector("".join(x for x in text if x.isprintable()), quiet=True).languages[0].name

train_data["lang"] = train_data["comment_text"].progress_apply(get_language)


# In[10]:


lang_list = sorted(list(set(train_data["lang"])))
counts = [list(train_data["lang"]).count(cont) for cont in lang_list]
df = pd.DataFrame(np.transpose([lang_list, counts]))
df.columns = ["Language", "Count"]
df["Count"] = df["Count"].apply(int)

df_en = pd.DataFrame(np.transpose([["English", "Non-English"], [max(counts), sum(counts) - max(counts)]]))
df_en.columns = ["Language", "Count"]

fig = px.bar(df_en, x="Language", y="Count", title="Language of comments", color="Language", text="Count")
fig.update_layout(template="plotly_white")
fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.5
fig.data[1].marker.line.color = 'rgb(0, 0, 0)'
fig.data[1].marker.line.width = 0.5
fig.data[0].textfont.color = "black"
fig.data[0].textposition = "outside"
fig.data[1].textfont.color = "black"
fig.data[1].textposition = "outside"
fig


# In[11]:


fig = px.bar(df.query("Language != 'English' and Language != 'un'").query("Count >= 50"),
             y="Language", x="Count", title="Language of non-English comments", template="plotly_white", color="Language", text="Count", orientation="h")
fig.update_traces(marker=dict(line=dict(width=0.75,
                                        color='black')),  textposition="outside")
fig.update_layout(showlegend=False)
fig


# In[12]:


fig = go.Figure([go.Pie(labels=df.query("Language != 'English' and Language != 'un'").query("Count >= 50")["Language"],
           values=df.query("Language != 'English' and Language != 'un'").query("Count >= 50")["Count"])])
fig.update_layout(title_text="Pie chart of non-English languages", template="plotly_white")
fig.data[0].marker.colors = [px.colors.qualitative.Plotly[2:]]
fig.data[0].textfont.color = "black"
fig.data[0].textposition = "outside"
fig.show()


# In[13]:


def get_country(language):
    if language == "German":
        return "Germany"
    if language == "Scots":
        return "Scotland"
    if language == "Danish":
        return "Denmark"
    if language == "Arabic":
        return "Saudi Arabia"
    if language == "Spanish":
        return "Spain"
    if language == "Persian":
        return "Iran"
    if language == "Greek":
        return "Greece"
    if language == "Portuguese":
        return "Portugal"
    if language == "English":
        return "United Kingdom"
    if language == "Hindi":
        return "India"
    if language == "Albanian":
        return "Albania"
    if language == "Bosnian":
        return "Bosnia and Herzegovina"
    if language == "Croatian":
        return "Croatia"
    if language == "Dutch":
        return "Netherlands"
    if language == "Russian":
        return "Russia"
    if language == "Vietnamese":
        return "Vietnam"
    if language == "Somali":
        return "Somalia"
    if language == "Turkish":
        return "Turkey"
    if language == "Serbian":
        return "Serbia"
    if language == "Indonesian":
        return "Indonesia"
    if language == "Manx":
        return "Ireland"
    if language == "Scots":
        return "Scotland"
    if language == "Latin":
        return "Holy See (Vatican City State)"
    if language == "Afrikaans":
        return "South Africa"
    return "None"
    
df["country"] = df["Language"].progress_apply(get_country)


# In[14]:


fig = px.choropleth(df.query("Language != 'English' and Language != 'un' and country != 'None'").query("Count >= 5"), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="Countries of non-English languages", color="Count",
                     template="plotly", color_continuous_scale="agsunset")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# In[15]:


fig = px.choropleth(df.query("Language != 'English' and Language != 'un' and country != 'None'"), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="Non-English European countries", color="Count",
                     template="plotly", color_continuous_scale="aggrnyl", scope="europe")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# In[16]:


fig = px.choropleth(df.query("Language != 'English' and Language != 'un' and country != 'None'"), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="Asian countries", color="Count",
                     template="plotly", color_continuous_scale="spectral", scope="asia")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# In[17]:


fig = px.choropleth(df.query("Language != 'English' and Language != 'un' and country != 'None'").query("Count >= 5"), locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="African countries", color="Count",
                     template="plotly", color_continuous_scale="agsunset", scope="africa")
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.2
fig.show()


# In[18]:


def new_len(x):
    if type(x) is str:
        return len(x.split())
    else:
        return 0

train_data["comment_words"] = train_data["comment_text"].apply(new_len)
nums = train_data.query("comment_words != 0 and comment_words < 200").sample(frac=0.1)["comment_words"]
fig = ff.create_distplot(hist_data=[nums],
                         group_labels=["All comments"],
                         colors=["coral"])

fig.update_layout(title_text="Comment words", xaxis_title="Comment words", template="simple_white", showlegend=False)
fig.show()


# In[19]:


df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["comment_words"]]))
df.columns = ["Language", "Average_comment_words"]
df["Average_comment_words"] = df["Average_comment_words"].apply(float)
df = df.query("Average_comment_words < 500")
fig = go.Figure(go.Bar(x=df["Language"], y=df["Average_comment_words"]))

fig.update_layout(xaxis_title="Language", yaxis_title="Average comment words", title_text="Average comment words vs. language", template="plotly_white")
fig.show()


# In[20]:


df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                     projection="natural earth", locationmode="country names", title="Average comment length vs. Country", color="Average_comment_words",
                     template="plotly", color_continuous_scale="aggrnyl")
fig


# In[21]:


def polarity(x):
    if type(x) == str:
        return SIA.polarity_scores(x)
    else:
        return 1000
    
SIA = SentimentIntensityAnalyzer()
train_data["polarity"] = train_data["comment_text"].progress_apply(polarity)


# In[22]:


fig = go.Figure(go.Histogram(x=[pols["neg"] for pols in train_data["polarity"] if pols["neg"] != 0], marker=dict(
            color='seagreen')
    ))

fig.update_layout(xaxis_title="Negativity sentiment", title_text="Negativity sentiment", template="simple_white")
fig.show()


# In[23]:


train_data["negativity"] = train_data["polarity"].apply(lambda x: x["neg"])
df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["negativity"].tolist()]))
df.columns = ["Language", "Negativity"]
df["Negativity"] = df["Negativity"].apply(float)
df = df.query("Negativity != 0")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average negative sentiment vs. Country", color="Negativity",
                    template="plotly", color_continuous_scale="greens")
fig.show()


# In[24]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["negativity"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["negativity"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Negativity vs. Toxicity", xaxis_title="Negativity", template="simple_white")
fig.show()


# In[25]:


fig = go.Figure(go.Histogram(x=[pols["pos"] for pols in train_data["polarity"] if pols["pos"] != 0], marker=dict(
            color='indianred')
    ))

fig.update_layout(xaxis_title="Positivity sentiment", title_text="Positivity sentiment", template="simple_white")
fig.show()


# In[26]:


train_data["positivity"] = train_data["polarity"].apply(lambda x: x["pos"])
df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["positivity"].tolist()]))
df.columns = ["Language", "Positivity"]
df["Positivity"] = df["Positivity"].apply(float)
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average positive sentiment vs. Country", color="Positivity",
                    template="plotly", color_continuous_scale="reds")
fig.show()


# In[27]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["positivity"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["positivity"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Positivity vs. Toxicity", xaxis_title="Positivity", template="simple_white")
fig.show()


# In[28]:


fig = go.Figure(go.Histogram(x=[pols["neu"] for pols in train_data["polarity"] if pols["neu"] != 1], marker=dict(
            color='dodgerblue')
    ))

fig.update_layout(xaxis_title="Neutrality sentiment", title_text="Neutrality sentiment", template="simple_white")
fig.show()


# In[29]:


train_data["neutrality"] = train_data["polarity"].apply(lambda x: x["neu"])
df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["neutrality"].tolist()]))
df.columns = ["Language", "Neutrality"]
df["Neutrality"] = df["Neutrality"].apply(float)
df = df.query("Neutrality != 1")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average neutral sentiment vs. Country", color="Neutrality",
                    template="plotly", color_continuous_scale="blues")
fig.show()


# In[30]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["neutrality"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["neutrality"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Neutrality vs. Toxicity", xaxis_title="Neutrality", template="simple_white")
fig.show()


# In[31]:


fig = go.Figure(go.Histogram(x=[pols["compound"] for pols in train_data["polarity"] if pols["compound"] != 0], marker=dict(
            color='orchid')
    ))

fig.update_layout(xaxis_title="Compound sentiment", title_text="Compound sentiment", template="simple_white")
fig.show()


# In[32]:


train_data["compound"] = train_data["polarity"].apply(lambda x: x["compound"])
df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["compound"].tolist()]))
df.columns = ["Language", "Compound"]
df["Compound"] = df["Compound"].apply(float)
df = df.query("Compound != 0")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average compound sentiment vs. Country", color="Compound",
                    template="plotly", color_continuous_scale="purples")
fig.show()


# In[33]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["compound"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["compound"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Compound vs. Toxicity", xaxis_title="Compound", template="simple_white")
fig.show()


# In[34]:


train_data["flesch_reading_ease"] = train_data["comment_text"].progress_apply(textstat.flesch_reading_ease)
train_data["automated_readability"] = train_data["comment_text"].progress_apply(textstat.automated_readability_index)
train_data["dale_chall_readability"] = train_data["comment_text"].progress_apply(textstat.dale_chall_readability_score)


# In[35]:


fig = go.Figure(go.Histogram(x=train_data.query("flesch_reading_ease > 0")["flesch_reading_ease"], marker=dict(
            color='darkorange')
    ))

fig.update_layout(xaxis_title="Flesch reading ease", title_text="Flesch reading ease", template="simple_white")
fig.show()


# In[36]:


df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["flesch_reading_ease"].tolist()]))
df.columns = ["Language", "flesch_reading_ease"]
df["flesch_reading_ease"] = df["flesch_reading_ease"].apply(float)
df = df.query("flesch_reading_ease > 0")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average Flesch reading ease vs. Country", color="flesch_reading_ease",
                    template="plotly", color_continuous_scale="oranges")
fig.show()


# In[37]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["flesch_reading_ease"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["flesch_reading_ease"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Flesch reading ease vs. Toxicity", xaxis_title="Flesch reading ease", template="simple_white")
fig.show()


# In[38]:


fig = go.Figure(go.Histogram(x=train_data.query("automated_readability < 100")["automated_readability"], marker=dict(
            color='mediumaquamarine')
    ))

fig.update_layout(xaxis_title="Automated readability", title_text="Automated readability", template="simple_white")
fig.show()


# In[39]:


df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["automated_readability"].tolist()]))
df.columns = ["Language", "automated_readability"]
df["automated_readability"] = df["automated_readability"].apply(float)
df = df.query("automated_readability < 100")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Automated readability vs. Country", color="automated_readability",
                    template="plotly", color_continuous_scale="GnBu")
fig.show()


# In[40]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["automated_readability"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["automated_readability"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Automated readability vs. Toxicity", xaxis_title="Automated readability", template="simple_white")
fig.show()


# In[41]:


fig = go.Figure(go.Histogram(x=train_data.query("dale_chall_readability < 20")["dale_chall_readability"], marker=dict(
            color='deeppink')
    ))

fig.update_layout(xaxis_title="Dale-Chall readability", title_text="Dale-Chall readability", template="simple_white")
fig.show()


# In[42]:


df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["dale_chall_readability"].tolist()]))
df.columns = ["Language", "dale_chall_readability"]
df["dale_chall_readability"] = df["dale_chall_readability"].apply(float)
df = df.query("dale_chall_readability < 20")
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Dale-Chall readability vs. Country", color="dale_chall_readability",
                    template="plotly", color_continuous_scale="PuRd")
fig.show()


# In[43]:


nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["dale_chall_readability"]
nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["dale_chall_readability"]

fig = ff.create_distplot(hist_data=[nums_1, nums_2],
                         group_labels=["Toxic", "Non-toxic"],
                         colors=["darkorange", "dodgerblue"], show_hist=False)

fig.update_layout(title_text="Dale-Chall readability vs. Toxicity", xaxis_title="Dale-Chall readability", template="simple_white")
fig.show()


# In[44]:


clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))
clean_mask=clean_mask[:,:,1]

subset = train_data.query("toxic == 0")
text = subset.comment_text.values
wc = WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(7.5, 7.5))
plt.axis("off")
plt.title("Words frequented in Clean Comments", fontsize=16)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()

clean_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))
clean_mask=clean_mask[:,:,1]

subset = train_data.query("toxic == 1")
text = subset.comment_text.values
wc = WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(7.5, 7.5))
plt.axis("off")
plt.title("Words frequented in Toxic Comments", fontsize=16)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()


# In[45]:


toxic_mask=np.array(Image.open("../input/imagesforkernal/toxic-sign.png"))
toxic_mask=toxic_mask[:,:,1]
#wordcloud for clean comments
subset=train_data.query("obscene == 1")
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=4000,mask=toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(20,20))
plt.subplot(221)
plt.axis("off")
plt.title("Words frequented in Obscene Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)

#Severely toxic comments
plt.subplot(222)
severe_toxic_mask=np.array(Image.open("../input/imagesforkernal/bomb.png"))
severe_toxic_mask=severe_toxic_mask[:,:,1]
subset=train_data[train_data.severe_toxic==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=severe_toxic_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Severe Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244), alpha=0.98)

#Threat comments
plt.subplot(223)
threat_mask=np.array(Image.open("../input/imagesforkernal/anger.png"))
threat_mask=threat_mask[:,:,1]
subset=train_data[train_data.threat==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=threat_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Threatening Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'summer' , random_state=2534), alpha=0.98)

#insult
plt.subplot(224)
insult_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))
insult_mask=insult_mask[:,:,1]
subset=train_data[train_data.insult==1]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,mask=insult_mask,stopwords=stopword)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in insult Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)

plt.show()


# In[46]:


fig = go.Figure(data=[
    go.Pie(labels=train_data.columns[2:7],
           values=train_data.iloc[:, 2:7].sum().values, marker=dict(colors=px.colors.qualitative.Plotly))
])
fig.update_traces(textposition='outside', textfont=dict(color="black"))
fig.update_layout(title_text="Pie chart of labels")
fig.show()


# In[47]:


fig = go.Figure(data=[
    go.Bar(y=train_data.columns[2:7],
           x=train_data.iloc[:, 2:7].sum().values, marker=dict(color=px.colors.qualitative.Plotly))
])

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
fig.data[0].marker.line.width = 0.75
fig.update_traces(orientation="h")
fig.update_layout(title_text="Bar chart of labels", template="plotly_white")
fig.show()


# In[48]:


df = pd.DataFrame(np.transpose([lang_list, train_data.groupby("lang").mean()["toxic"].tolist()]))
df.columns = ["Language", "toxicity"]
df["toxicity"] = df["toxicity"].apply(float)
df["country"] = df["Language"].apply(get_country)
df = df.query("country != 'None'")

fig = px.choropleth(df, locations="country", hover_name="country",
                    projection="natural earth", locationmode="country names", title="Average toxicity vs. Country", color="toxicity",
                    template="plotly", color_continuous_scale="tealrose")
fig.show()


# In[49]:


val = val_data
train = train_data

def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

val["comment_text"] = clean(val["comment_text"])
test_data["content"] = clean(test_data["content"])
train["comment_text"] = clean(train["comment_text"])


# In[50]:


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


# In[51]:


def fast_encode(texts, tokenizer, chunk_size=240, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[52]:


AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

EPOCHS = 2
BATCH_SIZE = 32 * strategy.num_replicas_in_sync


# In[53]:


tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

save_path = '/kaggle/working/distilbert_base_uncased/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', 
                                        lowercase=True)


# In[54]:


x_train = fast_encode(train.comment_text.astype(str), 
                      fast_tokenizer, maxlen=512)
x_valid = fast_encode(val_data.comment_text.astype(str).values, 
                      fast_tokenizer, maxlen=512)
x_test = fast_encode(test_data.content.astype(str).values, 
                     fast_tokenizer, maxlen=512)

y_valid = val.toxic.values
y_train = train.toxic.values


# In[55]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# In[56]:


def build_vnn_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    
    conc = K.sum(embedding, axis=2)
    conc = Dense(128, activation='relu')(conc)
    conc = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=input_word_ids, outputs=conc)
    
    model.compile(Adam(lr=0.01), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# In[57]:


with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.    from_pretrained('distilbert-base-multilingual-cased')
    model_vnn = build_vnn_model(transformer_layer, max_len=512)

model_vnn.summary()


# In[58]:


SVG(tf.keras.utils.model_to_dot(model_vnn, dpi=70).create(prog='dot', format='svg'))


# In[59]:


def callback():
    cb = []

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',  
                                    factor=0.3, patience=3, 
                                    verbose=1, mode='auto', 
                                    epsilon=0.0001, cooldown=1, min_lr=0.000001)
    cb.append(reduceLROnPlat)
    log = CSVLogger('log.csv')
    cb.append(log)

    RocAuc = RocAucEvaluation(validation_data=(x_valid, y_valid), interval=1)
    cb.append(RocAuc)
    
    return cb


# In[60]:


N_STEPS = x_train.shape[0] // BATCH_SIZE
calls = callback()

train_history = model_vnn.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)


# In[61]:


translator = Translator()

def visualize_model_preds(model, indices=[0, 17, 1, 24]):
    comments = val_data.comment_text.loc[indices].values.tolist()
    preds = model.predict(x_valid[indices].reshape(len(indices), -1))

    for idx, i in enumerate(indices):
        if y_valid[i] == 0:
            label = "Non-toxic"
            color = f'{Fore.GREEN}'
            symbol = '\u2714'
        else:
            label = "Toxic"
            color = f'{Fore.RED}'
            symbol = '\u2716'

        print('{}{} {}'.format(color, str(idx+1) + ". " + label, symbol))
        print(f'{Style.RESET_ALL}')
        print("ORIGINAL")
        print(comments[idx]); print("")
        print("TRANSLATED")
        print(translator.translate(comments[idx]).text)
        fig = go.Figure()
        if list.index(sorted(preds[:, 0]), preds[idx][0]) > 1:
            yl = [preds[idx][0], 1 - preds[idx][0]]
        else:
            yl = [1 - preds[idx][0], preds[idx][0]]
        fig.add_trace(go.Bar(x=['Non-Toxic', 'Toxic'], y=yl, marker=dict(color=["seagreen", "indianred"])))
        fig.update_traces(name=comments[idx])
        fig.update_layout(xaxis_title="Labels", yaxis_title="Probability", template="plotly_white", title_text="Predictions for validation comment #{}".format(idx+1))
        fig.show()
        
visualize_model_preds(model_vnn)


# In[62]:


def build_cnn_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    
    embedding = SpatialDropout1D(0.3)(embedding)
    conv_1 = Conv1D(64, 2)(embedding)
    conv_2 = Conv1D(64, 3)(embedding)
    conv_3 = Conv1D(64, 4)(embedding)
    conv_4 = Conv1D(64, 5)(embedding)
    
    maxpool_1 = GlobalAveragePooling1D()(conv_1)
    maxpool_2 = GlobalAveragePooling1D()(conv_2)
    maxpool_3 = GlobalAveragePooling1D()(conv_3)
    maxpool_4 = GlobalAveragePooling1D()(conv_4)
    conc = concatenate([maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)

    conc = Dense(64, activation='relu')(conc)
    conc = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=input_word_ids, outputs=conc)
    
    model.compile(Adam(lr=0.01), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# In[63]:


with strategy.scope():
    model_cnn = build_cnn_model(transformer_layer, max_len=512)

model_cnn.summary()


# In[64]:


SVG(tf.keras.utils.model_to_dot(model_cnn, dpi=70).create(prog='dot', format='svg'))


# In[65]:


train_history = model_cnn.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)


# In[66]:


visualize_model_preds(model_cnn)


# In[67]:


class AttentionWeightedAverage(Layer):

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


# In[68]:


def build_lstm_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    
    embedding = SpatialDropout1D(0.3)(embedding)
    lstm_1 = LSTM(128, return_sequences=True)(embedding)
    lstm_2 = LSTM(128, return_sequences=True)(lstm_1)
    
    attention = AttentionWeightedAverage()(lstm_2)
    conc = Dense(64, activation='relu')(attention)
    conc = Dense(1, activation='sigmoid')(conc)
    
    model = Model(inputs=input_word_ids, outputs=conc)
    
    model.compile(Adam(lr=0.01), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# In[69]:


with strategy.scope():
    model_lstm = build_lstm_model(transformer_layer, max_len=512)

model_lstm.summary()


# In[70]:


SVG(tf.keras.utils.model_to_dot(model_lstm, dpi=70).create(prog='dot', format='svg'))


# In[71]:


train_history = model_lstm.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)


# In[72]:


visualize_model_preds(model_lstm)


# In[73]:


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

class Capsule(Layer):

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 initializer='glorot_uniform',
                 activation=None,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        self.activation = activations.get(activation)
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializers.get(initializer)
        self.constraint = constraints.get(constraint)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule *
                                            self.dim_capsule),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     constraint=self.constraint,
                                     trainable=True)

        self.build = True

    def call(self, inputs):
        if self.share_weights:
            u_hat_vectors = K.conv1d(inputs, self.W)
        else:
            u_hat_vectors = K.local_conv1d(inputs, self.W, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        u_hat_vectors = K.reshape(u_hat_vectors, (batch_size,
                                                  input_num_capsule,
                                                  self.num_capsule,
                                                  self.dim_capsule))

        u_hat_vectors = K.permute_dimensions(u_hat_vectors, (0, 2, 1, 3))
        routing_weights = K.zeros_like(u_hat_vectors[:, :, :, 0])

        for i in range(self.routings):
            capsule_weights = K.softmax(routing_weights, 1)
            outputs = K.batch_dot(capsule_weights, u_hat_vectors, [2, 2])
            if K.ndim(outputs) == 4:
                outputs = K.sum(outputs, axis=1)
            if i < self.routings - 1:
                outputs = K.l2_normalize(outputs, -1)
                routing_weights = K.batch_dot(outputs, u_hat_vectors, [2, 3])
                if K.ndim(routing_weights) == 4:
                    routing_weights = K.sum(routing_weights, axis=1)

        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[74]:


def build_capsule_model(transformer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    
    embedding = SpatialDropout1D(0.3)(embedding)
    capsule = Capsule(num_capsule=5, dim_capsule=5,
                      routings=4, activation=squash)(embedding)

    capsule = Flatten()(capsule)
    output = Dense(128, activation='relu')(capsule)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(inputs=input_word_ids, outputs=output)
    
    model.compile(Adam(lr=1.5e-5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# In[75]:


with strategy.scope():
    model_capsule = build_capsule_model(transformer_layer, max_len=512)

model_capsule.summary()


# In[76]:


train_history = model_capsule.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)


# In[77]:


SVG(tf.keras.utils.model_to_dot(model_capsule, dpi=70).create(prog='dot', format='svg'))


# In[78]:


visualize_model_preds(model_capsule)


# In[79]:


def build_distilbert_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    cls_token = Dense(500, activation="elu")(cls_token)
    cls_token = Dropout(0.1)(cls_token)
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    
    model.compile(Adam(lr=1.5e-5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# In[80]:


with strategy.scope():
    model_distilbert = build_distilbert_model(transformer_layer, max_len=512)

model_distilbert.summary()


# In[81]:


train_history = model_distilbert.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)


# In[82]:


SVG(tf.keras.utils.model_to_dot(model_distilbert, dpi=70).create(prog='dot', format='svg'))


# In[83]:


visualize_model_preds(model_distilbert)


# In[84]:


sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
sub['toxic'] = model_distilbert.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)

