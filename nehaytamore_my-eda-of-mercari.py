#!/usr/bin/env python
# coding: utf-8

# In[1]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
get_ipython().system('pip install git+https://github.com/LIAAD/yake')
# For example, here's several helpful packages to load in 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
import plotly.offline as py
py.init_notebook_mode(connected=True)
import multiprocessing as mp

import string
import spacy 
import en_core_web_sm
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly.graph_objs as go


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


# In[2]:


train_data = pd.read_table("../input/train.tsv")
test_data = pd.read_table("../input/test_stg2.tsv")


# In[3]:


#As the size of the dataset could be too large to be able to run this notebook quickly,
#we have selected first 20K points for some of the experiments 
train_data_partial = train_data.loc[0:20000,:].copy()


# In[4]:


print("train data shape = ",train_data.shape)
print("test data shape = ", test_data.shape)


# In[5]:


train_data.dtypes


# In[6]:


train_data.head()


# In[7]:


train_data.isnull().sum()/train_data.shape[0]


# In[8]:


train_data.info()


# In[9]:


train_data.describe(include = 'all')


# In[10]:


train_data.head()


# In[11]:


train_data.duplicated(subset = ['name', 'item_description', 'brand_name', 'shipping','category_name', 'price']).sum()  #name, description, brand, shipping and price


# In[12]:


train_data.duplicated(subset = ['item_description', 'shipping', 'price']).sum()  #name, description, brand, shipping and price


# In[13]:


train_data.duplicated(subset = ['name', 'brand_name', 'shipping', 'price']).sum()  #name, description, brand, shipping and price


# In[14]:


train_data[train_data.duplicated(subset = ['name', 'brand_name', 'category_name', 'shipping', 'price'])].head()  #name, description, brand, shipping and price


# In[15]:


train_data[(train_data.name == "Giffin 25 rdta full tank kit") & (train_data.price == 25.0) ]


# In[16]:


print(train_data.loc[1199,'item_description'])
print(train_data.loc[1630,'item_description'])
#mostly same items!


# In[17]:


#Lets try couple of more
train_data[(train_data.name == "Too Faced Better Than Sex Mascara") & (train_data.price == 18.0) ]


# In[18]:


train_data[(train_data.name == "PINK by Victoria's Secret lace bandeau") & (train_data.price == 7.0) ]


# In[19]:


print(train_data.loc[53,'item_description'])
print(train_data.loc[3329,'item_description'])


# In[20]:


train_data.price.describe()


# In[21]:


#Price 0 is not possible, thus we will remove those points
train_data = train_data[train_data.price!=0]


# In[22]:


fig, ax = plt.subplots(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train_data['price'], bins = 50)
plt.xlabel('price+', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Price Distribution - Training Set', fontsize=17)

plt.subplot(1, 2, 2)
sns.distplot(np.log(train_data['price'] + 1), bins = 50)
plt.xlabel('log(price+1)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Log(Price) Distribution - Training Set', fontsize=17)
plt.show()


# In[23]:


fig, ax = plt.subplots(figsize=(10, 10))
#How many values have price above 500?
print((train_data.price>500).sum())

# Very small proportion. This could be better explored with a box plot.
sns.boxplot(y = 'price', data = train_data)
plt.ylim(0,300)


# In[24]:


# For almost 95% the of the datapoints, the price is less than 75
# 99% data is covered by price less than 170
train_data.price.describe(percentiles=[0.8,0.9,0.95,0.99])

#These prices could be outliers or very expensive gadgets/gold and other comodities 


# In[25]:


train_data.shipping.value_counts(normalize=True)


# In[26]:


train_data.groupby('shipping')['price'].describe(percentiles = [0.9,0.95,0.99])


# In[27]:


fig, ax = plt.subplots(figsize=(20, 10))
sns.distplot(np.log(train_data.loc[train_data['shipping']==1,"price"])+1 , color="red", label="shipping 1",hist=False, rug=True)
sns.distplot(np.log(train_data.loc[train_data['shipping']==0,"price"])+1 , color="skyblue", label="Shipping 0",hist=False, rug = True)


# In[28]:


print(train_data.item_condition_id.value_counts())
train_data.item_condition_id.value_counts(normalize=True)


# In[29]:


train_data.groupby('item_condition_id')['price'].describe()


# In[30]:


# the distribution of condition vs price in box plot could be helpful
fig, ax = plt.subplots(figsize=(20, 10))
sns.violinplot(x='item_condition_id', y='price', data=train_data)
plt.ylim(0,200)

train_data.groupby('item_condition_id').price.describe()


# In[31]:


fig, ax = plt.subplots(figsize=(20, 10))
sns.distplot(train_data.loc[train_data['item_condition_id']==5,'price'])


# In[32]:


sns.lmplot(x = 'item_condition_id',y = 'price',data = train_data,x_estimator = np.median,col='shipping',fit_reg=False)
sns.lmplot(x = 'item_condition_id',y = 'price',data = train_data,x_estimator = np.mean,col='shipping',fit_reg=False)


# In[33]:


g = sns.FacetGrid(train_data,col='item_condition_id',row='shipping')
g.map(plt.hist,"price")
plt.xlim(0,500)


# In[34]:


train_data.loc[(train_data['item_condition_id']==5) & (train_data['price']>100)].groupby('brand_name').describe()
# Apple products with bad condition still have high price!


# In[35]:


pd.set_option('display.max_colwidth',1000)
train_data.loc[(train_data['item_condition_id']==5) & (train_data['price']>100) & (train_data['brand_name']=='American Girl ®')]


# In[36]:


pd.crosstab(train_data.item_condition_id,train_data.shipping,normalize=True)


# In[37]:




sns.lmplot(x='item_condition_id', y='price', col='shipping',data=train_data)


# In[38]:


print("Brand name is absent in ", train_data.brand_name.isnull().sum(), "number of samples")
print("Equivalnt to ", (train_data['brand_name'].isnull().sum()/train_data.shape[0]*100), "%")


# In[39]:


# Many a times brand names contribute a lot to the pricing of the products. Let's verify this hypothesis
print("Number of unique brands = ",train_data.brand_name.nunique())
train_data.brand_name.value_counts().head(10)


# In[40]:


# brand_name 
# What is the average price of the most frequent brands?
train_data.groupby('brand_name').price.agg({"mean_price": np.mean,"count":'count'}).sort_values(
    "count", ascending=False).head(10).reset_index()


# In[41]:


train_data.groupby('brand_name').price.agg({'mean' : np.mean, 'count':'count'}).sort_values(by = 'mean', ascending=False).head(10)


# In[42]:


get_ipython().run_cell_magic('time', '', '# attempt to find missing brand names\n\ntrain_data.brand_name.fillna(value="missing", inplace=True)\ntest_data.brand_name.fillna(value="missing", inplace=True)\nunique_brand_names = set(train_data.brand_name.unique()).union(set(test_data.brand_name.unique()))\n\n# get to finding!\npremissing = len(train_data.loc[train_data[\'brand_name\'] == \'missing\'])\ndef brandfinder(line):\n   brand = line[0]\n   name = line[1]\n   if brand == \'missing\':\n       for brand_name in unique_brand_names:\n           if brand_name in name and len(brand_name)>2:\n               return brand_name\n   return brand')


# In[43]:


get_ipython().run_cell_magic('time', '', "train_data['brand_name'] = train_data[['brand_name','name']].apply(brandfinder, axis = 1)\ntest_data['brand_name'] = test_data[['brand_name','name']].apply(brandfinder, axis = 1)\nfound = premissing-len(train_data.loc[train_data['brand_name'] == 'missing'])\nprint(found)")


# In[44]:


#Going through the values of brand name, the filled values seem to be sensible
train_data.head()


# In[45]:


top_10_categories = train_data.category_name.value_counts()[0:10]


# In[46]:


top_10_categories


# In[47]:


# def create_churn_trace(col, visible=False):
#     return go.Histogram(
#         x=churn[col],
#         name='churn',
#         marker = dict(color = colors[1]),
#         visible=visible
#     )

# def create_no_churn_trace(col, visible=False):
#     return go.Histogram(
#         x=no_churn[col],
#         name='no churn',
#         marker = dict(color = colors[0]),
#         visible = visible,
#     )

# features_not_for_hist = ["state", "phone_number", "churn"]
# features_for_hist = [x for x in pre_df.columns if x not in features_not_for_hist]
# active_idx = 0
# traces_churn = [(create_churn_trace(col) if i != active_idx else create_churn_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]
# traces_no_churn = [(create_no_churn_trace(col) if i != active_idx else create_no_churn_trace(col, visible=True)) for i, col in enumerate(features_for_hist)]
# data = traces_churn + traces_no_churn

# n_features = len(features_for_hist)
# steps = []
# for i in range(n_features):
#     step = dict(
#         method = 'restyle',  
#         args = ['visible', [False] * len(data)],
#         label = features_for_hist[i],
#     )
#     step['args'][1][i] = True # Toggle i'th trace to "visible"
#     step['args'][1][i + n_features] = True # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active = active_idx,
#     currentvalue = dict(
#         prefix = "Feature: ", 
#         xanchor= 'center',
#     ),
#     pad = {"t": 50},
#     steps = steps,
# )]

# layout = dict(
#     sliders=sliders,
#     yaxis=dict(
#         title='#samples',
#         automargin=True,
#     ),
# )

# fig = dict(data=data, layout=layout)

# iplot(fig, filename='histogram_slider')
# 0


# In[48]:


def drawBarGraph(data_x,data_y, hover_text, graph_title,x_title,y_title,colors):
    trace1 = go.Bar(y=data_y, x=data_x, text=hover_text,marker=dict(
                color = data_y,colorscale=colors,showscale=True,
                reversescale = False
                ))
    layout = dict(title= graph_title,
                yaxis = y_title,
                xaxis = x_title)
    fig=dict(data=[trace1], layout=layout)
    py.iplot(fig)
    
hover_text = [("%.2f"%(v*100/len(train_data)))+"%"for v in (top_10_categories.values)]
data_x = top_10_categories.index
data_y = top_10_categories.values
title = 'Number of items by category'
x_title = dict(title='Category')
y_title = dict(title='Count')
drawBarGraph(data_x, data_y,hover_text, title,x_title,y_title,colors = 'Picnic')


# In[49]:


# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train_data['first_category'], train_data['second_category'], train_data['third_category'] = zip(*train_data['category_name'].apply(lambda x: split_cat(x)))
train_data.head()


# In[50]:


first_category_value_counts = train_data.first_category.value_counts()[0:15]
hover_text = [("%.2f"%(v*100/len(train_data)))+"%"for v in (first_category_value_counts.values)]
data_x = first_category_value_counts.index
data_y = first_category_value_counts.values
title = 'Number of items by primary category'
x_title = dict(title='Primary Category')
y_title = dict(title='Count')
drawBarGraph(data_x, data_y,hover_text, title,x_title,y_title,colors = 'Viridis')


# In[51]:


women_products = train_data.loc[train_data.first_category=='Women',['brand_name','price']]
first_N_brands = 20
top_women_brands = women_products.brand_name.value_counts()[:first_N_brands] 
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, first_N_brands)]
i=0
data = [{
    'y': women_products.loc[women_products['brand_name']==brand_name, 'price'], 
    'type':'box',
    'name':brand_name,
    'marker':{'color': c[freq%first_N_brands]}
    } for brand_name, freq in top_women_brands.iteritems()]
    
layout = {'xaxis': {'title':'brand name','showgrid':True,'zeroline':False, 'tickangle':60,'showticklabels':False},
          'yaxis': {'title':"price",'zeroline':False,'gridcolor':'white'},
          'paper_bgcolor': 'rgb(233,233,233)',
          'plot_bgcolor': 'rgb(233,233,233)',
          }
py.iplot(data)


# In[52]:


second_category_value_counts = train_data.second_category.value_counts()[0:15]
hover_text = [("%.2f"%(v*100/len(train_data)))+"%"for v in (second_category_value_counts.values)]
data_x = second_category_value_counts.index
data_y = second_category_value_counts.values
title = 'Number of items by secondary category'
x_title = dict(title='Secondary Category')
y_title = dict(title='Count')
drawBarGraph(data_x, data_y,hover_text, title,x_title,y_title,colors = 'Viridis')


# In[53]:


third_category_value_counts = train_data.third_category.value_counts()[0:15]
hover_text = [("%.2f"%(v*100/len(train_data)))+"%"for v in (third_category_value_counts.values)]
data_x = third_category_value_counts.index
data_y = third_category_value_counts.values
title = 'Number of items by third category'
x_title = dict(title='Third Category')
y_title = dict(title='Count')
drawBarGraph(data_x, data_y,hover_text, title,x_title,y_title,colors = 'Viridis')


# In[54]:


print(train_data['first_category'].nunique())
print(train_data['second_category'].nunique())
print(train_data['third_category'].nunique())


# In[55]:


data = []
colors = ['red','pink','blue','green','orange']
i=0
for category,count in second_category_value_counts[0:5].iteritems():
    grouped_brand_data = train_data.loc[train_data['second_category']==category].groupby('brand_name')                               .price                               .agg({'mean':np.mean,'count':'count'})                               .sort_values(by = 'count', ascending=False)[0:20]    
    trace = { "x": grouped_brand_data.loc[:,'mean'], 
              "y": grouped_brand_data.index, 
              "marker": {"color": colors[i], "size": 12}, 
              "mode": "markers", 
              "name": category, 
              "type": "scatter", 
        }
    data.append(trace)
    i=i+1
                              
layout = go.Layout(
    title="Price distribution of top 20 brands in top 5 secondary categories ",
    xaxis=dict(
        title='Price',
        showgrid=True,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        titlefont=dict(
            color='rgb(204, 204, 204)'
        ),
        tickfont=dict(
            color='rgb(102, 102, 102)',
        ),
        showticklabels=True,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    yaxis = dict(title='brand names'),
    margin=dict(
        l=140,
        r=40,
        b=50,
        t=80
    ),
    legend=dict(
        font=dict(
            size=10,
        ),
        yanchor='bottom',
        xanchor='right',
    ),
    width=800,
    height=600,
    paper_bgcolor='rgb(254, 247, 234)',
    plot_bgcolor='rgb(254, 247, 234)',
    hovermode='closest',
)


fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
                                                                     


# In[56]:


#Before handling the brand_name lets preprocess it

def wordCount(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ")                  if not w in stop_words.ENGLISH_STOP_WORDS]
        return len(words), ' '.join(words)
    except: 
        return 0, ""


# In[57]:


train_data.name.isnull().sum()


# In[58]:


print("Proportion of unique product names is = ",train_data['name'].str.lower().str.strip().nunique()/train_data.shape[0])


# In[59]:


train_data['name'].value_counts().head(10)


# In[60]:


#Pipeline for text preprocessing
#This is very generic list of contractions and most of the words may not even appear in an item description
contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", 
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 
                "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 
                "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
                "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 
                "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
                "you're": "you are", "you've": "you have" }


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 contractions={},
                 stop_words={},
                 spellings={},
                 user_abbrevs={},
                 n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. contractions
            3. Punctuation removal
            4. Stop words removal - words like not are excluded from stop words
        """
       
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs
        self.contractions = contractions
        self.stop_words = stop_words
        self.spellings = spellings
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)   # split data for parallel processing
        pool = mp.Pool(cores)                           # create pools
        data = pd.concat(pool.map(self._preprocess_part, data_split))   # concatenate results
        pool.close()                                  
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        lowercase_text = self._lowercase(text)
        expanded_contractions = self._expand_contactions(lowercase_text)
        removed_punct = self._remove_punct(expanded_contractions)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return (removed_stop_words)
   
    def _lowercase(self, text):
        return text.lower()
    
    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, user_abbrevs=self.user_abbrevs, verbose=True))
        except:
            return text

    def _expand_contactions(self, doc):
        new_text = ""
        for t in doc.split():
            if t in contractions:
                new_text = new_text + " " + (contractions[t])
            else: 
                new_text = new_text + " " + t
        return new_text
 
    def _remove_punct(self, doc):
        return ' '.join([t for t in doc.split() if t not in string.punctuation])

    def _remove_stop_words(self, doc):
        return ' '.join([t for t in doc.split() if t not in self.stop_words])    


# In[61]:


from nltk.corpus import stopwords
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
from sklearn.feature_extraction import stop_words


refined_stop_words = stop_words.ENGLISH_STOP_WORDS - {"not", "none", "nothing", "nowhere", "never", "cannot",
                                "cant", "couldnt", "except", "hasnt", "neither", "no", 
                                 "nobody", "nor", "without" }

get_ipython().run_line_magic('time', '')
textPreprocessor = TextPreprocessor(n_jobs=-1, contractions=contractions,
                 stop_words=refined_stop_words)
    
train_data['name'] = textPreprocessor.transform(train_data['name'])


# In[62]:


get_ipython().run_cell_magic('time', '', "train_data['name_length'] = train_data['name'].apply(lambda x: len(x))")


# In[63]:



#Could their be corelation between the length of name and the price?
# so first let's get the average price of the products having same length names.

name_length_price_df = train_data.groupby('name_length').price.agg(['mean','median'])


# In[64]:


name_length_price_df.head()


# In[65]:


trace1 = go.Scatter(
    x = name_length_price_df.index,
    y = name_length_price_df['mean'],
    #mode = 'lines mean',
    name = 'price mean'
) 

trace2 = go.Scatter(
    x = name_length_price_df.index,
    y = name_length_price_df['median'],
    #mode = 'lines median',
    name = 'price median'
)
layout = dict(title= 'Average (Price) by Name Length',
              yaxis = dict(title='(Price)'),
              xaxis = dict(title='Name Length'))
fig=dict(data=[trace1, trace2], layout=layout)
py.iplot(fig)


# In[66]:


print("Proportion of unique product names is = ",train_data['item_description'].str.lower().str.strip().nunique()/train_data.shape[0])


# In[67]:


train_data['item_description'].value_counts().head(10)


# In[68]:


def handle_missing(text):
    if (text == "No description yet") or (text == np.NAN) or text=="" or text == None:
        return "missing"
    return text
train_data['item_description'].fillna("missing",inplace=True)
train_data['item_description'] = train_data['item_description'].apply(lambda x: handle_missing(x))
test_data['item_description'].fillna("missing",inplace=True)
test_data['item_description'] = test_data['item_description'].apply(lambda x: handle_missing(x))


# In[69]:


train_data.head()


# In[70]:


get_ipython().run_cell_magic('time', '', "train_data['item_description'] = textPreprocessor.transform(train_data['item_description'])\ntest_data['item_description'] = textPreprocessor.transform(test_data['item_description'])")


# In[71]:


get_ipython().run_cell_magic('time', '', "train_data['item_description_length'] = train_data['item_description'].apply(lambda x: len(x))\ntest_data['item_description_length'] = test_data['item_description'].apply(lambda x: len(x))")


# In[72]:


item_description_price_df = train_data.groupby('item_description_length').price.agg(['mean','median'])


# In[73]:


item_description_price_df.head()


# In[74]:


trace1 = go.Scatter(
    x = item_description_price_df.index,
    y = item_description_price_df['mean'],
    #mode = 'lines mean',
    name = 'price mean'
) 

trace2 = go.Scatter(
    x = item_description_price_df.index,
    y = item_description_price_df['median'],
    #mode = 'lines median',
    name = 'price median'
)
layout = dict(title= 'Average (Price) by Description Length',
              yaxis = dict(title='(Price)'),
              xaxis = dict(title='Name Length'))
fig=dict(data=[trace1, trace2], layout=layout)
py.iplot(fig)


# In[75]:


def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          max_words=50, max_font_size=40,
                          random_state=42
                         ).generate(str(tup))
    return wordcloud


# In[76]:


fig,axes = plt.subplots(1, 2, figsize=(30, 15))

ax = axes[0]
ax.imshow(generate_wordcloud(train_data['name']), interpolation="bilinear")

ax = axes[1]
ax.imshow(generate_wordcloud(train_data['item_description']), interpolation="bilinear")


# In[77]:


# # Using YAAAKE!
# import yake
# train_data['keywords'] = ""
# # assuming default parameters
# simple_kwextractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, features=None)
# train_data.loc[0:10000,'keywords'] = train_data.loc[0:10000,'item_description'].apply(lambda x:simple_kwextractor.extract_keywords(x))


# In[ ]:




