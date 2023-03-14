#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

pyLDAvis.enable_notebook()
np.random.seed(2018)
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_cell_magic('time', '', 'JIGSAW_PATH = "../input/"\ntrain = pd.read_csv(os.path.join(JIGSAW_PATH,\'train.csv\'), index_col=\'id\')\ntest = pd.read_csv(os.path.join(JIGSAW_PATH,\'test.csv\'), index_col=\'id\')')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


print("Train and test shape: {} {}".format(train.shape, test.shape))


# In[6]:


train.info()


# In[7]:


get_ipython().run_cell_magic('html', '', '<style>\n.container { width:1000px !important; }\n</style>')


# In[8]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
sns.distplot(train.target, kde=False, bins=40).set_title("Histogram Plot of target", fontsize=15)


# In[9]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
sns.kdeplot(train.target).set_title("Kernel Density Estimate(kde) Plot of target", fontsize=15)


# In[10]:


fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of toxicity subtype attributes in train data', fontsize=15)
sns.kdeplot(train['severe_toxicity'], ax=axarr[0][0])
sns.kdeplot(train['obscene'], ax=axarr[0][1])
sns.kdeplot(train['threat'], ax=axarr[1][0])
sns.kdeplot(train['insult'], ax=axarr[1][1])
sns.kdeplot(train['identity_attack'], ax=axarr[2][0])
sns.kdeplot(train['sexual_explicit'], ax=axarr[2][1])
sns.despine()


# In[11]:


train['target_binarized'] = train['target'].apply(lambda x : 'Toxic' if  x >= 0.5 else 'Non-Toxic')
fig, axarr = plt.subplots(1,1,figsize=(12, 4))
train['target_binarized'].value_counts().plot.bar(fontsize=10).set_title("Toxic vs Non-Toxic Comments Count", 
                                                                         fontsize=15)
sns.despine(bottom=True,  left=True)


# In[12]:


#train = train.drop(columns='target_binarized')


# In[13]:


f = (
    train.loc[:, ['target', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']]
        .applymap(lambda v: float(v))
        .dropna()
)


# In[14]:


f.head()


# In[15]:


f_corr=f.corr()


# In[16]:


fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.heatmap(f_corr, annot=True)


# In[17]:


fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of race and ethnicity features values in the train set', fontsize=15)
sns.kdeplot(train['asian'], ax=axarr[0][0], color='mediumvioletred')
sns.kdeplot(train['black'], ax=axarr[0][1], color='mediumvioletred')
sns.kdeplot(train['jewish'], ax=axarr[1][0], color='mediumvioletred')
sns.kdeplot(train['latino'], ax=axarr[1][1], color='mediumvioletred')
sns.kdeplot(train['other_race_or_ethnicity'], ax=axarr[2][0], color='mediumvioletred')
sns.kdeplot(train['white'], ax=axarr[2][1], color='mediumvioletred')
sns.despine()


# In[18]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "asian", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "black", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "jewish", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "latino", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_race_or_ethnicity", color='mediumvioletred')

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "white", color='mediumvioletred')


# In[19]:


fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
fig.suptitle('Distribution of toxicity target for every race/ethnicity sample that has been annotated with value of 1', 
             fontsize=14)
sns.violinplot(train[train['asian'] == np.max(train.asian)]['target'], ax=axarr[0]).set_title("asian")
sns.violinplot(train[train['black'] == np.max(train.black)]['target'], ax=axarr[1]).set_title("black")
sns.despine()
fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
sns.violinplot(train[train['jewish'] == np.max(train.jewish)]['target'], ax=axarr[0]).set_title("jewish")
sns.violinplot(train[train['latino'] == np.max(train.latino)]['target'], ax=axarr[1]).set_title("latino")
sns.despine()
fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
sns.violinplot(train[train['other_race_or_ethnicity'] == np.max(train.other_race_or_ethnicity)]['target'], ax=axarr[0]).set_title("other_race_or_ethnicity")
sns.violinplot(train[train['white'] == np.max(train.white)]['target'], ax=axarr[1]).set_title("white")
sns.despine()


# In[20]:


np.min(train.target)


# In[21]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of gender in the train set', fontsize=15)
sns.kdeplot(train['female'], ax=axarr[0][0], color='violet')
sns.kdeplot(train['male'], ax=axarr[0][1], color='violet')
sns.kdeplot(train['transgender'], ax=axarr[1][0], color='violet')
sns.kdeplot(train['other_gender'], ax=axarr[1][1], color='violet')
sns.despine()


# In[22]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "female", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "male", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "transgender", color="violet")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_gender", color="violet")


# In[23]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of sexual orientation in the train set', fontsize=15)
sns.kdeplot(train['bisexual'], ax=axarr[0][0], color='red')
sns.kdeplot(train['heterosexual'], ax=axarr[0][1], color='red')
sns.kdeplot(train['homosexual_gay_or_lesbian'], ax=axarr[1][0], color='red')
sns.kdeplot(train['other_sexual_orientation'], ax=axarr[1][1], color='red')
sns.despine()


# In[24]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "bisexual", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "heterosexual", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "homosexual_gay_or_lesbian", color="red")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_sexual_orientation", color="red")


# In[25]:


fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
fig.suptitle('Distribution of disability in the train set', fontsize=15)
sns.kdeplot(train['intellectual_or_learning_disability'], ax=axarr[0][0], color='green')
sns.kdeplot(train['physical_disability'], ax=axarr[0][1], color='green')
sns.kdeplot(train['psychiatric_or_mental_illness'], ax=axarr[1][0], color='green')
sns.kdeplot(train['other_disability'], ax=axarr[1][1], color='green')
sns.despine()


# In[26]:


g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "intellectual_or_learning_disability", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "physical_disability", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "psychiatric_or_mental_illness", color="green")

g = sns.FacetGrid(train.dropna(), col="target_binarized", size=4, aspect=1.5)
g.map(sns.violinplot, "other_disability", color="green")


# In[27]:


train['created_date_time'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')
#datetime64[Y] ==> Month and Date is always 1
#datetime64[M] ==> Date is always 1
#datetime64[D] ==> Year, Month and Data not neglected


# In[28]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
train['created_date_time'].value_counts().sort_values().plot.line(fontsize=10).set_title("Number of comments vs Year-Month", 
                                                                                           fontsize=15)
sns.despine(bottom=True,  left=True)


# In[29]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
train['created_date_time'].value_counts().resample('Y').sum().plot.line(fontsize=10).set_title("Number of comments vs Year", 
                                                                                           fontsize=15)
sns.despine(bottom=True,  left=True)


# In[30]:


fig, axarr = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
lag_plot(train['target']).set_title("Lag Plot", fontsize=15)


# In[31]:


train['comment_text_length'] = train['comment_text'].apply(lambda x : len(x))
fig, axarr = plt.subplots(1,1,figsize=(12, 6))
sns.kdeplot(train['comment_text_length']).set_title("Distribution of comment_text_length", fontsize=15)


# In[32]:


g = sns.FacetGrid(train, col="target_binarized", size=4, aspect=1.5)
g.map(sns.kdeplot, "comment_text_length", color='red')


# In[33]:


(
    ggplot(train.sample(100000))
        + geom_point()
        + aes(color='comment_text_length')
        + aes('comment_text_length', 'target')
)


# In[34]:


sns.jointplot(x='comment_text_length', y='target', data=train, kind='hex', gridsize=20, size=8)


# In[35]:


stopwords = set(STOPWORDS)

def plot_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[36]:


plot_wordcloud(train['comment_text'], title = 'Frequently used words in train data')


# In[37]:


plot_wordcloud(train[train['target'] == np.max(train.target)]['comment_text'], title = 'Frequent Words : Toxicity target value = 1 #Toxic')


# In[38]:


plot_wordcloud(train[train['target'] == np.min(train.target)]['comment_text'], title = 'Frequent Words : Toxicity target value = 0 #Non-Toxic')


# In[39]:


plot_wordcloud(train[(train['female'] >0)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and Female')


# In[40]:


plot_wordcloud(train[(train['male'] >0)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and Male')


# In[41]:


plot_wordcloud(train[(train['insult'] >0.8)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and insult > 0.8')


# In[42]:


plot_wordcloud(train[(train['threat'] >0.8)&(train['target']>0.8)]['comment_text'], title = 'Frequent Words : toxicity target > 0.8 and threat > 0.8')

