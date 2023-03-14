#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import numpy as np 
import pandas as pd 
import os
import json
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from PIL import Image
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
from catboost import CatBoostClassifier
from tqdm import tqdm_notebook
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import random
import warnings
warnings.filterwarnings("ignore")
from functools import partial
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
import scipy as sp
from math import sqrt
from collections import Counter
from sklearn.metrics import confusion_matrix as sk_cmatrix

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import RandomForestClassifier
import langdetect
import eli5
from IPython.display import display 

from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# In[2]:


breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
colors = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
states = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sub = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])


# In[3]:


train.drop('Description', axis=1).head(10)


# In[4]:


test.drop('Description', axis=1).head(10)


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


print(colors.shape)
colors


# In[8]:


print(breeds.shape)
breeds.head()


# In[9]:


print(states.shape)
states.head()


# In[10]:


def plot_target(x, data, hue, title, alldata = False): 
# Plot count and rate in dataset
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(22, 8))

    # Plot count number of target in dataset
    g = sns.countplot(x=x, data=data, hue = hue, ax=axes[0]);
    g.set_title(f'Count number of {title}');
    g.set_xlabel(f'{title} Type')
    ax_g=g.axes
    for p in ax_g.patches:
        ax_g.annotate(f'{p.get_height()}',
                    xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    color = 'black',
                    ha='center', va='bottom')
    
    # Plot rate of target in dataset    
    k = sns.countplot(x=x, data=data,hue = hue, ax=axes[1]);
    k.set_title(f'Rate of {title}');
    k.set_xlabel(f'{title} Type')
    k.set_ylabel('Rate')
    ax_k=k.axes
    
    if alldata == True:
        # Annotate train set and test set seperately
        i = 0
        for p in ax_k.patches:
            y_value = p.get_height()
            if i%2 == 0:
                y_value = y_value/train.shape[0]*100
            else:
                y_value = y_value/test.shape[0]*100   
            ax_k.annotate(f"{y_value:.2f}%", 
                        xy= (p.get_x() + p.get_width() / 2., p.get_height()),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        color = 'black',
                        ha='center', va='bottom')
            i += 1
    
    if alldata == False:
        for p in ax_k.patches:
            y_value = p.get_height() * 100 / data.shape[0]
            ax_k.annotate(f"{y_value:.2f}%", 
                        xy= (p.get_x() + p.get_width() / 2., p.get_height()),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        color = 'black',
                        ha='center', va='bottom')


# In[11]:


plot_target(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'], hue = None,title='Adoption Speed')


# In[12]:


# Change data type to dog, cat
all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')


# In[13]:


plot_target(x='dataset_type', data=all_data, hue = 'Type',title='Cat and Dog', alldata= True)


# In[14]:


plot_target(x='Type', data=all_data.loc[all_data['dataset_type'] == 'train'],hue = "AdoptionSpeed",title='Cat and Dog')


# In[15]:


# Visualize dog and cat in all train data
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(22, 8))

# Plot number of cat and dog in traing set
g = sns.countplot(x='Type', data=all_data.loc[all_data['dataset_type'] == 'train'], hue='AdoptionSpeed', ax=axes[0]);
g.set_title('Number of cats and dogs adoption speed in train data');
ax_g=g.axes
for p in ax_g.patches:
    ax_g.annotate(f'{p.get_height()}',
                    xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    color = 'black',
                    ha='center', va='bottom')

# Plot rate of cat and dog in training set
k = sns.countplot(x='Type', data=all_data.loc[all_data['dataset_type'] == 'train'], hue='AdoptionSpeed', ax=axes[1]);
k.set_title('Rate [%] cats and dogs adoption speed in train data')
k.set_ylabel('rate')
ax_k=k.axes
i=0
for p in ax_k.patches:
    y_value = p.get_height()
    if i%2 == 0:
        y_value = y_value/train.loc[train['Type']==2].shape[0]*100
    else:
        y_value = y_value/train.loc[train['Type']==1].shape[0]*100 
    ax_k.annotate("{:.2f}%".format(y_value), 
                xy= (p.get_x() + p.get_width() / 2., p.get_height()),
                xytext=(0, 3),  
                textcoords="offset points",
                color = 'black',
                ha='center', va='bottom')
    i +=1


# In[16]:


fig, ax = plt.subplots(figsize = (22, 8))
plt.subplot(1, 2, 1)
text_cat = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top cat names')
plt.axis("off")

plt.subplot(1, 2, 2)
text_dog = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_dog)
plt.imshow(wordcloud)
plt.title('Top dog names')
plt.axis("off")

plt.show()


# In[17]:


train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')
all_data['Name'] = all_data['Name'].fillna('Unnamed')

train['No_name'] = 0
train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
test['No_name'] = 0
test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
all_data['No_name'] = 0
all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1

print(f"Rate of unnamed pets in train data: {train['No_name'].sum() * 100 / train['No_name'].shape[0]:.4f}%.")
print(f"Rate of unnamed pets in test data: {test['No_name'].sum() * 100 / test['No_name'].shape[0]:.4f}%.")


# In[18]:


pd.crosstab(train['No_name'], train['AdoptionSpeed'], normalize='index')


# In[19]:


fig, ax = plt.subplots(figsize = (22, 8))
plt.subplot(1, 2, 1)
plt.title('Distribution of pets age');
train['Age'].plot('hist', label='train');
test['Age'].plot('hist', label='test');
plt.legend()
plt.xlabel('Days');

plt.subplot(1, 2, 2)
plt.title('Distribution of pets age (log)');
np.log1p(train['Age']).plot('hist', label='train');
np.log1p(test['Age']).plot('hist', label='test');
plt.legend()
plt.xlabel('Log(days)');


# In[20]:


plt.figure(figsize=(22, 8));
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and age');


# In[21]:


data = []
for a in range(5):
    df = train.loc[train['AdoptionSpeed'] == a]

    data.append(go.Scatter(
        x = df['Age'].value_counts().sort_index().index,
        y = df['Age'].value_counts().sort_index().values,
        name = str(a)
    ))
    
layout = go.Layout(dict(title = "AdoptionSpeed trends by Age",
                  xaxis = dict(title = 'Age (days)'),
                  yaxis = dict(title = 'Counts'),
                  )
                  )
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[22]:


train['Pure_breed'] = 0
train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
test['Pure_breed'] = 0
test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
all_data['Pure_breed'] = 0
all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1

print(f"Rate of pure breed pets in train data: {train['Pure_breed'].sum() * 100 / train['Pure_breed'].shape[0]:.4f}%.")
print(f"Rate of pure breed pets in test data: {test['Pure_breed'].sum() * 100 / test['Pure_breed'].shape[0]:.4f}%.")


# In[23]:


plot_target(x='Pure_breed', data=all_data.loc[all_data['dataset_type'] == 'train'],hue = "AdoptionSpeed",title='Pure_breed vs AdoptionSpeed')


# In[24]:


plot_target(x='dataset_type', data=all_data,hue = "Pure_breed",title='Pure_breed', alldata=True)


# In[25]:


plot_target(x='Pure_breed', data=train.loc[train['Type'] == 1],hue = "AdoptionSpeed",title='pure_breed for dog')


# In[26]:


plot_target(x='Pure_breed', data=train.loc[train['Type'] == 2],hue = "AdoptionSpeed",title='pure_breed for cat')


# In[27]:


breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}


# In[28]:


train['Breed1_name'] = train['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
train['Breed2_name'] = train['Breed2'].apply(lambda x: '_'.join(breeds_dict[x]) if x in breeds_dict else '-')

test['Breed1_name'] = test['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
test['Breed2_name'] = test['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

all_data['Breed1_name'] = all_data['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
all_data['Breed2_name'] = all_data['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')


# In[29]:


fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2')
plt.axis("off")
plt.show()


# In[30]:


(all_data['Breed1_name'] + '__' + all_data['Breed2_name']).value_counts().head(15)


# In[31]:


def plot_overview(x,data,hue,title):
    plot_target(x=x, data=data, hue = hue, title = title)
    plot_target(x='dataset_type', data=all_data ,hue = x,title='dataset_type', alldata= True)


# In[32]:


plot_overview(x='Gender', data=train,hue = "AdoptionSpeed",title='Gender')


# In[33]:


sns.factorplot('Type', col='Gender', data=all_data, kind='count', hue='dataset_type');
plt.subplots_adjust(top=.8)
plt.suptitle('Count of cats and dogs in train and test set by gender');


# In[34]:


colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')


# In[35]:


def make_factor_plot(df, x, col, title, hue=None, ann=True, col_wrap=4):
    """
    Plotting countplot.
    Making annotations is a bit more complicated, because we need to iterate over axes.
    """
    if hue:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap, hue=hue);
    else:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap);
    plt.subplots_adjust(top=0.9);
    plt.suptitle(title);
    ax = g.axes
    if ann:
        for a in ax:
            for p in a.patches:
                a.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                     textcoords='offset points') 


# In[36]:


sns.factorplot('dataset_type', col='Type', data=all_data, kind="count", hue='Color1_name',palette=['Black', 'Brown', '#FFFDD0', 'Gray', 'Gold', 'White', 'Yellow'],size=6);
plt.subplots_adjust(top=0.9)
plt.suptitle('Counts of pets in datasets by main color');


# In[37]:


make_factor_plot(df=train, x='Color1_name', col='AdoptionSpeed', title='Counts of pets by main color and Adoption Speed')


# In[38]:


train['full_color'] = (train['Color1_name'] + '__' + train['Color2_name'] + '__' + train['Color3_name']).str.replace('__', '')
test['full_color'] = (test['Color1_name'] + '__' + test['Color2_name'] + '__' + test['Color3_name']).str.replace('__', '')
all_data['full_color'] = (all_data['Color1_name'] + '__' + all_data['Color2_name'] + '__' + all_data['Color3_name']).str.replace('__', '')

make_factor_plot(df=train.loc[train['full_color'].isin(list(train['full_color'].value_counts().index)[:12])], x='full_color', col='AdoptionSpeed', title='Counts of pets by color and Adoption Speed')


# In[39]:


gender_dict = {1: 'Male', 2: 'Female', 3: 'Mixed'}
for i in all_data['Type'].unique():
    for j in all_data['Gender'].unique():
        df = all_data.loc[(all_data['Type'] == i) & (all_data['Gender'] == j)]
        top_colors = list(df['full_color'].value_counts().index)[:5]
        j = gender_dict[j]
        print(f"Most popular colors of {j} {i}s: {' '.join(top_colors)}")


# In[40]:


plot_overview(x='MaturitySize', data=train,hue = "AdoptionSpeed",title='MaturitySize')


# In[41]:


make_factor_plot(df=all_data, x='MaturitySize', col='Type', title='Count of cats and dogs in train and test set by MaturitySize', hue='dataset_type', ann=True)


# In[42]:


images = [i.split('-')[0] for i in os.listdir('../input/petfinder-adoption-prediction/train_images/')]
size_dict = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large'}
for t in all_data['Type'].unique():
    for m in all_data['MaturitySize'].unique():
        df = all_data.loc[(all_data['Type'] == t) & (all_data['MaturitySize'] == m)]
        top_breeds = list(df['Breed1_name'].value_counts().index)[:5]
        m = size_dict[m]
        print(f"Most common Breeds of {m} {t}s:")
        
        fig = plt.figure(figsize=(25, 4))
        
        for i, breed in enumerate(top_breeds):
            # excluding pets without pictures
            b_df = df.loc[(df['Breed1_name'] == breed) & (df['PetID'].isin(images)), 'PetID']
            if len(b_df) > 1:
                pet_id = b_df.values[1]
            else:
                pet_id = b_df.values[0]
            ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

            im = Image.open("../input/petfinder-adoption-prediction/train_images/" + pet_id + '-1.jpg')
            plt.imshow(im)
            ax.set_title(f'Breed: {breed}')
        plt.show();


# In[43]:


plot_overview(x='FurLength', data=train,hue = "AdoptionSpeed",title='FurLength')


# In[44]:


fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with medium fur')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2 with medium fur')
plt.axis("off")
plt.show()


# In[45]:


c = 0
strange_pets = []
for i, row in all_data[all_data['Breed1_name'].str.contains('air')].iterrows():
    if 'Short' in row['Breed1_name'] and row['FurLength'] == 1:
        pass
    elif 'Medium' in row['Breed1_name'] and row['FurLength'] == 2:
        pass
    elif 'Long' in row['Breed1_name'] and row['FurLength'] == 3:
        pass
    else:
        c += 1
        strange_pets.append((row['PetID'], row['Breed1_name'], row['FurLength']))
        
print(f"There are {c} pets whose breed and fur length don't match")


# In[46]:


strange_pets = [p for p in strange_pets if p[0] in images]
fig = plt.figure(figsize=(25, 12))
fur_dict = {1: 'Short', 2: 'Medium', 3: 'long'}
for i, s in enumerate(random.sample(strange_pets, 12)):
    ax = fig.add_subplot(3, 4, i+1, xticks=[], yticks=[])

    im = Image.open("../input/petfinder-adoption-prediction/train_images/" + s[0] + '-1.jpg')
    plt.imshow(im)
    ax.set_title(f'Breed: {s[1]} \n Fur length: {fur_dict[s[2]]}')
plt.show();


# In[47]:


plot_target(data=train, x='Vaccinated', title='Vaccinated', hue='AdoptionSpeed')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Vaccinated');

plot_target(data=train, x='Sterilized', title='Sterilized', hue='AdoptionSpeed')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Sterilized');

plot_target(data=train, x='Health', title='Health', hue='AdoptionSpeed')
plt.xticks([0, 1, 2], ['Healthy', 'Minor Injury', 'Serious Injury']);
plt.title('AdoptionSpeed and Health');


# In[48]:


train['health'] = train['Vaccinated'].astype(str) + '_' + train['Dewormed'].astype(str) + '_' + train['Sterilized'].astype(str) + '_' + train['Health'].astype(str)
test['health'] = test['Vaccinated'].astype(str) + '_' + test['Dewormed'].astype(str) + '_' + test['Sterilized'].astype(str) + '_' + test['Health'].astype(str)


make_factor_plot(df=train.loc[train['health'].isin(list(train.health.value_counts().index[:5]))], x='health', col='AdoptionSpeed', title='Counts of pets by main health conditions and Adoption Speed')


# In[49]:


plt.figure(figsize=(20, 16))
plt.subplot(3, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="Age", data=train);
plt.title('Age distribution by Age');
plt.subplot(3, 2, 3)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Vaccinated", data=train);
plt.title('Age distribution by Age and Vaccinated');
plt.subplot(3, 2, 4)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Dewormed", data=train);
plt.title('Age distribution by Age and Dewormed');
plt.subplot(3, 2, 5)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Sterilized", data=train);
plt.title('Age distribution by Age and Sterilized');
plt.subplot(3, 2, 6)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Health", data=train);
plt.title('Age distribution by Age and Health');


# In[50]:


train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
plot_overview(x='Free', title='Number of pets by Free in train and test data', hue = 'AdoptionSpeed', data = train)
plot_target(x='Free', title = 'Dog', data= train.loc[train['Type']==1], hue = 'AdoptionSpeed')
plot_target(x='Free', title = 'Cat', data= train.loc[train['Type']==2], hue = 'AdoptionSpeed')


# In[51]:


plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(train.loc[train['Fee'] < 400, 'Fee']);
plt.title('Distribution of fees lower than 400');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="Fee", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and Fee');


# In[52]:


plt.figure(figsize=(16, 10));
sns.scatterplot(x="Fee", y="Quantity", hue="Type",data=all_data);
plt.title('Quantity of pets and Fee');


# In[53]:


states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['State_name'] = train['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
test['State_name'] = test['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
all_data['State_name'] = all_data['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')


# In[54]:


all_data['State_name'].value_counts(normalize=True).head()


# In[55]:


make_factor_plot(df=train.loc[train['State_name'].isin(list(train.State_name.value_counts().index[:3]))], x='State_name', col='AdoptionSpeed', title='Counts of pets by states and Adoption Speed')


# In[56]:


all_data['RescuerID'].value_counts().head()


# In[57]:


make_factor_plot(df=train.loc[train['RescuerID'].isin(list(train.RescuerID.value_counts().index[:5]))], x='RescuerID', col='AdoptionSpeed', title='Counts of pets by rescuers and Adoption Speed', col_wrap=5)


# In[58]:


train['VideoAmt'].value_counts()


# In[59]:


print(F'Maximum amount of photos in {train["PhotoAmt"].max()}')
train['PhotoAmt'].value_counts().head()


# In[60]:


make_factor_plot(df=train.loc[train['PhotoAmt'].isin(list(train.PhotoAmt.value_counts().index[:5]))], x='PhotoAmt', col='AdoptionSpeed', title='Counts of pets by PhotoAmt and Adoption Speed', col_wrap=5)


# In[61]:


plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(train['PhotoAmt']);
plt.title('Distribution of PhotoAmt');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="PhotoAmt", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and PhotoAmt');


# In[62]:


fig, ax = plt.subplots(figsize = (12, 8))
text_cat = ' '.join(all_data['Description'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top words in description');
plt.axis("off");


# In[63]:


tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

vectorizer.fit(all_data['Description'].fillna('').values)
X_train = vectorizer.transform(train['Description'].fillna(''))

rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, train['AdoptionSpeed'])


# In[64]:


for i in range(5):
    print(f'Example of Adoption speed {i}')
    text = train.loc[train['AdoptionSpeed'] == i, 'Description'].values[0]
    print(text)
    display(eli5.show_prediction(rf, doc=text, vec=vectorizer, top=10))


# In[65]:


train['Description'] = train['Description'].fillna('')
test['Description'] = test['Description'].fillna('')
all_data['Description'] = all_data['Description'].fillna('')

train['desc_length'] = train['Description'].apply(lambda x: len(x))
train['desc_words'] = train['Description'].apply(lambda x: len(x.split()))

test['desc_length'] = test['Description'].apply(lambda x: len(x))
test['desc_words'] = test['Description'].apply(lambda x: len(x.split()))

all_data['desc_length'] = all_data['Description'].apply(lambda x: len(x))
all_data['desc_words'] = all_data['Description'].apply(lambda x: len(x.split()))

train['averate_word_length'] = train['desc_length'] / train['desc_words']
test['averate_word_length'] = test['desc_length'] / test['desc_words']
all_data['averate_word_length'] = all_data['desc_length'] / all_data['desc_words']


# In[66]:


plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="desc_length", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and description length');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="desc_words", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and count of words in description');


# In[67]:


sentiment_dict = {}
for filename in os.listdir('../input/petfinder-adoption-prediction/train_sentiment/'):
    with open('../input/petfinder-adoption-prediction/train_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

for filename in os.listdir('../input/petfinder-adoption-prediction/test_sentiment/'):
    with open('../input/petfinder-adoption-prediction/test_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']


# In[68]:


train['lang'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

test['lang'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

all_data['lang'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
all_data['magnitude'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)


# In[69]:


plot_overview(x='lang', title='lang', hue = 'AdoptionSpeed', data = train)
plot_target(x='lang', title = 'Dog', data= train.loc[train['Type']==1], hue = 'AdoptionSpeed')
plot_target(x='lang', title = 'Cat', data= train.loc[train['Type']==2], hue = 'AdoptionSpeed')


# In[70]:


plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="score", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and score');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="magnitude", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and magnitude of sentiment');


# In[71]:


cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'health', 'Free', 'score',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'desc_length', 'desc_words', 'averate_word_length', 'magnitude']
train = train[[col for col in cols_to_use if col in train.columns]]
test = test[[col for col in cols_to_use if col in test.columns]]


# In[72]:


cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'RescuerID',
       'No_name', 'Pure_breed', 'health', 'Free']
cat_cols


# In[73]:


more_cols = []
for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 != col2 and col1 not in ['RescuerID', 'State'] and col2 not in ['RescuerID', 'State']:
            train[col1 + '_' + col2] = train[col1].astype(str) + '_' + train[col2].astype(str)
            test[col1 + '_' + col2] = test[col1].astype(str) + '_' + test[col2].astype(str)
            more_cols.append(col1 + '_' + col2)
            
cat_cols = cat_cols + more_cols
cat_cols


# In[74]:


get_ipython().run_cell_magic('time', '', 'indexer = {}\nfor col in cat_cols:\n    # print(col)\n    _, indexer[col] = pd.factorize(train[col].astype(str))\n    \nfor col in tqdm_notebook(cat_cols):\n    # print(col)\n    train[col] = indexer[col].get_indexer(train[col].astype(str))\n    test[col] = indexer[col].get_indexer(test[col].astype(str))')


# In[75]:


y = train['AdoptionSpeed']
train = train.drop(['AdoptionSpeed'], axis=1)


# In[76]:


test.shape


# In[77]:


n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)


# In[78]:


def train_model(X=train, X_test=test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, averaging='usual', make_oof=False):
    result_dict = {}
    # Prepare data
    if make_oof:
        oof = np.zeros((len(X), 5))
    prediction = np.zeros((len(X_test), 5))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # LightGBM model
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature = cat_cols)
            
            model = lgb.train(params,
                    train_data,
                    num_boost_round=20000,
                    valid_sets = [train_data, valid_data],
                    verbose_eval=500,
                    early_stopping_rounds = 200)

            del train_data, valid_data
            
            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
            del X_valid
            gc.collect()
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
             # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        # XGBoost model
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, 
                              num_boost_round=20000, 
                              evals=watchlist, 
                              early_stopping_rounds=200, 
                              verbose_eval=500, 
                              params=params)
            
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
            
            # feature importance
            fold_importance = pd.DataFrame(list(model.get_fscore().items()),columns=['feature','importance'])
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        # CatBoost model
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000,  loss_function='MultiClass',early_stopping_rounds = 200, **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test)
        
             # feature importance
            fold_importance = model.get_feature_importance(prettified=True).rename(columns={'Feature Id': 'feature', 'Importances': 'importance'})
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        if make_oof:
            oof[valid_index] = y_pred_valid
            
        scores.append(kappa(y_valid, y_pred_valid.argmax(1)))
        print('Fold kappa:', kappa(y_valid, y_pred_valid.argmax(1)))
        print('')
        
        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values
        
       
    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if plot_feature_importance: 
                 
        feature_importance["importance"] /= n_fold
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
             by="importance", ascending=False)[:50].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        
        if model_type == 'xgb':
            plt.title('XGB Features (avg over folds)');
        
        if model_type == 'lgb':
            plt.title('LGB Features (avg over folds)');
        
        if model_type == 'cat':
            plt.title('CAT Features (avg over folds)');
            
        result_dict['feature_importance'] = feature_importance
            
    result_dict['prediction'] = prediction
    if make_oof:
        result_dict['oof'] = oof
    
    return result_dict, scores , best_features


# In[79]:


start = time.time()
lgb_params = {'num_leaves': 512,
        #  'min_data_in_leaf': 60,
         'objective': 'multiclass',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 3,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
        #  "lambda_l1": 0.1,
         # "lambda_l2": 0.1,
         "random_state": 42,          
         "verbosity": -1,
         "num_class": 5}
result_dict_lgb, scores_lgb, best_features_lgb = train_model(X=train, X_test=test, y=y, params=lgb_params, model_type='lgb', plot_feature_importance=True, make_oof=True)
end = time.time()
time_spend_lgb = end - start
print(f'Time spend: {time_spend_lgb}')


# In[80]:


best_features_lgb_top = best_features_lgb.loc[best_features_lgb['fold'] == 5].sort_values(by=['importance'], ascending=False)[:5].drop(['fold','importance'], axis=1).reset_index(drop=True)
best_features_lgb_top['model'] = 'LightGBM'
best_features_lgb_top


# In[81]:


final_score_lgb = pd.DataFrame([[np.mean(scores_lgb),np.std(scores_lgb),'LightGBM',time_spend_lgb]], columns=['mean','std','model','time_spend'])
final_score_lgb


# In[82]:


start = time.time()
xgb_params = {'eta': 0.01, 'max_depth': 9, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'multi:softprob', 'eval_metric': 'merror', 'silent': True, 'nthread': 4, 'num_class': 5}
result_dict_xgb, scores_xgb, best_features_xgb= train_model(X=train, X_test=test, y=y,params=xgb_params, model_type='xgb',plot_feature_importance=True, make_oof=True)
end = time.time()
time_spend_xgb = end - start
print(f'Time spend: {time_spend_xgb}')


# In[83]:


best_features_xgb_top = best_features_xgb.loc[best_features_xgb['fold'] == 5].sort_values(by=['importance'], ascending=False)[:5].drop(['fold','importance'], axis=1).reset_index(drop=True)
best_features_xgb_top['model'] = 'XGBoost'
best_features_xgb_top


# In[84]:


final_score_xgb = pd.DataFrame([[np.mean(scores_xgb),np.std(scores_xgb),'XGBoost',time_spend_xgb]], columns=['mean','std','model','time_spend'])
final_score_xgb


# In[85]:


start = time.time()
cat_params = {'learning_rate':0.03}
result_dict_cat,scores_cat, best_features_cat = train_model(X=train, X_test=test, y=y, model_type='cat',params=cat_params,plot_feature_importance=True, make_oof=True)
end = time.time()
time_spend_cat = end - start
print(f'Time spend: {time_spend_cat}')


# In[86]:


best_features_cat_top = best_features_cat.loc[best_features_cat['fold'] == 5].sort_values(by=['importance'], ascending=False)[:5].drop(['fold','importance'], axis=1).reset_index(drop=True)
best_features_cat_top['model'] = 'CatBoost'
best_features_cat_top


# In[87]:


final_score_cat = pd.DataFrame([[np.mean(scores_cat),np.std(scores_cat),'CatBoost',time_spend_cat]], columns=['mean','std','model','time_spend'])
final_score_cat


# In[88]:


feature_list = [best_features_cat_top, best_features_xgb_top, best_features_lgb_top] 
feature_impotant = pd.concat(feature_list)
feature_impotant


# In[89]:


score_list = [final_score_cat, final_score_lgb, final_score_xgb] 
final_score = pd.concat(score_list).sort_values(by='mean').reset_index(drop=True)
final_score


# In[90]:


prediction_cat = result_dict_cat['prediction'].argmax(1)
submission_cat = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_lgb_cat]})
submission_cat.head()


# In[91]:


prediction_lgb_cat = (result_dict_lgb['prediction'] + result_dict_cat['prediction']).argmax(1)
submission_lgb_cat = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_lgb_cat]})
submission_lgb_cat.head()


# In[92]:


prediction_all = (result_dict_lgb['prediction'] + result_dict_xgb['prediction'] + result_dict_cat['prediction']).argmax(1)
submission_all = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction_all]})
submission_all.head()


# In[93]:


submission_lgb_cat.to_csv('submission.csv', index=False)

