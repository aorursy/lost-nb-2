#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import gc

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
from datetime import datetime, timedelta
import time
import fancyimpute
from itertools import chain

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[2]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Data is loaded')


# In[3]:


(market_train_df, news_train_df) = env.get_training_data()


# In[4]:


print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features ')


# In[5]:


start = datetime(2012, 1, 1, 0, 0, 0).date()
market_train_df = market_train_df.loc[market_train_df['time'].dt.date >= start].reset_index(drop=True)
news_train_df = news_train_df.loc[news_train_df['time'].dt.date >= start].reset_index(drop=True)
del start

#collect residual garbage
gc.collect()


# In[6]:



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


# In[7]:


reduce_mem_usage(market_train_df)
reduce_mem_usage(news_train_df)


# In[8]:


market_train_df.head()


# In[9]:


market_train_df.dtypes


# In[10]:


market_train_df.isna().sum()


# In[11]:


market_train_df.nunique()


# In[12]:


volumeByAssets = market_train_df.groupby(market_train_df['assetCode'])['volume'].sum()
highestVolumes = volumeByAssets.sort_values(ascending=False)[0:10]

trace1 = go.Pie(
    labels = highestVolumes.index,
    values = highestVolumes.values
)

layout = dict(title = "Highest trading volumes")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[13]:


market_train_df['assetName'].describe()


# In[14]:


print("There are {:,} records with assetName = `Unknown` in the training set".format(market_train_df[market_train_df['assetName'] == 'Unknown'].size))


# In[15]:


assetNameGB = market_train_df[market_train_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))


# In[16]:


unknownAssets.head()


# In[17]:


data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 5):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 5 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[18]:


data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['close'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of closing prices by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),
    annotations=[
        dict(
            x='2008-09-01 22:00:00+0000',
            y=82,
            xref='x',
            yref='y',
            text='Collapse of Lehman Brothers',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2011-08-01 22:00:00+0000',
            y=85,
            xref='x',
            yref='y',
            text='Black Monday',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2014-10-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Another crisis',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=-20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2016-01-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Oil prices crash',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        )
    ])
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[19]:


market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()


# In[20]:


print(f"Average standard deviation of price change within a day in {grouped['price_diff']['std'].mean():.4f}.")


# In[21]:


g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[22]:


market_train_df.sort_values('price_diff')[:10]


# In[23]:


market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])


# In[24]:


print(f"In {(market_train_df['close_to_open'] >= 1.2).sum()} lines price increased by 20% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.80).sum()} lines price decreased by 20% or more.")


# In[25]:


print(f"In {(market_train_df['close_to_open'] >= 2).sum()} lines price increased by 100% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.5).sum()} lines price decreased by 100% or more.")


# In[26]:


market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')
market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')

# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']
        
for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']


# In[27]:


market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby(['time']).agg({'price_diff': ['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values * 5,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[28]:


market_train_df['returnsOpenNextMktres10'].describe()


# In[29]:


noOutliers = market_train_df[(market_train_df['returnsOpenNextMktres10'] < 1) &  (market_train_df['returnsOpenNextMktres10'] > -1)]


# In[30]:


trace1 = go.Histogram(
    x = noOutliers.sample(n=10000)['returnsOpenNextMktres10'].values
)

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[31]:


data = []
for col in ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10']:
    df = market_train_df.groupby('time')[col].mean().reset_index()
    data.append(go.Scatter(
        x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = df[col].values,
        name = col
    ))
    
layout = go.Layout(dict(title = "Treand of mean values",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[32]:


market_train_df.isna().sum().to_frame()


# In[33]:


num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                   'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                   'returnsOpenPrevMktres10']


# In[34]:


from fancyimpute import IterativeImputer


# In[35]:


imp_cols = market_train_df[num_cols].columns.values
market_train_df[num_cols] = pd.DataFrame(IterativeImputer(verbose=True).fit_transform(market_train_df[num_cols]),columns= imp_cols)


# In[36]:


market_train_df.isna().sum().to_frame()


# In[37]:


scaler = StandardScaler()
market_train_df[num_cols] = scaler.fit_transform(market_train_df[num_cols])


# In[38]:


market_train_df.isna().sum().to_frame()


# In[39]:


def generate_lag_features(df):
    df['MA7MA'] = df['close'].rolling(window=7).mean()
    df['MA_15MA'] = df['close'].rolling(window=15).mean()
    df['MA_30MA'] = df['close'].rolling(window=30).mean()
    df['MA_60MA'] = df['close'].rolling(window=60).mean()
    ewma = pd.Series.ewm
    df['close_30EMA'] = ewma(df["close"], span=30).mean()
    df['close_26EMA'] = ewma(df["close"], span=26).mean()
    df['close_12EMA'] = ewma(df["close"], span=12).mean()

    df['MACD'] = df['close_12EMA'] - df['close_26EMA']

    no_of_std = 2

    df['MA7MA'] = df['close'].rolling(window=7).mean()
    df['MA_7MA_std'] = df['close'].rolling(window=7).std() 
    df['MA_7MA_BB_high'] = df['MA7MA'] + no_of_std * df['MA_7MA_std']
    df['MA_7MA_BB_low'] = df['MA7MA'] - no_of_std * df['MA_7MA_std']


    df['VMA_7MA'] = df['volume'].rolling(window=7).mean()
    df['VMA_15MA'] = df['volume'].rolling(window=15).mean()
    df['VMA_30MA'] = df['volume'].rolling(window=30).mean()
    df['VMA_60MA'] = df['volume'].rolling(window=60).mean()
    
    new_col = df["close"] - df["open"]
    df.insert(loc=6, column="daily_diff", value=new_col)
    df['close_to_open'] =  np.abs(df['close'] / df['open'])
   
    return df


# In[40]:


generate_lag_features(market_train_df)


# In[41]:


num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                   'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                   'returnsOpenPrevMktres10','MA7MA','MA_15MA','MA_30MA','MA_60MA','close_30EMA','close_26EMA','close_12EMA','MACD','MA_7MA','MA_7MA_std','MA_7MA_BB_high','MA_7MA_BB_low','close_to_open','VMA_7MA','VMA_15MA','VMA_30MA','VMA_60MA']


# In[42]:


market_train_df.head()


# In[43]:


news_train_df.head()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords


# In[45]:


vectorizer = CountVectorizer(max_features=1000, stop_words={"english"})

X = vectorizer.fit_transform(news_train_df['headline'].values)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
X_train_vals = X_train_tf.mean(axis=1)


del vectorizer
del X
del X_train_tf

#mean tf-idf score for news article.
d = pd.DataFrame(data=X_train_vals)
news_train_df['tf_score'] = d


# In[46]:


from wordcloud import WordCloud, STOPWORDS


# In[47]:


news_train_df_positive = news_train_df[news_train_df.sentimentPositive> 0.5]


# In[48]:


news_train_df_negative = news_train_df[(news_train_df.sentimentNegative + news_train_df.sentimentNeutral) > 0.5]


# In[49]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size= 40)
wc.generate(" ".join(news_train_df_positive['headline']))
plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=20)
# plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
plt.imshow(wc.recolor( colormap= 'pink_r' , random_state=17), alpha=0.98)
plt.axis('off')


# In[50]:


gc.collect()


# In[51]:


news_train_df["rise_fall"]=(news_train_df.headline.str.lower().str.contains('research|roundup|raises|public offering|target|second quarter')).astype(int)
news_train_df.head()


# In[52]:


news_agg_cols = [f for f in news_train_df.columns if 'novelty' in f or
                'volume' in f or
                'sentiment' in f or
                'bodySize' in f or
                'Count' in f or
                'marketCommentary' in f or
                'tf_score' in f or
                'rise_fall' in f or
                'relevance' in f]
news_agg_dict = {}
for col in news_agg_cols:
    news_agg_dict[col] = ['mean', 'sum', 'max', 'min']
news_agg_dict['urgency'] = ['min', 'count']
news_agg_dict['takeSequence'] = ['max']


# In[53]:


# update market dataframe to only contain the specific rows with matching indecies.
def check_index(index, indecies):
    if index in indecies:
        return True
    else:
        return False

# note to self: fill int/float columns with 0
def fillnulls(X):
    
    # fill headlines with the string null
    X['headline'] = X['headline'].fillna('null')
    
def generalize_time(X):
    # convert time to string and get rid of Hours, Minutes, and seconds
    X['time'] = X['time'].dt.strftime('%Y-%m-%d %H:%M:%S').str.slice(0,16) #(0,10) for Y-m-d, (0,13) for Y-m-d H

# this function checks for potential nulls after grouping by only grouping the time and assetcode dataframe
# returns valid news indecies for the next if statement.
def partial_groupby(market_df, news_df, df_assetCodes):
    
    # get new dataframe
    temp_news_df_expanded = pd.merge(df_assetCodes, news_df[['time', 'assetCodes']], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # groupby dataframes
    temp_news_df = temp_news_df_expanded.copy()[['time', 'assetCode']]
    temp_market_df = market_df.copy()[['time', 'assetCode']]

    # get indecies on both dataframes
    temp_news_df['news_index'] = temp_news_df.index.values
    temp_market_df['market_index'] = temp_market_df.index.values

    # set multiindex and join the two
    temp_news_df.set_index(['time', 'assetCode'], inplace=True)

    # join the two
    temp_market_df_2 = temp_market_df.join(temp_news_df, on=['time', 'assetCode'])
    del temp_market_df, temp_news_df

    # drop nulls in any columns
    temp_market_df_2 = temp_market_df_2.dropna()

    # get indecies
    market_valid_indecies = temp_market_df_2['market_index'].tolist()
    news_valid_indecies = temp_market_df_2['news_index'].tolist()
    del temp_market_df_2

    # get index column
    market_df = market_df.loc[market_valid_indecies]
    
    return news_valid_indecies

def join_market_news(market_df, news_df, nulls=False):
    
    # convert time to string
    generalize_time(market_df)
    generalize_time(news_df)
    
    # Fix asset codes (str -> list)
    news_df['assetCodes'] = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")

    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_df['assetCodes']))
    assetCodes_index = news_df.index.repeat( news_df['assetCodes'].apply(len) )
    
    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
    
    if not nulls:
        news_valid_indecies = partial_groupby(market_df, news_df, df_assetCodes)
    
    # create dataframe based on groupby
    news_col = ['time', 'assetCodes', 'headline'] + sorted(list(news_agg_dict.keys()))
    news_df_expanded = pd.merge(df_assetCodes, news_df[news_col], left_on='level_0', right_index=True, suffixes=(['','_old']))
    
    # check if the columns are in the index
    if not nulls:
        news_df_expanded = news_df_expanded.loc[news_valid_indecies]

    def news_df_feats(x):
        if x.name == 'headline':
            return list(x)
    
    # groupby time and assetcode
    news_df_expanded = news_df_expanded.reset_index()
    news_groupby = news_df_expanded.groupby(['time', 'assetCode'])
    
    # get aggregated df
    news_df_aggregated = news_groupby.agg(news_agg_dict).apply(np.float32).reset_index()
    news_df_aggregated.columns = ['_'.join(col).strip() for col in news_df_aggregated.columns.values]
    
    # get any important string dataframes
    news_df_cat = news_groupby.transform(lambda x: news_df_feats(x))['headline'].to_frame()
    new_news_df = pd.concat([news_df_aggregated, news_df_cat], axis=1)
    
    # cleanup
    del news_df_aggregated
    del news_df_cat
    del news_df
    
    # rename columns
    new_news_df.rename(columns={'time_': 'time', 'assetCode_': 'assetCode'}, inplace=True)
    new_news_df.set_index(['time', 'assetCode'], inplace=True)
    
    # Join with train
    market_df = market_df.join(new_news_df, on=['time', 'assetCode'])

    # cleanup
    fillnulls(market_df)

    return market_df


# In[54]:


X_train = join_market_news(market_train_df, news_train_df, nulls=False)


# In[55]:


X_train.head()


# In[56]:


# first get dates
def split_time(df):
    # split date_time into categories
    df['time_day'] = df['time'].str.slice(8,10)
    df['time_month'] = df['time'].str.slice(5,7)
    df['time_year'] = df['time'].str.slice(0,4)
    df['time_hour'] = df['time'].str.slice(11,13)
    df['time_minute'] = df['time'].str.slice(14,16)
    
    # source: https://www.kaggle.com/nicapotato/taxi-rides-time-analysis-and-oof-lgbm
    df['temp_time'] = df['time'].str.replace(" UTC", "")
    df['temp_time'] = pd.to_datetime(df['temp_time'], format='%Y-%m-%d %H')
    
    df['time_day_of_year'] = df.temp_time.dt.dayofyear
    df['time_week_of_year'] = df.temp_time.dt.weekofyear
    df["time_weekday"] = df.temp_time.dt.weekday
    df["time_quarter"] = df.temp_time.dt.quarter
    
    del df['temp_time']
    gc.collect()
    
    # convert to non-object columns
    time_feats = ['time_day', 'time_month', 'time_year','time_hour','time_minute','time_day_of_year','time_week_of_year',"time_weekday","time_quarter"]
    df[time_feats] = df[time_feats].apply(pd.to_numeric)
    df['time'] = pd.to_datetime(df['time'])
    
    del time_feats
    gc.collect()


# In[57]:


split_time(market_train_df)


# In[58]:


def remove_cols(X):
    del_cols = [f for f in X.columns if X[f].dtype == 'object']
    for f in del_cols:
        del X[f]


# In[59]:


gc.collect()


# In[60]:


remove_cols(X_train)


# In[61]:


X_train.head()

