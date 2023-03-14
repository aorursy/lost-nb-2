#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

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
corpus = set(stopwords.words('english'))


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(train_df_market, train_df_news) = env.get_training_data()


# In[ ]:


print(f'There are {train_df_market.shape[0]} samples and {train_df_market.shape[1]} features in the training market dataset.')


# In[ ]:


train_df_market.head()


# In[ ]:


d = []
for assetData in np.random.choice(train_df_market['assetName'].unique(), 15):
    df_asset = train_df_market[(train_df_market['assetName'] == assetData)]

    d.append(go.Scatter(
        x = df_asset['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = df_asset['close'].values,
        name = assetData
    ))
l = go.Layout(dict(title = "Closing prices for 15 randomly selected assets",
                  xaxis = dict(title = 'Month(M)'),
                  yaxis = dict(title = 'Price in $(USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=d, layout=l), filename='basic-line')


# In[ ]:


d = []
for val in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    dfPrice = train_df_market.groupby('time')['close'].quantile(val).reset_index()

    d.append(go.Scatter(
        x = dfPrice['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = dfPrice['close'].values,
        name = f'{val} quantile'
    ))
l = go.Layout(dict(title = "Trends of closing prices by quantiles",
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
            arrowcolor='#000000',
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
            arrowcolor='#000000',
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
            arrowcolor='#000000',
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
            arrowcolor='#000000',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        )
    ])
py.iplot(dict(data=d, layout=l), filename='basic-line')


# In[ ]:


train_df_market['price_diff'] = train_df_market['close'] - train_df_market['open']
grp = train_df_market.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()


# In[ ]:


print(f"Average standard deviation of price change within a day in {grp['price_diff']['std'].mean():.5f}.")


# In[ ]:


group = grp.sort_values(('price_diff', 'std'), ascending=False)[:20]
group['min_text'] = 'Maximum price drop: ' + (-1 * group['price_diff']['min']).astype(str)
t = go.Scatter(
    x = group['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = group['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = group['price_diff']['std'].values,
        color = group['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = group['min_text'].values
)
d = [t]

l= go.Layout(
    autosize= True,
    title= 'Best 20 months regarding standard deviation of price difference within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 6,
        gridwidth= 3,
    ),
    showlegend= False
)
f = go.Figure(data=d, layout=l)
py.iplot(f,filename='scatter2010')


# In[ ]:


train_df_market.sort_values('price_diff')[:10]


# In[ ]:



train_df_market['close_to_open'] =  np.abs(train_df_market['close'] / train_df_market['open'])

print(f"In {(train_df_market['close_to_open'] >= 1.2).sum()} lines price increased by 20% or more.")
print(f"In {(train_df_market['close_to_open'] <= 0.8).sum()} lines price decreased by 20% or more.")

print(f"In {(train_df_market['close_to_open'] >= 2).sum()} lines price increased by 100% or more.")
print(f"In {(train_df_market['close_to_open'] <= 0.5).sum()} lines price decreased by 100% or more.")


# In[ ]:


train_df_market['assetName_mean_open'] = train_df_market.groupby('assetName')['open'].transform('mean')
train_df_market['assetName_mean_close'] = train_df_market.groupby('assetName')['close'].transform('mean')

# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in train_df_market.loc[train_df_market['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        train_df_market.iloc[i,5] = row['assetName_mean_open']
    else:
        train_df_market.iloc[i,4] = row['assetName_mean_close']
        
for i, row in train_df_market.loc[train_df_market['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        train_df_market.iloc[i,5] = row['assetName_mean_open']
    else:
        train_df_market.iloc[i,4] = row['assetName_mean_close']


# In[ ]:


train_df_market['price_diff'] = train_df_market['close'] - train_df_market['open']
grouped = train_df_market.groupby(['time']).agg({'price_diff': ['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:15]
g['min_text'] = 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)
t = go.Scatter(
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
d = [t]

layout= go.Layout(
    autosize= True,
    title= 'After Preprocessing',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
f = go.Figure(data=d, layout=l)
py.iplot(f,filename='scatter2010')


# In[ ]:


d = []
for val in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    dfPrice = train_df_market.groupby('time')['returnsOpenNextMktres10'].quantile(val).reset_index()

    d.append(go.Scatter(
        x = dfPrice['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = dfPrice['returnsOpenNextMktres10'].values,
        name = f'{val} quantile'
    ))
l = go.Layout(dict(title = "Plot of returnsOpenNextMktres10 by quantiles",
                  xaxis = dict(title = 'Month(M)'),
                  yaxis = dict(title = 'Price in $(USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=d, layout=l), filename='basic-line')


# In[ ]:


d = []
train_df_market = train_df_market.loc[train_df_market['time'] >= '2010-01-01 22:00:00+0000']

dfPrice = train_df_market.groupby('time')['returnsOpenNextMktres10'].mean().reset_index()

d.append(go.Scatter(
    x = dfPrice['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = dfPrice['returnsOpenNextMktres10'].values,
    name = f'{val} quantile'
))
l = go.Layout(dict(title = "Plot of returnsOpenNextMktres10 mean",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=d, layout=l), filename='basic-line')


# In[ ]:


d = []
for column in ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10']:
    dataFrame = train_df_market.groupby('time')[column].mean().reset_index()
    d.append(go.Scatter(
        x = dataFrame['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = dataFrame[column].values,
        name = column
    ))
    
l = go.Layout(dict(title = "Plot of mean values",
                  xaxis = dict(title = 'Month(M)'),
                  yaxis = dict(title = 'Price in $(USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=d, layout=l), filename='basic-line')


# In[ ]:


train_df_news .head()


# In[ ]:


print(f'There are {train_df_news.shape[0]} samples and {train_df_news.shape[1]} features in the training news dataset.')


# In[ ]:


string = ' '.join(train_df_news['headline'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, stopwords=corpus, background_color='white',
                      width=1200, height=1000).generate(string)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Word Cloud for frequent words in business news')
plt.axis("off")
plt.show()


# In[ ]:


train_df_news = train_df_news.loc[train_df_news['time'] >= '2010-01-01 22:00:00+0000']


# In[ ]:


(train_df_news['urgency'] .value_counts() / 1000000).plot('bar');
plt.xticks(rotation=30);
plt.title('Urgency counts (mln)');


# In[ ]:


train_df_news['sentence_word_count'] =  train_df_news['wordCount'] / train_df_news['sentenceCount']
plt.boxplot(train_df_news['sentence_word_count'][train_df_news['sentence_word_count'] < 40]);


# In[ ]:


train_df_news['provider'].value_counts().head(10) 


# In[ ]:


(train_df_news['headlineTag'].value_counts() / 1000)[:10].plot('barh');
plt.title('headlineTag counts (thousands)');


# In[ ]:


for i, j in zip([-1, 0, 1], ['negative', 'neutral', 'positive']):
    sentiment = train_df_news.loc[train_df_news['sentimentClass'] == i, 'assetName']
    print(f'The companies mentioned the most  for {j} sentiment are:')
    print(sentiment.value_counts().head(5))
    print('')


# In[ ]:



def data_prep(market_df,news_df):
    market_df['time'] = market_df.time.dt.date
    market_df['returnsOpenPrevRaw1_to_volume'] = market_df['returnsOpenPrevRaw1'] / market_df['volume']
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df['volume_to_mean'] = market_df['volume'] / market_df['volume'].mean()
    news_df['sentence_word_count'] =  news_df['wordCount'] / news_df['sentenceCount']
    news_df['time'] = news_df.time.dt.hour
    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['headlineLen'] = news_df['headline'].apply(lambda x: len(x))
    news_df['assetCodesLen'] = news_df['assetCodes'].apply(lambda x: len(x))
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news_df['asset_sentence_mean'] = news_df.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')
    lbl = {k: v for v, k in enumerate(news_df['headlineTag'].unique())}
    news_df['headlineTagT'] = news_df['headlineTag'].map(lbl)
    kcol = ['firstCreated', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()

    market_df = pd.merge(market_df, news_df, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])

    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    market_df['assetCodeT'] = market_df['assetCode'].map(lbl)
    
    market_df = market_df.dropna(axis=0)
    
    return market_df

train_df_market.drop(['price_diff', 'assetName_mean_open', 'assetName_mean_close'], axis=1, inplace=True)
market_train = data_prep(train_df_market, train_df_news)
print(market_train.shape)
up = market_train.returnsOpenNextMktres10 >= 0

fcol = [c for c in market_train.columns if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'assetCodeT',
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)


# In[ ]:


X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)

xgb_up = XGBClassifier(n_jobs=8,
                        n_estimators=300,
                        max_depth=12,
                        booster='gbtree',
                        learning_rate=0.01,
                        objective='binary:logistic',
                        eta=0.15,
                        random_state=42
                        )


# In[ ]:


#params = {'learning_rate': 0.01, 'max_depth': 12, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'is_training_metric': True, 'seed': 42}
xgb_up.fit(X_train,up_train)


# In[ ]:


y_pred = xgb_up.predict(X_test)


# In[ ]:


predictions = [round(value) for value in y_pred]


# In[ ]:


accuracy = accuracy_score(up_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


print(classification_report(up_test, predictions))


# In[ ]:


def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))
    return color

df = pd.DataFrame({'imp': xgb_up.feature_importances_, 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
data = [df]
for dd in data:  
    colors = []
    for i in range(len(dd)):
         colors.append(generate_color())

    data = [
        go.Bar(
        orientation = 'h',
        x=dd.imp,
        y=dd.col,
        name='Features',
        textfont=dict(size=20),
            marker=dict(
            color= colors,
            line=dict(
                color='#000000',
                width=0.5
            ),
            opacity = 0.87
        )
    )
    ]
    l= go.Layout(
        title= 'Feature Importance of LGB',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )

    py.iplot(dict(data=data,layout=l), filename='horizontal-bar')


# In[ ]:


days = env.get_prediction_days()
import time

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')
    
    t = time.time()
    market_obs_df = data_prep(market_obs_df, news_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = model.predict(X_live)
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2 * lp -1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()

