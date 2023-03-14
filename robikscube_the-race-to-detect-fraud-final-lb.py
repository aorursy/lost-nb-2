#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
import datetime
import colorlover as cl
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# Format the data
df = pd.read_csv('../input/ieee-fraud-leaderboard/ieee-fraud-detection-publicleaderboard_10_3_Final.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.max().sort_values(ascending=False)[15]
FIFTYTH_SCORE = df.max().sort_values(ascending=False)[50]
TOP_SCORE = df.max().sort_values(ascending=False)[0]


# In[2]:


# Missing "Young for you"
gold_teams = ['FraudSquad','2 uncles and 3 puppies','T.U.V', 'Lions','The Zoo',
              'Grand Rookie(Done!Good luck to everyone!)）', 'S.A.R.M.A', 'AlKo', 'M5',
              'Flying Whales','邦盛科技小分队','YuyaYamamoto','MaYa','Mr Lonely ♬','spongebob',
              'Taemyung Heo','conundrum.ai & Evgeny', 'クソザコちゃうねん','Our AUC says nothing to us',
              'bird and fish', '天行健,君子以自强不息 地势坤,君子以厚德载物']

gold_df = df[gold_teams]
gold_scores = [0.945884, 0.944210, 0.942580, 0.942453, 0.942391, 0.942314, 0.942268, 0.942129, 0.941750,
              0.941638, 0.941413, 0.941338, 0.941153, 0.941096, 0.941011, 0.940934, 0.940756, 0.940730,
              0.940526, 0.940250, 0.940076]

gold_scores_df = pd.DataFrame(index=gold_teams,
                             data=gold_scores,
                             columns=['Private Score'])


# In[3]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 21 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.max().loc[df.max() > FIFTEENTH_SCORE].index.values
df_filtered = gold_df.ffill()
df_filtered = df_filtered.iloc[df_filtered.index >= '08-01-2019']
team_ordered = df_filtered.max(axis=0)     .sort_values(ascending=False).index.tolist()

data = []
i = 0
for col in df_filtered[team_ordered].columns:
    data.append(go.Scatter(
                        x = df_filtered.index,
                        y = df_filtered[col],
                        name=col,
                        line=dict(color=colors[i], width=2),)
               )
    i += 1

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Gold Medal Teams Private LB Journey',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[0.945, TOP_SCORE+0.001]),
                   hovermode='x',
                   plot_bgcolor='white',
                  annotations=annotations,
                  )
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    legend=go.layout.Legend(
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2,
    )
)

fig.update_layout(legend_orientation="h")
fig.update_layout(template="plotly_white")
#fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
fig.update_xaxes(showgrid=False)

iplot(fig)


# In[4]:


gold_scores_df.sort_values('Private Score', ascending=True).     plot(kind='barh',
         xlim=(0.94, 0.946),
         figsize=(15, 10),
         title='Final Private Board Scores of Gold Teams',
         color='lightgoldenrodyellow')
plt.show()


# In[5]:


# Interative Plotly
mypal = cl.scales['9']['div']['Spectral']
colors = cl.interp( mypal, 15 )
annotations = []
init_notebook_mode(connected=True)
TOP_TEAMS = df.max().loc[df.max() > FIFTEENTH_SCORE].index.values
df_filtered = df[TOP_TEAMS].ffill()
df_filtered = df_filtered.iloc[df_filtered.index >= '08-01-2019']
team_ordered = df_filtered.max(axis=0)     .sort_values(ascending=False).index.tolist()

data = []
i = 0
for col in df_filtered[team_ordered].columns:
    data.append(go.Scatter(
                        x = df_filtered.index,
                        y = df_filtered[col],
                        name=col,
                        line=dict(color=colors[i], width=2),)
               )
    i += 1

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='IEEE Fraud Detection Leaderboard Tracking',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

layout = go.Layout(yaxis=dict(range=[0.945, TOP_SCORE+0.001]),
                   hovermode='x',
                   plot_bgcolor='white',
                  annotations=annotations,
                  )
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    legend=go.layout.Legend(
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2,
    )
)

fig.update_layout(legend_orientation="h")
fig.update_layout(template="plotly_white")
#fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
fig.update_xaxes(showgrid=False)

iplot(fig)


# In[6]:


# Scores of top teams over time
plt.rcParams["font.size"] = "12"
ALL_TEAMS = df.columns.values
df_ffill = df[ALL_TEAMS].ffill()

# This is broken
df_ffill.T.sample(1000).T.plot(figsize=(20, 10),
                           color=color_pal[0],
                           legend=False,
                           alpha=0.05,
                           ylim=(0.925, TOP_SCORE+0.001),
                           title='All Teams Public LB Scores over Time')

df.ffill().max(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)
plt.show()


# In[7]:


plt.rcParams["font.size"] = "12"
ax =df.ffill()     .count(axis=1)     .plot(figsize=(20, 8),
          title='Number of Teams in the Competition by Date',
         color=color_pal[5], lw=5)
ax.set_ylabel('Number of Teams')
plt.axvline('09-23-2019', color='orange', linestyle='-.')
plt.text('09-23-2019', 4000,'Merger Deadline',rotation=-90)
plt.axvline('10-1-2019', color='orange', linestyle='-.')
plt.text('10-1-2019', 4000,'Original Deadline',rotation=-90)
plt.axvline('10-3-2019', color='orange', linestyle='-.')
plt.text('10-3-2019', 4000,'Extended Deadline',rotation=-90)
plt.show()


# In[8]:


# plt.style.use('ggplot')
# team_over_time = df.ffill() \
#     .count(axis=1)

# lr = LinearRegression()
# _ = lr.fit(np.array(pd.to_numeric(team_over_time.index).tolist()).reshape(-1, 1),
#            team_over_time.values)

# teamcount_df = pd.DataFrame(team_over_time)

# teamcount_pred_df = pd.DataFrame(index=pd.date_range('07-15-2019','10-05-2019'))
# teamcount_pred_df['Forecast Using All Data'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

# lr = LinearRegression()
# _ = lr.fit(np.array(pd.to_numeric(team_over_time[-5000:].index).tolist()).reshape(-1, 1),
#            team_over_time[-5000:].values)

# teamcount_pred_df['Forecast Using Recent Data'] = lr.predict(np.array(pd.to_numeric(teamcount_pred_df.index).tolist()).reshape(-1, 1))

# plt.rcParams["font.size"] = "12"
# ax =df.ffill() \
#     .count(axis=1) \
#     .plot(figsize=(20, 8),
#           title='Forecasting the Final Number of Teams',
#          color=color_pal[5], lw=5,
#          xlim=('07-13-2019','10-02-2019'))
# teamcount_pred_df['Forecast Using All Data'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using All Data')
# teamcount_pred_df['Forecast Using Recent Data'].plot(ax=ax, style='.-.', alpha=0.5, label='Regression Using last 1000 observations')
# ax.set_ylabel('Number of Teams')
# teamcount_pred_df.plot(ax=ax, style='.-.', alpha=0.5)
# plt.axvline('09-23-2019', color='orange', linestyle='-.')
# plt.text('09-23-2019', 4000,'Merger Deadline',rotation=-90)
# plt.axvline('10-1-2019', color='orange', linestyle='-.')
# plt.text('10-1-2019', 4000,'Original Deadline',rotation=-90)
# plt.axvline('10-3-2019', color='orange', linestyle='-.')
# plt.text('10-3-2019', 4000,'Extended Deadline',rotation=-90)
# plt.show()


# In[9]:


plt.rcParams["font.size"] = "12"
# Create Top Teams List
TOP_TEAMS = df.max().loc[df.max() > FIFTYTH_SCORE].index.values
df[TOP_TEAMS].max().sort_values(ascending=True).plot(kind='barh',
                                       xlim=(FIFTYTH_SCORE-0.001,TOP_SCORE+0.001),
                                       title='Top 50 Public LB Teams',
                                       figsize=(12, 15),
                                       color=color_pal[3])
plt.show()


# In[10]:


plt.rcParams["font.size"] = "12"
df[TOP_TEAMS].nunique().sort_values().plot(kind='barh',
                                           figsize=(12, 15),
                                           color=color_pal[1],
                                           title='Count of Submissions improving LB score by Team')
plt.show()


# In[11]:


plt.rcParams["font.size"] = "7"
n_weeks = (datetime.date(2019, 10, 3) - datetime.date(2019, 7, 14)).days / 7 # Num days of the comp
n_weeks = int(n_weeks)
fig, axes = plt.subplots(n_weeks, 1, figsize=(15, 25), sharex=True)
#plt.subplots_adjust(top=8, bottom=2)
for x in range(n_weeks):
    date2 = df.loc[df.index.date == datetime.date(2019, 7, 15) + datetime.timedelta(x*7+1)].index.min()
    num_teams = len(df.ffill().loc[date2].dropna())
    max_cutoff = df.ffill().loc[date2] < 5
    df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',
                               bins=500,
                               ax=axes[x],
                               title='{} ({} Teams)'.format(date2.date().isoformat(),
                                                            num_teams),
                                              xlim=(0.9, 0.96))
    y_axis = axes[x].yaxis
    y_axis.set_label_text('')
    y_axis.label.set_visible(False)

