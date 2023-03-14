#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as py

py.init_notebook_mode(connected=True)

# Data analysis custom settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


base_layout = dict(
    colorway = ['#ff5200', '#6f0000', '#00263b'] +
    [ '#ffa41b', '#000839', '#005082', '#00a8cc']+
    ['#000839', '#00a8cc']
    + ['#eb4559', '#f78259', '#522d5b'],
)
base_fig = go.Figure(
    layout = base_layout
)
template_fig = pio.to_templated(base_fig)
pio.templates['m5'] = template_fig.layout.template
pio.templates.default = 'm5'
pio.renderers.default = 'kaggle'


# In[3]:


os.getcwd()


# In[4]:


sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

for d in np.arange(1914, 1970):
    col = 'd_'+str(d)
    sales[col] = 0
    sales[col] = sales[col].astype(np.int16)


# In[5]:


# Downcasting

def downcast(df):
    dtypes = df.dtypes
    cols = dtypes.index.tolist()
    types = dtypes.values.tolist()
    
    for col, typ in zip(cols, types):
        
        if 'int' in str(typ):
            if df[col].min() > np.iinfo(np.int8).min and                 df[col].max() < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            
            elif df[col].min() > np.iinfo(np.int16).min and                 df[col].max() < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
                
            elif df[col].min() > np.iinfo(np.int32).min and                 df[col].max() < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
            else:
                df[col] = df[col].astype(np.int64)
                
        elif 'float' in str(typ):
            if df[col].min() > np.finfo(np.float16).min and                 df[col].max() < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
                
            elif df[col].min() > np.finfo(np.float32).min and                 df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
                
            else:
                df[col] = df[col].astype(np.float64)
                
        elif typ == np.object:
            if col == 'date':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                
            else:
                df[col] = df[col].astype('category')
    
    return df


# In[6]:


sales = downcast(sales)
calendar = downcast(calendar)
prices = downcast(prices)


# In[7]:


df = pd.melt(
    sales,
    id_vars=[
        'id',
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id'
    ],
    var_name='d',
    value_name='sold'
).dropna()

df = pd.merge(df, calendar, on='d', how='left')

df = pd.merge(
    df,
    prices,
    on=['store_id','item_id','wm_yr_wk'],
    how='left') 


# In[8]:


day = [
    'Monday',
    'Tuesday',
    'Thursday',
    'Wednesday',
    'Friday',
    'Saturday',
    'Sunday'
]

df['weekday'] = pd.Categorical(
    df['weekday'],
    categories=day,
    ordered=True
)
df['revenue'] = df.sold*df.sell_price


# In[9]:


categories = pd.DataFrame(
    df.cat_id.unique(),
    columns=['cat_id']
)
categories['color'] = ['#ff5200', '#6f0000', '#00263b']
categories.set_index('cat_id', inplace=True)

event_t1_df = pd.DataFrame(
    calendar.event_type_1.dropna().unique(),
    columns=['event_type_1']
)
event_t1_df['color'] = [
    '#ffa41b',
    '#000839',
    '#005082',
    '#00a8cc'
]
event_t1_df.set_index('event_type_1', inplace=True)

event_t2_df = pd.DataFrame(
    calendar.event_type_2.dropna().unique(),
    columns=['event_type_2']
)
event_t2_df['color'] = ['#000839', '#00a8cc']
event_t2_df.set_index('event_type_2', inplace=True)


states = pd.DataFrame(
    df.state_id.unique(),
    columns=['state_id']
)
states['color'] = ['#eb4559', '#f78259', '#522d5b']
states.set_index('state_id', inplace=True)


# In[10]:


bar_data = df.groupby(
    ['cat_id', 'dept_id']
)['sold'].sum().dropna()

fig = go.Figure()

for cat in [ 'HOBBIES','HOUSEHOLD','FOODS']:
    
    bar_data_fil = bar_data.loc[(cat, )].sort_values()
    trace = go.Bar(
        y = bar_data_fil.index.get_level_values(0),
        x = bar_data_fil.values,
        marker_color=categories.loc[(cat), 'color'],
        orientation='h',
        name = cat,
        texttemplate = '<b>%{x}</b>',
        textposition='inside',
    )
    fig.add_trace(trace)
fig.update_layout(
    title = dict(text = 'UNITS SOLD BY DEPTARTMENT'),
    legend = dict(x=0.85, y=0.1),
)
fig.show()


# In[11]:


bar_data = df.groupby(
    ['state_id', 'store_id']
)['sold'].sum().dropna()

fig = go.Figure()

for state in list(bar_data.index.levels[0])[::-1]:
    
    bar_data_fil = bar_data.loc[(state, )].sort_values()
    trace = go.Bar(
        y = bar_data_fil.index.get_level_values(0),
        x = bar_data_fil.values,
        marker_color=states.loc[(state), 'color'],
        orientation='h',
        name = state,
        texttemplate = '<b>%{x}</b>',
        textposition='inside',
    )
    fig.add_trace(trace)
fig.update_layout(
    title = dict(text = 'UNITS SOLD BY STORE'),
    legend = dict(x=0.9, y=0.1),
)
fig.show()


# In[12]:


data = df.groupby(
    ['state_id', 'store_id', 'cat_id', 'dept_id']
)['sold'].sum()

data = data.unstack(level=[-2,-1])        .dropna(axis=1, how='all')        .dropna(axis=0, how='all')

fig = go.Figure()
trace = go.Heatmap(
    y = [
        data.index.get_level_values(0),
        data.index.get_level_values(1)
    ],
    x = [
        data.columns.get_level_values(0),
        data.columns.get_level_values(1)
    ],
    z = data.values,
    coloraxis = 'coloraxis'
    
)
fig.add_trace(trace)
fig.update_layout(
    title = dict(
        text = 'UNITS SOLD BY STORE AND DEPARTMENT'
    ),
    coloraxis = dict(colorscale = 'Cividis')
)
fig.show()


# In[13]:


data = df.groupby(['cat_id', 'dept_id', pd.Grouper(key='date', freq='M')])['sold'].sum()
data = data[data>0]
fig = px.line(
    data_frame=data.reset_index(),
    x = 'date',
    y = 'sold',
    color = 'dept_id',
    facet_col='cat_id'
)
fig.update_xaxes(nticks=7)
fig.update_layout(
    title = dict(text = 'UNITS SOLD BY MONTH-YEAR')
)
fig.show()


# In[14]:


data = df.loc[df.sold>0].groupby(['cat_id', 'dept_id', 'weekday'])['sold'].mean().reset_index()
fig = px.line(
    data_frame=data,
    x = 'weekday',
    y = 'sold',
    color = 'dept_id',
    facet_col = 'cat_id',
)
fig.update_layout(
    title = dict(text='AVERAGE UNITS SOLD BY DEPARTMENT')
)

fig.show()


# In[15]:


data = df.loc[df.sold>0].groupby(
    ['cat_id', 'dept_id', pd.Grouper(key='date', freq='M')]
)['sold'].mean()

fig = px.line(
    data_frame=data.reset_index(),
    x = 'date',
    y = 'sold',
    color = 'dept_id',
    facet_col = 'cat_id',
    render_mode='svg'
)

fig.update_xaxes(nticks=7)
fig.update_layout(
    title = dict(text = 'AVERAGE UNITS SOLD BY MONTH-YEAR')
)
fig.show()


# In[16]:


selling_days = df.loc[df.sold>0].groupby(['cat_id', 'dept_id', pd.Grouper(key='date', freq='M')]).size()
active_days = df.groupby(['cat_id', 'dept_id', pd.Grouper(key='date', freq='M')]).size()
data = active_days.div(selling_days).reset_index()
data.rename(columns={0:'sold'}, inplace = True)

fig = px.line(
    data_frame=data,
    x = 'date',
    y = 'sold',
    color = 'dept_id',
    facet_col = 'cat_id',
    render_mode='svg'
)
fig.update_xaxes(nticks=7)
fig.update_layout(
    title = dict(text = 'AVERAGE FREQUENCY ASSISTANCE BY YEAR-MONTH')
)
fig.show()


# In[17]:


data = df.groupby(['state_id', 'store_id', pd.Grouper(key='date', freq='M')])['sold'].sum()
data = data[data>0]

metadata = dict()
metadata['traces'] = []
metadata['rows'] = []
metadata['cols'] = []
metadata['titles'] = []

nrows = 1
ncols = 3

for idx, cat in enumerate(data.index.levels[0]): 
    row = (idx//ncols)+ 1
    col = (idx%ncols)+ 1
    fil_stores = data.loc[(cat, )]
    
    for store in fil_stores.index.remove_unused_levels().levels[0]:
        fil_data = fil_stores.loc[(store, )]
        trace = go.Scatter(
            x = fil_data.index,
            y = fil_data.values,
            name = store,
            showlegend=False,
            hovertemplate= f'<b>Store:</b> {store}<br>' +
                            '<b>Units Sold:</b> %{y}<br>' +
                            '<b>Date:</b> %{x}'
        )
        metadata['traces'].append(trace)
        metadata['rows'].append(row)
        metadata['cols'].append(col)
    metadata['titles'].append(cat)
        
fig = make_subplots(rows = nrows, cols=ncols, subplot_titles=metadata['titles'], shared_yaxes=True)
fig.add_traces(data = metadata['traces'], rows=metadata['rows'], cols = metadata['cols'])
fig.update_layout(
    title = dict(text = 'UNITS SOLD BY STORE')
)
fig.show()


# In[18]:


# Data 
data = df.groupby(
    ['cat_id', 'date']
)['sold'].sum().reset_index()

data['year'] = data.date.dt.year
data['month'] = pd.Categorical(
    data.date.dt.month_name().str.slice(stop=3), 
                               categories = [
                                   'Jan',
                                   'Feb',
                                   'Mar',
                                   'Apr',
                                   'May',
                                   'Jun',
                                   'Jul',
                                   'Aug',
                                   'Sep',
                                   'Oct',
                                   'Nov',
                                   'Dec'
                               ],
                               ordered = True
)

data['weekday'] = pd.Categorical(
    data.date.dt.day_name(),
    categories=day,
    ordered=True
)

# subplots info
metadata = dict()
metadata['traces'] = []
metadata['rows'] = []
metadata['cols'] = []
metadata['titles'] = []

nrows = 6
ncols = 1

for idx, year in enumerate(data.year.unique()):
    
    row_mask = data.year==year
    col_mask = ['month', 'weekday', 'sold']
    fil_data = data.loc[row_mask, col_mask]                        .pivot_table(columns='month', index='weekday')                        .sort_index(ascending=False)
    trace = go.Heatmap(
        x = fil_data.columns.get_level_values(1),
        y = fil_data.index,
        z = fil_data.values,
        coloraxis = 'coloraxis',
        name=''
    )
    fig.add_trace(trace)
    
    #updating subplots info
    metadata['traces'].append(trace)
    metadata['rows'].append(idx+1)
    metadata['cols'].append(1)
    metadata['titles'].append(str(year))
    
fig = make_subplots(
    rows = nrows,
    cols = ncols,
    subplot_titles=metadata['titles'],
    shared_xaxes=True,
    shared_yaxes=True,
)

fig.add_traces(
    data = metadata['traces'],
    rows = metadata['rows'],
    cols = metadata['cols'],
)

fig.update_layout(
    title = 'UNITS SOLD BY DAY OF WEEK',
    height=900,
    coloraxis=dict(colorscale='Cividis'), showlegend=False
)

fig.show()


# In[19]:


data = df.groupby(['date', 'cat_id'])['sold'].sum().reset_index()
data1 = df.groupby(['date', 'cat_id'])['sold'].sum().rolling(14).mean().reset_index()
data = data.merge(data1, how='outer', on=['date', 'cat_id'])
data.columns = ['date', 'cat_id', 'sold', 'sold_ma']
event_cols = list(calendar.columns[calendar.columns.str.contains('event|snap')]) + ['date']
data = data.merge(calendar[event_cols], how='outer', on='date')
del(data1)

metadata = dict()
metadata['traces'] = []
metadata['rows'] = []
metadata['cols'] = []
metadata['titles'] = []

nrows = 4
ncols = 1
for idx, event in enumerate(data.event_type_1.unique().dropna()):
    for cat in data.cat_id.unique():
        mask = (data.cat_id == cat)
        mask1 = (data.event_type_1 == event)
        fil_data = data.loc[mask]
        fil_data2 = data.loc[mask&mask1]
        if idx == 0:
            showlegend = True
        else:
            showlegend = False
        trace = go.Scatter(
            x = fil_data.date,
            y = fil_data.sold,
            marker_color = categories.loc[(cat), 'color'],
            legendgroup = f'Items sold MA - {cat}',
            showlegend=showlegend,
            name = f'Items sold - {cat}',
            mode = 'lines',
            hovertemplate = f'<b>Category: </b>{cat}<br>'+
                            '<b>Sold Units: </b>%{y}<br>'+
                            '<b>Date:</b>%{x}<br>'
        )
        metadata['traces'].append(trace)
        metadata['rows'].append(idx+1)
        metadata['cols'].append(1)
        
        trace2 = go.Scatter(
            x = fil_data2.date,
            y = fil_data2.sold,
            marker_color = 'gold',
            name = f'Items sold MA - {cat}',
            legendgroup = f'Items sold MA - {cat}',
            showlegend=False,
            mode = 'markers',
            text = fil_data2.event_name_1,
            hovertemplate = '<b>Event Name:</b>%{text}',
            texttemplate = '<b>%{text}'
        )
        metadata['traces'].append(trace2)
        metadata['rows'].append(idx+1)
        metadata['cols'].append(1)
    metadata['titles'].append(f'Event type = {event}')
    
fig = make_subplots(
    rows=nrows,
    cols=ncols,
    subplot_titles=metadata['titles'],
    shared_xaxes=True,
    shared_yaxes=True,
)
fig.add_traces(
    data = metadata['traces'],
    rows = metadata['rows'],
    cols = metadata['cols'],
)
fig.update_layout(
    height=900,
    legend = dict(x=0.5, y=1.07, orientation='h'),
    title = dict(text='IMPACT OF EVENT TYPES ON SOLD UNITS')
)
fig.show()        


# In[20]:


def describe_moments(df):
    my_aggs = dict(
        sold = ['mean', 'median','std','skew', pd.DataFrame.kurt, 'sum', 'size'],
        sell_price = ['mean', 'median','std','skew', pd.DataFrame.kurt],
        revenue = ['mean', 'sum']
    )
    moments = df.groupby('item_id').agg(my_aggs)
    moments.columns = moments.columns.get_level_values(0)+ '_' + moments.columns.get_level_values(1)
    
    moment_label = ['mean', 'median', 'std', 'skew', 'kurt']
    for moment in moment_label:
        col_min = moments[f'sold_{moment}'].min()
        col_max = moments[f'sold_{moment}'].max()
        print(f'{moment} {col_min}, {col_max}')
    return moments


# In[21]:


non_cero_moments = describe_moments(df[df.sold>0])
non_cero_moments['cat_id'] = np.array(non_cero_moments.index.str.extract('([A-Z]+)')[0])
non_cero_moments['dept_id'] = np.array(non_cero_moments.index.str.extract('([A-Z]+_\d)')[0])

selling_days = df[df.sold>0].groupby(['item_id'])['date'].size()
activity_horizon = df.groupby(['item_id'])['date'].size().div(selling_days)
non_cero_moments['avg_sold_days'] = non_cero_moments.index.map(activity_horizon)

non_cero_moments.head()


# In[22]:


fig=go.Figure()
for cat in non_cero_moments.cat_id.unique():
    fil_data = non_cero_moments.loc[non_cero_moments.cat_id == cat]
    trace = go.Box(
        y = fil_data.dept_id,
        x = fil_data.sold_mean,
        marker_color = categories.loc[(cat), 'color'],
        orientation = 'h',
        name = cat,
        hovertext = fil_data.index,
        hovertemplate='<b>Item: </b>%{hovertext}<br>'+
                        '<b>Department: </b>%{y}<br>'+
                        '<b>Avg Sold Units: </b>%{x}<br>'
    )
    fig.add_trace(trace)

fig.show()


# In[23]:


sold_cols = non_cero_moments.columns[non_cero_moments.columns.str.contains('sold')]

metadata = dict()
metadata['traces'] = []
metadata['rows'] = []
metadata['cols'] = []
metadata['titles'] = []

nrows = 2
ncols = 4

for cat in non_cero_moments.cat_id.unique():
    fil_cat_df = non_cero_moments.loc[non_cero_moments.cat_id == cat]

    for idx, stat in enumerate(sold_cols):
        row = (idx//ncols) + 1
        col = (idx%ncols) + 1
        color = categories.loc[cat, 'color'] 
        
        if row ==1 and col==1:
            showlegend = True
        else:
            showlegend = False
            
        trace = go.Histogram(
            x = fil_cat_df[stat],
            marker_color = color,
            showlegend = showlegend,
            legendgroup=cat,
            opacity=0.4,
            cumulative_enabled = True,
            histnorm = 'probability',
            name = cat
        )
        metadata['traces'].append(trace)
        metadata['rows'].append(row)
        metadata['cols'].append(col)
        metadata['titles'].append(stat)
    
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles = metadata['titles'])
fig.add_traces(data=metadata['traces'], rows = metadata['rows'],
               cols = metadata['cols'])
fig.update_layout(
    title = dict(text='DISTRIBUTIONS BY ITEM BEHAVIOR'),
    xaxis_zeroline=False,
    barmode='overlay'
)
# fig.update_yaxes(showticklabels=False)
fig.show()


# In[24]:


fig = px.scatter(
    data_frame = non_cero_moments.reset_index(),
    x = 'sold_mean',
    y = 'sold_std',
    color = 'sold_kurt',
    size = 'sold_skew',
    facet_col='cat_id',
    render_mode='svg'
)

fig.update_layout(
#     title = 'UNITS SOLD BY DAY OF WEEK',
    coloraxis=dict(colorscale='Cividis'), showlegend=False
)

fig.update_traces(
    text = non_cero_moments.reset_index().item_id,
    hovertemplate = '<b>Item: </b>%{text}<br>'+
                    '<b>Sold Mean: </b>%{x:.2f}<br>'+
                    '<b>Sold Std: </b>%{y:.2f}<br>'+
                    '<b>Sold Skew: </b>%{marker.size:.2f}<br>'+
                    '<b>Sold kurtosis: </b>%{marker.color:.2f}'
)

fig.update_layout(
    title = dict(text='ITEMS BEHAVIOR BY CATEGORY')
)
fig.show()


# In[25]:



fig = px.scatter(
    data_frame = non_cero_moments.reset_index(),
    x = 'sold_mean',
    y = 'sold_std',
    size = 'revenue_sum',
    color = 'sold_sum',
    facet_col='cat_id',
    render_mode='svg'
)
fig.update_layout(
#     title = 'UNITS SOLD BY DAY OF WEEK',
    coloraxis=dict(colorscale='Cividis'), showlegend=False
)
fig.update_traces(
    text = non_cero_moments.reset_index().item_id,
    hovertemplate = '<b>Item: </b>%{text}<br>'+
                    '<b>Sold Mean: </b>%{x:.2f}<br>'+
                    '<b>Sold Std: </b>%{y:.2f}<br>'+
                    '<b>Total Revenue: </b>%{marker.size:,.0f}<br>'+
                    '<b>Sold Units: </b>%{marker.color:,}'
)

fig.update_layout(
    title = dict(text='ITEMS BEHAVIOR BY CATEGORY')
)
fig.show()


# In[26]:


top5_items = non_cero_moments.sold_sum.nlargest(5).index

def top5_plot(variable):
    mycolors = ['#4d3e3e', '#bb3b0e', '#dd7631', '#708160', '#d8c593']
    fig = make_subplots(rows = 2, cols = 1, subplot_titles=[f'Top 5 Items {variable} by Date', f'Top Items 5 {variable} Cumulative Distributions'])
    for color, top in zip(mycolors, top5_items):
        fil_df = df.loc[df.item_id == top].sort_values('date')

        trace = go.Scatter(
            x = fil_df.date,
            y = fil_df[variable],
            opacity=0.6,
            marker_color = color,
            legendgroup=top,
            showlegend=False,
            name = top,
        )

        fig.add_trace(trace, row=1, col=1)
        trace1 = go.Histogram(
            x = fil_df[variable],
            opacity=0.6,
            cumulative_enabled = True,
            histnorm='probability',
            marker_color = color,
            legendgroup=top,
            name = top,
        )
        fig.add_trace(trace1, row=2, col=1)
    fig.update_layout(title = dict(text = f'Top 5 {variable}'), barmode='overlay')
    fig.show()


# In[27]:


top5_plot('sold')


# In[ ]:




