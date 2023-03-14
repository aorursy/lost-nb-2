#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


stv = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
s_sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
s_price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')


# In[3]:


stv


# In[4]:


print("Categories {}".format(stv['cat_id'].unique()))
print("States {}".format(stv['state_id'].unique()))
print("Stores {}".format(stv['store_id'].unique()))



# In[5]:


cal.head(5)


# In[6]:


stv.mean(axis=1).sort_values()


# In[7]:


# to get only days columns
d_cols = [col for col in stv.columns if 'd_' in col]
# and keeping ids aside we will only look into sales
stv.loc[stv['id'] == 'FOODS_3_090_CA_3_validation'].set_index('id')[d_cols].T.plot(figsize=(15, 5),title='FOODS_3_090_CA_3 sales by "d" number')
plt.legend('')
plt.show()


# In[8]:


examples = ['FOODS_3_090_CA_3','FOODS_3_586_TX_3','FOODS_3_586_TX_2']
ex1= stv.loc[stv['id'] == 'FOODS_3_090_CA_3_validation'][d_cols].T
ex1 = ex1.rename(columns={8412:'FOODS_3_090_CA_3'})
ex2= stv.loc[stv['id'] == 'FOODS_3_586_TX_3_validation'][d_cols].T
ex2 = ex2.rename(columns={21104:'FOODS_3_586_TX_3'})
ex3= stv.loc[stv['id'] == 'FOODS_3_586_TX_2_validation'][d_cols].T
ex3 = ex3.rename(columns={18055:'FOODS_3_586_TX_2'})
examples_df = [ex1,ex2,ex3]
ex1 = ex1.reset_index().rename(columns={'index':'d'})
ex1 = ex1.merge(cal,how='left',validate='1:1')
for i in [0,1,2]:
    examples_df[i] = examples_df[i].reset_index().rename(columns={'index':'d'})
    examples_df[i] = examples_df[i].merge(cal,how='left',validate='1:1')   
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,3))
    examples_df[i].groupby('wday').mean()[examples[i]].plot(kind='line',title='average sale: day of week',ax=ax1)
    examples_df[i].groupby('month').mean()[examples[i]].plot(kind='line',title='average monthly sale',ax=ax2)
    examples_df[i].groupby('year').mean()[examples[i]].plot(kind='line',title='average yearly sale',ax=ax3)


# In[9]:


samples = stv.sample(20, random_state=200).set_index('id')[d_cols].T            .merge(cal.set_index('d')['date'],left_index=True,right_index=True,validate='1:1').set_index('date')
fig,axs = plt.subplots(10, 2, figsize=(16, 24))
axs = axs.flatten()
for i in range(len(samples.columns)):
    samples[samples.columns[i]].plot(title=samples.columns[i],ax=axs[i])
plt.tight_layout()
plt.show()
    

There are many days when the sale is zero.
# In[10]:


sns.countplot(data=stv,x='cat_id')


# In[11]:


cats = ['FOODS','HOBBIES','HOUSEHOLD']
id_cols = ['id','item_id','dept_id','store_id','state_id']
daily_sale = stv.groupby('cat_id').sum().T.reset_index().rename(columns={'index':'d'}).merge(cal,how='left',validate='1:1')
daily_sale = daily_sale.set_index('date')
fig, axs =  plt.subplots(3, figsize=(16, 24))
axs = axs.flatten()
for i in range(len(cats)):
    daily_sale[cats[i]].plot(title= cats[i], ax = axs[i])


# In[12]:


from matplotlib.pyplot import figure
def plot_Graph3Series(series, title,labels):
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(series[0],'-b', label=labels[0])
    plt.plot(series[1],'-r', label=labels[1])
    plt.plot(series[2],'-g', label=labels[2])
    plt.title(title)
    plt.legend(framealpha=1, frameon=True)
    plt.show()


# In[13]:


#monthly_sale = daily_sale['FOODS'].groupby('month')
daily_sale = stv.groupby('cat_id').sum().T.reset_index().rename(columns={'index':'d'}).rename(columns={'index':'d'})#
fig, axs =  plt.subplots(3, figsize=(6, 14))
axs = axs.flatten()
for i in range(len(cats)):   
    sale = daily_sale[['d',cats[i]]].merge(cal,how='left',validate='1:1').set_index('date')
    sale.groupby('month')[cats[i]].mean().plot(title = cats[i],ax=axs[i])


# In[14]:


#daily_sale['date'] = pd.to_datetime(daily_sale['date'])
#everydaydf.groupby([everydaydf[date_column].dt.to_period("M")]).sum()
daily_sale = daily_sale.merge(cal,how='left',validate='1:1').set_index('date')
daily_sale['date'] = pd.to_datetime(daily_sale.index)
#daily_sale = daily_sale.groupby([daily_sale['date'].dt.to_period("M")])['FOODS'].sum()
#.index = monthlyWithoutOutliersdf.index.to_timestamp()


# In[15]:


series = []
for i in range(len(cats)):
    x = daily_sale.groupby([daily_sale['date'].dt.to_period("M")])[cats[i]].sum()
    x.index = x.index.to_timestamp()
    series.append(x)
plot_Graph3Series(series,'Monthly Sale in categories',cats)


# In[16]:


past_sales = stv.set_index('id')[d_cols]     .T     .merge(cal.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')
for i in stv['cat_id'].unique():
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col]         .sum(axis=1)         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Total Sales by Item Type')
plt.legend(stv['cat_id'].unique())
plt.show()


# In[17]:


state_list = stv['state_id'].unique()
for s in state_list:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(30).mean()         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Rolling 90 Day Average Total Sales (10 stores)')
plt.legend(state_list)
plt.show()


# In[18]:


# ----------------------------------------------------------------------------
# Author:  Nicolas P. Rougier
# License: BSD
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from datetime import datetime
from dateutil.relativedelta import relativedelta

def calmap(ax, year, data):
    ax.tick_params('x', length=0, labelsize="medium", which='major')
    ax.tick_params('y', length=0, labelsize="x-small", which='major')

    # Month borders
    xticks, labels = [], []
    start = datetime(year,1,1).weekday()
    for month in range(1,13):
        first = datetime(year, month, 1)
        last = first + relativedelta(months=1, days=-1)

        y0 = first.weekday()
        y1 = last.weekday()
        x0 = (int(first.strftime("%j"))+start-1)//7
        x1 = (int(last.strftime("%j"))+start-1)//7

        P = [ (x0,   y0), (x0,    7),  (x1,   7),
              (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
              (x0+1,  0), (x0+1,  y0) ]
        xticks.append(x0 +(x1-x0+1)/2)
        labels.append(first.strftime("%b"))
        poly = Polygon(P, edgecolor="black", facecolor="None",
                       linewidth=1, zorder=20, clip_on=False)
        ax.add_artist(poly)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(0.5 + np.arange(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("{}".format(year), weight="semibold")
    
    # Clearing first and last day from the data
    valid = datetime(year, 1, 1).weekday()
    data[:valid,0] = np.nan
    valid = datetime(year, 12, 31).weekday()
    # data[:,x1+1:] = np.nan
    data[valid+1:,x1] = np.nan

    # Showing data
    ax.imshow(data, extent=[0,53,0,7], zorder=10, vmin=-1, vmax=1,
              cmap="RdYlBu_r", origin="lower", alpha=.75)


# In[19]:


from sklearn.preprocessing import StandardScaler
sscale = StandardScaler()
past_sales.index = pd.to_datetime(past_sales.index)
for i in stv['cat_id'].unique():
    fig, axes = plt.subplots(3, 1, figsize=(20, 8))
    items_col = [c for c in past_sales.columns if i in c]
    sales2013 = past_sales.loc[past_sales.index.isin(pd.date_range('31-Dec-2012',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2013.values.reshape(-1, 1)))
    calmap(axes[0], 2013, vals.reshape(53,7).T)
    sales2014 = past_sales.loc[past_sales.index.isin(pd.date_range('30-Dec-2013',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2014.values.reshape(-1, 1)))
    calmap(axes[1], 2014, vals.reshape(53,7).T)
    sales2015 = past_sales.loc[past_sales.index.isin(pd.date_range('29-Dec-2014',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2015.values.reshape(-1, 1)))
    calmap(axes[2], 2015, vals.reshape(53,7).T)
    plt.suptitle(i, fontsize=30, x=0.4, y=1.01)
    plt.tight_layout()
    plt.show()


# In[20]:


#fig, axes = plt.subplots(3, 1, figsize=(20, 8))
#items_col = [c for c in past_sales.columns if i in c]
items_col = []
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
index=0
for i in stv['state_id'].unique():
    items_col = [c for c in past_sales.columns if i in c]
    state_sales = past_sales[items_col].sum(axis=1)
    state_sales.plot(ax=axes[index],title=i)
    index = index+1


# In[21]:


cal


# In[22]:


state_cols=[]
cat_cols = []


# In[23]:


past_sales['date'] = pd.to_datetime(past_sales.index)
#past_sales.index


# In[24]:


past_sales = stv.set_index('id')[d_cols].T
fig,axs = plt.subplots(1,3,figsize=(16,3))
axs = axs.flatten()
i=0
for s in stv['state_id'].unique():
    series = pd.DataFrame()
    state_cols= [c for c in past_sales.columns if s in c]
    for cat in stv['cat_id'].unique():
        cat_cols= [c for c in state_cols if cat in c]
        x = pd.DataFrame(past_sales[cat_cols].sum(axis=1))
        x['d'] = past_sales[cat_cols].sum(axis=1).index 
        x= x.merge(cal.set_index('d'),how='left',left_index=True,right_index=True,validate='1:1')
        x = x.groupby('wday').mean()[0].rename(cat)
        series = pd.concat([series,x],axis=1) 
    series.plot(ax=axs[i],title='Average Sale on day of week for '+ s)
    i=i+1
       


# In[25]:


fig,axs = plt.subplots(1,3,figsize=(16,3))
axs = axs.flatten()
i=0
for s in stv['state_id'].unique():
    series = pd.DataFrame()
    state_cols= [c for c in past_sales.columns if s in c]
    for cat in stv['cat_id'].unique():
        cat_cols= [c for c in state_cols if cat in c]
        x = pd.DataFrame(past_sales[cat_cols].sum(axis=1))
        x['d'] = past_sales[cat_cols].sum(axis=1).index 
        x= x.merge(cal.set_index('d'),how='left',left_index=True,right_index=True,validate='1:1')
        x = x.groupby('month').mean()[0].rename(cat)
        series = pd.concat([series,x],axis=1) 
    series.plot(ax=axs[i],title='Average monthly Sale for '+ s)
    i=i+1
       


# In[26]:


#snap_count = cal.groupby(['weekday','snap_TX'])['snap_TX'].count()
#snap_count['weekday'] = snap_count.index
#snap_count = snap_count.rename({0:'off',1:'on'})
fig,axs = plt.subplots(1,3,figsize=(16,3))
sns.countplot(data=cal,x='weekday',hue='snap_CA', ax=axs[0])
sns.countplot(data=cal,x='weekday',hue='snap_TX', ax=axs[1])
sns.countplot(data=cal,x='weekday',hue='snap_WI', ax=axs[2])


# In[27]:


#import datetime
#date1 = datetime.datetime.strptime('2012-31-12',"%Y-%d-%m") 
date_range = pd.date_range('31-Dec-2012',periods=371)
cal['date'] = pd.to_datetime(cal['date']) 
sales2013 = cal.loc[(cal['date'] >= min(date_range)) & (cal['date'] <= max(date_range))]['snap_TX']
vals = np.hstack(sscale.fit_transform(sales2013.values.reshape(-1, 1)))
fig, axes = plt.subplots(figsize=(20, 8))
calmap(axes, 2013, vals.reshape(53,7).T)


# In[28]:



fig,axs = plt.subplots(1,3,figsize=(16,3))
axs = axs.flatten()
i=0
#for s in stv['state_id'].unique():
#    series = pd.DataFrame()
#    state_cols= [c for c in past_sales.columns if s in c]
for cat in stv['cat_id'].unique():
    cat_cols= [c for c in past_sales.columns if cat in c]
    x = pd.DataFrame(past_sales[cat_cols].sum(axis=1))
    x['d'] = past_sales[cat_cols].sum(axis=1).index 
    x= x.merge(cal.set_index('d'),how='left',left_index=True,right_index=True,validate='1:1')
    sns.boxplot(y=x[0], x=x['event_type_1'], ax=axs[i]).set_title(cat)
    i=i+1
       


# In[29]:



fig,axs = plt.subplots(3,3,figsize=(16,16))
#axs = axs.flatten()
i=0
j=0
for s in stv['state_id'].unique():
    series = pd.DataFrame()
    state_cols= [c for c in past_sales.columns if s in c]
    j=0
    for cat in stv['cat_id'].unique():
        cat_cols= [c for c in state_cols if cat in c]
        x = pd.DataFrame(past_sales[cat_cols].sum(axis=1))
        x['d'] = past_sales[cat_cols].sum(axis=1).index 
        x= x.merge(cal.set_index('d'),how='left',left_index=True,right_index=True,validate='1:1')
        col = 'snap_'+s
        sns.boxplot(y=x[0], x=x[col], ax=axs[i][j]).set_title(cat + " "+ s)
        j=j+1
    i=i+1
       


# In[30]:


sns.boxplot(y=x[0], x=x['event_type_1'])


# In[31]:


fig, axes = plt.subplots(1,3,figsize=(16,6))
sns.countplot(data=cal, x=cal['snap_CA'], ax=axes[0]) 
sns.countplot(data=cal, x=cal['snap_WI'], ax=axes[1]) 
sns.countplot(data=cal, x=cal['snap_TX'], ax=axes[2]) 


# In[32]:


sns.countplot(data=cal['event_name_1'], x=cal['event_type_1']) #.isnull().sum()

