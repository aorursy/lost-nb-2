#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle

# 参考：グラフ作成のためのチートシートとPythonによる各種グラフの実装
# https://qiita.com/4m1t0/items/76b0033edb545a78cef5
# 最大列
pd.set_option('max_columns', 50)
# https://matplotlib.org/3.1.3/gallery/style_sheets/bmh.html
# 「Bayesian Methods for Hackers style sheet」
plt.style.use('bmh')
# 「rcParams」はmatplotlibのデフォルトセット
# 「axes.prop_cycle」はcycler.Cycler型
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# In[2]:


get_ipython().system('ls -GFlash --color ../input/m5-forecasting-accuracy/')


# In[3]:


# データ読み込み
INPUT_DIR = '../input/m5-forecasting-accuracy'
# cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
# stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
# ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
# sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')

df_calendar = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
df_sales_train_validation = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
df_sample_submission = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
df_sell_prices = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')


# In[4]:


df_calendar.head()


# In[5]:


df_sales_train_validation.head()


# In[6]:


df_sample_submission.head()


# In[7]:


df_sell_prices.head()


# In[8]:


df_sample_submission.head()


# In[9]:


df_sales_train_validation.head()


# In[10]:


# 「日付」の列を取り出す
d_cols = [column for column in df_sales_train_validation.columns if 'd_' in column]


# In[11]:


# ↓にpandasで以下を実行する
# 1. 商品を選択
# 2. IDにindexを設定し、売上データのみの列にする
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html
# 3. 転置する
# 4. データをプロットする

df_sales_train_validation.loc[
    df_sales_train_validation['id'] == 'FOODS_3_090_CA_3_validation'
] \
.set_index('id')[d_cols] \
.T \
.plot(figsize=(15, 5),
      title='FOODS_3_090_CA_3 sales by "d" number',
      color=next(color_cycle))
plt.legend('')
plt.show()

# 見た感じは250日目ぐらいまでほとんど売れてないね...


# In[12]:


# カレンダーではこのようになっている
# 列は今注目している列のみ
df_calendar[
    [
    # 日付ID（d_1〜）
    'd',
    # 実際の日付
    'date',
    # イベント名1
    'event_name_1',
    # イベント名2
    'event_name_2',
    # イベントタイプ1
    'event_type_1',
    # イベントタイプ2
    'event_type_2', 
    # カリフォルニア州のSNAP
    'snap_CA',
    ]
].head()


# In[13]:


# 商品データとカレンダーをマージする
# 先程のデータ取り出し
df_example = df_sales_train_validation.loc[
    df_sales_train_validation['id'] == 'FOODS_3_090_CA_3_validation'] \
[d_cols].T
# 列名（元のindex名）を正しくする
df_example = df_example.rename(columns={8412:'FOODS_3_090_CA_3'})
# インデックス値をリセット
# 名称を「d」とする
df_example = df_example.reset_index().rename(columns={'index': 'd'}) # make the index "d"
# カレンダー情報のマージ
# 「validate」は1:1のマージであることをチェックする
df_example = df_example.merge(df_calendar, how='left', validate='1:1')
# indexにdate（calendar.csvの日付）を設定する
df_example.set_index('date')['FOODS_3_090_CA_3']     .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='FOODS_3_090_CA_3 sales by actual sale dates')
plt.show()

# 2011-10-05あたりから売上てるのがわかる


# In[14]:


# 他のトップセールスの例を見てみよう！(1)
# HOBBIES_1_234_CA_3_validation
df_example2 = df_sales_train_validation.loc[
    df_sales_train_validation['id'] == 'HOBBIES_1_234_CA_3_validation'
][d_cols].T
df_example2 = df_example2.rename(columns={6324:'HOBBIES_1_234_CA_3'}) # Name it correctly
df_example2 = df_example2.reset_index().rename(columns={'index': 'd'}) # make the index "d"
df_example2 = df_example2.merge(df_calendar, how='left', validate='1:1')
df_example2.set_index('date')['HOBBIES_1_234_CA_3']     .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='HOBBIES_1_234_CA_3 sales by actual sale dates')
plt.show()


# In[15]:


# 他のトップセールスの例を見てみよう！(2)
# HOUSEHOLD_1_118_CA_3_validation
df_example3 = df_sales_train_validation.loc[
    df_sales_train_validation['id'] == 'HOUSEHOLD_1_118_CA_3_validation'
][d_cols].T
df_example3 = df_example3.rename(columns={6776:'HOUSEHOLD_1_118_CA_3'}) # Name it correctly
df_example3 = df_example3.reset_index().rename(columns={'index': 'd'}) # make the index "d"
df_example3 = df_example3.merge(df_calendar, how='left', validate='1:1')
df_example3.set_index('date')['HOUSEHOLD_1_118_CA_3']     .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='HOUSEHOLD_1_118_CA_3 sales by actual sale dates')
plt.show()


# In[16]:


# サンプル商品IDリスト
example_ids = ['FOODS_3_090_CA_3','HOBBIES_1_234_CA_3','HOUSEHOLD_1_118_CA_3']
# サンプル商品の売上データ
df_example_all = [
    df_example, 
    df_example2, 
    df_example3
]

# 各商品データをプロットしていく
for i in [0, 1, 2]:
    # グラフ準備
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    
    # 週単位で平均を取る
    df_example_all[i].groupby('wday').mean()[
        example_ids[i]] \
        .plot(
            kind='line',
            title='average sale: day of week',
            lw=5,
            color=color_pal[0],
            ax=ax1
    )
    
    # 月単位で平均を取る
    df_example_all[i].groupby('month').mean()[
        example_ids[i]] \
        .plot(kind='line',
              title='average sale: month',
              lw=5,
              color=color_pal[4],
              ax=ax2)
    
    # 年で平均を取る
    df_example_all[i].groupby('year').mean()[example_ids[i]]         .plot(kind='line',
              lw=5,
              title='average sale: year',
              color=color_pal[2],
              ax=ax3)
    
    # サブタイトル
    fig.suptitle(f'Trends for item: {example_ids[i]}',
                 size=20,
                 y=1.1)
    plt.tight_layout()
    plt.show()


# In[17]:


# 20例のデータ取り出し
twenty_examples = df_sales_train_validation.sample(
        20, random_state=529
    ) \
    .set_index('id')[d_cols] \
    .T \
    .merge(df_calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')


# In[18]:


# 20例のデータをプロット
fig, axs = plt.subplots(10, 2, figsize=(15, 20))
axs = axs.flatten()
ax_idx = 0
for item in twenty_examples.columns:
    twenty_examples[item].plot(title=item,
                              color=next(color_cycle),
                              ax=axs[ax_idx])
    ax_idx += 1
plt.tight_layout()
plt.show()


# In[19]:


# 何の商品タイプがあるか
df_sales_train_validation['cat_id'].unique()


# In[20]:


# カテゴリ別でどれだけ商品数があるか
df_sales_train_validation.groupby('cat_id').count()['id']     .sort_values()     .plot(kind='barh', figsize=(15, 5), title='Count of Items by Category')
plt.show()


# In[21]:


# 過去の売上
# index: 日付ID
# column: 商品ID
past_sales = df_sales_train_validation.set_index('id')[d_cols]     .T     .merge(df_calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')


# カテゴリ別で各商品データを取り出して
for i in df_sales_train_validation['cat_id'].unique():
    # 対象カテゴリの商品列を取り出す
    items_col = [c for c in past_sales.columns if i in c]
    # 対象カテゴリの商品数を合計してプロット
    past_sales[items_col]         .sum(axis=1)         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Total Sales by Item Type')
plt.legend(df_sales_train_validation['cat_id'].unique())
plt.show()


# In[22]:


# 店リスト
store_list = df_sell_prices['store_id'].unique()

# 店単位で集計
for s in store_list:
    # 店単位で商品取り出し
    # ※ IDに店のIDが含まれている
    store_items = [c for c in past_sales.columns if s in c]
    # 店単位で移動平均を出す
    # https://note.nkmk.me/python-pandas-rolling/
    # 店で売っている商品を日付ごとに合計して90日ずつの移動平均を表示
    past_sales[store_items]         .sum(axis=1)         .rolling(90).mean()         .plot(figsize=(15, 5),
              alpha=0.8,
              title='Rolling 90 Day Average Total Sales (10 stores)')
plt.legend(store_list)
plt.show()


# In[23]:


# 店で売っている商品を日付ごとに合計して7日ずつの移動平均を表示
fig, axes = plt.subplots(5, 2, figsize=(15, 10), sharex=True)
axes = axes.flatten()
ax_idx = 0
for s in store_list:
    store_items = [c for c in past_sales.columns if s in c]
    past_sales[store_items]         .sum(axis=1)         .rolling(7).mean()         .plot(alpha=1,
              ax=axes[ax_idx],
              title=s,
              lw=3,
              color=next(color_cycle))
    ax_idx += 1
# plt.legend(store_list)
plt.suptitle('Weekly Sale Trends by Store ID')
plt.tight_layout()
plt.show()


# In[24]:


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


# In[25]:


# 最低売上
print('The lowest sale date was:', past_sales.sum(axis=1).sort_values().index[0],
     'with', past_sales.sum(axis=1).sort_values().values[0], 'sales')
# 最高売上 ※ 元はlowestだけどhighestに変更
print('The highest sale date was:', past_sales.sum(axis=1).sort_values(ascending=False).index[0],
     'with', past_sales.sum(axis=1).sort_values(ascending=False).values[0], 'sales')


# In[26]:


# ヒートマップ作成
from sklearn.preprocessing import StandardScaler
sscale = StandardScaler()
past_sales.index = pd.to_datetime(past_sales.index)

for i in df_sales_train_validation['cat_id'].unique():
    # 2013年
    fig, axes = plt.subplots(3, 1, figsize=(20, 8))
    items_col = [c for c in past_sales.columns if i in c]
    sales2013 = past_sales.loc[past_sales.index.isin(pd.date_range('31-Dec-2012',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2013.values.reshape(-1, 1)))
    calmap(axes[0], 2013, vals.reshape(53,7).T)
    
    # 2014年
    sales2014 = past_sales.loc[past_sales.index.isin(pd.date_range('30-Dec-2013',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2014.values.reshape(-1, 1)))
    calmap(axes[1], 2014, vals.reshape(53,7).T)
    
    # 2015年
    sales2015 = past_sales.loc[past_sales.index.isin(pd.date_range('29-Dec-2014',
                                                                   periods=371))][items_col].mean(axis=1)
    vals = np.hstack(sscale.fit_transform(sales2015.values.reshape(-1, 1)))
    calmap(axes[2], 2015, vals.reshape(53,7).T)
    
    plt.suptitle(i, fontsize=30, x=0.4, y=1.01)
    plt.tight_layout()
    plt.show()


# In[27]:


fig, ax = plt.subplots(figsize=(15, 5))
stores = []

# 1商品の店舗・週別の価格をプロット
for store, d in df_sell_prices.query('item_id == "FOODS_3_090"').groupby('store_id'):
    d.plot(x='wm_yr_wk',
          y='sell_price',
          style='.',
          color=next(color_cycle),
          figsize=(15, 5),
          title='FOODS_3_090 sale price over time',
         ax=ax,
          legend=store)
    stores.append(store)
    plt.legend()
plt.legend(stores)
plt.show()


# In[28]:


df_sell_prices['Category'] = df_sell_prices['item_id'].str.split('_', expand=True)[0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
i = 0

# 店舗・カテゴリ別の価格をプロット
for cat, d in df_sell_prices.groupby('Category'):
    ax = d['sell_price'].apply(np.log1p)         .plot(kind='hist',
                         bins=20,
                         title=f'Distribution of {cat} prices',
                         ax=axs[i],
                                         color=next(color_cycle))
    ax.set_xlabel('Log(price)')
    i += 1
plt.tight_layout()


# In[29]:


# 過去30日間の平均を取った辞書データ
thirty_day_avg_map = df_sales_train_validation.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()

# 予測日の列リスト
fcols = [f for f in df_sample_submission.columns if 'F' in f]

for f in fcols:
    # 各商品単位で平均の値を設定
    df_sample_submission[f] = df_sample_submission['id'].map(thirty_day_avg_map).fillna(0)
    
df_sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




