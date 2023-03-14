#!/usr/bin/env python
# coding: utf-8

# In[1]:


merge_order_product_dsimport pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.options.display.max_rows = 20
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )
#sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'AppleGothic'} )

def toReadable(v):
    value = round(v,2) if isinstance(v, float) else v

    if value < 1000:
        return str(value)
    elif value<1000000:
        return str(round(value/1000,1))+'K'
    elif value>=1000000:
        return str(round(value/1000000,1))+'M'
    return value


# In[2]:


raw_order_ds = pd.read_csv('../input/orders.csv')
order_product_ds = pd.read_csv('../input/order_products__prior.csv')
product_ds = pd.read_csv('../input/products.csv')

order_product_cnt_ds = order_product_ds.groupby('order_id').count()[['product_id']]
order_product_cnt_ds.columns = ['product_cnt']

## join product count 
order_ds = raw_order_ds.merge(order_product_cnt_ds, left_on='order_id', right_index=True)


# In[3]:


total_user = len(order_ds.user_id.unique())
total_order = len(order_ds)
total_ordered_product = len(order_product_ds)
unique_products = len(order_product_ds.product_id.unique())

print("total user = {}".format(toReadable(total_user)))
print("total order = {} ({} orders per a user )".format(toReadable(total_order), toReadable(total_order/total_user) ))
print("total product = ", toReadable(unique_products))
print("total ordered product  = {} ({} orders per a product )".format(
    toReadable(total_ordered_product), toReadable(total_ordered_product/unique_products) ))


# In[4]:


index2day = "Dom Sab Vie Jue Mie Mar Lun".split()


# In[5]:


def drawWeekHour(ds, values,  aggfunc=len, title=None, figsize=(18,5) , cmap=None):
    weekhour_ds = ds.pivot_table(index='order_dow', columns='order_hour_of_day', values=values, aggfunc=aggfunc).fillna(0)
    weekhour_ds.index =  [  index2day[index] for index in weekhour_ds.index]
    sns.set(style="whitegrid", palette="colorblind", font_scale=1 )

    plt.figure(figsize=figsize)
    f = sns.heatmap(weekhour_ds, annot=True, fmt="1.1f", linewidths=.5, cmap=cmap) 
    plt.xlabel("Hora")
    plt.ylabel("Día de la semana")
    if title:
        plt.title(title, fontsize=15)


# In[6]:


drawWeekHour(order_ds, values='order_id', title="Total Order Frequency(unit:1k)", aggfunc=lambda x: len(x)/1000)


# In[7]:


avg_users = round(order_ds.groupby(['order_dow','order_hour_of_day']).agg({'user_id':lambda x: len(x.unique())/1000}).mean().values[0],2)
drawWeekHour(order_ds, values='user_id', title="Total Unique Users(unit:1k) / Avg Users= {}k".format(avg_users), aggfunc=lambda x: len(x.unique())/1000)


# In[8]:


drawWeekHour(order_ds, values='user_id', title="Total Unique Users(unit:1k)"
             , aggfunc=lambda x: len(x)/len(x.unique()))


# In[9]:


merge_order_product_ds = order_product_ds.merge(order_ds, on='order_id' )
merge_order_product_ds = merge_order_product_ds.merge(product_ds, on='product_id')


# In[10]:


hour_9_order_product_ds = merge_order_product_ds[merge_order_product_ds.order_hour_of_day==9]
grouped = hour_9_order_product_ds[:].groupby(['order_dow'])


# In[11]:


topn = 5
hour_9_popluar_product = []
for (dow,rows) in grouped:
    sub_ds = rows.groupby('product_id', as_index=False).agg({'order_id':len}).sort_values('order_id', ascending=False)[:topn]
    sub_ds['dow'] = dow
    sub_ds['rank'] = list(range(0,topn))
    hour_9_popluar_product.append(sub_ds)

# pd.options.display.max_rows=200
hour_9_popluar_product_ds = pd.concat(hour_9_popluar_product).sort_values(['rank','dow']).merge(product_ds, on='product_id').pivot(index='dow',columns='rank',values='product_name')
hour_9_popluar_product_ds.index = index2day


# In[12]:


hour_9_popluar_product_ds


# In[13]:


def topItemEachGroup(ds, group_name, sort_name, topn):
    concat_list = []
    for (key, rows) in ds.groupby(group_name):
        sub_ds = rows.sort_values(sort_name, ascending=False)[:topn]
        sub_ds['rank'] = list(range(1,topn+1))
        concat_list.append(sub_ds)

    return pd.concat(concat_list)


# In[14]:


def drawRankTrend(pivot_ds, ylabel='Rank'):
    sns.set(style="whitegrid", palette="colorblind", font_scale=1.3)

    index_max = pivot_ds.index.max()
    rank_max = pivot_ds.max().max()
    pivot_ds = pivot_ds.applymap(lambda x:rank_max-x+1)
    pivot_ds.plot(marker='o', figsize=(16,12), cmap='Dark2', xticks=pivot_ds.index, legend=None )
    
    plt.yticks(np.arange(rank_max,0,-1), np.arange(1,rank_max+1))
    for name, rank in pivot_ds.loc[index_max].sort_values(ascending=False).dropna().iteritems():
        plt.text(index_max*1.01,rank,name)
    plt.ylabel(ylabel)
    plt.show()
    


# In[15]:


hour_product_ds = merge_order_product_ds.groupby(['product_name','order_hour_of_day'], as_index=False).agg({'order_id':len})
hour_top_product_ds = topItemEachGroup(hour_product_ds, 'order_hour_of_day', 'order_id' , 20)
hour_top_product_pivot_ds = hour_top_product_ds.pivot(index='order_hour_of_day', columns='product_name', values='rank') 


# In[16]:


drawRankTrend(hour_top_product_pivot_ds)


# In[17]:


rank_ds = merge_order_product_ds.groupby(['product_name','order_dow'], as_index=False).agg({'order_id':len})
rank_ds = topItemEachGroup(rank_ds, 'order_dow', 'order_id' , 20)
rank_pivot_ds = rank_ds.pivot(index='order_dow', columns='product_name', values='rank') 


# In[18]:


drawRankTrend(rank_pivot_ds)


# In[19]:


drawWeekHour(order_ds, values='product_cnt', title="Product cnt per a order", aggfunc=lambda x: np.mean(x), cmap='YlGn')


# In[20]:


drawWeekHour(order_ds, values='days_since_prior_order', title="prior orders", aggfunc=lambda x: np.mean(x), cmap='YlGn')


# In[21]:


sns.set(style="whitegrid", palette="colorblind", font_scale=1.4, rc={'font.family':'NanumGothic'} )


# In[22]:


print("Avg days_since_prior_order {} Days".format( round(order_ds.days_since_prior_order.mean(),2)))


# In[23]:


order_ds.groupby('order_number').agg({'days_since_prior_order':np.mean, 'product_cnt':np.mean}).plot(figsize=(16,6), 
                                title="Order sequence # vs day_since_prior_order", marker='o')
plt.tight_layout()
plt.show()


# In[24]:


merge_order_product_ds = order_product_ds.merge(order_ds, on='order_id' )


# In[25]:


reordered_since_days_ds = merge_order_product_ds.groupby(['days_since_prior_order','reordered']).agg({'product_id':len})
reordered_since_days_ds = reordered_since_days_ds.reset_index().pivot(index='days_since_prior_order', columns='reordered', values='product_id')
reordered_since_days_ds['reorder_rate'] = reordered_since_days_ds[1] /reordered_since_days_ds.sum(axis=1)
avg_reordered_rate = round(reordered_since_days_ds[1].sum() / reordered_since_days_ds[[0,1]].sum().sum(),2)


# In[26]:


reordered_since_days_ds[['reorder_rate']].plot(kind='line', marker='o',figsize=(16,6))
plt.title("Reordered Rate (Avg {})".format(avg_reordered_rate), fontsize=20)
plt.tight_layout()
plt.show()


# In[27]:


reordered_order_num_ds = merge_order_product_ds.groupby(['order_number','reordered']).agg({'product_id':len})
reordered_order_num_ds = reordered_order_num_ds.reset_index().pivot(index='order_number', columns='reordered', values='product_id')
reordered_order_num_ds['reorder_rate'] = reordered_order_num_ds[1] /reordered_order_num_ds.sum(axis=1)
avg_reordered_rate = round(reordered_order_num_ds[1].sum() / reordered_order_num_ds[[0,1]].sum().sum(),2)
reordered_order_num_ds.fillna(0, inplace=True)


# In[28]:


reordered_order_num_ds[['reorder_rate']].plot(kind='line', marker='o',figsize=(16,6))
plt.title("Reordered Rate (Avg {})".format(avg_reordered_rate), fontsize=20)
plt.show()


# In[29]:


product_reorder_ds = merge_order_product_ds.groupby(['product_id']).agg({'order_id':len,
                                                                         'reordered':lambda x: len(x[x>0]),
                                                                         'user_id':lambda x: len(x.unique())})


# In[30]:


convert_colnames = {'user_id':'unique_users','reordered':'reorder' , 'order_id':'total_order'}
product_reorder_ds.columns = [  convert_colnames[col] for col in product_reorder_ds.columns]


# In[31]:


product_reorder_ds['reorder_rate'] = round(product_reorder_ds.reorder / product_reorder_ds.total_order,2)
product_reorder_ds['orders_per_user'] = round(product_reorder_ds.total_order/product_reorder_ds.unique_users,2)
product_reorder_ds = product_reorder_ds.merge(product_ds, left_index=True, right_on='product_id')


# In[32]:


product_reorder_ds[product_reorder_ds.total_order>1000].sort_values('reorder_rate', ascending=False)        [['product_name','total_order', 'reorder_rate', 'aisle_id','orders_per_user']][:20]


# In[33]:


# product_reorder_ds.groupby('aisle_id').agg({'product_name':                                           lambda x: })
from collections import defaultdict
import operator

def popularWords(names, topn=2):
    wordFrequency = defaultdict(int)
    def updateWords(words):
        for word in words :
            if len(word)>1:
                wordFrequency[word] += 1
    names.apply(lambda x: updateWords(x.split()))
    tops = sorted(wordFrequency.items(), key=operator.itemgetter(1),reverse=True)[:topn]
    return " ".join([n[0] for n in tops])


# In[34]:


aisle_ds = product_ds.groupby('aisle_id').agg({'product_name':popularWords
                                               , 'product_id':lambda x:len(x.unique())})
# aisle_ds.columns = ['products','product_names']


# In[35]:


aisle_order_stat_ds = product_reorder_ds.groupby('aisle_id').agg({'total_order':sum, 'reorder':sum})
aisle_order_stat_ds['reorder_rate'] = round(aisle_order_stat_ds.reorder / aisle_order_stat_ds.total_order, 2)
aisle_order_stat_ds = aisle_order_stat_ds.merge(aisle_ds, left_index=True, right_index=True).sort_values('reorder_rate', ascending=False)


# In[36]:


sns.set(style="whitegrid", palette="colorblind", font_scale=1.4, rc={'font.family':'NanumGothic'} )

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

top20_ds = aisle_order_stat_ds.set_index('product_name')[['reorder_rate']][:20]
top20_ds.plot(kind='bar', figsize=(16,6), alpha=.7, ax=ax1
             , title='Top 20 reorder rate Aisle (avg={})'.format(toReadable(top20_ds.reorder_rate.mean())))

bottom20_ds = aisle_order_stat_ds.set_index('product_name')[['reorder_rate']][-20:]
bottom20_ds.plot(kind='bar', figsize=(16,6), alpha=.7, ax=ax2
                , title='Bottom 20 reorder rate Aisle (avg={})'.format(toReadable(bottom20_ds.reorder_rate.mean())))
plt.show()


# In[37]:


from scipy.stats import spearmanr
g = sns.jointplot("reorder_rate", "total_order", kind="reg", marker='.', ylim=(0,100000), size=8, ratio=8
                  , stat_func=spearmanr
                  , data=product_reorder_ds)


# In[38]:


order_product_list = merge_order_product_ds    .sort_values(['user_id','order_id','add_to_cart_order'])[['order_id','product_id']]    .values.tolist()

product_corpus = []
sentence = []
new_order_id = order_product_list[0][0]
for (order_id, product_id) in order_product_list:
    if new_order_id != order_id:
        product_corpus.append(sentence)
        sentence = []
        new_order_id = order_id
    sentence.append(str(product_id))


# In[39]:


from gensim.models import Word2Vec

model100D = Word2Vec(product_corpus, window=6, size=100, workers=4, min_count=200)
# model100D.save('./resource/prod2vec.100d.model')
# model = Word2Vec.load('./resource/prod2vec.100d.model')


# In[40]:


def toProductName(id):
    return product_ds[product_ds.product_id==id]['product_name'].values.tolist()[0]
toProductName(24852)


# In[41]:


def most_similar_readable(model, product_id):
    similar_list = [(product_id,1.0)]+model.wv.most_similar(str(product_id))
    
    return [( toProductName(int(id)), similarity ) for (id,similarity) in similar_list]


# In[42]:


pd.DataFrame(most_similar_readable(model, 24852), columns=['product','similarity'])


# In[43]:


pd.DataFrame(most_similar_readable(model, 27845), columns=['product','similarity'])


# In[44]:


pd.DataFrame(most_similar_readable(model, 40939), columns=['product','similarity'])


# In[45]:


pd.DataFrame(most_similar_readable(model, 48697), columns=['product','similarity'])


# In[46]:


import kmeans


# In[47]:


def clustering(model, k=500, delta=0.00000001, maxiter=200):
    movie_vec = model.wv.syn0
    centres, index2cid, dist = kmeans.kmeanssample(movie_vec, k, 
                                                   metric = 'cosine', 
                                                   delta = delta, 
                                                   nsample = 0, maxiter = maxiter,)
    clustered_ds = pd.DataFrame( [ (a, b, c) for a, b, c in zip(model.wv.index2word, index2cid, dist )],
                 columns=['product_id', 'cid', 'dist'] ).sort_values(['cid','dist'], ascending=True)

    prod2cid = { product_id:cid for product_id,cid in zip(model.wv.index2word, index2cid) }

    return (centres, index2cid, dist, clustered_ds, prod2cid)


# In[48]:


(centres, index2cid, dist, clustered_ds, prod2cid) = clustering(model)


# In[49]:


clustered_ds.product_id = clustered_ds.product_id.apply(pd.to_numeric)


# In[50]:


def idToProductDesc(id):
    return product_ds[product_ds.product_id==id][['product_name','aisle_id']].values.tolist()[0]
    
def getProductNames(product_id_list):
    return [ idToProductDesc(int(product_id)) for  product_id in product_id_list ]

import urllib
def printClusterMembers(cluster_id, topn=10):
    members = getProductNames(clustered_ds[clustered_ds.cid==cluster_id].product_id[:topn].tolist())
    for member in members:
        print("{aisle} / {name}  https://www.google.co.kr/search?tbm=isch&q={q}".format( 
            aisle=member[1], name=member[0], q=urllib.parse.quote_plus(member[0]) ) 
        )


# In[51]:


printClusterMembers(1, topn=10)


# In[52]:


printClusterMembers(100, topn=10)


# In[53]:


printClusterMembers(200, topn=10)


# In[54]:


printClusterMembers(300, topn=10)


# In[55]:


printClusterMembers(400, topn=10)


# In[56]:


printClusterMembers(499, topn=10)


# In[57]:


clusterIdToKeywords = { cid: popularWords(sub_ds.product_name,3) for cid, sub_ds in clustered_ds.merge(product_ds, on='product_id').groupby('cid')}


# In[58]:


product_hod_ds = merge_order_product_ds.pivot_table(index='product_id', columns='order_hour_of_day', values='order_id', aggfunc=len, fill_value=0)

orderByHotHour = clustered_ds.merge(product_hod_ds, left_on='product_id', right_index=True)    .groupby('cid').sum()[np.arange(0,24)].idxmax(axis=1).sort_values().index


# In[59]:


sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )

def drawHODCluster(ncols, nrows, startClusterNumber, step):
    fig, axes = plt.subplots(ncols=ncols, nrows = nrows, figsize=(ncols*2.5,nrows*2), sharex=True, sharey=True)

    for cid, ax  in enumerate(axes.flatten()):
        cid = startClusterNumber + (cid*step)
        if cid>=500:
            break
        cid = orderByHotHour[cid]

        product_id_list = clustered_ds[clustered_ds.cid==cid].product_id.values
        tmp_ds = product_hod_ds.loc[product_id_list].T
        hot_hour = tmp_ds.sum(axis=1).argmax()
        normalized_ds =(tmp_ds/tmp_ds.max())
        title = "{cid}th {n} products \n({keyword})".format(cid=cid, n=normalized_ds.shape[1],  keyword=clusterIdToKeywords[cid][:23])
        normalized_ds.plot(linewidth=.3, legend=False, alpha=.4, ax=ax, title=title, color='r' if hot_hour<13 else 'k')
        ax.plot((hot_hour,hot_hour),(1,0), '-.', linewidth=1, color='b')
        ax.text(hot_hour,0,"{h}h(hot)".format(h=hot_hour),color='b')

    fig.tight_layout()


# In[60]:


ncols, nrows=(6,4)
step = 3
for n in np.arange(0,500,ncols*nrows*step):
    drawHODCluster(ncols, nrows, n, step)


# In[61]:


product_dow_ds = merge_order_product_ds.pivot_table(index='product_id', columns='order_dow', values='order_id', aggfunc=len, fill_value=0)

orderByHotDay = clustered_ds.merge(product_dow_ds, left_on='product_id', right_index=True)    .groupby('cid').sum()[np.arange(0,6)].idxmax(axis=1).sort_values().index


# In[62]:


def drawDOWCluster(ncols, nrows, startClusterNumber, step):
    sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )
    week_day = "Sun Mon Tue Wed Thu Fri Sat".split()
    fig, axes = plt.subplots(ncols=ncols, nrows = nrows, figsize=(ncols*2.5,nrows*2), sharex=True, sharey=True)

    for cid, ax  in enumerate(axes.flatten()):
        cid = startClusterNumber + (cid*step)
        if cid>=500:
            break
        cid = orderByHotDay[cid]    
        product_id_list = clustered_ds[clustered_ds.cid==cid].product_id.values
        tmp_ds = product_dow_ds.loc[product_id_list].T
        hot_day = tmp_ds.sum(axis=1).argmax()
        normalized_ds =(tmp_ds/tmp_ds.max())
        normalized_ds.index = week_day
        title = "{cid}th \n({keyword})".format(cid=cid, h=hot_day,  keyword=clusterIdToKeywords[cid][:23])
        normalized_ds.plot(kind='bar', linewidth=.1, legend=False, alpha=.4, ax=ax, title=title, color='r' if hot_day in(0,6) else 'k')
        ax.plot((hot_day,hot_day),(1,0), '-.', linewidth=2, color='b')
        # ax.text(hot_day+.3,-.5,"{h}".format(h=week_day[hot_day]),color='b')
    
    fig.tight_layout()


# In[63]:


ncols, nrows=(6,4)
step = 3
for n in np.arange(0,500,ncols*nrows*step):
    drawDOWCluster(ncols, nrows, n, step)


# In[64]:




