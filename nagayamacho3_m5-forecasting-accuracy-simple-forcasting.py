#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')


# In[2]:


# カレンダーデータ
df_cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
# 製品および店舗ごとの過去の毎日の販売台数データ[d_1 - d_1941]（パブリックリーダーボードに使用されるラベル）
df_eval = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
# 製品および店舗ごとの過去の毎日の販売台数データ[d_1 - d_1913]
# df_val = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
# 店舗および日付ごとに販売された製品の価格に関する情報が含まれています。
df_price = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
# サンプルアウトプット
df_sample_output = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')


# In[3]:


df_sample_output.head()


# In[4]:


df_sample_output.describe().T


# In[5]:


df_cal.head()


# In[6]:


# 宗教別で祝日などが異なるため、きれいな特徴量にはならなそう→後ほどチューニング?
holiday = ['NewYear', 'OrthodoxChristmas', 'MartinLutherKingDay', 'SuperBowl', 'PresidentsDay', 'StPatricksDay', 'Easter', 'Cinco De Mayo', 'IndependenceDay', 'EidAlAdha', 'Thanksgiving', 'Christmas']
weekend = ['Saturday', 'Sunday']

def is_holiday(x):
    if x in holiday:
        return 1
    else:
        return 0

def is_weekend(x):
    if x in weekend:
        return 1
    else:
        return 0


# In[7]:


df_cal['is_holiday_1'] = df_cal['event_name_1'].apply(is_holiday)
df_cal['is_holiday_2'] = df_cal['event_name_2'].apply(is_holiday)
df_cal['is_holiday'] = df_cal[['is_holiday_1','is_holiday_2']].max(axis=1)
df_cal['is_weekend'] = df_cal['weekday'].apply(is_weekend)


# In[8]:


df_cal.head()


# In[9]:


df_cal = df_cal.drop(['weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis='columns')


# In[10]:


df_price.head()


# In[11]:


df_price.describe()


# In[12]:


df_eval.head()


# In[13]:


del_col = []
for x in range(1851):
    del_col.append('d_' + str(x+1))


# In[14]:


df_eval = df_eval.drop(del_col, axis='columns')


# In[15]:


df_eval = df_eval.melt(['id','item_id','dept_id','cat_id','store_id','state_id'], var_name='d', value_name='qty')
print(df_eval.shape)
df_eval.head()


# In[16]:


df_eval = pd.merge(df_eval, df_cal, how='left', on='d')
df_eval.head()


# In[17]:


df_eval = pd.merge(df_eval, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])
df_eval.head()


# In[18]:


df_eval.shape


# In[19]:


df_eval.tail()


# In[20]:


df_eval.head()


# In[21]:


df_eval.tail()


# In[22]:


df_eval_test = df_eval.query('d == "d_1852"')


# In[23]:


df_eval_test.head()


# In[24]:


df_eval_test = df_eval_test[['id', 'store_id', 'item_id', 'dept_id', 'cat_id', 'state_id', 'd', 'qty', 'sell_price']]


# In[25]:


df_eval_test.head()


# In[26]:


df_eval_test.shape


# In[27]:


df_eval_test['qty'] = df_eval_test['d'].apply(lambda x: int(x.replace(x, '0')))


# In[28]:


tmp_df = df_eval_test


# In[29]:


for x in range(28):
    df_eval_test = df_eval_test.append(tmp_df)


# In[30]:


df_eval_test = df_eval_test.reset_index(drop=True)


# In[31]:


df_eval_test.head()


# In[32]:


df_eval_test.tail()


# In[33]:


# ※ここに日付を直す処理を入れる、今はすべて同じ※
lst_d = []
i = 0
lst_index = df_eval_test.index
for x in lst_index:
    lst_d.append('d_' + str(((lst_index[i]) // 30490) + 1942))
    i = i + 1

lst_d


# In[34]:


df_eval_test['d'] = lst_d


# In[35]:


df_eval_test.head()


# In[36]:


df_eval_test.tail()


# In[37]:


df_eval_test.shape


# In[38]:


df_eval_test = pd.merge(df_eval_test, df_cal, how='left', on='d')


# In[39]:


df_eval_test = pd.merge(df_eval_test, df_price, how='left', on=['item_id', 'wm_yr_wk', 'store_id'])


# In[40]:


df_eval_test.head()


# In[41]:


import gc
del tmp_df
gc.collect()


# In[42]:


df_eval = pd.get_dummies(data=df_eval, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])
df_eval_test = pd.get_dummies(data=df_eval_test, columns=['dept_id', 'cat_id', 'store_id', 'state_id'])


# In[43]:


df_eval.info()


# In[44]:


df_eval_test.info()


# In[45]:


df_eval_test.head(10).T


# In[46]:


df_eval_test = df_eval_test.drop(['sell_price_x', 'snap_CA', 'snap_TX', 'snap_WI'], axis='columns')
df_eval_test = df_eval_test.rename(columns={'sell_price_y': 'sell_price'})
df_eval = df_eval.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis='columns')                                   


# In[47]:


df_eval.info()


# In[48]:


df_eval_test.info()


# In[49]:


from sklearn.model_selection import train_test_split

# 目的変数
target_col = 'qty'

# 除外する説明変数
exclude_cols = ['id', 'item_id', 'd', 'date', 'wm_yr_wk']

# 説明変数
feature_cols = [col for col in df_eval.columns if col not in exclude_cols]

# ndarrayに変換
y = np.array(df_eval[target_col])
X = np.array(df_eval[feature_cols])

# 学習データとテストデータに分割
# ramdom_state 固定で再現性の高い結果にする
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=1234)

# 学習データを更に分割
# X_train1, X_train2, y_train1, y_train2 = \
#  train_test_split(X_train, y_train, test_size=0.3, random_state=1234)


# In[50]:


import lightgbm as lgb

#LGB用のデータに変形
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'n_jobs': -1,
    'seed': 236,
    'learning_rate': 0.01,
    'bagging_fraction': 0.75,
    'bagging_freq': 10, 
    'colsample_bytree': 0.75}

model = lgb.train(params, lgb_train, num_boost_round=2500, early_stopping_rounds=50, valid_sets = [lgb_train, lgb_eval], verbose_eval=100)


# In[51]:


pred = model.predict(df_eval_test[feature_cols])


# In[52]:


pred


# In[53]:


len(pred)


# In[54]:


df_eval_test['pred_qty'] = pred


# In[55]:


df_eval_test


# In[56]:


predictions = df_eval_test[['id', 'date', 'pred_qty']]
predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'pred_qty').reset_index()
predictions


# In[57]:


predictions.describe()


# In[58]:


predictions = predictions.drop(predictions.columns[1], axis=1)
predictions


# In[59]:


predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
predictions


# In[60]:


x = 2744099 + 1 - 853720
df_val = df_eval[x:]


# In[61]:


predictions_v = df_val[['id', 'date', 'qty']]
predictions_v = pd.pivot(predictions_v, index = 'id', columns = 'date', values = 'qty').reset_index()
predictions_v


# In[62]:


predictions_v['id'] = predictions['id'].apply(lambda x: x.replace('evaluation', 'validation'))
predictions_v.head()


# In[63]:


predictions_v.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
predictions_v.head()


# In[64]:


predictions_concat = pd.concat([predictions, predictions_v], axis=0)


# In[65]:


predictions_concat


# In[66]:


predictions_concat.to_csv('submission.csv', index=False)

