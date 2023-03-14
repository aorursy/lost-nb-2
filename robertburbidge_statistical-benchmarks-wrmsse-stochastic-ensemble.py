#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import gc
import os
from tqdm import tqdm_notebook as tqdm
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import random
from statsmodels.tsa.api import SimpleExpSmoothing

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


def read_data():
    print('Reading files...')
    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    
    sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    
    sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0], sales_train_val.shape[1]))
    
    submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
    
    return calendar, sell_prices, sales_train_val, submission


# In[4]:


import IPython

def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


# In[5]:


calendar, sell_prices, sales_train_val, submission = read_data()


# In[6]:


# 予測期間とitem数の定義
NUM_ITEMS = sales_train_val.shape[0]  # 30490
DAYS_PRED = submission.shape[1] - 1  # 28


# In[7]:


def explanatory_variables(df):
    
    df['ex1'] = df['snap_CA'] + df['snap_TX'] + df['snap_WI']
    df['ex2'] = 1 * (pd.notnull(df['event_name_1']) | pd.notnull(df['event_name_2']))

    return df


calendar = explanatory_variables(calendar).pipe(reduce_mem_usage)


# In[8]:


calendar.head()


# In[9]:


nrows = 365 * 2 * NUM_ITEMS


# In[10]:


#加工前  
display(sales_train_val.head(5))


# In[11]:


sales_train_val = pd.melt(sales_train_val,
                                     id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                     var_name = 'day', value_name = 'demand')


# In[12]:


#加工後  
display(sales_train_val.head(5))
print('Melted sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0],
                                                                            sales_train_val.shape[1]))


# In[13]:


sales_train_val = sales_train_val.iloc[-nrows:,:]


# In[14]:


# seperate test dataframes

# submission fileのidのvalidation部分と, ealuation部分の名前を取得
test1_rows = [row for row in submission['id'] if 'validation' in row]
test2_rows = [row for row in submission['id'] if 'evaluation' in row]

# submission fileのvalidation部分をtest1, ealuation部分をtest2として取得
test1 = submission[submission['id'].isin(test1_rows)]
test2 = submission[submission['id'].isin(test2_rows)]

# test1, test2の列名の"F_X"の箇所をd_XXX"の形式に変更
test1.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
test2.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

# test2のidの'_evaluation'を置換
#test1['id'] = test1['id'].str.replace('_validation','')
test2['id'] = test2['id'].str.replace('_evaluation','_validation')

# sales_train_valからidの詳細部分(itemやdepartmentなどのid)を重複なく一意に取得。
product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

# idをキーにして, idの詳細部分をtest1, test2に結合する.
test1 = test1.merge(product, how = 'left', on = 'id')
test2 = test2.merge(product, how = 'left', on = 'id')

# test1, test2をともにmelt処理する.（売上数量:demandは0）
test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                var_name = 'day', value_name = 'demand')

test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                var_name = 'day', value_name = 'demand')

# validation部分と, evaluation部分がわかるようにpartという列を作り、 test1,test2のラベルを付ける。
sales_train_val['part'] = 'train'
test1['part'] = 'test1'
test2['part'] = 'test2'

# sales_train_valとtest1, test2の縦結合.
data = pd.concat([sales_train_val, test1, test2], axis = 0)

# memoryの開放
del test1, test2, sales_train_val

# delete test2 for now(6/1以前は, validation部分のみ提出のため.)
data = data[data['part'] != 'test2']

gc.collect()


# In[15]:


#calendarの結合
# drop some calendar features(不要な変数の削除:weekdayやwdayなどはdatetime変数から後ほど作成できる。)
calendar.drop(['weekday', 'wday', 'month', 'year'], 
              inplace = True, axis = 1)

# notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)(dayとdをキーにdataに結合)
data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data.drop(['d', 'day'], inplace = True, axis = 1)

# memoryの開放
del calendar
gc.collect()

#sell priceの結合
# get the sell price data (this feature should be very important)
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

# memoryの開放
del sell_prices
gc.collect()


# In[16]:


data.head(3)


# In[17]:


display(data.head())


# In[18]:


data = data[['date', 'demand', 'id', 'ex1', 'ex2', 'sell_price']]

# going to evaluate with the last 28 days
x_train = data[data['date'] <= '2016-03-27']
y_train = x_train['demand']
x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
y_val = x_val['demand']
test = data[(data['date'] > '2016-04-24')]

#dataの削除（メモリの削除）
#del data
#gc.collect()


# In[19]:


weight_mat = np.c_[np.identity(NUM_ITEMS).astype(np.int8), #item :level 12
                   np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                   pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values
                   ].T

weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; gc.collect()

def weight_calc(data,product):

    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    
    sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
    
    d_name = ['d_' + str(i+1) for i in range(1913)]
    
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    
    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
    
    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    
    
    # denominator of RMSSE / RMSSEの分母
    weight1 = np.sum((np.diff(sales_train_val,axis=1)**2),axis=1)/(1913-start_no)
    
    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum).values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)
    
    del sales_train_val
    gc.collect()
    
    return weight1, weight2


weight1, weight2 = weight_calc(data,product)

def wrmsse(preds, data):
    
    preds = preds.astype('int32')
    
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()
    y_true = y_true.astype('int32')
    
    # number of columns
    num_col = len(y_true)//NUM_ITEMS
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = np.array(preds).reshape(num_col, NUM_ITEMS).T
    reshaped_true = np.array(y_true).reshape(num_col, NUM_ITEMS).T
    
    x_name = ['pred_' + str(i) for i in range(num_col)]
    x_name2 = ["act_" + str(i) for i in range(num_col)]
          
    train = np.array(weight_mat_csr*np.c_[reshaped_preds, reshaped_true])
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False

def wrmsse_simple(preds, data):
    
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()
    
    # number of columns
    num_col = len(y_true)//NUM_ITEMS
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = np.array(preds).reshape(num_col, NUM_ITEMS).T
    reshaped_true = np.array(y_true).reshape(num_col, NUM_ITEMS).T
    
    train = np.c_[reshaped_preds, reshaped_true]
    
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2/np.sum(weight2_2)
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) /  weight1[:NUM_ITEMS])*weight2_2)
    
    return 'wrmsse', score, False


# In[20]:


# useful dataset for speeding up the following calculations
uids = pd.concat([pd.Series(np.arange(30490)), pd.Series(x_train['id'].unique())], axis=1)
uids.columns = ['idx', 'id']
demand = x_train.merge(uids, on='id').sort_values(by=['idx', 'date'], axis=0)['demand']
num_vals = int(len(demand)/30490)


# In[21]:


# helper function
def intervals(ts):
    y = np.zeros(len(ts))
    k = 0
    counter = 0
    for tmp in range(len(ts)):
        if(ts[tmp]==0):
            counter = counter + 1
        else:
            k = k + 1
            y[k] = counter
            counter = 1
    y = np.array(y)
    y[np.isnan(y)] = 1
    y = y[y > 0]
    return y


# In[22]:


# Simple Exponential Smoothing
def SES(ts, alpha=0.1, h=56):
    if ts is None or len(ts)==0:
        return np.zeros(h)
    
    y = np.zeros(len(ts)+1)
    y[0] = ts[0]
        
    for t in range(len(ts)):
        y[t+1] = alpha*ts[t]+(1-alpha)*y[t]
  
    return np.concatenate([y[0:len(ts)], np.repeat(y[-1:],h)])


# In[23]:


# 1. Naive (LB: 1.46378)
y_pred_Naive = np.tile(x_train[x_train['date'] == '2016-03-27']['demand'], 28)
y_test_Naive = np.tile(x_train[x_train['date'] == '2016-03-27']['demand'], 28)
wrmsse_Naive = wrmsse(y_pred_Naive, lgb.Dataset(x_val, y_val))[1]
print(f'WRMSSE for Naive method: {wrmsse_Naive}')
del y_pred_Naive


# In[24]:


# 2. Seasonal Naive (LB: 0.86967)
y_pred_sNaive = np.tile(x_train[(x_train['date'] > '2016-03-20') & (x_train['date'] <= '2016-03-27')]['demand'], 4)
y_test_sNaive = np.tile(x_train[(x_train['date'] > '2016-03-20') & (x_train['date'] <= '2016-03-27')]['demand'], 4)
wrmsse_sNaive = wrmsse(y_pred_sNaive, lgb.Dataset(x_val, y_val))[1]
print(f'WRMSSE for Seasonal Naive method: {wrmsse_sNaive}')
del y_pred_sNaive


# In[25]:


# 3. Simple Exponential Smoothing (LB: 1.07202)
def mySES(ts):
    if ts is None or len(ts)==0:
        ts = np.asarray([0, 0])
    
    start_period = np.argmin(ts!=0)
    ts = ts[start_period:]
    MSE = {}
    for alpha in np.arange(0.1,0.3,0.1):
        MSE[str(alpha)] = np.mean(np.square(SES(ts, alpha)[0:len(ts)] - ts))
    opt_alpha = float(list(MSE.keys())[np.argmin(MSE.values())])
    return SES(ts, opt_alpha)[-56:]
    

y_pred_SES = Parallel(n_jobs=-1)(delayed(mySES)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_SES = np.concatenate(y_pred_SES)
y_pred_SES = y_pred_SES.reshape([56, 30490], order='F')
y_test_SES = y_pred_SES[28:56,:].reshape([28*30490,1])
y_pred_SES = y_pred_SES[0:28,:].reshape([28*30490,1])
y_test_SES = np.concatenate(y_test_SES)
y_pred_SES = np.concatenate(y_pred_SES)
wrmsse_SES = wrmsse(y_pred_SES, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for SES method: {wrmsse_SES}")
del y_pred_SES


# In[26]:


# 4. Moving Averages (LB: 1.09815)
def MA(ts, h=56):
    mse = np.ones(14)*np.inf
    for k in np.arange(2,16):
        y = np.repeat(np.nan, len(ts))
        for i in np.arange(k+1,len(ts)+1):
            y[i-1] = np.mean(ts[(i-k-1):i])
        mse[k-2] = np.mean(np.square(y[~np.isnan(y)]-ts[~np.isnan(y)]))
        k = np.argmin(mse)+2
    forecast = np.repeat(np.mean(ts[-k:]), h)
    return forecast


y_pred_MA = Parallel(n_jobs=-1)(delayed(MA)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_MA = np.concatenate(y_pred_MA)
y_pred_MA = y_pred_MA.reshape([56, 30490], order='F')
y_test_MA = y_pred_MA[28:56,:].reshape([28*30490,1])
y_pred_MA = y_pred_MA[0:28,:].reshape([28*30490,1])
y_test_MA = np.concatenate(y_test_MA)
y_pred_MA = np.concatenate(y_pred_MA)
wrmsse_MA = wrmsse(y_pred_MA, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for MA method: {wrmsse_MA}")
del y_pred_MA


# In[27]:


# 5. Croston's (LB: 1.05648)
def Croston(ts, h=56, alpha=0.1, debias=1.0):
    yd = np.mean(SES(ts[ts!=0])[-56:])
    yi = np.mean(SES(intervals(ts))[-56:])
    return np.repeat(yd/yi, h)*debias


y_pred_Croston = Parallel(n_jobs=-1)(delayed(Croston)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_Croston = np.concatenate(y_pred_Croston)
y_pred_Croston = y_pred_Croston.reshape([56, 30490], order='F')
y_test_Croston = y_pred_Croston[28:56,:].reshape([28*30490,1])
y_pred_Croston = y_pred_Croston[0:28,:].reshape([28*30490,1])
y_test_Croston = np.concatenate(y_test_Croston)
y_pred_Croston = np.concatenate(y_pred_Croston)
wrmsse_Croston = wrmsse(y_pred_Croston, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for Croston's method: {wrmsse_Croston}")
del y_pred_Croston


# In[28]:


# 6. Optimized Croston's (LB: 1.05804)
def optCroston(ts, h=56, debias=1.0):
    yd = np.mean(mySES(ts[ts!=0])[-56:])
    yi = np.mean(mySES(intervals(ts))[-56:])
    return np.repeat(yd/yi, h)*debias


y_pred_optCroston = Parallel(n_jobs=-1)(delayed(optCroston)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_optCroston = np.concatenate(y_pred_optCroston)
y_pred_optCroston = y_pred_optCroston.reshape([56, 30490], order='F')
y_test_optCroston = y_pred_optCroston[28:56,:].reshape([28*30490,1])
y_pred_optCroston = y_pred_optCroston[0:28,:].reshape([28*30490,1])
y_test_optCroston = np.concatenate(y_test_optCroston)
y_pred_optCroston = np.concatenate(y_pred_optCroston)
wrmsse_optCroston = wrmsse(y_pred_optCroston, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for Optimized Croston's method: {wrmsse_optCroston}")
del y_pred_optCroston


# In[29]:


# 7. Syntetos-Boylan Approximation (LB: 1.09166)
y_pred_SBA = Parallel(n_jobs=-1)(delayed(lambda x: Croston(x, debias=0.95))(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_SBA = np.concatenate(y_pred_SBA)
y_pred_SBA = y_pred_SBA.reshape([56, 30490], order='F')
y_test_SBA = y_pred_SBA[28:56,:].reshape([28*30490,1])
y_pred_SBA = y_pred_SBA[0:28,:].reshape([28*30490,1])
y_test_SBA = np.concatenate(y_test_SBA)
y_pred_SBA = np.concatenate(y_pred_SBA)
wrmsse_SBA = wrmsse(y_pred_SBA, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for SBA method: {wrmsse_SBA}")
del y_pred_SBA


# In[30]:


# 8. Teunter-Syntetos-Babai method (LB: 1.06812)
# This should be optimized over the smoothing parameters by grid search
def TSB(ts,extra_periods=56,alpha=0.4,beta=0.4):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    
    #level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]
                 
    # Create all the t+1 forecasts
    for t in range(0,cols): 
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = beta*(1) + (1-beta)*p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]       
        f[t+1] = p[t+1]*a[t+1]
        
    # Future Forecast
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
                      
    #df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return f[-extra_periods:]


y_pred_TSB = Parallel(n_jobs=-1)(delayed(TSB)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_TSB = np.concatenate(y_pred_TSB)
y_pred_TSB = y_pred_TSB.reshape([56, 30490], order='F')
y_test_TSB = y_pred_TSB[28:56,:].reshape([28*30490,1])
y_pred_TSB = y_pred_TSB[0:28,:].reshape([28*30490,1])
y_test_TSB = np.concatenate(y_test_TSB)
y_pred_TSB = np.concatenate(y_pred_TSB)
wrmsse_TSB = wrmsse(y_pred_TSB, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for TSB method: {wrmsse_TSB}")
del y_pred_TSB


# In[31]:


# 9. Aggregate-Disaggregate Intermittent Demand Approach (LB: 1.07268)
def ADIDA(ts, h=56):
    a1 = np.ceil(np.mean(intervals(ts)))
    idx = [0 if (np.isnan(a1)) or (a1==0) else -int((len(ts) // int(a1)) * a1)][0]
    if np.isnan(a1):
        a1 = 1
    agg_ser = pd.Series(ts[idx:]).rolling(int(a1)).sum()
    agg_ser = agg_ser[np.arange(0, len(agg_ser), int(a1))] # non-overlapping
    agg_ser[np.isnan(agg_ser)] = np.mean(agg_ser)
    forecast = mySES(agg_ser.values)[-56:]/a1
    return forecast


y_pred_ADIDA = Parallel(n_jobs=-1)(delayed(ADIDA)(
        demand[i*num_vals:(i+1)*num_vals].values
    ) for i in tqdm(range(30490)))
y_pred_ADIDA = np.concatenate(y_pred_ADIDA)
y_pred_ADIDA = y_pred_ADIDA.reshape([56, 30490], order='F')
y_test_ADIDA = y_pred_ADIDA[28:56,:].reshape([28*30490,1])
y_pred_ADIDA = y_pred_ADIDA[0:28,:].reshape([28*30490,1])
y_test_ADIDA = np.concatenate(y_test_ADIDA)
y_pred_ADIDA = np.concatenate(y_pred_ADIDA)
wrmsse_ADIDA = wrmsse(y_pred_ADIDA, lgb.Dataset(x_val, y_val))[1]
print(f"WRMSSE for ADIDA method: {wrmsse_ADIDA}")
del y_pred_ADIDA


# In[32]:


# stochastic ensemble
del demand, x_train, y_train, x_val, y_val 
gc.collect()

probs = 1.0/np.array([wrmsse_Naive, wrmsse_sNaive, wrmsse_SES, wrmsse_MA, 
                    wrmsse_Croston, wrmsse_optCroston, wrmsse_SBA, wrmsse_TSB, wrmsse_ADIDA]) / \
    sum(1.0/np.array([wrmsse_Naive, wrmsse_sNaive, wrmsse_SES, wrmsse_MA, 
                    wrmsse_Croston, wrmsse_optCroston, wrmsse_SBA, wrmsse_TSB, wrmsse_ADIDA]))
probs = np.cumsum(probs)

rs = [random.random() for i in range(30490*28)]
def f(rs): 
    return np.argmax(rs<probs)
rcols = np.vectorize(f)(rs)

test['demand'] = np.select([rcols==0, rcols==1, rcols==2, rcols==3, rcols==4, rcols==5, rcols==6, rcols==7, rcols==8], 
         [y_test_Naive, y_test_sNaive, y_test_SES, y_test_MA, y_test_Croston, y_test_optCroston, y_test_SBA, y_test_TSB, y_test_ADIDA])
test.fillna(0, inplace=True)

predictions = test[['id', 'date', 'demand']]
predictions = predictions.pivot(index = 'id', columns = 'date', values = 'demand').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
evaluation = submission[submission['id'].isin(evaluation_rows)]

validation = submission[['id']].merge(predictions, on = 'id')
final = pd.concat([validation, evaluation])
final.to_csv('submission.csv', index = False)

