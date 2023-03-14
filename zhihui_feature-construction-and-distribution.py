#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')
df_members = pd.read_csv('../input/members.csv')
df_transactions = pd.read_csv('../input/transactions.csv')
df_sample = pd.read_csv('../input/sample_submission_zero.csv')
# Any results you write to the current directory are saved as output.


# In[2]:


df_user_logs_1 = pd.read_csv('../input/user_logs.csv', nrows = 1e5)
df_user_logs_2 = pd.read_csv('../input/user_logs.csv', skiprows = int(1e7), nrows = 1e5)
df_user_logs_3 = pd.read_csv('../input/user_logs.csv', skiprows = int(5e7), nrows = 1e5)


# In[3]:


df_train.head()


# In[4]:


df_train.info()


# In[5]:


# examine if there are duplicates for msno
df_train.msno.unique().shape, df_train.shape


# In[6]:


df_train_test_merge = df_train.merge(df_sample, on = 'msno', how = 'outer')
df_train_test_merge.head()


# In[7]:


df_train_test_merge.info()


# In[8]:


df_com = df_train_test_merge[(~pd.isnull(df_train_test_merge['is_churn_y']))&(~pd.isnull(df_train_test_merge['is_churn_x']))]
df_com.info()


# In[9]:


print (df_com.shape)
df_com


# In[10]:


print ('Percentage of is_churn in test dataset also labeled in train dataset: {:.2f}'.format(df_com.shape[0]*100.0/df_sample.shape[0]))


# In[11]:


df_com['is_churn_x'].plot(kind='hist')


# In[12]:


df_train['is_churn'].plot(kind='hist')


# In[13]:


sum(df_train['is_churn'] == 0), sum(df_train['is_churn'] == 1)


# In[14]:


df_members.head()


# In[15]:


df_members.info()


# In[16]:


df_members.describe()


# In[17]:


df_members['reg_year'] = df_members['registration_init_time'].astype(str).apply(lambda x : int(x[:4]))
df_members['reg_month'] = df_members['registration_init_time'].astype(str).apply(lambda x : int(x[4:6]))
df_members['reg_day'] = df_members['registration_init_time'].astype(str).apply(lambda x : int(x[6:]))
df_members['exp_year'] = df_members['expiration_date'].astype(str).apply(lambda x : int(x[:4]))
df_members['exp_month'] = df_members['expiration_date'].astype(str).apply(lambda x : int(x[4:6]))
df_members['exp_day'] = df_members['expiration_date'].astype(str).apply(lambda x : int(x[6:]))


# In[18]:


df_members['exp_reg'] = (df_members['exp_year'] - df_members['reg_year'])*365                             + (df_members['exp_month'] - df_members['reg_month'])*30 +                                 (df_members['exp_day'] - df_members['reg_day'])


# In[19]:


# separate out 10% data only present in train file but not in sample submission
df_train_only = df_train_test_merge[pd.isnull(df_train_test_merge['is_churn_y'])]


# In[20]:


df_train_only = df_train_only.iloc[:, [0,1]]
df_train_only.columns = ['msno', 'is_churn']
df_train_only.head()


# In[21]:


gender_dict = {np.nan:0, 'female': 1, 'male':2}
df_members['gender'] = df_members['gender'].apply(lambda x : gender_dict[x])


# In[22]:


df_members_train = df_train_only.merge(df_members, on = 'msno', how = 'left')


# In[23]:


df_members_test = df_sample.merge(df_members, on = 'msno', how = 'left')
del df_members


# In[24]:


df_members_merge = df_members_train.merge(df_members_test, on = 'msno', how ='outer')


# In[25]:


df_members_train_not_churn = df_members_train[df_members_train['is_churn'] == 0]
df_members_train_churn = df_members_train[df_members_train['is_churn'] == 1]


# In[26]:


import matplotlib.pyplot as plt
def dis_1(df1, df2, cols = None):
    '''
    generate distribution plots for churn and not churn in train
    '''
    if cols:
        for col in cols:
            plt.figure()
            df1[col].plot(kind='hist', bins = 200, logy=True, legend = True, label = 'Not Churn', figsize=(10, 4))
            df2[col].plot(kind='hist', bins = 200, logy=True, legend = True, label = 'Churn', figsize=(10, 4))
            plt.title('Distribubtion of {} in train'.format(col))


# In[27]:


members_cols = ['city', 'bd', 'gender', 'registered_via', 'reg_year', 'reg_month', 'reg_day', 
               'exp_year', 'exp_month', 'exp_day', 'exp_reg']
dis_1(df_members_train_not_churn, df_members_train_churn, members_cols)


# In[28]:


def dis_2(df, cols = None):
    '''
    generate distribution plots in train and test
    '''
    if cols:
        for col in cols:
            plt.figure()
            df[[col + '_x']].plot(kind='hist', bins=100, logy=True, legend = True, label = 'Train', figsize=(10, 4))
            df[[col + '_y']].plot(kind='hist', bins=100, logy=True, legend = True, label = 'Test', figsize=(10, 4))
            plt.title('Distribubtion of {} in train and test'.format(col))


# In[29]:


dis_2(df_members_merge, members_cols)


# In[30]:


df_transactions.info()


# In[31]:


df_transactions['trans_year'] = df_transactions['transaction_date'].astype(str).apply(lambda x : int(x[:4]))
df_transactions['trans_month'] = df_transactions['transaction_date'].astype(str).apply(lambda x : int(x[4:6]))
df_transactions['trans_day'] = df_transactions['transaction_date'].astype(str).apply(lambda x : int(x[6:]))
df_transactions['mem_exp_year'] = df_transactions['membership_expire_date'].astype(str).apply(lambda x : int(x[:4]))
df_transactions['mem_exp_month'] = df_transactions['membership_expire_date'].astype(str).apply(lambda x : int(x[4:6]))
df_transactions['mem_exp_day'] = df_transactions['membership_expire_date'].astype(str).apply(lambda x : int(x[6:]))


# In[32]:


df_transactions['exp_trans'] = (df_transactions['mem_exp_year'] - df_transactions['trans_year'])*365                             + (df_transactions['mem_exp_month'] - df_transactions['trans_month'])*30 +                                 (df_transactions['mem_exp_day'] - df_transactions['trans_day'])


# In[33]:


df_transactions['discount'] = df_transactions['actual_amount_paid'] - df_transactions['plan_list_price']


# In[34]:


df_transactions['price_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']
df_transactions.replace(np.inf, 0, inplace = True)
df_transactions.fillna(0, inplace=True)


# In[35]:


df_transactions.head()


# In[36]:


# get mean for payment_plan_days, exp_trans, discount plan_list_price, 
# actual_amount_paid and price_per_day
trans_mean_cols = ['msno', 'payment_plan_days', 'exp_trans', 'discount', 
             'plan_list_price', 'actual_amount_paid', 'price_per_day']
df_transactions_mean = df_transactions[trans_mean_cols].groupby(['msno']).mean().reset_index()
print(df_transactions_mean.shape)
df_transactions_mean.head()


# In[37]:


# Counts for is_auto_renew and is_cancel
trans_count_cols = ['msno', 'is_auto_renew', 'is_cancel']
df_transactions_count = df_transactions[trans_count_cols].groupby(['msno']).sum().reset_index()
print(df_transactions_count.shape)
df_transactions_count.head()


# In[38]:


# frequency for payment_method_id, trans_year, trans_month, trans_day, mem_exp_year, 
# mem_exp_month, mem_exp_day
trans_count_freq = ['msno', 'trans_year', 'trans_month', 'trans_day', 'mem_exp_year', 
                    'mem_exp_month', 'mem_exp_day']
df_transactions_freq = df_transactions[trans_count_freq].groupby(['msno']).count().reset_index()
del df_transactions
df_transactions_freq.head()


# In[39]:


df_transactions_proc = df_transactions_mean.merge(df_transactions_count, on = 'msno', 
                                                  how = 'outer').merge(df_transactions_freq, 
                                                                       on = 'msno', how = 'outer')
df_transactions_proc.replace(np.inf, 0, inplace = True)
df_transactions_proc.fillna(0, inplace=True)


# In[40]:


df_transactions_train = df_train_only.merge(df_transactions_proc, on = 'msno', how = 'left')
df_transactions_test = df_sample.merge(df_transactions_proc, on = 'msno', how = 'left')
df_transactions_merge = df_transactions_train.merge(df_transactions_test, on = 'msno', how = 'outer')
df_trans_train_not_churn = df_transactions_train[df_transactions_train['is_churn'] == 0]
df_trans_train_churn = df_transactions_train[df_transactions_train['is_churn'] == 1]
del df_transactions_proc


# In[41]:


trans_cols = ['payment_plan_days', 'exp_trans', 'discount', 
             'plan_list_price', 'actual_amount_paid', 'price_per_day',
              'is_auto_renew', 'is_cancel', 'trans_year', 'trans_month', 'trans_day', 
              'mem_exp_year', 'mem_exp_month', 'mem_exp_day']


# In[42]:


dis_1(df_trans_train_not_churn, df_trans_train_churn, trans_cols)


# In[43]:


dis_2(df_transactions_merge, trans_cols)


# In[44]:


df_user_logs_2.columns = df_user_logs_1.columns
df_user_logs_3.columns = df_user_logs_1.columns
df_user_logs = pd.concat([df_user_logs_1, df_user_logs_2, df_user_logs_3], axis=0)


# In[45]:


df_user_logs.head()


# In[46]:


df_user_logs.shape


# In[47]:


df_user_logs.msno.unique().shape


# In[48]:


df_user_logs['log_year'] = df_user_logs['date'].astype(str).apply(lambda x : int(x[:4]))
df_user_logs['log_month'] = df_user_logs['date'].astype(str).apply(lambda x : int(x[4:6]))
df_user_logs['log_day'] = df_user_logs['date'].astype(str).apply(lambda x : int(x[6:]))


# In[49]:


df_user_logs['avg_secs'] = df_user_logs['total_secs']/df_user_logs[['num_25', 'num_50', 'num_75', 'num_985', 'num_100']].sum(axis=1)


# In[50]:


df_user_logs['num_dup'] = df_user_logs[['num_25', 'num_50', 'num_75', 'num_985', 'num_100']].sum(axis=1) - df_user_logs['num_unq']


# In[51]:


# mean for 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'num_dup'
user_logs_mean_cols = ['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'num_dup']
df_user_logs_mean = df_user_logs[user_logs_mean_cols].groupby(['msno']).mean().reset_index()
df_user_logs_mean.head()


# In[52]:


# frequency for log_year, log_month and log_day
user_logs_count_cols = ['msno', 'log_year', 'log_month', 'log_day']
df_user_logs_count = df_user_logs[user_logs_count_cols].groupby(['msno']).count().reset_index()
df_user_logs_count.head()


# In[53]:


df_logs_proc = df_user_logs_mean.merge(df_user_logs_count, on = 'msno', 
                                                  how = 'outer')
df_logs_proc.replace(np.inf, 0, inplace = True)
df_logs_proc.fillna(0, inplace=True)


# In[54]:


df_logs_train = df_train_only.merge(df_logs_proc, on = 'msno', how = 'left')
df_logs_test = df_sample.merge(df_logs_proc, on = 'msno', how = 'left')
df_logs_merge = df_logs_train.merge(df_logs_test, on = 'msno', how = 'outer')
df_logs_train_not_churn = df_logs_train[df_logs_train['is_churn'] == 0]
df_logs_train_churn = df_logs_train[df_logs_train['is_churn'] == 1]
del df_logs_proc


# In[55]:


log_cols = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'num_dup',
           'log_year', 'log_month', 'log_day']
dis_1(df_logs_train_not_churn, df_logs_train_churn, log_cols)


# In[56]:


dis_2(df_logs_merge, log_cols)


# In[57]:


import seaborn as sns
def corr_analysis(df = None, corr_method = 'pearson', show_graph = False):
    '''
    function to analyze the correlation between features
    '''
    assert(not df.empty)
    # compute correlation
    df_corr = df.corr(method = corr_method)
    print(df_corr['is_churn'])
    # show heatmap graph
    if show_graph:
        plt.figure(figsize = (16,14))
        mask = np.zeros_like(df_corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            #matplotlib.rcParams.update({"font.size": 8})
            sns.set(font_scale=1.2)
            sns.heatmap(df_corr, cmap = "coolwarm", annot = False, mask = mask)
            plt.yticks(rotation=0) 
            plt.xticks(rotation=90) 
            plt.title('Correlation Analysis Between Features')


# In[58]:


df = df_members_train.merge(df_transactions_train, on = ['msno', 'is_churn'], how = 'outer')
df.head()


# In[59]:


corr_analysis(df = df, corr_method = 'pearson', show_graph = True)


# In[60]:




