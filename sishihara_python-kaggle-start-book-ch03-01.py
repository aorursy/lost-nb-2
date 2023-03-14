#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


application_train =     pd.read_csv('../input/home-credit-default-risk/application_train.csv')
application_train.head()


# In[2]:


bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau.head()


# In[3]:


previous_loan_counts =     bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns={'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()


# In[4]:


application_train =     pd.merge(application_train, previous_loan_counts, on='SK_ID_CURR', how='left')

application_train['previous_loan_counts'].fillna(0, inplace=True)
application_train.head()


# In[ ]:




