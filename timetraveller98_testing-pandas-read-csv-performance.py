#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/csv-large/csv_large.csv')")


# In[ ]:


df.head()


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/csv-large/csv_large.csv',engine='python')")


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/csv-large/csv_large.csv',engine='c')")


# In[ ]:


df=pd.read_csv('../input/csv-large/csv_large.csv',engine='c',sep=None)


# In[ ]:


df=pd.read_csv('../input/csv-large/csv_large.csv',engine='python',sep=None)


# In[ ]:


df.head()


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/csv-large/csv_large.csv',verbose=True)")


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/csv-large/csv_large.csv',low_memory=False,verbose=True)")


# In[ ]:


def get_linedata(lines):
    return [
                    [line for line in lines if line.split()[0]=='Tokenization' ],
                    [line for line in lines if line.split()[0]=='Type' ],
                    [line for line in lines if line.split()[0]=='Parser' ],
    ]

def sum_times():
    total_tokenization=0
    total_type=0
    total_parser=0
    with open('../input/token-time/tokenization_time.txt') as handle:
        lines=handle.readlines()
        tokenizations,types,parsers=get_linedata(lines)
        for tz,ty,pr in zip(tokenizations,types,parsers):
            total_tokenization+=float(tz[19:24])
            total_type+=float(ty[22:26])
            total_parser+=float(pr[28:32])
    return {'total tokenization time':'{:.2f} ms'.format(total_tokenization),
                    'total type conversion time':'{:.2f} ms'.format(total_type) ,
                    'total parser time':'{:.2f} ms'.format(total_parser) 
           }     
        
sum_times()


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/bluebook-for-bulldozers/train/Train.csv')")


# In[ ]:


print(df.columns[13],df.columns[39],df.columns[40],df.columns[41])
print(f"Total columns: {df.columns.size}")


# In[ ]:


get_ipython().system('wc -l ../input/bluebook-for-bulldozers/train/Train.csv')


# In[ ]:


get_ipython().run_line_magic('time', "df=pd.read_csv('../input/bluebook-for-bulldozers/train/Train.csv',low_memory=False)")


# In[ ]:


get_ipython().system('wc -l ../input/vv-large/vv_large.csv')


# In[ ]:


names_=['id','name','sex']
get_ipython().run_line_magic('time', "df=pd.read_csv('../input/vv-large/vv_large.csv', names=names_)")
print(
    type(df.id[2]),
    type(df.sex[2]),
    type(df.name[2]),
     )


# In[ ]:


df.head()


# In[ ]:


names_=['id','name','sex']
dtypes={'id':np.int8,'name':'str','sex':'str'}
get_ipython().run_line_magic('time', "df=pd.read_csv('../input/vv-large/vv_large.csv',names=names_,dtype=dtypes)")


# In[ ]:


names_=['id','name','sex']
dtypes={'id':np.int32,'name':'str','sex':'str'}
get_ipython().run_line_magic('time', "df=pd.read_csv('../input/vv-large/vv_large.csv',names=names_,dtype=dtypes,low_memory=False)")


# In[ ]:


names_=['id','name','sex']
dtypes={'id':'object','name':'object','sex':'object'}
get_ipython().run_line_magic('time', "df=pd.read_csv('../input/vv-large/vv_large.csv',names=names_,dtype=dtypes)")


# In[ ]:


names_=['id','name','sex']
dtypes={'id':'object','name':'object','sex':'object'}
get_ipython().run_line_magic('time', "df=pd.read_csv('../input/vv-large/vv_large.csv',names=names_,dtype=dtypes,verbose=True)")


# In[ ]:


get_ipython().system('ls -sh ../input/favorita-grocery-sales-forecasting/train.csv')


# In[ ]:


get_ipython().system('wc -l ../input/favorita-grocery-sales-forecasting/train.csv')


# In[ ]:


df=pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv',nrows=10**3)
df.head()


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'pd.io.parsers.TextFileReader')


# In[ ]:


get_ipython().run_line_magic('time', "TextFileReaderObject=pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv',chunksize=10**5)")
#Reading 100k rows in each chunk


# In[ ]:


print(next(TextFileReaderObject).shape)
next(TextFileReaderObject).head()


# In[ ]:


TextFileReaderObject=pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv',chunksize=10**5)
get_ipython().run_line_magic('time', 'df = pd.concat(chunk for chunk in TextFileReaderObject)')


# In[ ]:


print(df.shape)
df.head()

