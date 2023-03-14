#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)
import cufflinks as cf
import plotly.offline as pyo
cf.go_offline()
pyo.init_notebook_mode()
#print(__version__)


# In[3]:


df_train= pd.read_csv('../input/sf-crime/train.csv.zip', parse_dates=['Dates'])
df_test=pd.read_csv('../input/sf-crime/test.csv.zip')


# In[4]:


print('First date: ', str(df_train.Dates.describe()['first']))
print('Last date: ', str(df_train.Dates.describe()['last']))


# In[5]:


df_train.head()


# In[6]:


df_test


# In[7]:


df_train.info()


# In[8]:


df_test.info()


# In[9]:


df_train['Category'].describe()


# In[10]:


df_train.dtypes


# In[11]:


df_train.duplicated().sum()


# In[12]:


df_train.drop_duplicates(keep='first', inplace=True)


# In[13]:


df_train.duplicated().sum()


# In[14]:


df_train['Category'].value_counts()


# In[15]:


df_train['Descript'].value_counts()


# In[16]:


df_train['Category'].isnull().sum()


# In[17]:


df_train['Category'].nunique()


# In[18]:


df_train['Descript'].nunique()


# In[19]:


df_train['PdDistrict'].value_counts()


# In[20]:


#plt.figure(figsize=(14,10))
#sns.countplot(x='PdDistrict', data=df_train, palette='viridis')


# In[21]:


df_train['PdDistrict'].value_counts().iplot(kind='bar', colors='red')


# In[22]:


#plt.figure(figsize=(20,10))
#ax=sns.countplot(x='Category', data=df_train, palette='viridis')
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')


# In[23]:


df_train['Category'].value_counts().iplot(kind='bar', colors='darkblue' ,title='SAN FRAN CRIME')


# In[24]:


df_train['PdDistrict'].value_counts().iplot(kind='bar',colors='Black')


# In[25]:


df_train['Category'].value_counts().iplot(kind='box', colors='darkblue' ,title='SAN FRAN CRIME')


# In[26]:


df_train['PdDistrict'].value_counts().iplot(kind='box', colors='darkblue' ,title='SAN FRAN CRIME')


# In[27]:


type(df_train['Dates'][0])


# In[28]:


type(df_test['Dates'][0])


# In[29]:


import datetime
df_test['Dates']=pd.to_datetime(df_test['Dates'],infer_datetime_format=True)


# In[30]:


type(df_test['Dates'][0])


# In[31]:


#df_train['Dates']


# In[32]:


#df_test['Dates']


# In[33]:


df_train['Hour']=df_train['Dates'].apply(lambda time: time.hour)
df_test['Hour']=df_train['Dates'].apply(lambda time: time.hour)


# In[34]:


df_train['Year']=df_train['Dates'].apply(lambda time: time.year)
df_test['Year']=df_train['Dates'].apply(lambda time: time.year)


# In[35]:


df_train['Month']=df_train['Dates'].apply(lambda time: time.month)
df_test['Month']=df_test['Dates'].apply(lambda time: time.month)


# In[36]:


df_train['Month'].value_counts()


# In[37]:


Month_dict= {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
df_train['Month']=df_train['Month'].map(Month_dict)
df_train['Month'].unique()


# In[38]:


df_train['Month'].value_counts().iplot(kind='bar', color='darkblue', title='Average crimes per month')


# In[39]:


df_train['Hour'].value_counts().iplot(kind='bar', title='Crimes by Hour', color='Darkred')


# In[40]:


df_train['DayOfWeek'].value_counts()


# In[41]:


df_train['DayOfWeek'].value_counts().iplot(kind='line',color='black', title='Crimes per day')


# In[42]:


df_train.head()


# In[43]:


plt.figure(figsize=(18,12))
sns.countplot(x='Year',hue='Category',data=df_train)
plt.legend(loc=10,bbox_to_anchor=(1.1, 0.5))


# In[44]:


plt.figure(figsize=(18,12))
sns.countplot(x='Year',hue='PdDistrict',data=df_train)
plt.legend(loc=10,bbox_to_anchor=(1.1, 0.5))


# In[45]:


#top 13
bob=df_train['Category'].value_counts().head(13)
bob


# In[46]:


monthyear= df_train.groupby(by=['Month', 'Year']).count()['Category'].unstack()


# In[47]:


monthyear


# In[48]:


monthyear.iloc[0,:].iplot(kind='line', title='January over the years')


# In[49]:


monthyear=monthyear.reindex(["January","February","March","April","May","June","July","August","September","October","November","December"])


# In[50]:


monthyear


# In[51]:


monthyear.iloc[:,11].iplot(kind='line', title='2004')


# In[52]:


monthyear.sum()


# In[53]:


monthyear.sum().iplot(kind='line')


# In[54]:


mycorr=monthyear.corr()


# In[55]:


plt.figure(figsize=(18,12))
sns.heatmap(mycorr, annot=True)


# In[56]:


plt.figure(figsize=(14,10))
sns.heatmap(monthyear)


# In[57]:


df_train=df_train.drop(['Dates','Descript'], axis=1)


# In[58]:


df_train.head()


# In[59]:


df_train=df_train.drop(['Resolution'], axis=1)


# In[60]:


df_train.head()


# In[61]:


df_test=df_test.drop(['Id', 'Dates'], axis=1)


# In[62]:


df_test.head()


# In[63]:


df_train.head()


# In[64]:


#Month_dict= {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
#df_test['Month']=df_test['Month'].map(Month_dict)


# In[65]:


df_test.head()


# In[66]:


type(df_test['Hour'][0])


# In[67]:


type(df_train['Hour'][0])


# In[68]:


df_test['Hour']=df_test['Hour'].fillna(18)


# In[69]:


df_test['Hour']=df_test['Hour'].astype(int)


# In[70]:


df_test['Year']=df_test['Year'].fillna(2014)


# In[71]:


df_test['Year']=df_test['Year'].astype(int)


# In[72]:


df_test.head()


# In[73]:


df_train.head()


# In[74]:


Month_num= {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
df_train['Month']=df_train['Month'].map(Month_num)
df_train['Month'].unique()


# In[75]:


pd_map={'NORTHERN':100,'PARK':200,'INGLESIDE':300,'BAYVIEW':400,'RICHMOND':500,'CENTRAL':600,'TARAVAL':700,'TENDERLOIN':800,'MISSION':900,'SOUTHERN':1000}
df_train['PdDistrict']=df_train['PdDistrict'].map(pd_map)
df_train['PdDistrict'].unique()


# In[76]:


df_test['PdDistrict']=df_test['PdDistrict'].map(pd_map)
df_test['PdDistrict'].unique()


# In[77]:


day_map={'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Friday':6,'Saturday':7}
df_train['DayOfWeek']=df_train['DayOfWeek'].map(day_map)
df_test['DayOfWeek']=df_test['DayOfWeek'].map(day_map)


# In[78]:


df_test['DayOfWeek'].unique()


# In[79]:


y=df_train['Category']


# In[80]:


df_test=df_test.drop('Address', axis=1)


# In[81]:


X=df_train.drop(['Category','Address'],axis=1)


# In[82]:


X.head()


# In[83]:


df_test.head()


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[85]:


from sklearn.ensemble import RandomForestClassifier
from time import time


# In[86]:


rfc = RandomForestClassifier(15)
print('Random Forest...')
start = time()
rfc.fit(X_train, y_train)
end = time()
print('Trained model in {:3f} seconds...'.format(end - start))


# In[87]:


rfc.score(X_train, y_train)


# In[88]:


predictions=rfc.predict(df_test)


# In[89]:


predictions


# In[ ]:





# In[ ]:





# In[ ]:




