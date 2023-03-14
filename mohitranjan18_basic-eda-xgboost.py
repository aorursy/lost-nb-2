#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# importing train and test data and submission 

train=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test=pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
population_info=pd.read_csv('../input/population/population_by_country_2020.csv')
submission_df=pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')


# In[3]:


submission_df.info()


# In[4]:


train.head()


# In[5]:


submission_df.head()


# In[6]:


test.head()


# In[7]:


population_info.head()


# In[8]:


train.info()


# In[9]:


test.info()


# In[10]:


# Converting date column to date object

train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])


# In[11]:


# looking for null values in train a
train.isnull().sum()


# In[12]:


train.isnull().sum()


# In[13]:


# Lets drop Province_State   from test and train as it has high number of null values

# Dropping province column
train.drop(columns='Province_State',axis=1,inplace=True)
test.drop(columns='Province_State',axis=1,inplace=True)


# In[14]:


time_df=train.groupby('Date')['ConfirmedCases','Fatalities'].max()


# In[15]:


plt.figure(figsize=(16,8))
sns.lineplot(data=time_df['ConfirmedCases'])
plt.show()


# In[16]:


# Method to split date into day,month and year
def createDateDetails(df,col_name):
    df[col_name+'_day']=df[col_name].dt.day
    df[col_name+'_month']=df[col_name].dt.month
    df[col_name+'_year']=df[col_name].dt.year


# In[17]:


# splitting ddate columns
createDateDetails(train,'Date')
createDateDetails(test,'Date')


# In[18]:


train.head()


# In[19]:


plt.figure(figsize=(16,8))
sns.lineplot(data=time_df['Fatalities'])
plt.title('Fatalities vs Time')
plt.show()


# In[20]:


# Creating few pivot for better visualization of fatality and confirmed cases 

month_day_pivot=pd.pivot_table(data=train,values='Fatalities',columns='Date_month',index='Date_day',aggfunc='sum')


# In[21]:


month_day_pivot.fillna(value=0,inplace=True)


# In[22]:


plt.figure(figsize=(20,10))
sns.heatmap(data=month_day_pivot,cmap='YlGnBu')
plt.title('DEATH IN MONTH WORLD WIDE')


# In[23]:


month_day_cases_pivot=pd.pivot_table(data=train,values='ConfirmedCases',columns='Date_month',index='Date_day',aggfunc='sum')
plt.figure(figsize=(20,10))
sns.heatmap(data=month_day_cases_pivot,cmap='YlGnBu')


# In[24]:


# Method to find the total number of cases in each country
count_dict={}
def total_case(df):
    for val in list(train.Country_Region.unique()):
        count_dict[val]=df[df['Country_Region']==val]['ConfirmedCases'].max()


# In[25]:


total_case(train)


# In[26]:


# Converting values taken into count_dict  country as key and cases as values into dictionary
count_df=pd.DataFrame(count_dict.items(),columns=['countries','case'])
count_df=count_df.sort_values(by=['case'],ascending=False)


# In[27]:


plt.figure(figsize=(20,50))
sns.barplot(data=count_df.iloc[:50,:],x='case',y='countries')#Top 50 countries only included
plt.title('TOTAL CASES ACROSS THE WORLD')


# In[28]:


list(population_info.columns)


# In[29]:


list(train.columns)


# In[30]:


# Lets convert the country name in all the table to lower case so that we can merge the table

population_info["Country (or dependency)"]=population_info["Country (or dependency)"].apply(lambda x:x.lower())
train['Country_Region']=train['Country_Region'].apply(lambda x:x.lower())
test['Country_Region']=test['Country_Region'].apply(lambda x:x.lower())


# In[31]:


population_info.head()


# In[32]:


train.head()


# In[33]:


test.head()


# In[34]:


# lets check the non mataching countries in population_info and train table

not_match=train[~train['Country_Region'].isin(population_info["Country (or dependency)"])]


# In[35]:


not_match.head()


# In[36]:


not_match.Country_Region.unique()


# In[37]:


# replacing new values of 'congo (kinshasa)' to congo in training file and doing similar for other non matching values in countries column

train['Country_Region']=train.Country_Region.replace({"congo (kinshasa)": "dr congo","congo (brazzaville)":"congo","korea, south": "south korea","burma":"myanmar","cote d'ivoire":"côte d'ivoire","us":"united states","saint Kitts & nevis":"saint kitts & nevis","saint vincent and the grenadines":"st. vincent & grenadines","taiwan*":"taiwan","west bank and gaza":"state of palestine","Czech Republic (Czechia)":"czech republic (czechia)"})


# In[38]:


not_match2=train[~train['Country_Region'].isin(population_info["Country (or dependency)"])]


# In[39]:


not_match2.Country_Region.unique()


# In[40]:


test['Country_Region']=test.Country_Region.replace({"congo (kinshasa)": "dr congo","congo (brazzaville)":"congo","korea, south": "south korea","burma":"myanmar","cote d'ivoire":"côte d'ivoire","us":"united states","saint Kitts & nevis":"saint kitts & nevis","saint vincent and the grenadines":"st. vincent & grenadines","taiwan*":"taiwan","west bank and gaza":"state of palestine","Czech Republic (Czechia)":"czech republic (czechia)"})


# In[41]:


# As above values are mostly cruise we can ignore and proceed to join the tables 
# Using left join 
population_info=population_info.rename(columns={"Country (or dependency)":"Country_Region"})#renaming column of population_info to do the join


# In[42]:


# Merging the columns

train_pop_df=train.merge(population_info,how='left',on='Country_Region')
test_pop_df=test.merge(population_info,how='left',on='Country_Region')


# In[43]:


train_pop_df.info()


# In[44]:


test_pop_df.info()


# In[45]:


(train_pop_df.isnull().sum()/len(train_pop_df.index))*100


# In[46]:


(test_pop_df.isnull().sum()/len(train_pop_df.index))*100


# In[47]:


# Dropping few columns such as World Share,fert Rate as it seems to be irrelevant in this case

train_pop_df.drop(columns=['World Share','Fert. Rate','Yearly Change','Migrants (net)','Net Change'],inplace=True,axis=1)
test_pop_df.drop(columns=['World Share','Fert. Rate','Yearly Change','Migrants (net)','Net Change'],inplace=True,axis=1)


# In[48]:


train_pop_df['Urban Pop %']=train_pop_df['Urban Pop %'].astype(str)
train_pop_df['Med. Age']=train_pop_df['Med. Age'].astype(str)


# In[49]:


test_pop_df['Urban Pop %']=test_pop_df['Urban Pop %'].astype(str)
test_pop_df['Med. Age']=test_pop_df['Med. Age'].astype(str)


# In[50]:


# train_pop_df['Urban Pop %'].fillna(train_pop_df['Urban Pop %'].mean())
train_pop_df['Urban Pop %']=train_pop_df['Urban Pop %'].apply(lambda x:x[:-1])
test_pop_df['Urban Pop %']=test_pop_df['Urban Pop %'].apply(lambda x:x[:-1])


# In[51]:


# Lets convert 'Med. Age' AND 'Urban Pop %' to numeric columns

train_pop_df['Urban Pop %']=train_pop_df['Urban Pop %'].apply(lambda x:'0' if x.isdigit()==False else x)
train_pop_df['Med. Age']=train_pop_df['Med. Age'].apply(lambda x:'0' if x.isdigit()==False else x)


# In[52]:



test_pop_df['Urban Pop %']=test_pop_df['Urban Pop %'].apply(lambda x:'0' if x.isdigit()==False else x)
test_pop_df['Med. Age']=test_pop_df['Med. Age'].apply(lambda x:'0' if x.isdigit()==False else x)


# In[53]:


train_pop_df['Urban Pop %']=train_pop_df['Urban Pop %'].astype('int32')
train_pop_df['Med. Age']=train_pop_df['Med. Age'].astype('int32')


# In[54]:


test_pop_df['Urban Pop %']=test_pop_df['Urban Pop %'].astype('int32')
test_pop_df['Med. Age']=test_pop_df['Med. Age'].astype('int32')


# In[55]:


train_pop_df['Urban Pop %']


# In[56]:


train_pop_df.iloc[8265]


# In[57]:


train_pop_df['Urban Pop %'].unique()


# In[58]:


train_pop_df.info()


# In[59]:


train_pop_df.head()


# In[60]:


test_pop_df.head()


# In[61]:


(test_pop_df.isnull().sum()/len(train_pop_df.index))*100


# In[62]:


(train_pop_df.isnull().sum()/len(train_pop_df.index))*100


# In[63]:


train_pop_df.describe()


# In[64]:


train_pop_df.isnull().sum()


# In[65]:


train_pop_df[train_pop_df['Population (2020)'].isnull()].index.tolist()


# In[66]:


train_pop_df.iloc[8455:8456,]


# In[67]:


train_pop_df[train_pop_df['Country_Region']=='czechia']['Population (2020)'].fillna(10649800)


# In[68]:


train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Population (2020)']=train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Population (2020)'].fillna(10649800)


# In[69]:


test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Population (2020)']=test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Population (2020)'].fillna(10649800)


# In[70]:


train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Density (P/Km²)']=train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Density (P/Km²)'].fillna(134)
test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Density (P/Km²)']=test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Density (P/Km²)'].fillna(134)


# In[71]:


train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Land Area (Km²)']=train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Land Area (Km²)'].fillna(78866)
test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Land Area (Km²)']=test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Land Area (Km²)'].fillna(78866)


# In[72]:


train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Urban Pop %']=train_pop_df.loc[train_pop_df['Country_Region']=='czechia','Urban Pop %'].fillna(73.5)
test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Urban Pop %']=test_pop_df.loc[test_pop_df['Country_Region']=='czechia','Urban Pop %'].fillna(73.5)


# In[73]:


train_pop_df.isnull().sum()


# In[74]:


# train_pop_df.Country_Region in ['diamond princess', 'kosovo', 'ms zaandam','saint kitts and nevis', 'sao tome and principe']].index.values
exclude_list=['diamond princess', 'kosovo', 'ms zaandam','saint kitts and nevis', 'sao tome and principe']
idx_list=[]
idx_list_test=[]
for val in exclude_list:
    idx_list.extend(train_pop_df[train_pop_df.Country_Region==val].index.values)

for val in exclude_list:
    idx_list_test.extend(test_pop_df[train_pop_df.Country_Region==val].index.values)


# In[75]:


train_pop_df.columns.to_list()


# In[76]:


train_pop_df.loc[idx_list, ['Date','ConfirmedCases', 'Fatalities','Date_day', 'Date_month','Date_year', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]=0
test_pop_df.loc[idx_list_test,['Date','Date_day', 'Date_month','Date_year', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]=0


# In[77]:


train_pop_df.info()


# In[78]:


test_pop_df.info()


# In[79]:



#     Method for Label Encoding
 def label_encoding(df,col):
     label_encoders=dict()
     output_df = df.copy(deep=True)
     for c in col:
         lbl = preprocessing.LabelEncoder()
         lbl.fit(df[c].values)
         output_df.loc[:, c] = lbl.transform(df[c].values)
         label_encoders[c] = lbl
     return output_df


# In[80]:


train_pop_df=label_encoding(train_pop_df,['Country_Region'])


# In[81]:


train_pop_df.head()


# In[82]:


test_pop_df=label_encoding(test_pop_df,['Country_Region'])


# In[83]:


train_pop_df.columns


# In[84]:


train_pop_df['Fatalities']


# In[85]:


X_train_final=train_pop_df.drop(['ConfirmedCases','Fatalities','Id','Date'],axis=1)
y_train1=train_pop_df['ConfirmedCases']
y_train2=train_pop_df['Fatalities']


# In[86]:


import xgboost as xgb
from xgboost import plot_importance, plot_tree


# In[87]:


#Fitting XGB regressor 
model1 = xgb.XGBRegressor()
model2=xgb.XGBRegressor()


# In[88]:


model1.fit(X_train_final,y_train1)
print (model1)


# In[89]:


model2.fit(X_train_final,y_train2)
print (model1)


# In[90]:


test_pop_df.columns


# In[91]:


X_train_final.columns


# In[92]:


test_pop_df.columns


# In[93]:


test_pop_df.drop(columns=['Date'],axis=1,inplace=True)


# In[94]:


plot = plot_importance(model1, height=0.9, max_num_features=5)


# In[95]:


plot = plot_importance(model2, height=0.9, max_num_features=5)


# In[96]:


xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})


# In[97]:


x_test_Id = test_pop_df.loc[:, 'ForecastId']


# In[98]:


test_pop_df.drop(columns=['ForecastId'],axis=1,inplace=True)


# In[99]:


#Predict 
output1 = model1.predict(data=test_pop_df)
output2 = model2.predict(data=test_pop_df)
# final_df = pd.DataFrame()
# final_df["ID"] = id_vals
# final_df["Prediction"] = output
# #final_df.to_csv("Output_1.csv")
# final_df.head()


# In[100]:


sub_data = pd.DataFrame({'ForecastId': x_test_Id, 'ConfirmedCases': output1, 'Fatalities': output2})


# In[101]:


xout = pd.concat([xout, sub_data], axis=0)


# In[102]:


xout


# In[103]:


xout['ConfirmedCases'] = xout['ConfirmedCases'].apply(int)
xout['Fatalities'] = xout['Fatalities'].apply(int)


# In[104]:


xout.reindex()


# In[105]:


xout.to_csv('submission.csv', index=False)


# In[106]:


# ## Hyper Parameter Optimization


# n_estimators = [100, 500, 900, 1100, 1500]
# max_depth = [2, 3, 5, 10, 15]
# booster=['gbtree','gblinear']
# learning_rate=[0.05,0.1,0.15,0.20]
# min_child_weight=[1,2,3,4]

# # Define the grid of hyperparameters to search
# hyperparameter_grid = {
#     'n_estimators': n_estimators,
#     'max_depth':max_depth,
#     'learning_rate':learning_rate,
#     'min_child_weight':min_child_weight,
#     'booster':booster,
#     'base_score':base_score
#     }


# In[107]:


# from sklearn.model_selection import RandomizedSearchCV


# In[108]:


# # Set up the random search with 5-fold cross validation
# random_cv = RandomizedSearchCV(estimator=regressor,
#             param_distributions=hyperparameter_grid,
#             cv=5, n_iter=50,
#             scoring = 'neg_mean_absolute_error',n_jobs = 4,
#             verbose = 5, 
#             return_train_score = True,
#             random_state=42)


# In[109]:


# random_cv.fit(X_train_final,y_train1)

