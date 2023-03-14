#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#Importing the libraries for the garphs
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

#importing the libraries for datapeprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler

#importing the libraries for splitting the data
from sklearn.model_selection import train_test_split

#Libraries for model building
from sklearn import linear_model
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

#import the libraries for checking the metrics
from sklearn.metrics import r2_score,mean_squared_error


# In[3]:


df_train = pd.read_csv("/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_train.csv")
df_validate = pd.read_csv("/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_validate.csv")
df_ss=pd.read_csv("/kaggle/input/dphi-amsterdam-airbnb-data/sample_submission.csv")


# In[4]:


def aboutdf (df):
    count_null = df.isnull().sum()
    df_stats = pd.DataFrame(index = df.columns, data =
                           {'datatype': df.dtypes,
                            'unique_values': df.nunique(),
                            'have_null?': df.isnull().any(),
                            'Number of null values' : count_null,
                            'percentage of null values' : count_null/df.shape[0]*100 })
    return df_stats


# In[5]:


aboutdf(df_train)


# In[6]:


df_train=df_train.drop(columns='neighbourhood_group',axis=1)


# In[7]:


bool_series = pd.notnull(df_train["name"])
df_train[bool_series].name


# In[8]:


df_train.name.describe()


# In[9]:


df_train[bool_series].groupby(['name']).size().sort_values(ascending=False).reset_index(name='count').head(100)


# In[10]:


d=df_train[bool_series].groupby(['name']).size().sort_values(ascending=False).reset_index(name='count').head(100)
fig=px.bar(d,
                           y='name',
                           x='count',
                           #size='count',
                           #color='name',
                           range_x=[0,30]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[11]:


df_train.host_name.describe()


# In[12]:


bool_series1 = pd.notnull(df_train["host_name"])
df_train[bool_series1].host_name


# In[13]:


df_train[bool_series1].groupby(['host_name']).size().sort_values(ascending=False).reset_index(name='count').head(100)


# In[14]:


d=df_train[bool_series1].groupby(['host_name']).size().sort_values(ascending=False).reset_index(name='count').head(100)
fig=px.bar(d,
                           y='host_name',
                           x='count',
                           #size='count',
                           #color='name',
                           range_x=[0,80]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[15]:


df_train.neighbourhood.describe()


# In[16]:


df_train.groupby(['neighbourhood']).size().sort_values(ascending=False).reset_index(name='count')


# In[17]:


d=df_train.groupby(['neighbourhood']).size().sort_values(ascending=False).reset_index(name='count')
fig=px.bar(d,
                           y='neighbourhood',
                           x='count',
                           #size='count',
                           #color='name',
                           range_x=[0,2500]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[18]:


df_train.room_type.unique


# In[19]:


d=df_train.groupby(['room_type']).size().sort_values(ascending=False).reset_index(name='count')
fig=px.bar(d,
                           y='room_type',
                           x='count',
                           #size='count',
                           #color='name',
                           #range_y=[0,5]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[20]:


df_train.minimum_nights.describe()


# In[21]:


d=df_train.groupby(['minimum_nights']).size().sort_values(ascending=False).reset_index(name='count').head(100)
fig=px.scatter(d,
                           x='minimum_nights',
                           y='count',
                           size='minimum_nights',
                           color='minimum_nights',
                           range_x=[0,1005],
                           range_y=[0,80]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[22]:


df_train.number_of_reviews.describe()


# In[23]:


df_train.groupby(['number_of_reviews']).size().sort_values(ascending=False).reset_index(name='count')


# In[24]:


d=df_train.groupby(['number_of_reviews']).size().sort_values(ascending=False).reset_index(name='count')
fig=px.scatter(d,
                           x='number_of_reviews',
                           y='count',
                           size='count',
                           color='number_of_reviews',
                           range_y=[0,1700],
                           range_x=[0,900]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[25]:


df_train.last_review.describe()


# In[26]:


df_train.last_review


# In[27]:


d=df_train.groupby(['last_review']).size().sort_values(ascending=False).reset_index(name='count').head(100)
fig=px.scatter(d,
                           x='last_review',
                           y='count',
                           size='count',
                           color='last_review',
                           range_y=[0,300],
                           #range_x=[0,900]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[28]:


df_train.reviews_per_month.describe()


# In[ ]:





# In[29]:


df_train.calculated_host_listings_count.describe()


# In[30]:


df_train.availability_365.describe()


# In[31]:


d=df_train.groupby(['availability_365']).size().sort_values(ascending=False).reset_index(name='count')
fig=px.scatter(d,
                           x='availability_365',
                           y='count',
                           size='availability_365',
                           color='availability_365',
                           range_y=[0,200],
                           #range_x=[0,900]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[32]:


df_train.price.describe()


# In[33]:


df_train[['name','host_name','price']].sort_values(by='price',ascending=False)


# In[34]:


d=df_train[['name','host_name','price']].sort_values(by='price',ascending=False).head(50)
fig=px.scatter(d,
                           x='name',
                           y='price',
                           size='price',
                           color='host_name',
                           #range_y=[0,200],
                           #range_x=[0,900]
                           )
fig.update_layout(autosize=False,
                  height=500,
                  width=1400,
                  font=dict(size=15,color="#0f0f0f",family="Courier New, monospace"),
                 )
fig.show()


# In[35]:


plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(),cmap='bwr',annot=True)


# In[36]:


df_train.cp=df_train.copy()


# In[37]:


df_train_dup=df_train[df_train.duplicated()]
df_train_dup.shape


# In[38]:


df_train['last_review'].mode()


# In[39]:


df_train['last_review'].describe()


# In[40]:


df_train['reviews_per_month'].describe()


# In[41]:


#df_train['reviews_per_month']=df_train['reviews_per_month'].apply(np.round)


# In[42]:


df_train['reviews_per_month'].describe()


# In[43]:


df_train['last_review']=pd.to_datetime(df_train['last_review'])


# In[44]:


df_train['name'].fillna(df_train['name'].mode()[0], inplace=True)
df_train['host_name'].fillna(df_train['host_name'].mode()[0],inplace=True)
df_train['last_review'].fillna(df_train['last_review'].mean(),inplace=True)
df_train['reviews_per_month'].fillna(df_train['reviews_per_month'].mean(),inplace=True)


# In[45]:


df_train["day"] = df_train['last_review'].map(lambda x: x.day)
df_train["month"] = df_train['last_review'].map(lambda x: x.month)
df_train["year"] = df_train['last_review'].map(lambda x: x.year)


# In[46]:


aboutdf(df_train)


# In[47]:


le = LabelEncoder()


# In[48]:


df_train['name']=LabelEncoder().fit_transform(df_train['name'])
df_train['host_name']=LabelEncoder().fit_transform(df_train['host_name'])
df_train['neighbourhood']=LabelEncoder().fit_transform(df_train['neighbourhood'])
df_train['room_type']=LabelEncoder().fit_transform(df_train['room_type'])


# In[49]:


# sns.pairplot(df_train, kind="reg")


# In[50]:



# # sns.regpairplot(x=df_train[col],y='price',data=df_train_plot,
#          scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[51]:


# sns.regplot(x='id',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[52]:


# sns.regplot(x='name',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[53]:


# sns.regplot(x='host_id',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[54]:


# sns.regplot(x='host_name',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[55]:


# sns.regplot(x='neighbourhood',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[56]:


# sns.regplot(x='latitude',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[57]:


# sns.regplot(x='longitude',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[58]:


# sns.regplot(x='room_type',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[59]:


sns.regplot(x='minimum_nights',y='price',data=df_train,
            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[60]:


# sns.regplot(x='number_of_reviews',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[61]:


# sns.regplot(x='reviews_per_month',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[62]:


# sns.regplot(x='calculated_host_listings_count',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[63]:


# sns.regplot(x='availability_365',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[64]:


# sns.regplot(x='day',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[65]:


# sns.regplot(x='month',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[66]:


# sns.regplot(x='year',y='price',data=df_train,
#            scatter_kws={'alpha':0.3},line_kws={'color':'orange'})


# In[67]:


#X=df_train.drop(columns=['price','last_review','id','host_id'],axis=1)
X=df_train.drop(columns=['price','last_review'],axis=1)# just check feature selection
y=df_train['price']


# In[68]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)


# In[69]:


from xgboost import XGBClassifier
from xgboost import plot_importance

# fit model to training data
xgb_model = XGBClassifier(random_state = 0 )
xgb_model.fit(X_train, y_train)

print("Feature Importances : ", xgb_model.feature_importances_)

# plot feature importance
plot_importance(xgb_model)
plt.show()


# In[70]:


from sklearn.feature_selection import SelectFromModel


# In[71]:


selection = SelectFromModel(xgb_model)
print(selection)
selection.fit(X_train, y_train)

# Transform the train and test features
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test) 

# train model
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)


# In[72]:


select_X_train


# In[73]:


y_train_pred=xgb_model.predict(X_train)
mse_train=mean_squared_error(y_train,y_train_pred)
rmse_train=math.sqrt(mse_train)
print(rmse_train)


# In[74]:


y_pred = xgb_model.predict(X_test)



mse_test=mean_squared_error(y_test,y_pred)
rmse_test=math.sqrt(mse_test)
print(rmse_test)


# In[75]:


predictions = selection_model.predict(select_X_test)


# In[76]:


mse_test1=mean_squared_error(y_test,predictions)
rmse_test1=math.sqrt(mse_test)
print(rmse_test1)


# In[77]:


lgbm =LGBMRegressor()


# In[78]:


lgbm =LGBMRegressor(random_state=4)
lgbm.fit(X_train,y_train)
bpred_train=lgbm.predict(X_train)
bpred=lgbm.predict(X_test)


# In[79]:


bmse_train=mean_squared_error(y_train,bpred_train)
brmse_train=math.sqrt(bmse_train)
print(brmse_train)


# In[80]:


bmse_test=mean_squared_error(y_test,bpred)
brmse_test=math.sqrt(bmse_test)
print(brmse_test)


# In[81]:


pip install lazypredict


# In[82]:


import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(X_train, X_test, y_train, y_test)


# In[83]:


models,predictions 


# In[84]:


from sklearn import ensemble
#model_LP = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=10, max_features=0.3, n_jobs=-1, random_state=0)
#model_LP = ensemble.ExtraTreesRegressor(n_estimators=100,max_depth=15,max_features=0.8,n_jobs=-2,random_state=0)--217.6253937376234
model_LP = ensemble.ExtraTreesRegressor(n_estimators=100,max_depth=15, n_jobs=-2, min_samples_split=2,min_samples_leaf=2, max_features=0.8,random_state=0)


# In[85]:


model_LP.fit(X_train,y_train)


# In[86]:


LP_pred_train=model_LP.predict(X_train)
LP_pred_test=model_LP.predict(X_test)


# In[87]:


LP_train=mean_squared_error(y_train,LP_pred_train)
MLP_train=math.sqrt(LP_train)
print(MLP_train)


# In[88]:


LP_test=mean_squared_error(y_test,LP_pred_test)
MLP_test=math.sqrt(LP_test)
print(MLP_test)


# In[89]:


scaler =StandardScaler()


# In[90]:


aboutdf(X)


# In[91]:


df_train['availability_365'].describe()


# In[92]:


X_scaler=scaler.fit_transform(X)


# In[93]:


X_scaler


# In[94]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)


# In[95]:


reg = linear_model.Ridge(alpha=.5)


# In[96]:


model=reg.fit(X_train,y_train)


# In[97]:


pred=model.predict(X_test)


# In[98]:


pred_train=model.predict(X_train)


# In[99]:


mse_train=mean_squared_error(y_train,pred_train)
rmse_train=math.sqrt(mse_train)
print(rmse_train)


# In[100]:


mse = mean_squared_error(y_test, pred)

rmse = math.sqrt(mse)
print(rmse)


# In[101]:


from tqdm import tqdm
import math
from math import sqrt


# In[102]:


alpha = [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7] 
cv_rmsle_array=[] 
for i in tqdm(alpha):
    model =linear_model.Ridge(solver="sag", random_state=42, alpha=i)
    model.fit(X_train, y_train)
    preds_cv = model.predict(X_test)
    mse=mean_squared_error(y_test, preds_cv)
    a=sqrt(mse)
    cv_rmsle_array.append(a)
    


for i in range(len(cv_rmsle_array)):
     print ('RMSLE for alpha = ',alpha[i],'is',cv_rmsle_array[i])

best_alpha = np.argmin(cv_rmsle_array)
fig, ax = plt.subplots()
ax.plot(alpha, cv_rmsle_array)
ax.scatter(alpha, cv_rmsle_array)
for i, txt in enumerate(np.round(cv_rmsle_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_rmsle_array[i]))
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()


# In[103]:


print("Best alpha: ",  alpha[best_alpha])
model = linear_model.Ridge(solver="sag", random_state=42, alpha=alpha[best_alpha])
model.fit(X_train, y_train)
ridge_preds_tr = model.predict(X_train)
ridge_preds_cv = model.predict(X_test)
#ridge_preds_te = model.predict(X_test)
mse=mean_squared_error(y_train, ridge_preds_tr)
print('Train RMSLE:', sqrt(mse))
mse_test=mean_squared_error(y_test, ridge_preds_cv)
ridge_rmsle = sqrt((mse_test))
print("Cross validation RMSLE: ", ridge_rmsle)


# In[104]:


from scipy.stats import uniform
from scipy.stats import randint as sp_randint


# In[105]:



lgb_model = LGBMRegressor(subsample=0.9)

params = {'learning_rate': uniform(0, 1),
          'n_estimators': sp_randint(200, 1500),
          'num_leaves': sp_randint(20, 200),
          'max_depth': sp_randint(2, 15),
          'min_child_weight': uniform(0, 2),
          'colsample_bytree': uniform(0, 1),
         }
lgb_random = RandomizedSearchCV(lgb_model, param_distributions=params, n_iter=10, cv=3, random_state=42, 
                                scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True)
lgb_random = lgb_random.fit(X_train, y_train)

best_params = lgb_random.best_params_
print(best_params)


# In[106]:



model_lgbm = LGBMRegressor(**best_params, subsample=0.9, random_state=42, n_jobs=-1)
model_lgbm.fit(X_train, y_train)

lgb_preds_tr = model_lgbm.predict(X_train)
#lgb_preds_cv = model.predict(X_test)
lgb_preds_te = model_lgbm.predict(X_test)
mse_tr=mean_squared_error(y_train,lgb_preds_tr)
print("mse_tr:",mse_tr)
print('Train RMSLE:', sqrt(mse_tr))
mse_te=mean_squared_error(y_test,lgb_preds_te)
print(mse_te)
lgb_rmsle = sqrt(mse_te)
print("Test RMSLE: ", lgb_rmsle)


# In[107]:


from prettytable import PrettyTable
x=PrettyTable()
x.field_names=["Model","methods","Train_RMSLE", "Test_RMSLE", "Kaggle_RMSLE(Public)"]
x.add_row(["Ridge","-id,host_id,+standardscalar","222.7798385525557","246.32535554500578","235.30133"])
x.add_row(["LightGBM","-id,host_id,+standardscalar","6.491644344317403","227.68104533754683","223.08112"])
x.add_row(["Ridge","+id,host_id,-standardscalar","222.24334838932626","245.89118907576744","-"])
x.add_row(["LightGBM","+id,host_id,-standardscalar","3.6579076181175987","224.6785629027711","217.04551"])
x.add_row(["XGB","+id,host_id,-standardscalar +fs","78.35747725276444","224.9775499003927","-"])
x.add_row(["LightGBM","+id,host_id,-standardscalar -fs","114.65801828043752","220.28635168900416","211.71734"])
x.add_row(["ExtraTreesRegressor","+id,host_id,-standardscalar -fs","0.010229915092057028","218.65119082233184","211.66401"])
x.add_row(["ExtraTreesRegressor","+id,host_id,-standardscalar-fs+tune","82.37541294218074","217.095978804005","211.24184"])
print(x)


# In[108]:


aboutdf(df_validate)


# In[109]:


df_validate=df_validate.drop(columns=['neighbourhood_group'],axis=1)


# In[110]:


df_validate['last_review']=pd.to_datetime(df_validate['last_review'])


# In[111]:


df_validate['name'].fillna(df_validate['name'].mode()[0], inplace=True)
df_validate['host_name'].fillna(df_validate['host_name'].mode()[0],inplace=True)
df_validate['last_review'].fillna(df_validate['last_review'].mean(),inplace=True)
df_validate['reviews_per_month'].fillna(df_validate['reviews_per_month'].mean(),inplace=True)


# In[112]:


df_validate["day"] = df_validate['last_review'].map(lambda x: x.day)
df_validate["month"] = df_validate['last_review'].map(lambda x: x.month)
df_validate["year"] = df_validate['last_review'].map(lambda x: x.year)


# In[113]:


df_validate=df_validate.drop(columns=['last_review'],axis=1)


# In[114]:


df_validate['name']=LabelEncoder().fit_transform(df_validate['name'])
df_validate['host_name']=LabelEncoder().fit_transform(df_validate['host_name'])
df_validate['neighbourhood']=LabelEncoder().fit_transform(df_validate['neighbourhood'])
df_validate['room_type']=LabelEncoder().fit_transform(df_validate['room_type'])


# In[115]:


X_val=df_validate


# In[116]:


X_valscaler=scaler.fit_transform(X_val)
pred_val=model.predict(X_valscaler)


# In[117]:


pred_lgbm=model_lgbm.predict(X_val)


# In[118]:


select_X_val = selection.transform(X_val)


# In[119]:


pred_xgb=selection_model.predict(select_X_val)


# In[120]:


bpred_lgbm=lgbm.predict(X_val)


# In[121]:


pred_lp=model_LP.predict(X_val)


# In[122]:


output = pd.DataFrame({'Id': df_validate.id,
                      'Price': pred_lp})
output.to_csv('submission.csv', index=False)
output

