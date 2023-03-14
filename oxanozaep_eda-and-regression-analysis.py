#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import pandas_profiling
import datetime
import sqlite3
import calendar
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[2]:


#We willl load all the csv files into Pandas dataframes, properly parsing dates

air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
hpg_reserve = pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])
air_store_info = pd.read_csv('../input/air_store_info.csv')
hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')
store_relation = pd.read_csv('../input/store_id_relation.csv')
date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date'])
air_visit = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])
sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[3]:


pandas_profiling.ProfileReport(air_reserve)


# In[4]:


pandas_profiling.ProfileReport(hpg_reserve)


# In[5]:


pandas_profiling.ProfileReport(air_store_info)


# In[6]:


pandas_profiling.ProfileReport(hpg_store_info)


# In[7]:


pandas_profiling.ProfileReport(store_relation)


# In[8]:


pandas_profiling.ProfileReport(date_info)


# In[9]:


pandas_profiling.ProfileReport(air_visit)


# In[10]:


print((air_reserve['reserve_datetime']>air_reserve['visit_datetime']).value_counts())

print((hpg_reserve['reserve_datetime']>hpg_reserve['visit_datetime']).value_counts())


# In[11]:


hpg_reserve['visit_year'] = hpg_reserve['visit_datetime'].dt.year
hpg_reserve['visit_month'] = hpg_reserve['visit_datetime'].dt.month
hpg_reserve['visit_day'] = hpg_reserve['visit_datetime'].dt.day
hpg_reserve['reserve_year'] = hpg_reserve['reserve_datetime'].dt.year
hpg_reserve['reserve_month'] = hpg_reserve['reserve_datetime'].dt.month
hpg_reserve['reserve_day'] = hpg_reserve['reserve_datetime'].dt.day

hpg_reserve.drop(['visit_datetime','reserve_datetime'], axis=1, inplace=True)

hpg_reserve = hpg_reserve.groupby(['hpg_store_id', 'visit_year', 'visit_month',                                   'visit_day','reserve_year','reserve_month','reserve_day'], as_index=False).sum()


# In[12]:


air_reserve['visit_year'] = air_reserve['visit_datetime'].dt.year
air_reserve['visit_month'] = air_reserve['visit_datetime'].dt.month
air_reserve['visit_day'] = air_reserve['visit_datetime'].dt.day
air_reserve['reserve_year'] = air_reserve['reserve_datetime'].dt.year
air_reserve['reserve_month'] = air_reserve['reserve_datetime'].dt.month
air_reserve['reserve_day'] = air_reserve['reserve_datetime'].dt.day

air_reserve.drop(['visit_datetime','reserve_datetime'], axis=1, inplace=True)

date_info['calendar_year'] = date_info['calendar_date'].dt.year
date_info['calendar_month'] = date_info['calendar_date'].dt.month
date_info['calendar_day'] = date_info['calendar_date'].dt.day

date_info.drop(['calendar_date'], axis=1, inplace=True)

air_visit['visit_year'] = air_visit['visit_date'].dt.year
air_visit['visit_month'] = air_visit['visit_date'].dt.month
air_visit['visit_day'] = air_visit['visit_date'].dt.day

air_visit.drop(['visit_date'], axis=1, inplace=True)


# In[13]:


hpg_reserve = pd.merge(hpg_reserve, store_relation, on='hpg_store_id', how='inner')
hpg_reserve.drop(['hpg_store_id'], axis=1, inplace=True)

air_reserve = pd.concat([air_reserve, hpg_reserve])


# In[14]:


air_reserve = air_reserve.groupby(['air_store_id', 'visit_year', 'visit_month','visit_day'],                as_index=False).sum().drop(['reserve_day','reserve_month','reserve_year'], axis=1)


# In[15]:


air_reserve = pd.merge(air_reserve, date_info, left_on=['visit_year','visit_month','visit_day'], right_on=['calendar_year','calendar_month','calendar_day'], how='left')
air_reserve.drop(['calendar_year','calendar_month','calendar_day'], axis=1, inplace=True)


# In[16]:


air_reserve = pd.merge(air_reserve, air_store_info, on='air_store_id', how='left')

df = pd.merge(air_reserve, air_visit, on=['air_store_id','visit_year','visit_month','visit_day'], how='left')


# In[17]:


pandas_profiling.ProfileReport(df)


# In[18]:


df.air_genre_name = df.air_genre_name.replace(' ', '_', regex=True)
df.air_genre_name = df.air_genre_name.replace('/', '_', regex=True)
df=df.rename(columns = {'air_genre_name':'genre','day_of_week':'dow'})

df.sort_values(by=['visit_year','visit_month','visit_day','air_store_id'],               ascending=[True,True,True,True], inplace=True)

data_train = df[df.visitors.notnull()]
data_test = df[df.visitors.isnull()]


# In[19]:


data_train['log_visitors'] = data_train.visitors.apply(lambda x: np.log(x))


# In[20]:


fig, ax = plt.subplots(figsize=(14,12));
ax = sns.violinplot(x='dow', y="visitors", hue='holiday_flg',data=df, palette="muted", split=True)


# In[21]:


sns.jointplot(x='visitors', y='reserve_visitors', data=data_train, color='navy',              size=10, space=0, kind='reg',marginal_kws={'hist_kws': {'log': True}})


# In[22]:


data_train_month = data_train[['visit_month','visitors','visit_year']].groupby(['visit_year','visit_month']).sum()

data_train_month.plot(kind ="bar", y='visitors')


# In[23]:


data_train_month_av = data_train[['visit_month','visitors','visit_year']].groupby(['visit_month']).mean()

data_train_month_av.plot(kind ="bar", y='visitors')


# In[24]:


cor = data_train.corr()
plt.figure(figsize=(14,3))
sns.heatmap(cor.loc[['visitors'], list(df)[:-1]]);


# In[25]:


data_train_Pivot = pd.pivot_table(data_train, values='visitors', columns='dow', index='visit_month')
data_train_Pivot.plot();
plt.legend(bbox_to_anchor=(1,1), loc="upper left")


# In[26]:


#Definition of the formula that will show the goodness of the model.

def RMSLE(predicted, actual):
    msle = (np.log(predicted+1) - np.log(actual+1))**2
    rmsle = np.sqrt(msle.sum()/msle.count())
    return rmsle


# In[27]:


data_train = pd.get_dummies(data_train, columns=['genre','dow'])

#We will use the log of the visitors to get a more useful mean.
model_mean_pred = data_train.log_visitors.mean()

# And we'll store this value in the dataframe
data_train['visitors_mean'] = np.exp(model_mean_pred)

data_train.loc[:, ['visitors','visitors_mean']].plot(color=['#bbbbbb','r'], figsize=(16,8));


# In[28]:


model_mean_RMSLE = RMSLE(data_train.visitors_mean, data_train.visitors)

results_df = pd.DataFrame(columns=["Model", "RMSLE"])

results_df.loc[0,"Model"] = "Mean"
results_df.loc[0,"RMSLE"] = model_mean_RMSLE
results_df.head()


# In[29]:


data_train = pd.merge(data_train, data_train[['air_store_id','visitors']].groupby(['air_store_id'], as_index=False).mean(), on='air_store_id', how='left')

data_train=data_train.rename(columns = {'visitors_y':'visitors_rest_mean','visitors_x':'visitors'})

model_mean_rest_RMSLE = RMSLE(data_train.visitors_rest_mean, data_train.visitors)

results_df.loc[1,"Model"] = "Mean_by_rest"
results_df.loc[1,"RMSLE"] = model_mean_rest_RMSLE
results_df.head()


# In[30]:


model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors',                             'log_visitors', 'air_store_id','visitors_mean'])), data_train)
result = model.fit()
print(result.summary())


# In[31]:


data_train["linear_regr"] = result.predict()

model_lin_RMSLE = RMSLE(data_train.linear_regr, data_train.visitors)

results_df.loc[2,"Model"] = "Multiple linear regressors"
results_df.loc[2,"RMSLE"] = model_lin_RMSLE
results_df


# In[32]:


dows = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for dow in dows:
    data_train['past_'+dow]= 0
    
data_train.sort_values(by=['air_store_id','visit_year','visit_month','visit_day'], ascending=[True,True,True,True], inplace=True)

data_train['store_change'] = (data_train.air_store_id!=data_train.air_store_id.shift())
data_train['past_dow_visitors'] = data_train['visitors_rest_mean']
data_train.reset_index(drop=True, inplace=True)

for index, row in data_train.iterrows():
    if not row.store_change:
        for dow in dows:
            if data_train.iloc[index-1, data_train.columns.get_loc('dow_'+dow)]:
                data_train.set_value(index,'past_'+dow,data_train.iloc[index-1, data_train.columns.get_loc('visitors')])
            else:
                data_train.set_value(index,'past_'+dow,data_train.iloc[index-1, data_train.columns.get_loc('past_'+dow)])


# In[33]:


for index, row in data_train.iterrows():
    for dow in dows:
        if row['dow_'+dow] and row['past_'+dow]>0:
            data_train.set_value(index,'past_dow_visitors', row['past_'+dow])

for dow in dows:
    data_train.drop(['past_'+dow], axis=1, inplace=True)


# In[34]:


model = sm.OLS.from_formula('visitors ~ past_dow_visitors * reserve_visitors * holiday_flg',data_train)
result = model.fit()
print(result.summary())


# In[35]:


model_pred = result.predict()
data_train['past_dow_predict'] = model_pred

model_past_dow_RMSLE = RMSLE(data_train.past_dow_predict, data_train.visitors)

results_df.loc[3,"Model"] = "Past_DoW"
results_df.loc[3,"RMSLE"] = model_past_dow_RMSLE
results_df


# In[36]:


s_residuals = pd.Series(result.resid_pearson, name="S. Residuals")
fitted_values = pd.Series(result.fittedvalues, name="Fitted Values")
sns.regplot(fitted_values, s_residuals,  fit_reg=False)


# In[37]:


def forward(predictors):
    remaining_predictors = [p for p in X.columns if p not in predictors]    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors + [p]))
    
    models = pd.DataFrame(results)
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors.")
    return models.loc[models['RSS'].argmin()]

def processSubset(feature_set):
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}


# In[38]:


models = pd.DataFrame(columns=["RSS", "model"])

predictors = []
y=data_train.visitors
X = data_train[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude',                'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday',                 'visitors_rest_mean','past_dow_visitors']].astype('float64')

for i in range(1, len(X.columns) + 1):    
    models.loc[i] = forward(predictors)
    predictors = models.loc[i]["model"].model.exog_names


# In[39]:


models.apply(lambda row: row[1].rsquared, axis=1)


# In[40]:


plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})
plt.subplot(4, 1, 1)

plt.plot(models["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

rsquared_adj = models.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(4, 1, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

aic = models.apply(lambda row: row[1].aic, axis=1)

plt.subplot(4, 1, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models.apply(lambda row: row[1].bic, axis=1)

plt.subplot(4, 1, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('BIC')


# In[41]:


data_train["subset_selection"] = models.loc[8, "model"].predict()
model_subset_RMSLE = RMSLE(data_train.subset_selection, data_train.visitors)

results_df.loc[4,"Model"] = "Subset selection"
results_df.loc[4,"RMSLE"] = model_subset_RMSLE
results_df


# In[42]:


poly_1 = smf.ols(formula='visitors ~ 1 + past_dow_visitors', data=data_train).fit()

poly_2 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0)', data=data_train).fit()

poly_3 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0)', data=data_train).fit()

poly_4 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0)', data=data_train).fit()

poly_5 = smf.ols(formula='visitors ~ 1 + past_dow_visitors + I(past_dow_visitors ** 2.0) + I(past_dow_visitors ** 3.0) + I(past_dow_visitors ** 4.0) + I(past_dow_visitors ** 5.0)', data=data_train).fit()


# In[43]:


print(sm.stats.anova_lm(poly_1, poly_2, poly_3, poly_4, poly_5, typ=1))


# In[44]:


plt.figure(figsize=(6 * 1.618, 6))
plt.scatter(data_train.past_dow_visitors, data_train.visitors, s=10, alpha=0.3)
plt.xlabel('past_dow_visitors')
plt.ylabel('visitors')

x = pd.DataFrame({'past_dow_visitors': np.linspace(data_train.past_dow_visitors.min(), data_train.past_dow_visitors.max(), 100)})
plt.plot(x.past_dow_visitors, poly_1.predict(x), 'b-', label='Poly n=1 $R^2$=%.2f' % poly_1.rsquared, alpha=0.9)
plt.plot(x.past_dow_visitors, poly_2.predict(x), 'g-', label='Poly n=2 $R^2$=%.2f' % poly_2.rsquared, alpha=0.9)
plt.plot(x.past_dow_visitors, poly_3.predict(x), 'r-', alpha=0.9,label='Poly n=3 $R^2$=%.2f' % poly_3.rsquared)
plt.plot(x.past_dow_visitors, poly_4.predict(x), 'y-', alpha=0.9,label='Poly n=4 $R^2$=%.2f' % poly_4.rsquared)
plt.plot(x.past_dow_visitors, poly_5.predict(x), 'k-', alpha=0.9,label='Poly n=5 $R^2$=%.2f' % poly_5.rsquared)

plt.legend()


# In[45]:


data_train["poly_regr"] = poly_5.predict()
model_poly_RMSLE = RMSLE(data_train.poly_regr, data_train.visitors)

results_df.loc[5,"Model"] = "Polynomial Regressor"
results_df.loc[5,"RMSLE"] = model_poly_RMSLE
results_df


# In[46]:


df_time = data_train[data_train.air_store_id == 'air_6b15edd1b4fbb96a']

df_time.set_index(pd.to_datetime(df_time.visit_year*10000+df_time.visit_month*100                                 +df_time.visit_day,format='%Y%m%d'), inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 7))

axes[0].plot(df_time.visitors, color='navy', linewidth=4)
axes[1].plot(df_time.visitors[df_time.visit_month > 10], color='navy', linewidth=4)


# In[47]:


df_time.visitors.plot(kind = "hist", bins = 30)


# In[48]:


df_time.log_visitors.plot(kind = "hist", bins = 30);


# In[49]:


model_mean_RMSLE = RMSLE(df_time.visitors_mean, df_time.visitors)
model_rest_mean_RMSLE = RMSLE(df_time.visitors.mean(), df_time.visitors)

results_df_time = pd.DataFrame(columns=["Model", "RMSLE"])
results_df_time.loc[0,"Model"] = "Total Mean"
results_df_time.loc[0,"RMSLE"] = model_mean_RMSLE
results_df_time.loc[1,"Model"] = "Restaurant Mean"
results_df_time.loc[1,"RMSLE"] = model_rest_mean_RMSLE

results_df_time


# In[50]:


decomposition = seasonal_decompose(df_time.log_visitors, model="additive", freq=6)
decomposition.plot();


# In[51]:


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

df_time['power_decomp'] = np.exp(trend + seasonal)


# In[52]:


model_Decomp_RMSLE = RMSLE(df_time.power_decomp, df_time.visitors)

results_df_time.loc[2,"Model"] = "Time Decomposition"
results_df_time.loc[2,"RMSLE"] = model_Decomp_RMSLE
results_df_time


# In[53]:


models_time = pd.DataFrame(columns=["RSS", "model"])

predictors = []
y=df_time.visitors
X = df_time[['visit_year', 'visit_month', 'visit_day', 'reserve_visitors','holiday_flg','latitude','longitude',                'dow_Friday','dow_Monday','dow_Tuesday','dow_Wednesday','dow_Thursday','dow_Saturday','dow_Sunday',                 'visitors_rest_mean','past_dow_visitors']].astype('float64')

for i in range(1, len(X.columns) + 1):    
    models_time.loc[i] = forward(predictors)
    predictors = models_time.loc[i]["model"].model.exog_names


# In[54]:


plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})
plt.subplot(4, 1, 1)

plt.plot(models_time["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

rsquared_adj = models_time.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(4, 1, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

aic = models_time.apply(lambda row: row[1].aic, axis=1)

plt.subplot(4, 1, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_time.apply(lambda row: row[1].bic, axis=1)

plt.subplot(4, 1, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "ob")
plt.xlabel('# Predictors')
plt.ylabel('BIC')


# In[55]:


df_time["subset_selection"] = models_time.loc[10, "model"].predict()
model_subset_RMSLE = RMSLE(df_time.subset_selection, df_time.visitors)

results_df_time.loc[3,"Model"] = "Subset selection"
results_df_time.loc[3,"RMSLE"] = model_subset_RMSLE
results_df_time


# In[56]:


#We get rid of the genres, as they do not help making a better model
df_time.drop(list(df_time.filter(regex = 'genre_')), axis = 1, inplace = True)
df_time.dropna(axis=0,how='any',inplace=True)

model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_time.columns.difference(['visitors', 'log_visitors','air_store_id','visitors_mean', 'subset_selection','past_dow_predict','power_decomp','poly_regr'])), df_time)

result = model.fit()
print(result.summary())


# In[57]:


df_time["linear_regr"] = result.predict()

# RMSLE for linear regressor
model_lin_RMSLE = RMSLE(df_time.linear_regr, df_time.visitors)

results_df_time.loc[4,"Model"] = "Linear Regressor"
results_df_time.loc[4,"RMSLE"] = model_lin_RMSLE
results_df_time


# In[58]:


#Let's get rid of the columns that won't be used in the final predictions.
data_train.drop(data_train[['air_area_name', 'latitude','past_dow_visitors','longitude','visitors_mean','linear_regr','store_change','past_dow_predict','subset_selection','poly_regr','log_visitors']], axis=1, inplace=True)
data_train.drop(list(data_train.filter(regex = 'genre_')), axis = 1, inplace = True)


# In[59]:


restaurants = data_train.air_store_id.unique()
RMSLEs = []
models_dict = {}

for i,restaurant in enumerate(restaurants):
    if i%100 == 0 or i==(len(restaurants)-1):
        print("Model {} of {}".format(i+1,len(restaurants)))
        
    df_temp = data_train[data_train.air_store_id == restaurant]
    df_temp.dropna(axis=0,how='any',inplace=True)
    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors',                                'air_store_id'])), df_temp).fit()
    RMSLEs.append(RMSLE(model.predict(), df_temp.visitors))
    models_dict[restaurant] = model


# In[60]:


RMSLEhalf = []
half_models_dict = {}

for i,restaurant in enumerate(restaurants):
    if i%100 == 0 or i==(len(restaurants)-1):
        print("Model {} of {}".format(i+1,len(restaurants)))
        
    df_temp = data_train[data_train.air_store_id == restaurant]
    df_temp.dropna(axis=0,how='any',inplace=True)
    model = sm.OLS.from_formula('visitors ~ ' + '+'.join(df_temp.columns.difference(['visitors',                                'air_store_id','reserve_visitors'])), df_temp).fit()
    RMSLEhalf.append(RMSLE(model.predict(), df_temp.visitors))
    half_models_dict[restaurant] = model


# In[61]:


nodata_model = sm.OLS.from_formula('visitors ~ ' + '+'.join(data_train.columns.difference(['visitors',                                   'air_store_id','reserve_visitors','visitors_rest_mean'])), data_train).fit()
RMSLE_rest = RMSLE(nodata_model.predict(), data_train.visitors)


# In[62]:


results_df.loc[6,"Model"] = "Regressor per id"
results_df.loc[6,"RMSLE"] = np.mean(RMSLEs)
results_df.loc[7,"Model"] = "Regressor per id w/o reserves"
results_df.loc[7,"RMSLE"] = np.mean(RMSLEs)
results_df.loc[8,"Model"] = "New id model"
results_df.loc[8,"RMSLE"] = RMSLE_rest

results_df

