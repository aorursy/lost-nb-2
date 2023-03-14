#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, gc
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import date, datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from scipy.optimize import nnls
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=6, suppress=True)


# In[2]:


# note: all data update 2020-04-01, china rule, smooth weights
mname = 'gbt1u'
path = '/kaggle/input/gbt1n-external/'
pathk = '/kaggle/input/covid19-global-forecasting-week-2/'
nhorizon = 30
skip = 0
# kv = [3]
# kv = [6]
kv = [6,11]
# kv = [13]
train_full = True
save_data = False

# booster = ['lgb','xgb']
booster = ['lgb','xgb','ctb']
# booster = ['cas']

# if using updated daily data, also update time-varying external data
# in COVID-19 and covid-19-data, git pull origin master 
# ecdc wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv
# weather: https://www.kaggle.com/davidbnn92/weather-data/output?scriptVersionId=31103959
# google trends: pytrends0b.ipynb


# In[3]:


train = pd.read_csv(pathk+'train.csv')
test = pd.read_csv(pathk+'test.csv')
ss = pd.read_csv(pathk+'submission.csv')


# In[4]:


train


# In[5]:


# tmax and dmax are the last day of training
tmax = train.Date.max()
dmax = datetime.strptime(tmax,'%Y-%m-%d').date()
print(tmax, dmax)


# In[6]:


# ddate is the last day of validation training
ddate = dmax - timedelta(days=nhorizon)
ddate


# In[7]:


test


# In[8]:


fmax = test.Date.max()
fdate = datetime.strptime(fmax,'%Y-%m-%d').date()
fdate


# In[9]:


tmin = train.Date.min()
fmin = test.Date.min()
tmin, fmin


# In[10]:


dmin = datetime.strptime(tmin,'%Y-%m-%d').date()
print(dmin)


# In[11]:


# train['ForecastId'] = train.Id - train.Id.max()
cp = ['Country_Region','Province_State']
cpd = cp + ['Date']
train = train.merge(test[cpd+['ForecastId']], how='left', on=cpd)
train['ForecastId'] = train['ForecastId'].fillna(0).astype(int)
train['y0_pred'] = np.nan
train['y1_pred'] = np.nan

test['Id'] = test.ForecastId + train.Id.max()
test['ConfirmedCases'] = np.nan
test['Fatalities'] = np.nan
# use zeros here instead of nans so monotonic adjustment fills final dates if necessary
test['y0_pred'] = 0.0
test['y1_pred'] = 0.0


# In[12]:


# concat non-overlapping part of test to train for feature engineering
d = pd.concat((train,test[test.Date > train.Date.max()])).reset_index(drop=True)
d


# In[13]:


(dmin + timedelta(30)).isoformat()


# In[14]:


d['Date'].value_counts().std()


# In[15]:


# fill missing province with blank, must also do this with external data before merging
d[cp] = d[cp].fillna('')

# create single location variable
d['Loc'] = d['Country_Region'] + ' ' + d['Province_State']
d['Loc'] = d['Loc'].str.strip()
d['Loc'].value_counts()


# In[16]:


# sort by location then date
d = d.sort_values(['Loc','Date']).reset_index(drop=True)


# In[17]:


d['Country_Region'].value_counts(dropna=False)


# In[18]:


d['Province_State'].value_counts(dropna=False)


# In[19]:


gt = pd.read_csv(path+'google_trends.csv')
gt[cp] = gt[cp].fillna('')
gt


# In[20]:


# since trends data lags behind a day or two, shift the date to make it contemporaneous
gmax = gt.Date.max()
gmax = datetime.strptime(gmax,'%Y-%m-%d').date()
goff = (dmax - gmax).days
print(dmax, gmax, goff)
gt['Date'] = (pd.to_datetime(gt.Date) + timedelta(goff)).dt.strftime('%Y-%m-%d')
gt['google_covid'] = gt['coronavirus'] + gt['covid-19'] + gt['covid19']
gt.drop(['coronavirus','covid-19','covid19'], axis=1, inplace=True)
google = ['google_covid']
gt


# In[21]:


d = d.merge(gt, how='left', on=['Country_Region','Province_State','Date'])
d


# In[22]:


d['google_covid'].describe()


# In[23]:


# merge country info
country = pd.read_csv(path+'covid19countryinfo1.csv')
# country["pop"] = country["pop"].str.replace(",","").astype(float)
country


# In[24]:


country.columns


# In[25]:


d.shape


# In[26]:


# first merge by country
d = d.merge(country.loc[country.medianage.notnull(),['country','pop','testpop','medianage']],
            how='left', left_on='Country_Region', right_on='country')
d


# In[27]:


# then merge by province
c1 = country.loc[country.medianage.isnull(),['country','pop','testpop']]
print(c1.shape)
c1.columns = ['Province_State','pop1','testpop1']
# d.update(c1)
d = d.merge(c1,how='left',on='Province_State')
d.loc[d.pop1.notnull(),'pop'] = d.loc[d.pop1.notnull(),'pop1']
d.loc[d.testpop1.notnull(),'testpop'] = d.loc[d.testpop1.notnull(),'testpop1']
d.drop(['pop1','testpop1'], axis=1, inplace=True)
print(d.shape)
print(d.loc[(d.Date=='2020-03-25') & (d['Province_State']=='New York')])


# In[28]:


# testing data time series, us states only, would love to have this for all countries
ct = pd.read_csv(path+'states_daily_4pm_et.csv')
si = pd.read_csv(path+'states_info.csv')
si = si.rename(columns={'name':'Province_State'})
ct = ct.merge(si[['state','Province_State']], how='left', on='state')
ct['Date'] = ct['date'].apply(str).transform(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
ct.loc[ct.Province_State=='US Virgin Islands','Province_State'] = 'Virgin Islands'
ct.loc[ct.Province_State=='District Of Columbia','Province_State'] = 'District of Columbia'
pd.set_option('display.max_rows', 20)
ct
# ct = ct['Date','state','total']


# In[29]:


ckeep = ['positive','negative','totalTestResults']
for c in ckeep: ct[c] = np.log1p(ct[c])


# In[30]:


d = d.merge(ct[['Province_State','Date']+ckeep], how='left',
            on=['Province_State','Date'])
d


# In[31]:


w = pd.read_csv(path+'training_data_with_weather_info_week_2.csv')
w.drop(['Id','Id.1','ConfirmedCases','Fatalities','country+province','day_from_jan_first'], axis=1, inplace=True)
w[cp] = w[cp].fillna('')
wf = list(w.columns[5:])
w


# In[32]:


# since weather data lags behind a day or two, adjust the date to make it contemporaneous
wmax = w.Date.max()
wmax = datetime.strptime(wmax,'%Y-%m-%d').date()
woff = (dmax - wmax).days
print(dmax, wmax, woff)
w['Date'] = (pd.to_datetime(w.Date) + timedelta(woff)).dt.strftime('%Y-%m-%d')
w


# In[33]:


# merge Lat and Long for all times and the time-varying weather data based on date
d = d.merge(w[cp+['Lat','Long']].drop_duplicates(), how='left', on=cp)
w.drop(['Lat','Long'],axis=1,inplace=True)
d = d.merge(w, how='left', on=cpd)
d


# In[34]:


# combine ecdc and nytimes data
ecdc = pd.read_csv(path+'ecdc.csv', encoding = 'latin')
ecdc


# In[35]:


# combine ecdc and nytimes data as extra y0 and y1
# https://opendata.ecdc.europa.eu/covid19/casedistribution/csv
ecdc['Date'] = pd.to_datetime(ecdc[['year','month','day']]).dt.strftime('%Y-%m-%d')
ecdc = ecdc.rename(mapper={'countriesAndTerritories':'Country_Region'}, axis=1)
ecdc['Country_Region'] = ecdc['Country_Region'].replace('_',' ',regex=True)
ecdc['Province_State'] = ''
ecdc['cc'] = ecdc.groupby(cp)['cases'].cummax()
ecdc['extra_y0'] = np.log1p(ecdc.cc)
ecdc['cd'] = ecdc.groupby(cp)['deaths'].cummax()
ecdc['extra_y1'] = np.log1p(ecdc.cd)
ecdc = ecdc[cpd + ['extra_y0','extra_y1']]
ecdc[::63]


# In[36]:


ecdc = ecdc[ecdc.Date >= '2020-01-22']
ecdc


# In[37]:


# https://github.com/nytimes/covid-19-data
nyt = pd.read_csv(path+'us-states.csv')
nyt['extra_y0'] = np.log1p(nyt.cases)
nyt['extra_y1'] = np.log1p(nyt.deaths)
nyt['Country_Region'] = 'US'
nyt = nyt.rename(mapper={'date':'Date','state':'Province_State'},axis=1)
nyt.drop(['fips','cases','deaths'],axis=1,inplace=True)
nyt


# In[38]:


extra = pd.concat([ecdc,nyt], sort=True)
extra


# In[39]:


d = d.merge(extra, how='left', on=cpd)
d


# In[40]:


d[['extra_y0','extra_y1']].describe()


# In[41]:


# recovered data from hopkins, https://github.com/CSSEGISandData/COVID-19
recovered = pd.read_csv(path+'time_series_covid19_recovered_global.csv')
recovered = recovered.rename(mapper={'Country/Region':'Country_Region','Province/State':'Province_State'}, axis=1)
recovered[cp] = recovered[cp].fillna('')
recovered = recovered.drop(['Lat','Long'], axis=1)
recovered


# In[42]:


# replace US row with identical rows for every US state
usp = d.loc[d.Country_Region=='US','Province_State'].unique()
print(usp, len(usp))
rus = recovered[recovered.Country_Region=='US']
rus


# In[43]:


rus = rus.reindex(np.repeat(rus.index.values,len(usp)))
rus.loc[:,'Province_State'] = usp
rus


# In[44]:


recovered =  recovered[recovered.Country_Region!='US']
recovered = pd.concat([recovered,rus]).reset_index(drop=True)
recovered


# In[45]:


# melt and merge
rm = pd.melt(recovered, id_vars=cp, var_name='d', value_name='recov')
rm


# In[46]:


rm['Date'] = pd.to_datetime(rm.d)
rm.drop('d',axis=1,inplace=True)
rm['Date'] = rm['Date'].dt.strftime('%Y-%m-%d')
rm


# In[47]:


d = d.merge(rm, how='left', on=['Country_Region','Province_State','Date'])
d


# In[48]:


d['recov'].describe()


# In[49]:


# approximate US state recovery via proportion of confirmed cases
d['ccsum'] = d.groupby(['Country_Region','Date'])['ConfirmedCases'].transform(lambda x: x.sum())
d.loc[d.Country_Region=='US','recov'] = d.loc[d.Country_Region=='US','recov'] *                                         d.loc[d.Country_Region=='US','ConfirmedCases'] /                                         (d.loc[d.Country_Region=='US','ccsum'] + 1)


# In[50]:


d.loc[:,'recov'] = np.log1p(d.recov)
# d.loc[:,'recov'] = d['recov'].fillna(0)


# In[51]:


d.loc[d.Province_State=='North Carolina','recov'][45:55]


# In[52]:


# log1p transform both targets
ynames = ['ConfirmedCases','Fatalities']
ny = len(ynames)
yv = []
for i in range(ny):
    v = 'y'+str(i)
    d[v] = np.log1p(d[ynames[i]])
    yv.append(v)
print(d[yv].describe())


# In[53]:


d['rate0'] = d.y0 - np.log(d['pop'])
d['rate1'] = d.y1 - np.log(d['pop'])


# In[54]:


d = d.sort_values(['Loc','Date']).reset_index(drop=True)
d.shape


# In[55]:


# compute nearest neighbors
regions = d[['Loc','Lat','Long']].drop_duplicates('Loc').reset_index(drop=True)
regions


# In[56]:


# regions.to_csv('regions.csv', index=False)


# In[57]:


# knn max features
k = kv[0]
nn = NearestNeighbors(k)
nn.fit(regions[['Lat','Long']])


# In[58]:


# first matrix is distances, second indices to nearest neighbors including self
# note two cruise ships are replicated and have identical lat, long values
knn = nn.kneighbors(regions[['Lat','Long']])
knn


# In[59]:


ns = d['Loc'].nunique()


# In[60]:


# time series matrix
ky = d['y0'].values.reshape(ns,-1)
print(ky.shape)

print(ky[0])

# use knn indices to create neighbors
knny = ky[knn[1]]
print(knny.shape)

knny = knny.transpose((0,2,1)).reshape(-1,k)
print(knny.shape)


# In[61]:


# knn max features
nk = len(kv)
kp = []
kd = []
ns = regions.shape[0]
for k in kv:
    nn = NearestNeighbors(k)
    nn.fit(regions[['Lat','Long']])
    knn = nn.kneighbors(regions[['Lat','Long']])
    kp.append('knn'+str(k)+'_')
    kd.append('kd'+str(k)+'_')
    for i in range(ny):
        yi = 'y'+str(i)
        kc = kp[-1]+yi
        # time series matrix
        ky = d[yi].values.reshape(ns,-1)
        # use knn indices to create neighbor matrix
        km = ky[knn[1]].transpose((0,2,1)).reshape(-1,k)
        
        # take maximum value over all neighbors to approximate spreading
        d[kc] = np.amax(km, axis=1)
        print(d[kc].describe())
        print()
        
        # distance to max
        kc = kd[-1]+yi
        ki = np.argmax(km, axis=1).reshape(ns,-1)
        kw = np.zeros_like(ki).astype(float)
        # inefficient indexing, surely some way to do it faster
        for j in range(ns): 
            kw[j] = knn[0][j,ki[j]]
        d[kc] = kw.flatten()
        print(d[kc].describe())
        print()


# In[62]:


ki[j]


# In[63]:


# range of dates for training
# dates = d[~d.y0.isnull()]['Date'].drop_duplicates()
dates = d[d.y0.notnull()]['Date'].drop_duplicates()
dates


# In[64]:


# correlations
cols = []
for i in range(ny):
    yi = yv[i]
    cols.append(yi)
    for k in kp:
        cols.append(k+yi)
d.loc[:,cols].corr()


# In[65]:


d['Date'] = pd.to_datetime(d['Date'])
d['Date'].describe()


# In[66]:


# days since beginning
# basedate = train['Date'].min()
# train['dint'] = train.apply(lambda x: (x.name.to_datetime() - basedate).days, axis=1)
d['dint'] = (d['Date'] - d['Date'].min()).dt.days
d['dint'].describe()


# In[67]:


d.shape


# In[68]:


# reference days since exp(j)th occurrence
for i in range(ny):
    
    for j in range(3):

        ij = str(i)+'_'+str(j)
        
        cut = 2**j if i==0 else j
        
        qd1 = (d[yv[i]] > cut) & (d[yv[i]].notnull())
        d1 = d.loc[qd1,['Loc','dint']]
        # d1.shape
        # d1.head()

        # get min for each location
        d1['dmin'] = d1.groupby('Loc')['dint'].transform(lambda x: x.min())
        # dintmax = d1['dint'].max()
        # print(i,j,'dintmax',dintmax)
        # d1.head()

        d1.drop('dint',axis=1,inplace=True)
        d1 = d1.drop_duplicates()
        d = d.merge(d1,how='left',on=['Loc'])
 
        # if dmin is missing then the series had no occurrences in the training set
        # go ahead and assume there will be one at the beginning of the test period
        # the average time between first occurrence and first death is 14 days
        # if j==0: d[dmi] = d[dmi].fillna(dintmax + 1 + i*14)

        # ref day is days since dmin, must clip at zero to avoid leakage
        d['ref_day'+ij] = np.clip(d.dint - d.dmin, 0, 100000)
        d.drop('dmin',axis=1,inplace=True)

        # asymptotic curve may bin differently
        d['recip_day'+ij] = 1 / (1 + (1 + d['ref_day'+ij])**(-1.0))
    

gc.collect()


# In[69]:


d['dint'].value_counts().std()


# In[70]:


# diffs and rolling means
e = 1
r = 5
for i in range(ny):
    yi = 'y'+str(i)
    dd = '_d'+str(e)
    rr = '_r'+str(r)
    
    d[yi+dd] = d.groupby('Loc')[yi].transform(lambda x: x.diff(e))
    d[yi+rr] = d.groupby('Loc')[yi].transform(lambda x: x.rolling(r).mean())
    d['rate'+str(i)+dd] = d.groupby('Loc')['rate'+str(i)].transform(lambda x: x.diff(e))
    d['rate'+str(i)+rr] = d.groupby('Loc')['rate'+str(i)].transform(lambda x: x.rolling(r).mean())
    d['extra_y'+str(i)+dd] = d.groupby('Loc')['extra_y'+str(i)].transform(lambda x: x.diff(e))
    d['extra_y'+str(i)+rr] = d.groupby('Loc')['extra_y'+str(i)].transform(lambda x: x.rolling(r).mean())

    for k in kp:
        d[k+yi+dd] = d.groupby('Loc')[k+yi].transform(lambda x: x.diff(e))
        d[k+yi+rr] = d.groupby('Loc')[k+yi].transform(lambda x: x.rolling(r).mean())

    for k in kd:
        d[k+yi+dd] = d.groupby('Loc')[k+yi].transform(lambda x: x.diff(e))
        d[k+yi+rr] = d.groupby('Loc')[k+yi].transform(lambda x: x.rolling(r).mean())
        
laglist = ['recov'] + google + wf

for v in laglist:
    d[v+dd] = d.groupby('Loc')[v].transform(lambda x: x.diff(e))
    d[v+rr] = d.groupby('Loc')[v].transform(lambda x: x.rolling(r).mean())


# In[71]:


# final sort before training
d = d.sort_values(['Loc','dint']).reset_index(drop=True)
d.shape


# In[72]:


# initial continuous and categorical features
dogs = []
for i in range(ny):
    for j in range(3):
        dogs.append('ref_day'+str(i)+'_'+str(j))
cats = ['Loc']
print(dogs, len(dogs))
print(cats, len(cats))


# In[73]:


# one-hot encode categorical features
ohef = []
for i,c in enumerate(cats):
    print(c, d[c].nunique())
    ohe = pd.get_dummies(d[c], prefix=c)
    ohec = [f.translate({ord(c): "_" for c in " !@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for f in list(ohe.columns)]
    ohe.columns = ohec
    d = pd.concat([d,ohe],axis=1)
    ohef = ohef + ohec


# In[74]:


d['Loc_US_North_Carolina'].describe()


# In[75]:


d['Loc_US_Colorado'].describe()


# In[76]:


# boosting hyperparameters
params = {}

params[('lgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}
params[('lgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}
params[('xgb','y0')] = {'lambda_l2': 1.9079933811271934, 'max_depth': 5}
params[('xgb','y1')] = {'lambda_l2': 1.690407455211948, 'max_depth': 3}
params[('ctb','y0')] = {'l2_leaf_reg': 1.9079933811271934, 'max_depth': 5}
params[('ctb','y1')] = {'l2_leaf_reg': 1.690407455211948, 'max_depth': 3}


# In[77]:


# must start cas server before running this cell
if 'cas' in booster:
    from swat import *
    s = CAS('server', 1)


# In[78]:


# single horizon validation using one day at a time for 28 days
nb = len(booster)
nls = np.zeros((nhorizon,ny,nb))
rallv = np.zeros((nhorizon,ny,nb))
iallv = np.zeros((nhorizon,ny,nb)).astype(int)
yallv = []
pallv = []
imps = []
 
# loop over horizons
for horizon in range(1+skip,nhorizon+1):
# for horizon in range(4,5):
    
    print()
#     print('*'*20)
#     print(f'horizon {horizon}')
#     print('*'*20)
    
    gc.collect()
    
    hs = str(horizon)
    if horizon < 10: hs = '0' + hs
    
    # build lists of features
    lags = []
    diffs = []
    for i in range(ny):
        yi = 'y'+str(i)
        lags.append(yi)
        lags.append('extra_'+yi)
        lags.append('rate'+str(i))
        lags.append(yi+dd)
        lags.append('extra_'+yi+dd)
        lags.append('rate'+str(i)+dd)
        lags.append(yi+rr)
        lags.append('extra_'+yi+rr)
        lags.append('rate'+str(i)+rr)
        for k in kp:
            lags.append(k+yi)
            lags.append(k+yi+dd)
            lags.append(k+yi+rr)
        for k in kd:
            lags.append(k+yi)
            lags.append(k+yi+dd)
            lags.append(k+yi+rr)
       
    lags.append('recov')
    
    lags = lags + google + wf + ckeep
    
#     cinfo = ['pop', 'tests', 'testpop', 'density', 'medianage',
#        'urbanpop', 'hospibed', 'smokers']
    cinfo0 = ['testpop']
    cinfo1 = ['testpop','medianage']
    
    f0 = dogs + lags + cinfo0 + ohef
    f1 = dogs + lags + cinfo1 + ohef
    
    # remove some features based on validation experiments
    f0 = [f for f in f0 if not f.startswith('knn11') and not f.startswith('kd')          and not f.startswith('rate') and not f.endswith(dd) and not f.endswith(rr)]
    f1 = [f for f in f1 if not f.startswith('knn6') and not f.startswith('kd6')]
    
    # remove any duplicates
    # f0 = list(set(f0))
    # f1 = list(set(f1))
    
    features = []
    features.append(f0)
    features.append(f1)
    
    nf = []
    for i in range(ny):
        nf.append(len(features[i]))
        # print(nf[i], features[i][:10])
        
    qtrain = d['Date'] <= ddate.isoformat()

    vdate = ddate + timedelta(days=horizon)
    qval = d['Date'] == vdate.isoformat()
    qvallag = d['Date'] == ddate.isoformat()
    
    x_train = d[qtrain].copy()
    # make y training data monotonic nondecreasing
    y_train = []
    for i in range(ny):
        y_train.append(pd.Series(d.loc[qtrain,['Loc',yv[i]]].groupby('Loc')[yv[i]].cummax()))

    x_val = d[qval].copy()
    y_val = [d.loc[qval,'y0'].copy(), d.loc[qval,'y1'].copy()]
    yallv.append(y_val)
    
    # lag features
    x_train.loc[:,lags] = x_train.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))
    x_val.loc[:,lags] = d.loc[qvallag,lags].values

    print()
    print(horizon, 'x_train', x_train.shape)
    print(horizon, 'x_val', x_val.shape)
    
    if train_full:
        
        qfull = (d['Date'] <= tmax)
        
        tdate = dmax + timedelta(days=horizon)
        qtest = d['Date'] == tdate.isoformat()
        qtestlag = d['Date'] == dmax.isoformat()
    
        x_full = d[qfull].copy()
        
        # make y training data monotonic nondecreasing
        y_full = []
        for i in range(ny):
            y_full.append(pd.Series(d.loc[qfull,['Loc',yv[i]]].groupby('Loc')[yv[i]].cummax()))
        
        x_test = d[qtest].copy()
        
        # lag features
        x_full.loc[:,lags] = x_full.groupby('Loc')[lags].transform(lambda x: x.shift(horizon))
        x_test.loc[:,lags] = d.loc[qtestlag,lags].values

        print(horizon, 'x_full', x_full.shape)
        print(horizon, 'x_test', x_test.shape)

    train_set = []
    val_set = []
    ny = len(y_train)

#     for i in range(ny):
#         train_set.append(xgb.DMatrix(x_train[features[i]], y_train[i]))
#         val_set.append(xgb.DMatrix(x_val[features[i]], y_val[i]))

    gc.collect()

    # loop over multiple targets
    mod = []
    pred = []
    rez = []
    iters = []
    
    for i in range(ny):
#     for i in range(1):
        print()
        print('*'*40)
        print(f'horizon {horizon} {yv[i]} {ynames[i]} {vdate}')
        print('*'*40)
        
        # use catboost only for y1
        # nb = 2 if i==0 else 3
       
        # matrices to store predictions
        vpm = np.zeros((x_val.shape[0],nb))
        tpm = np.zeros((x_test.shape[0],nb))
        
        for b in range(nb):
            
            if booster[b] == 'cas':
                
                x_train['Partition'] = 1
                x_val['Partition'] = 0
                x_cas_all = pd.concat([x_train, x_val], axis=0)
                # make copy of target since it is also used for lags
                x_cas_all['target'] = pd.concat([y_train[i], y_val[i]], axis=0).values
                s.upload(x_cas_all, casout="x_cas_val")

                target = 'target'
                inputs = features[i]
                inputs.append(target)

                s.loadactionset("autotune")
                res=s.autotune.tuneGradientBoostTree (
                    trainOptions = {
                        "table":{"name":'x_cas_val',"where":"Partition=1"},
                        "target":target,
                        "inputs":inputs,
                        "casOut":{"name":"model", "replace":True}
                    },
                    scoreOptions = {
                        "table":{"name":'x_cas_val', "where":"Partition=0"},
                        "model":{"name":'model'},
                        "casout":{"name":"x_valid_preds","replace":True},
                        "copyvars": ['Id','Loc','Date']
                    },
                    tunerOptions = {
                        "seed":54321,  
                        "objective":"RASE", 
                        "userDefinedPartition":True 
                    }
                )
                print()
                print(res.TunerSummary)
                print()
                print(res.BestConfiguration)        

                TunerSummary=pd.DataFrame(res['TunerSummary'])
                TunerSummary["Value"]=pd.to_numeric(TunerSummary["Value"])
                BestConf=pd.DataFrame(res['BestConfiguration'])
                BestConf["Value"]=pd.to_numeric(BestConf["Value"])
                vpt = s.CASTable("x_valid_preds").to_frame()
                #FG: resort the CAS predictions by Id
                vpt = vpt.sort_values(['Loc','Date']).reset_index(drop=True)
                vp = vpt['P_target'].values

                s.dropTable("x_cas_val")
                s.dropTable("x_valid_preds")
                
            else:
                # scikit interface automatically uses best model for predictions
                params[(booster[b],yv[i])]['n_estimators'] = 5000
                if booster[b]=='lgb':
                    model = lgb.LGBMRegressor(**params[(booster[b],yv[i])]) 
                elif booster[b]=='xgb':
                    model = xgb.XGBRegressor(**params[(booster[b],yv[i])])
                else:
                    # hack for categorical features, ctb must be last in booster list
                    features[i] = features[i][:-294] + ['Loc']
                    params[(booster[b],yv[i])]['cat_features'] = ['Loc']
                    model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])
                    
                model.fit(x_train[features[i]], y_train[i],
                                  eval_set=[(x_train[features[i]], y_train[i]),
                                            (x_val[features[i]], y_val[i])],
                                  early_stopping_rounds=30,
                                  verbose=False)

                vp = model.predict(x_val[features[i]])
                
                iallv[horizon-1,i,b] = model._best_iteration if booster[b]=='lgb' else                                        model.best_iteration if booster[b]=='xgb' else                                        model.best_iteration_

                gain = model.feature_importances_
        #         gain = model.get_score(importance_type='gain')
        #         split = model.get_score(importance_type='weight')   
            #     gain = model.feature_importance(importance_type='gain')
            #     split = model.feature_importance(importance_type='split').astype(float)  
            #     imp = pd.DataFrame({'feature':features,'gain':gain,'split':split})
                imp = pd.DataFrame({'feature':features[i],'gain':gain})
        #         imp = pd.DataFrame({'feature':features[i]})
        #         imp['gain'] = imp['feature'].map(gain)
        #         imp['split'] = imp['feature'].map(split)

                imp.set_index(['feature'],inplace=True)

                imp.gain /= np.sum(imp.gain)
        #         imp.split /= np.sum(imp.split)

                imp.sort_values(['gain'], ascending=False, inplace=True)

                print()
                print(imp.head(n=10))
                # print(imp.shape)

                imp.reset_index(inplace=True)
                imp['horizon'] = horizon
                imp['target'] = yv[i]
                imp['set'] = 'valid'
                imp['booster'] = booster[b]

                mod.append(model)
                imps.append(imp)
                
            # china rule, last observation carried forward, set to zero here
            qcv = (x_val['Country_Region'] == 'China') &                   (x_val['Province_State'] != 'Hong Kong') &                   (x_val['Province_State'] != 'Macau')
            vp[qcv] = 0.0

            # make sure horizon 1 prediction is not smaller than first lag
            # because we know series is monotonic
            # if horizon==1+skip:
            if True:
                a = np.zeros((len(vp),2))
                a[:,0] = vp
                a[:,1] = x_val[yv[i]].values
                vp = np.nanmax(a,axis=1)
            
            val_score = np.sqrt(mean_squared_error(vp, y_val[i]))
            vpm[:,b] = vp
            
            print()
            print(f'{booster[b]} validation rmse {val_score:.6f}')
            rallv[horizon-1,i,b] = val_score

            gc.collect()
    
#             break

            if train_full:
                
                print()
                print(f'{booster[b]} training with full data and predicting', tdate.isoformat())
                    
                if booster[b] == 'cas':
                    
                    x_full['target'] = y_full[i].values
                    s.upload(x_full, casout="x_full")
                    # use hyperparameters from validation fit
                    s.loadactionset("decisionTree")
                    result = s.gbtreetrain(
                        table={"name":'x_full'},
                        target=target,
                        inputs= inputs,
                        varimp=True,
                        ntree=BestConf.iat[0,2], 
                        m=BestConf.iat[1,2],
                        learningRate=BestConf.iat[2,2],
                        subSampleRate=BestConf.iat[3,2],
                        lasso=BestConf.iat[4,2],
                        ridge=BestConf.iat[5,2],
                        nbins=BestConf.iat[6,2],
                        maxLevel=BestConf.iat[7,2],
                        #quantileBin=True,
                        seed=326146718,
                        #savestate={"name":"aStore","replace":True}
                        casOut={"name":'fullmodel', "replace":True}
                        ) 

                    s.upload(x_test, casout="x_test_cas")

                    s.decisionTree.gbtreeScore(
                        modelTable={"name":"fullmodel"},        
                        table={"name":"x_test_cas"},
                        casout={"name":"x_test_preds","replace":True},
                        copyvars= ['Loc','Date']
                        ) 
                    # save test predictions back into main table
                    forecast = s.CASTable("x_test_preds").to_frame()
                    forecast = forecast.sort_values(['Loc','Date']).reset_index(drop=True)
                    tp = forecast['_GBT_PredMean_'].values
                    
                    s.dropTable("x_full")
                    s.dropTable("x_test_cas")
                     
                else:
                
                    # use number of iterations from validation fit
                    params[(booster[b],yv[i])]['n_estimators'] = iallv[horizon-1,i,b]
                    if booster[b]=='lgb':
                        model = lgb.LGBMRegressor(**params[(booster[b],yv[i])])
                    elif booster[b]=='xgb':
                        model = xgb.XGBRegressor(**params[(booster[b],yv[i])])
                    else:
                        model = ctb.CatBoostRegressor(**params[(booster[b],yv[i])])
                    
                    model.fit(x_full[features[i]], y_full[i], verbose=False)
                    
                    params[(booster[b],yv[i])]['n_estimators'] = 5000

                    tp = model.predict(x_test[features[i]])
                
                    gain = model.feature_importances_
            #         gain = model.get_score(importance_type='gain')
            #         split = model.get_score(importance_type='weight')   
                #     gain = model.feature_importance(importance_type='gain')
                #     split = model.feature_importance(importance_type='split').astype(float)  
                #     imp = pd.DataFrame({'feature':features,'gain':gain,'split':split})
                    imp = pd.DataFrame({'feature':features[i],'gain':gain})
            #         imp = pd.DataFrame({'feature':features[i]})
            #         imp['gain'] = imp['feature'].map(gain)
            #         imp['split'] = imp['feature'].map(split)

                    imp.set_index(['feature'],inplace=True)

                    imp.gain /= np.sum(imp.gain)
            #         imp.split /= np.sum(imp.split)

                    imp.sort_values(['gain'], ascending=False, inplace=True)

                    print()
                    print(imp.head(n=10))
                    # print(imp.shape)

                    imp.reset_index(inplace=True)
                    imp['horizon'] = horizon
                    imp['target'] = yv[i]
                    imp['set'] = 'full'
                    imp['booster'] = booster[b]

                    imps.append(imp)

                # china rule, last observation carried forward, set to zero here
                qct = (x_test['Country_Region'] == 'China') &                       (x_test['Province_State'] != 'Hong Kong') &                       (x_test['Province_State'] != 'Macau')
                tp[qct] = 0.0

                # make sure first horizon prediction is not smaller than first lag
                # because we know series is monotonic
                # if horizon==1+skip:
                if True:
                    a = np.zeros((len(tp),2))
                    a[:,0] = tp
                    a[:,1] = x_test[yv[i]].values
                    tp = np.nanmax(a,axis=1)

                tpm[:,b] = tp
                
                gc.collect()
                
        # nonnegative least squares to estimate ensemble weights
        x, rnorm = nnls(vpm, y_val[i])
        
        # smooth weights by shrinking towards all equal
        # x = (x + np.ones(3)/3.)/2

        # smooth weights with rolling mean, ewma
        # if horizon+skip > 1: x = (x + nls[horizon+skip-2,i])/2
        alpha = 0.1
        if horizon+skip > 1: x = alpha * x + (1 - alpha) * nls[horizon+skip-2,i]

        nls[horizon+skip-1,i] = x
        
        val_pred = np.matmul(vpm, x)
        test_pred = np.matmul(tpm, x)
        
        # china rule in case weights do not sum to 1
        # val_pred[qcv] = vpm[:,0][qcv]
        # test_pred[qcv] = tpm[:,0][qct]
        
        # save validation and test predictions back into main table
        d.loc[qval,yv[i]+'_pred'] = val_pred
        d.loc[qtest,yv[i]+'_pred'] = test_pred

        # ensemble validation score
        # val_score = np.sqrt(rnorm/vpm.shape[0])
        val_score = np.sqrt(mean_squared_error(val_pred, y_val[i]))
        
        rez.append(val_score)
        pred.append(val_pred)

    pallv.append(pred)
    
    # nnls weights
    w0 = ''
    w1 = ''
    for b in range(nb):
        w0 = w0 + f' {nls[horizon-1,0,b]:.2f}'
        w1 = w1 + f' {nls[horizon-1,1,b]:.2f}'
        
    print()
    print('Validation RMSLE')
    print(f'{ynames[0]} \t {rez[0]:.6f}  ' + w0)
    print(f'{ynames[1]} \t {rez[1]:.6f}  ' + w1)
    print(f'Mean \t \t {np.mean(rez):.6f}')

#     # break down RMSLE by day
#     rp = np.zeros((2,7))
#     for i in range(ny):
#         for di in range(50,57):
#             j = di - 50
#             qf = x_val.dint == di
#             rp[i,j] = np.sqrt(mean_squared_error(pred[i][qf], y_val[i][qf]))
#             print(i,di,f'{rp[i,j]:.6f}')
#         print(i,f'{np.mean(rp[i,:]):.6f}')
#         plt.plot(rp[i])
#         plt.title(ynames[i] + ' RMSLE')
#         plt.show()
        
    # plot actual vs predicted
    plt.figure(figsize=(10, 5))
    for i in range(ny):
        plt.subplot(1,2,i+1)
        plt.plot([0, 12], [0, 12], 'black')
        plt.plot(pred[i], y_val[i], '.')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(ynames[i])
        plt.grid()
    plt.show()
    
# save one big table of importances
impall = pd.concat(imps)

# remove number suffixes from lag names to aid in analysis
# impall['feature1'] = impall['feature'].replace(to_replace='lag..', value='lag', regex=True)

os.makedirs('imp', exist_ok=True)
fname = 'imp/' + mname + '_imp.csv'
impall.to_csv(fname, index=False)
print()
print(fname, impall.shape)

# save scores and weights
os.makedirs('rez', exist_ok=True)
fname = 'rez/' + mname+'_rallv.npy'
np.save(fname, rallv)
print(fname, rallv.shape)

fname = 'rez/' + mname+'_nnls.npy'
np.save(fname, nls)
print(fname, nls.shape)


# In[79]:


if 'cas' in booster: s.shutdown()


# In[80]:


tdate.isoformat()


# In[81]:


rf = [f for f in features[0] if f.startswith('ref')]
d[rf].describe()


# In[82]:


np.mean(iallv, axis=0)


# In[83]:


plt.figure(figsize=(10, 8))
for i in range(ny):
    plt.subplot(2,2,1+i)
    plt.plot(rallv[:,i])
    plt.title(ynames[i] + ' RMSLE vs Horizon')
    plt.grid()
    
    plt.subplot(2,2,3+i)
    plt.plot(nls[:,i])
    plt.title(ynames[i] + ' Ensemble Weights')
    plt.grid()
plt.show()


# In[84]:


# compute validation rmsle
m = 0
locs = d.loc[:,['Loc','Country_Region','Province_State']].drop_duplicates().reset_index(drop=True)
# locs = x_val.copy().reset_index(drop=True)
# print(locs.shape)
y_truea = []
y_preda = []

print(f'# {mname}')
for i in range(ny):
    y_true = []
    y_pred = []
    for j in range(nhorizon-skip):
        y_true.append(yallv[j][i])
        y_pred.append(pallv[j][i])
    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    # print(y_pred.shape)
    # make each series monotonic increasing
    for j in range(y_pred.shape[1]): 
        y_pred[:,j] = np.maximum.accumulate(y_pred[:,j])
    # copy updated predictions into main table
    for horizon in range(1+skip,nhorizon+1):
        vdate = ddate + timedelta(days=horizon)
        qval = d['Date'] == vdate.isoformat()
        d.loc[qval,yv[i]+'_pred'] = y_pred[horizon-1-skip]
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    print(f'# {rmse:.6f}')
    m += rmse/2
    locs['rmse'+str(i)] = np.sqrt(np.mean((y_true-y_pred)**2, axis=0))
    y_truea.append(y_true)
    y_preda.append(y_pred)
print(f'# {m:.6f}')


# In[85]:


# gbt1u
# 0.626124
# 0.404578
# 0.515351


# In[86]:


# sort to find worst predictions of y0
locs = locs.sort_values('rmse0', ascending=False)
locs[:10]


# In[87]:


# plot worst fits of y0
for i in range(5):
    li = locs.index[i]
    plt.plot(y_truea[0][:,li])
    plt.plot(y_preda[0][:,li])
    plt.title(locs.loc[li,'Loc'])
    plt.show()


# In[88]:


# plt.plot(d.loc[d.Loc=='Belgium','y0'][39:])


# In[89]:


# sort to find worst predictions of y1
locs = locs.sort_values('rmse1', ascending=False)
locs[:10]


# In[90]:


# plot worst fits of y1
for i in range(5):
    li = locs.index[i]
    plt.plot(y_truea[1][:,li])
    plt.plot(y_preda[1][:,li])
    plt.title(locs.loc[li,'Loc'])
    plt.show()


# In[91]:


tmax


# In[92]:


# enforce monotonicity of forecasts in test set after last date in training
loc = d['Loc'].unique()
for l in loc:
    # q = (d.Loc==l) & (d.ForecastId > 0)
    q = (d.Loc==l) & (d.Date > tmax)
    for yi in yv:
        yp = yi+'_pred'
        d.loc[q,yp] = np.maximum.accumulate(d.loc[q,yp])


# In[93]:


# plot actual and predicted curves over time for specific locations
locs = ['China Tibet','China Xinjiang','China Hong Kong', 'China Macau',
        'Spain','Italy','India',
        'US Washington','US New York','US California',
        'US North Carolina','US Ohio']
xlab = ['03-12','03-18','03-25','04-01','04-08','04-15','04-22']
for loc in locs:
    plt.figure(figsize=(14,2))
    
    # fig, ax = plt.subplots()
    # fig.autofmt_xdate()
    
    for i in range(ny):
    
        plt.subplot(1,2,i+1)
        plt.plot(d.loc[d.Loc==loc,[yv[i],'Date']].set_index('Date'))
        plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred','Date']].set_index('Date'))
        # plt.plot(d.loc[d.Loc==loc,[yv[i]]])
        # plt.plot(d.loc[d.Loc==loc,[yv[i]+'_pred']])
        # plt.xticks(np.arange(len(xlab)), xlab, rotation=-45)
        # plt.xticks(np.arange(12), calendar.month_name[3:5], rotation=20)
        # plt.xticks(rotation=-45)
        plt.xticks([])
        plt.title(loc + ' ' + ynames[i])
       
    plt.show()


# In[94]:


pd.set_option('display.max_rows', 100)
loc = 'China Xinjiang'
d.loc[d.Loc==loc,['Date',yv[0],yv[0]+'_pred']]


# In[95]:


tmax


# In[96]:


fmin


# In[97]:


# compute public lb score
q = (d.Date >= fmin) & (d.Date <= tmax)
print(f'# {tmax} {sum(q)/ns} {mname}')
s0 = np.sqrt(mean_squared_error(d.loc[q,'y0'],d.loc[q,'y0_pred']))
s1 = np.sqrt(mean_squared_error(d.loc[q,'y1'],d.loc[q,'y1_pred']))
print(f'# CC \t {s0:.6f}')
print(f'# Fa \t {s1:.6f}')
print(f'# Mean \t {(s0+s1)/2:.6f}')


# In[98]:


# 2020-03-31 13.0 gbt1u
# CC 	 0.744772
# Fa 	 0.567829
# Mean 	 0.656301


# In[99]:


# create submission
sub = d.loc[d.ForecastId > 0, ['ForecastId','y0_pred','y1_pred']]
sub['ConfirmedCases'] = np.expm1(sub['y0_pred'])
sub['Fatalities'] = np.expm1(sub['y1_pred'])
sub.drop(['y0_pred','y1_pred'],axis=1,inplace=True)
os.makedirs('sub',exist_ok=True)
# fname = 'sub/' + mname + '.csv'
fname = 'submission.csv'
sub.to_csv(fname, index=False)
print(fname, sub.shape)


# In[100]:


sub.describe()


# In[101]:


sub


# In[102]:


# save predictions
ovars = ['Id','ForecastId','Country_Region','State_Provice','Loc','y0','y1','y0_pred','y1_pred']
oof = d.loc[:,ovars]
oof = oof.rename(mapper={'y0_pred':mname+'0','y1_pred':mname+'1'}, axis=1)
os.makedirs('oof',exist_ok=True)
fname = 'oof/' + mname + '.csv'
oof.to_csv(fname, index=False)
print(fname, oof.shape)


# In[103]:


if save_data:
    os.makedirs('data',exist_ok=True)
    fname = 'data/' + mname + '_d.csv'
    d.to_csv(fname, index=False)
    print(fname, d.shape)


# In[ ]:




