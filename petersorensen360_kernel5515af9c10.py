#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.optimizers import Nadam
from datetime import timedelta
import csv
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

datapath       = '../input/covid19-global-forecasting-week-4/'
datapath2      = '../input/worldpopulationinfo/'
datapath3      = '../input/country-ppp/'
datapath4      = '../input/populationandcountryinfo/'
datapath5      = '../input/usstateland/'
datapath_week1 = '../input/covid19week1/'

add_other = True
CURVE_SMOOTHING = True
USE_NEW = False

normfactor = 1.0
NUM_SHIFT = 25
days_shift =[0,1,2,3,5,7,10,14,18,22,27,32,37,42]
days_shift =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,22,24,26,28,30,32,34,36]
#days_shift =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]  #virkede med 5
NUM_MODELS = 10
TRAIN_START_DAY = 1   # betyder reelt ingenting nå rvi tager så meget med bagud

TARGETS = ["ConfirmedCases", "Fatalities"]
if USE_NEW:
    TARGETS = ["NewCases", "NewFatalities"]
    
loc_group = ["Province_State", "Country_Region"] # in part1 

if add_other:
    base_features = ['ages 0-14', 'ages 15-64', 'ages 64-',
       'Population (2020)', 'Density (P/Km²)', 'Med. Age', 'Urban Pop %',
       'Apr', 'year', 'Low Temp', 'ppp', 'Season',
       'R_Africa','R_China',
       'R_Australia and New Zealand', 'R_Eastern Asia', 'R_Europe',
       'R_Latin America and the Caribbean', 'R_Northern America', 'R_Oceania',
       'R_Southern Asia', 'R_Western Asia',
       'I_High income', 'I_Low income', 'I_Lower middle income',
       'I_Upper middle income']
    base_features = ['ages 64-',
        'Urban Pop %',
       'Season','Apr','ppp']
#    ,
#       'R_Africa',
#       'R_Australia and New Zealand', 'R_Eastern Asia', 'R_Europe',
#       'R_Latin America and the Caribbean', 'R_Northern America', 'R_Oceania',
#       'R_Southern Asia', 'R_Western Asia']

    if USE_NEW:
        base_features =['NewFatalities_ppm','NewCases_ppm']    + base_features
    else:
        a=1
        #base_features =['Fatalities_ppm','ConfirmedCases_ppm'] + base_features
else:
    base_features=[]

def get_shift_features():
    shift_features = []
    for s in range(1, NUM_SHIFT+1):
        for col in TARGETS:
            shift_features.append("prev_{}_{}".format(col, days_shift[s]))
    return shift_features

shift_features = get_shift_features()
prev_targets   = shift_features[0:2] # f.eks. 'prev_ConfirmedCases_1', 'prev_Fatalities_1']
    

def fill_shift_columns(df,targets):
    for s in range(1, NUM_SHIFT+1):    #1->5   # Laver shiftede kolonner
        for col in targets:
            df["prev_{}_{}".format(col, days_shift[s])] = df.groupby(loc_group)[col].shift(days_shift[s])
    return df


def add_seasons(coor_df):
    coor_df["Season"] = 0
    mask = coor_df["Lat"]>20
    coor_df.loc[mask,"Season"] = 1
    mask = coor_df["Lat"]<-20
    coor_df.loc[mask,"Season"] = -1
    return coor_df

def preprocess(df):
    df["Date"] = df["Date"].astype("datetime64[ms]")
    df["days"] = (df["Date"] - pd.to_datetime("2020-01-01")).dt.days

    for col in loc_group:
        df[col].fillna("none", inplace=True)    # countries with no information in a "none" group
    return df

def print_rows_with_nan(df,features=False):
    if not features:
        dyt = df.isna()
        dyt2 = dyt.any(axis=1)
        dyt2.sum()
        #print(dyt2)
        print(df.loc[dyt2])
    else:
        dyt = df[features].isna()
        dyt2 = dyt.any(axis=1)
        dyt2.sum()
        #print(dyt2)
        print( df[features].loc[dyt2])
    #df.loc[dyt2,['Country_Region','Province_State','ConfirmedCases','Fatalities','Date','prev_NewCases_1']]
    return

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def has_nan(df,features=False):
    if not features:
        dyt = df.isnull()
    else:
        dyt = df[features].isnull()
    dyt2 = dyt.any(axis=1)
    return dyt2.sum()
# NUMPY VERSION:     if np.isnan(np.sum(df)):

def print_washinton(df,features,values):
    print(len(values))
    if len(values)== 1:
        mask1 = df[features[0]]==values[0]
        print(df.loc[mask1,features+["Date","ConfirmedCases","Fatalities"]].tail(50))
    elif len(values)== 2:
        mask1 = df[features[0]]==values[0]
        mask2 = df[features[1]]==values[1]
        mask4 = pd.concat((mask1,mask2), axis=1)
        mask5 = mask4.all(axis=1)
        print(df.loc[mask5,features+["Date","ConfirmedCases","Fatalities"]].tail(50))
    else: 
        mask1 = df[features[0]]==values[0]
        mask2 = df[features[1]]==values[1]
        mask3 = df[features[2]]==values[2]
        mask4 = pd.concat((mask1,mask2,mask3), axis=1)
        mask5 = mask4.all(axis=1)
        print(df.loc[mask5,["Date"]])
    return
    
   
    
def find_anomalities(df):
#    df[['Cases_test','Fatality_test']] =  df.groupby(['Country_Region', 'Province_State']) \
#                [['ConfirmedCases','Fatalities']].shift(1) 
    df[['Cases_test']] =  df.groupby([ 'Country_Region', 'Province_State'])                 [['ConfirmedCases']].shift(1) 
    df[['Fatality_test']] =  df.groupby(['Country_Region', 'Province_State'])                 [['Fatalities']].shift(1) 

    mask1 =  df['ConfirmedCases'] < df['Cases_test']
    mask2 =  df['Fatalities']     < df['Fatality_test']
    print("ANOMALITIES ConfirmedCases")
    print(df.loc[mask1])
    print("ANOMALITIES Fatalities")
    print(df.loc[mask2])
    print("SLUT")
    return

def correct_anomalities(df):
    df[['Cases_testm']] =  df.groupby([ 'Country_Region', 'Province_State'])                 [['ConfirmedCases']].shift(1) 
    df[['Fatalities_testm']] =  df.groupby(['Country_Region', 'Province_State'])                 [['Fatalities']].shift(1) 
    df[['Cases_testp']] =  df.groupby([ 'Country_Region', 'Province_State'])                 [['ConfirmedCases']].shift(-1) 
    df[['Fatalities_testp']] =  df.groupby(['Country_Region', 'Province_State'])                 [['Fatalities']].shift(-1) 
    print("CORRECTING ConfirmedCases")
    #dyk
    mask1 =  df['ConfirmedCases'] < df['Cases_testm']
    mask2 =  df['Cases_testp']    >= df['Cases_testm']
    mask3 =  pd.concat((mask1,mask2),axis=1)
    mask4 =  mask3.all(axis=1)
    df.loc[mask4,'ConfirmedCases'] = 0.5* (df.loc[mask4,'Cases_testp']+df.loc[mask4,'Cases_testm'])
    print(df.loc[mask4,['Country_Region', 'Province_State','Cases_testm','ConfirmedCases','Cases_testp']])
    
    #spids
    mask1 =  df['ConfirmedCases'] > df['Cases_testm']
    mask2 =  df['Cases_testp']    == df['Cases_testm']
    mask3 =  pd.concat((mask1,mask2),axis=1)
    mask4a =  mask3.all(axis=1)
    df.loc[mask4a,'ConfirmedCases'] = 0.5* (df.loc[mask4a,'Cases_testp']+df.loc[mask4a,'Cases_testm'])
    print(df.loc[mask4a,['Country_Region', 'Province_State','Cases_testm','ConfirmedCases','Cases_testp']])  
    print("CORRECTED ConfirmedCases:", mask4.sum(),"+", mask4a.sum())
    
    print("CORRECTING Fatalities")
    mask1 =  df['Fatalities']       < df['Fatalities_testm']
    mask2 =  df['Fatalities_testp'] >= df['Fatalities_testm']
    mask3 =  pd.concat((mask1,mask2),axis=1)
    mask4 =  mask3.all(axis=1)
    df.loc[mask4,'Fatalities'] = 0.5* (df.loc[mask4,'Fatalities_testp']+df.loc[mask4,'Fatalities_testm'])
    print(df.loc[mask4,['Country_Region', 'Province_State','Fatalities_testm','Fatalities','Fatalities_testp']])

    mask1 =  df['Fatalities']       >  df['Fatalities_testm']
    mask2 =  df['Fatalities_testp'] == df['Fatalities_testm']
    mask3 =  pd.concat((mask1,mask2),axis=1)
    mask4a =  mask3.all(axis=1)
    df.loc[mask4a,'Fatalities'] = 0.5* (df.loc[mask4a,'Fatalities_testp']+df.loc[mask4a,'Fatalities_testm'])
    print(df.loc[mask4a,['Country_Region', 'Province_State','Fatalities_testm','Fatalities','Fatalities_testp']])
    print("CORRECTED Fatalities:",mask4.sum(),"+",mask4a.sum())
    
    #MULTIPLE FEJL
    print("CORRECTING Series (in Mix)")
    cases = 1
    sumcases = 0
    while cases > 0:
        mask1 =  df['ConfirmedCases'] < df['Cases_testm']
        df.loc[mask1,'ConfirmedCases'] = df.loc[mask1,'Cases_testm']
        print("Cases",df.loc[mask1,['Country_Region', 'Province_State','Cases_testm','ConfirmedCases']])

        mask2 =  df['Fatalities']  < df['Fatalities_testm']
        df.loc[mask2,'Fatalities'] = df.loc[mask2,'Fatalities_testm']
        print("Fatas",df.loc[mask2,['Country_Region', 'Province_State','Fatalities_testm','Fatalities']])
        
        cases = mask1.sum()+mask2.sum()
        sumcases+=cases
        print
        df[['Cases_testm']] =  df.groupby([ 'Country_Region', 'Province_State'])                     [['ConfirmedCases']].shift(1) 
        df[['Fatalities_testm']] =  df.groupby(['Country_Region', 'Province_State'])                     [['Fatalities']].shift(1) 

    df.drop([ 'Cases_testm', 'Fatalities_testm', 'Cases_testp','Fatalities_testp'],axis=1,inplace=True)
    print("Corrected in Series in all",sumcases)
    return 


# In[2]:



population_by_age_df  = pd.read_csv(datapath4 + "population_age_info.csv")
population_by_age_df.drop('ID',axis=1,inplace=True)
population_by_age_df[['ages 0-14', 'ages 15-64','Density (P/Km²)','Med. Age', 'ages 64-','Urban Pop %']] =             population_by_age_df[['ages 0-14', 'ages 15-64','Density (P/Km²)','Med. Age', 'ages 64-','Urban Pop %']].apply(lambda x: x.fillna(x.mean()))

translations = [["Brunei Darussalam","Brunei"],
                ["Myanmar","Bruma"],
                ["Gambia, The","Gambia"],
                ["Egypt, Arab Rep.","Egypt"],
                ["Congo, Rep.","Congo (Brazzaville)"],
                ["Congo, Dem. Rep.","Congo (Kinshasa)"],
                ["Iran, Islamic Rep.","Iran"],
                ["Korea, Rep.","Korea, South"],
                ["Russian Federation","Russia"],
                ["Syrian Arab Republic","Syria"],
                ["Venezuela, RB","Venezuela"],
                ["Slovak Republic","Slovakia"]]
for post in translations:
    #print(post[0],post[1])
    mask = population_by_age_df["Country_Region"] == post[0]
    population_by_age_df.loc[mask,"Country_Region"] = post[1]

usstates_land       = pd.read_csv(datapath5 + "us-state-land.csv", sep='\t')
usstates_population = pd.read_csv(datapath2 + "us-state-population.csv", sep='\t')
usstates_info       = pd.merge(usstates_land,usstates_population, on=['Province_State'], how='left')#, indicator=True)
usstates_info['Country_Region'] = 'US'
usstates_info["Province_State"].replace( '_',' ', regex=True,inplace=True)  # _ var indført for at få den til at læse
#print(usstates_info)

supp_info = pd.read_csv(datapath2 + "population_info_supplement.csv")
supp_info["Country_Region"].replace( ':',',', regex=True,inplace=True)  # _ var indført for at få den til at læse
#print(supp_info)

CountryRegion         = pd.read_csv(datapath2+'CountryRegion.csv')
CountryRegion.drop('ID',axis=1,inplace=True)

state_temperatures    = pd.read_csv(datapath4+'state_temperature_info.csv')
state_temperatures.drop('ID',axis=1,inplace=True)
state_temperatures["Province_State"].fillna("", inplace=True)    # countries with no information in a "" group

#
ppp_tabel = pd.read_csv(datapath3 + 'Country_PPP.csv', sep='\s+')#.sort_values(by=['Country'])
ppp_tabel.drop('Id', 1,inplace=True)
ppp_tabel = ppp_tabel.append({'Country' : 'Burma' , 'ppp' : 8000} , ignore_index=True)
ppp_tabel = ppp_tabel.append({'Country' : 'MS_Zaandam' , 'ppp' : 40000} , ignore_index=True)
ppp_tabel = ppp_tabel.append({'Country' : 'West_Bank_and_Gaza' , 'ppp' : 20000} , ignore_index=True)
ppp_tabel["Country"].replace( '_',' ', regex=True,inplace=True)  # _ var indført for at få den til at læse
ppp_tabel["Country"].replace( 'United States','US', regex=True,inplace=True)  # _ var indført for at få den til at læse
ppp_tabel.rename(columns={'Country':'Country_Region'},inplace=True)
ppp_tabel.sort_values('Country_Region',inplace=True)


# Week1 data are aquired to get the (latitude,longitude) coordinates information for countries
coor_df = pd.read_csv(datapath_week1 + "train.csv").rename(columns={"Country/Region": "Country_Region","Province/State":"Province_State"})
coor_df["Province_State"].fillna("", inplace=True)    # countries with no information in a "" group
coor_df = coor_df[coor_df["Country_Region"].notnull()]
coor_df = coor_df.groupby(["Country_Region","Province_State"])[["Lat", "Long"]].mean().reset_index()
coor_df = add_seasons(coor_df)

coor_country = coor_df.groupby(["Country_Region"])[["Lat", "Long"]].mean().reset_index()
coor_country = add_seasons(coor_country)
coor_country.rename(columns={'Season':'Season2'},inplace=True)

coor_df = pd.merge(coor_df,coor_country, on=['Country_Region'], how='left')#, indicator=True)


# In[3]:



def add_other_info(df,istest_df,usstates_info,supp_info,population_by_age_df,CountryRegion,state_temperatures,ppp_tabel,coor_df):
    
    df = pd.merge(df, usstates_info, on=['Country_Region','Province_State'], how='left')#, indicator=True)

    df = pd.merge(df, population_by_age_df, on=['Country_Region'], how='left')#, indicator=True)
    df[['ages 0-14', 'ages 15-64','Density (P/Km²)','Med. Age', 'ages 64-','Urban Pop %']] =             df[['ages 0-14', 'ages 15-64','Density (P/Km²)','Med. Age', 'ages 64-','Urban Pop %']].apply(lambda x: x.fillna(x.mean()))
    print(df[df['IncomeGroup'].isnull()]["Country_Region"].unique())

    df = pd.merge(df, CountryRegion, on=['Country_Region'], how='left')#, indicator=True)
    
    mask = df['Country_Region']=='US'
    df.loc[mask,'Population (2020)'] = df.loc[mask,'population']
    df.loc[mask,'Density (P/Km²)']   = df.loc[mask,'population']/df.loc[mask,'land']
    df.drop(["population","land"],axis=1,inplace=True)

    df = pd.merge(df, supp_info, on=['Country_Region'], how='left')#, indicator=True)

    mask = df['Population (2020)'].isnull()
    df.loc[mask,'Population (2020)'] = df.loc[mask,'population']
    df.loc[mask,'Density (P/Km²)']   = df.loc[mask,'population']/df.loc[mask,'land']
    df.drop(["population","land"],axis=1,inplace=True)
       
    df = pd.merge(df, state_temperatures, on=['Country_Region','Province_State'], how='left')#, indicator=True)
    df[['Jan', 'Feb', 'Mar', 'Apr', 'May',
       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'year', 'Low Temp']] = \
            df[['Jan', 'Feb', 'Mar', 'Apr', 'May',
       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'year', 'Low Temp']].apply(lambda x: x.fillna(x.mean()))
    df.drop(['May','Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov', 'Dec'],axis=1,inplace=True )
    df.drop(['Jan', 'Feb', 'Mar'],axis=1,inplace=True )
    
    df = pd.merge(df,ppp_tabel, on=['Country_Region'], how='left')#, indicator=True)
    
    df = pd.merge(df,coor_df[["Country_Region",'Province_State',"Season","Season2"]], 
                          on=["Country_Region",'Province_State'], how="left")#, indicator=True)

#    coor_df.rename(columns={"Season2":"Season"},inplace= True)
    mask = df['Season'].isna()
    df.loc[mask,["Season"]]= df.loc[mask,"Season2"]
    df["Season"].fillna(1, inplace=True)
    
    mask = df['Country_Region']== 'China'
    df.loc[mask,["sub-region"]]= 'China'

    onehot = pd.get_dummies(df['sub-region'],prefix='R')
    onehot2 = pd.get_dummies(df['IncomeGroup'],prefix='I')
    df.drop(["Season2","sub-region",'IncomeGroup'],axis=1,inplace= True)

    df = pd.concat( [df,onehot],axis=1,ignore_index=False)
    df = pd.concat( [df,onehot2],axis=1,ignore_index=False)

    # Calculate ppm
    if istest_df:
        if USE_NEW:
            df["NewCases_ppm"]       = np.log1p(1000000/df["Population (2020)"])
            df["NewFatalities_ppm"]  = np.log1p(1000000/df["Population (2020)"])
        else:
            df["ConfirmedCases_ppm"] = np.log1p(1000000/df["Population (2020)"])
            df["Fatalities_ppm"]     = np.log1p(1000000/df["Population (2020)"])
    else:
        if USE_NEW:
            df["NewCases_ppm"]       = np.log1p(df["NewCases"]*1000000/df["Population (2020)"])
            df["NewFatalities_ppm"]  = np.log1p(df["NewFatalities"]*1000000/df["Population (2020)"])
        else:
            df["ConfirmedCases_ppm"] = np.log1p(df["ConfirmedCases"]*1000000/df["Population (2020)"])
            df["Fatalities_ppm"]     = np.log1p(df["Fatalities"]*1000000/df["Population (2020)"])

    #normeringer
    df['ages 0-14'] = df['ages 0-14']/100.
    df['ages 15-64'] = df['ages 15-64']/100.
    df['ages 64-'] = df['ages 64-']/100.
    df['Population (2020)'] = np.log1p(df['Population (2020)'])
    df['Density (P/Km²)'] = np.log1p(df['Density (P/Km²)'])
    df['Med. Age'] = df['Med. Age']/50.
    df['Urban Pop %'] = df['Urban Pop %']/100.
    df['Apr'] = df['Apr']/30.
    df['year'] = df['year']/30.
    df['Low Temp'] = df['Low Temp']/30.
    df['ppp'] = np.log1p(df['ppp'])
    return df


# In[ ]:





# In[4]:



df     = pd.read_csv(datapath + "train.csv")
sub_df = pd.read_csv(datapath + "test.csv")

df['Province_State'].fillna('', inplace=True)
sub_df['Province_State'].fillna('', inplace=True)

#tag kopi af TARGETS til senere brug
gem_targets = df[['Country_Region','Province_State','Date']+TARGETS]
gem_targets["Date"] = gem_targets["Date"].astype("datetime64[ms]")

#find alle combinationer af Countrt/State
combi= df.groupby(["Country_Region","Province_State"]).size().reset_index()
combi =combi[["Country_Region","Province_State"]].values.tolist()


mask1 = df["Country_Region"]=='Guyana'             # håndrettes
mask2 = df["Date"] > '2020-03-21'
mask3 = df["Date"] < '2020-03-28'
mask4 = pd.concat((mask1,mask2,mask3),axis=1)
mask5 = mask4.all(axis=1)
df.loc[mask5,"ConfirmedCases"] = 7.0

mask1 = df["Province_State"]=='Northern Territory' # håndrettes
mask2 = df["Date"] > '2020-03-05'
mask3 = df["Date"] < '2020-03-10'
mask4 = pd.concat((mask1,mask2,mask3),axis=1)
mask5 = mask4.all(axis=1)
df.loc[mask5,"ConfirmedCases"] = 1.0

mask1 = df["Country_Region"]=='Philippines'        # håndrettess
mask2 = df["Date"] > '2020-03-17'
mask3 = df["Date"] < '2020-03-19'
mask4 = pd.concat((mask1,mask2,mask3),axis=1)
mask5 = mask4.all(axis=1)
df.loc[mask5,"Fatalities"] = 14.0

correct_anomalities(df)

#find_anomalities(df)

if USE_NEW:
    df[['NewCases','NewFatalities']] = df.groupby(['Country_Region', 'Province_State'])                 [['ConfirmedCases','Fatalities']].transform(lambda x: x.diff()) 
    mask = df['NewCases'].isnull()
    df.loc[mask,['NewCases']] = 0
    df.loc[mask,['NewFatalities']] = 0
    provins = 'Virgin Islands'
    print("testprint for",provins)
    print(df[df['Province_State']== provins][['NewCases','NewFatalities']])

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)   

df     = preprocess(df)
sub_df = preprocess(sub_df)

GIVEN_FIRST_DATE = df["Date"].min()
GIVEN_LAST_DATE  = df["Date"].max()
SUBMISSION_FIRST_DATE =  sub_df["Date"].min()
ESTIMATE_FIRST_DATE = df["Date"].max() + timedelta(days=1)
ESTIMATE_LAST_DATE  = sub_df["Date"].max()
ESTIMATE_DAYS = (ESTIMATE_LAST_DATE - ESTIMATE_FIRST_DATE).days +1

if add_other:
    df     = add_other_info(df,False,usstates_info,supp_info,population_by_age_df,CountryRegion,state_temperatures,ppp_tabel,coor_df)
    sub_df = add_other_info(sub_df,True,usstates_info,supp_info,population_by_age_df,CountryRegion,state_temperatures,ppp_tabel,coor_df)

print("SUB_DF COLUMNS NYLAVET:")
print(sub_df.columns)


# In[5]:



if CURVE_SMOOTHING:
    #Add averages
    df['Cases_m'] =  df.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases']].transform(lambda x: x.shift(1)) 
    df['Cases_p']  = df.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases']].transform(lambda x: x.shift(-1)) 
    df['Cases_ave'] = 0.5*(df['ConfirmedCases']+0.5*(df['Cases_p']+df['Cases_m']))
    case_cols = ['ConfirmedCases','Cases_m','Cases_p','Cases_ave']

    df['Fatalities_m'] =  df.groupby(['Country_Region', 'Province_State'])[['Fatalities']].transform(lambda x: x.shift(1)) 
    df['Fatalities_p']  = df.groupby(['Country_Region', 'Province_State'])[['Fatalities']].transform(lambda x: x.shift(-1)) 
    df['Fatalities_ave'] = 0.5*(df['Fatalities']+0.5*(df['Fatalities_p']+df['Fatalities_m']))
    fata_cols = ['Fatalities','Fatalities_m','Fatalities_p','Fatalities_ave']

    if USE_NEW:
        df[['NewCases','NewFatalities']] = df.groupby(['Country_Region', 'Province_State'])                             [['ConfirmedCases','Fatalities']].transform(lambda x: x.diff()) 

        df['NewCases_m'] =  df.groupby(['Country_Region', 'Province_State'])[['NewCases']].transform(lambda x: x.shift(1)) 
        df['NewCases_p']  = df.groupby(['Country_Region', 'Province_State'])[['NewCases']].transform(lambda x: x.shift(-1)) 
        df['NewCases_m2'] =  df.groupby(['Country_Region', 'Province_State'])[['NewCases']].transform(lambda x: x.shift(2)) 
        df['NewCases_p2']  = df.groupby(['Country_Region', 'Province_State'])[['NewCases']].transform(lambda x: x.shift(-2)) 
#        df['NewCases_ave'] = 0.5*(df['NewCases']+0.5*(df['NewCases_p']+df['NewCases_m']))
        df['NewCases_ave'] = 0.2*(df['NewCases']+df['NewCases_p']+df['NewCases_m']+df['NewCases_p2']+df['NewCases_m2'])

        df['NewFatalities_m'] =  df.groupby(['Country_Region', 'Province_State'])[['NewFatalities']].transform(lambda x: x.shift(1)) 
        df['NewFatalities_p']  = df.groupby(['Country_Region', 'Province_State'])[['NewFatalities']].transform(lambda x: x.shift(-1)) 
        df['NewFatalities_m2'] =  df.groupby(['Country_Region', 'Province_State'])[['NewFatalities']].transform(lambda x: x.shift(2)) 
        df['NewFatalities_p2']  = df.groupby(['Country_Region', 'Province_State'])[['NewFatalities']].transform(lambda x: x.shift(-2)) 
        df['NewFatalities_ave'] =0.2*(df['NewFatalities']+df['NewFatalities_p']+df['NewFatalities_m']+df['NewFatalities_p2']+df['NewFatalities_m2'])

    date_max = df["Date"].max()
    mask = df["Date"] ==date_max
    mask2 = df["Date"]==date_max - timedelta(days=1)
    df.loc[mask,'Cases_ave']         = 0.75*df.loc[mask,'ConfirmedCases']+0.25*df.loc[mask,'Cases_m']
    df.loc[mask,'Fatalities_ave']    = 0.75*df.loc[mask,'Fatalities']    +0.25*df.loc[mask,'Fatalities_m']
    df.loc[mask,'Cases_ave']         = 0.75*df.loc[mask,'ConfirmedCases']+0.25*df.loc[mask,'Cases_m']
    df.loc[mask,'Fatalities_ave']    = 0.75*df.loc[mask,'Fatalities']    +0.25*df.loc[mask,'Fatalities_m']
    df.drop(['Cases_m', 'Cases_p', 'Fatalities_m','Fatalities_p','ConfirmedCases','Fatalities'],axis=1,inplace=True)
    df.rename(columns={'Cases_ave':'ConfirmedCases','Fatalities_ave':'Fatalities'},inplace=True)

    if USE_NEW:
        df.loc[mask2,'NewCases_ave']      = 0.3*df.loc[mask2,'NewCases_p']+0.3*df.loc[mask2,'NewCases']+                        0.3*df.loc[mask2,'NewCases_m']+0.1*df.loc[mask2,'NewCases_m2']
        df.loc[mask2,'NewFatalities_ave'] = 0.3*df.loc[mask2,'NewFatalities_p']+0.3*df.loc[mask2,'NewFatalities']+                        0.3*df.loc[mask2,'NewFatalities_m']+0.1*df.loc[mask2,'NewFatalities_m2']
        df.loc[mask,'NewCases_ave']      = 0.55*df.loc[mask,'NewCases']  +0.3*df.loc[mask,'NewCases_m']+                        0.2*df.loc[mask,'NewCases_m2']
        df.loc[mask,'NewFatalities_ave'] = 0.5*df.loc[mask,'NewFatalities'] +0.3*df.loc[mask,'NewFatalities_m']+                        0.2*df.loc[mask,'NewFatalities_m2']
        df.drop(['NewCases_m', 'NewCases_p', 'NewFatalities_m','NewFatalities_p','NewCases','NewFatalities'],axis=1,inplace=True)
        df.drop(['NewCases_m2', 'NewCases_p2', 'NewFatalities_m2','NewFatalities_p2'],axis=1,inplace=True)
        df.rename(columns={'NewCases_ave':'NewCases','NewFatalities_ave':'NewFatalities'},inplace=True)

        if add_other:
            df["NewCases_ppm"]      = np.log1p(df["NewCases"]*1000000/df["Population (2020)"])
            df["NewFatalities_ppm"] = np.log1p(df["NewFatalities"]*1000000/df["Population (2020)"])

df.fillna(0, inplace=True)
sub_df.fillna(0,inplace =True)
 
DEFAULT_VALUE = 0
if df is None:
    df = DEFAULT_VALUE
if sub_df is None:
    sub_df = DEFAULT_VALUE

if has_nan(df,base_features):
    print("DYT-DYYYYYT: der er NAN")
    print_rows_with_nan(df)
else:
    print("HURRA, HURRA")

if has_nan(df)>0:
    print("DOOOOOOOOOT: der er NAN")
    print_rows_with_nan(df)
    print("DOT DONE")
else:
    print("HIP HIP")


# In[6]:



df = df[df["Date"] >= df["Date"].min() + timedelta(days=days_shift[NUM_SHIFT])].copy() # Cutter af så alle shift er med i df (i princippet)

#df[features] = df[features].fillna(method='ffill',inplace=True)
#df[features] = df[features].fillna(method='bfill',inplace=True)
#df[features] = df[features].fillna(0)

for col in TARGETS:                        #Laver targets om til Logaritmer
    df[col] = np.log1p(df[col])/normfactor
        
df = df[df['days']>TRAIN_START_DAY]


# In[7]:



def nn_block(input_layer, size, dropout_rate, activation):
    out_layer = KL.Dense(size, activation=None)(input_layer)
    #out_layer = KL.BatchNormalization()(out_layer)
    out_layer = KL.Activation(activation)(out_layer)
    out_layer = KL.Dropout(dropout_rate)(out_layer)
    return out_layer

def get_model(feature_length,target_length,):
    inp = KL.Input(shape=(feature_length,))

    hidden_layer = nn_block(inp, 64, 0.0, "relu")
    gate_layer = nn_block(hidden_layer, 32, 0.0, "sigmoid")
    hidden_layer = nn_block(hidden_layer, 64, 0.0, "relu")
    hidden_layer = nn_block(hidden_layer, 32, 0.0, "relu")
    hidden_layer = KL.multiply([hidden_layer, gate_layer])

    out = KL.Dense(target_length, activation="linear")(hidden_layer)

    model = tf.keras.models.Model(inputs=[inp], outputs=out)
    return model


def get_input(df,features):
    return [df[features]]


def train_models(df,features,targets,save=False):
    print()
    print("TRAINING. Der regnes på denne model:")
    get_model(len(features),len(targets)).summary()
    models = []
    for i in range(NUM_MODELS):
        print("PHS1")
        model = get_model(len(features),len(targets))
        print("PHS2")
        model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))
        print("PHS3 - Targets:",targets)
        hist = model.fit(get_input(df,features), df[targets],
                         batch_size=2048, epochs=500, verbose=0, shuffle=True)
        print("PHS4")
        if save:
            print("PHS5")
            model.save_weights("model{}.h5".format(i))
            print("PHS6")
        models.append(model)
        print("PHS7")
    return models


# In[8]:



def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(df,targets):
    error = 0
    for col in targets:
        error += rmse(df[col].values, df["pred_{}".format(col)].values)
    return np.round(error/len(targets), 5)

def predict_one(df,features,prev_targets, models):
    pred = np.zeros((df.shape[0], 2))           
        
    for model in models:
        pred += model.predict(get_input(df,features))/len(models)
    if np.isnan(np.sum(pred)):
        print("DYYYYYYYYYYYYT: der er NAN")
        print(pred)
#    if USE_NEW:
#        pred[:, 0] = np.maximum(pred[:, 0], 0)          #new cases kan ikke gå under 0
#        pred[:, 1] = np.maximum(pred[:, 1], 0)          #new fatalities kan ikke gå under 0
#    else:
    pred = np.maximum(pred, df[prev_targets].values)
    #pred[:, 0] = np.maximum(pred[:, 0], df[prev_targets[0]].values)          #sum kan aldrig gå ned
    #pred[:, 1] = np.maximum(pred[:, 1], df[prev_targets[1]].values)          #sum kan aldrig gå ned#
    pred[:, 0] = np.log1p(np.expm1(normfactor*pred[:, 0]) + 0.1)/normfactor
    pred[:, 1] = np.log1p(np.expm1(normfactor*pred[:, 1]) + 0.01)/normfactor
         
    if np.isnan(np.sum(pred)):
        print("DYT DYT DYT: der er NAN")

    return np.clip(pred, None, 15)    # intet minimum men tilladt højst 15

def predict(test_df,features,targets,prev_targets,first_day, num_days, models, val=False):

    for d in range(0, num_days):
        print("DAY NO.",d)
        test_df = fill_shift_columns(test_df,targets)       

        date = first_day + timedelta(days=d)
        temp_df = test_df.loc[test_df["Date"] == date].copy()
        y_pred = predict_one(temp_df,features,prev_targets, models)

        for i, col in enumerate(targets):
            test_df.loc[test_df["Date"] == date, col] = y_pred[:, i]

        if add_other:   
            if USE_NEW:
                test_df.loc[test_df["Date"] == date,"NewCases_ppm"] =                                      np.log1p(np.expm1(y_pred[:, 0])*1000000/                                              np.expm1(test_df.loc[test_df["Date"] == date,"Population (2020)"]))
                test_df.loc[test_df["Date"] == date,"NewFatalities_ppm"] =                                          np.log1p(np.expm1(y_pred[:, 1])*1000000/                                              np.expm1(test_df.loc[test_df["Date"] == date,"Population (2020)"]))
            else:
                test_df.loc[test_df["Date"] == date,"ConfirmedCases_ppm"] =                                  np.log1p(np.expm1(y_pred[:, 0])*1000000/                                          np.expm1(test_df.loc[test_df["Date"] == date,"Population (2020)"]))
                test_df.loc[test_df["Date"] == date,"Fatalities_ppm"] =                                      np.log1p(np.expm1(y_pred[:, 1])*1000000/                                          np.expm1(test_df.loc[test_df["Date"] == date,"Population (2020)"]))

        if val:
            print(evaluate(test_df[test_df["Date"] == date]))

    return test_df


# In[9]:



all_features = base_features + shift_features

df = fill_shift_columns(df,TARGETS)
df[all_features] = df[all_features].fillna(0)

print("Kolonner i modelller",all_features)    
print("BEFORE TRAINING")
print(df[(df['Country_Region']=='Germany') & (df['days'] >75)])


final_models = train_models(df,all_features,TARGETS, save=True)          #final models


# In[10]:



sub_df_public  = sub_df[sub_df["Date"] <= GIVEN_LAST_DATE].copy()     # den del vi allerede kender
sub_df_private = sub_df[sub_df["Date"] >  GIVEN_LAST_DATE].copy()     # den del der skal etimeres

print("df Fra/Til  ",df["Date"].min(),df["Date"].max())
print("Public Fra/Til  ",sub_df_public["Date"].min(),sub_df_public["Date"].max())
print("Private Fra/Til ",sub_df_private["Date"].min(),sub_df_private["Date"].max())
print("GIVEN DATES FIRST/LAST",GIVEN_FIRST_DATE,GIVEN_LAST_DATE)
print("ESTIMATES DATES FIRST/LAST",ESTIMATE_FIRST_DATE,ESTIMATE_LAST_DATE)
print("ESTIMATE_DAYS",ESTIMATE_DAYS)

#lav ekstra kolonner
new_columns = diff(df.columns,sub_df_private.columns)
for a in new_columns:
   sub_df_private[a] = np.NaN
full_df = pd.concat([df,sub_df_private])
print ("Full Fra/Til ",full_df["Date"].min(),full_df["Date"].max())

#Flyt ForecastId over på full_df
sub_df.rename(columns={"ForecastId":"ForecastId2"},inplace=True)
full_df = full_df.merge(sub_df[["Date"] + loc_group+ ["ForecastId2"]], how="left", on=["Date"] + loc_group)
sub_df.rename(columns={"ForecastId2":"ForecastId"},inplace=True)
mask= full_df["ForecastId"].isna()
full_df.loc[mask,["ForecastId"]]= full_df.loc[mask,["ForecastId2"]]
full_df.drop(["ForecastId2"],axis=1,inplace=True)


# In[11]:



full_df_pred= predict(full_df,all_features,TARGETS,prev_targets,ESTIMATE_FIRST_DATE,ESTIMATE_DAYS, final_models)

for col in TARGETS:                                                      
    full_df_pred[col] = np.expm1(full_df_pred[col]) # regner tilbage fra logaritme


# In[12]:


gem_targets = gem_targets[gem_targets["Date"]>=SUBMISSION_FIRST_DATE]
print(gem_targets.head(15))

values_to_submit = full_df_pred[full_df_pred["Date"]>=ESTIMATE_FIRST_DATE]
values_to_submit = values_to_submit[['Date','Country_Region','Province_State','ConfirmedCases', 'Fatalities']]
print(values_to_submit.head(15))

values_to_submit = gem_targets.append(values_to_submit, sort=False)          # lægger pivate stykke til public
print(values_to_submit.tail(10))

print(sub_df[['Date','Country_Region','Province_State','ForecastId']])
values_to_submit   = pd.merge(values_to_submit,sub_df[['Date','Country_Region','Province_State','ForecastId']], on=['Date','Country_Region','Province_State'], how='left')#, indicator=True)
print(values_to_submit.head(20))
print(values_to_submit.tail(20))

sub2 =  values_to_submit[["ForecastId"] + TARGETS]
sub2["ForecastId"] = sub2["ForecastId"].astype(np.int16)


# In[13]:


sub2.sort_values("ForecastId", inplace=True)
sub2.to_csv("submission.csv", index=False)


# In[14]:


print(sub2)


# In[15]:


full_df_pred['Cases_Estimate']      =  full_df_pred['ConfirmedCases']
full_df_pred['Fatalities_Estimate'] =  full_df_pred['Fatalities']
full_df_pred2 = full_df_pred[['Date','Country_Region','Province_State']+TARGETS]

full_df2 = full_df[['Date','Country_Region','Province_State']+TARGETS]


full_df3 = pd.merge(full_df2,full_df_pred[['Date','Country_Region','Province_State','Cases_Estimate','Fatalities_Estimate']], on=['Date','Country_Region','Province_State'], how='left')#, indicator=True)
mask = full_df3['Cases_Estimate'].notna()
full_df3.loc[mask,'CasesConfirmed'] =  full_df3.loc[mask,'Cases_Estimate']
full_df3.loc[mask,'Fatalities']     =  full_df3.loc[mask,'Fatalities_Estimate']
             
split_on = 'Country_Region'
split_values = full_df3[split_on].unique()
kolonne = 'CasesConfirmed'
#print("ConfirmedCases / Private")
for imin in range(0,175,25):
    plt.figure(figsize=(30,30))
    imax = imin+25
    for i in range(imin,imax):
        plt.subplot(5,5,i-imin+1)
        idx = i
        df_interest = full_df3[full_df3[split_on]==split_values[idx]].reset_index(drop=True)
        tmp = df_interest[kolonne].values
    #    tmp = np.cumsum(tmp)
        sns.lineplot(x=df_interest['Date'], y=tmp, label='pred')
        df_interest2 = full_df3[(full_df3[split_on]==split_values[idx]) & (full_df3['Date']<=GIVEN_LAST_DATE)].reset_index(drop=True)
        sns.lineplot(x=df_interest2['Date'].values, y=df_interest2[kolonne].values, label='true')
        plt.title(split_on+'  '+str(split_values[idx]))
    plt.show()


# In[ ]:




