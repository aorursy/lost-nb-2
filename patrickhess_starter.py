#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import hashlib
import random
from math import exp
import xgboost as xgb
from sklearn.decomposition import PCA
from math import sin, cos, sqrt, atan2, radians


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import Birch


def dist(list_one, list_two):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(list_one['latitude'])
    lon1 = radians(list_one['longitude'])
    lat2 = radians(list_two['latitude'])
    lon2 = radians(list_two['longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance



def cluster_latlon(n_clusters, data):  
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    data_e=data[~((data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9))]
    #put it in matrix form
    coords=data_c.as_matrix(columns=['latitude', "longitude"])
    
    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)    
    #brc2 = Birch(branching_factor=100, n_clusters=n_clusters / 2, threshold=0.005,compute_labels=True)
    #brc4 = Birch(branching_factor=100, n_clusters=n_clusters / 4, threshold=0.005,compute_labels=True)
    brc.fit(coords)
    clusters=brc.predict(coords)
    #print clusters
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    #brc2.fit(coords)
    #clusters=brc2.predict(coords)
    #print clusters
    #data_c["cluster_"+str(n_clusters/2)]=clusters
    #data_e["cluster_"+str(n_clusters/2)]=-1 #assign cluster label -1 for the non NYC listings 
    #brc4.fit(coords)
    #clusters=brc4.predict(coords)
    #print clusters
    #data_c["cluster_"+str(n_clusters/4)]=clusters
    #data_e["cluster_"+str(n_clusters/4)]=-1 #assign cluster label -1 for the non NYC listings 


    data=pd.concat([data_c,data_e])
    #plt.scatter(data_c["longitude"], data_c["latitude"], c=data_c["cluster_"+str(n_clusters)], s=10, linewidth=0.1)
    #plt.title(str(n_clusters)+" Neighbourhoods from clustering")
    #plt.show()
    return data 




def preprocess(train_df, test_df):
    """Just a generic preprocessing function, feel free to substitute it with your custom function"""
    # encode target variable
    train_df['interest_level'] = train_df['interest_level'].apply(lambda x: {'high': 2, 'medium': 1, 'low': 0}[x])   
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    manager_score = [np.nan]*len(train_df)
    manager_low = [np.nan]*len(train_df)
    manager_medium = [np.nan]*len(train_df)
    manager_high = [np.nan]*len(train_df)
    manager_low_pct = [np.nan]*len(train_df)
    manager_medium_pct = [np.nan]*len(train_df)
    manager_high_pct = [np.nan]*len(train_df)    
    building_score = [np.nan]*len(train_df)
    building_low = [np.nan]*len(train_df)
    building_medium = [np.nan]*len(train_df)
    building_high = [np.nan]*len(train_df)
    building_low_pct = [np.nan]*len(train_df)
    building_medium_pct = [np.nan]*len(train_df)
    building_high_pct = [np.nan]*len(train_df)
    pct_low =  [np.nan]*len(train_df)
    pct_medium =  [np.nan]*len(train_df)
    pct_high =  [np.nan]*len(train_df)
    neigh_count = [np.nan]*len(train_df)
    for j in range(5):
        print j
        manager_sum = {}
        manager_high_tmp = {}
        manager_medium_tmp = {}
        manager_low_tmp = {}
        manager_count = {}        
        building_sum = {}
        building_high_tmp = {}
        building_medium_tmp = {}
        building_low_tmp = {}
        building_count = {}
        high_total = 0.0
        medium_total = 0.0
        low_total = 0.0
        manager_ct = 0.0 
        building_ct = 0.0
        sm = 0.0
        ct = 0.0        
        test_ind = index[int((j*train_df.shape[0])/5):int(((j+1)*train_df.shape[0])/5)]
        train_ind = list(set(index).difference(test_ind))
        print 'train ind'
        for i in train_ind:
            x = train_df.iloc[i]
            if x['manager_id'] not in manager_sum:
                manager_sum[x['manager_id']] = 0.0
                manager_count[x['manager_id']] = 0.0
                manager_ct += 1
            if x['building_id'] not in building_sum:
                building_sum[x['building_id']] = 0.0
                building_count[x['building_id']] = 0.0
                building_ct += 1
            building_sum[x['building_id']] += x['interest_level']
            manager_sum[x['manager_id']] += x['interest_level']
            if  x['interest_level'] == 0.0:
                if x['manager_id'] not in manager_low_tmp:
                    manager_low_tmp[x['manager_id']] = 0.0
                if x['building_id'] not in building_low_tmp:
                    building_low_tmp[x['building_id']] = 0.0
                manager_low_tmp[x['manager_id']] += 1
                building_low_tmp[x['building_id']] += 1
                low_total += 1.0
            if  x['interest_level'] == 1:
                if x['manager_id'] not in manager_medium_tmp:
                    manager_medium_tmp[x['manager_id']] = 0.0
                if x['building_id'] not in building_medium_tmp:
                    building_medium_tmp[x['building_id']] = 0.0
                manager_medium_tmp[x['manager_id']] += 1
                building_medium_tmp[x['building_id']] += 1
                medium_total += 1.0
            if  x['interest_level'] == 2:
                if x['manager_id'] not in manager_high_tmp:
                    manager_high_tmp[x['manager_id']] = 0.0
                if x['building_id'] not in building_high_tmp:
                    building_high_tmp[x['building_id']] = 0.0
                manager_high_tmp[x['manager_id']] += 1
                building_high_tmp[x['building_id']] += 1
                high_total += 1.0
            manager_count[x['manager_id']] += 1.0
            building_count[x['building_id']] += 1
            sm += x['interest_level']        
            ct += 1.0
        avg = sm / ct        
        neigh_low = {}
        neigh_medium = {}
        neigh_high = {}
        for i in train_ind:
            x = train_df.iloc[i]
            round_lat = round(x['latitude'], 2)
            round_long = round(x['longitude'], 2)
            hsh = str(round_lat) + "#" + str(round_long)
            if x['interest_level'] == 0.0:
                if hsh not in neigh_low:
                    neigh_low[hsh] = 0.0
                neigh_low[hsh] += 1
            if x['interest_level'] == 1:
                if hsh not in neigh_medium:
                    neigh_medium[hsh] = 0.0
                neigh_medium[hsh] += 1
            if x['interest_level'] == 2:
                if hsh not in neigh_high:
                    neigh_high[hsh] = 0.0
                neigh_high[hsh] += 1        
        neigh_pct_low_tmp = {}
        neigh_pct_medium_tmp = {}
        neigh_pct_high_tmp = {}
        neigh_count_tmp = {} 
        for i in train_ind:
            x = train_df.iloc[i]
            round_lat = round(x['latitude'], 2)
            round_long = round(x['longitude'], 2)
            lat_down = round_lat - 0.01
            lat_up = round_lat + 0.01
            long_down = round_long - 0.01
            long_up = round_long + 0.01            
            low_sum = 0.0    
            md_sum = 0.0
            high_sum = 0.0            
            pos = [str(lat_down) + "#" + str(long_down),  
                   str(round_lat) + "#" + str(long_down),  
                   str(lat_up) + "#" + str(long_down), 
                   str(lat_down) + "#" + str(round_long),
                   str(round_lat)  + "#" + str(round_long),
                   str(lat_up) + "#" + str(round_long),
                   str(lat_down)  + "#" + str(long_up), 
                   str(round_lat)  + "#" + str(long_up),
                   str(lat_up)+ "#" + str(long_up)]
            for ps in pos:
                if ps in neigh_low:    
                    low_sum += neigh_low[ps]
                if ps in neigh_medium:
                    md_sum += neigh_medium[ps]
                if ps in neigh_high:
                    high_sum += neigh_high[ps]
            hsh =  str(round_lat)  + "#" + str(round_long)
            neigh_pct_low_tmp[hsh] = low_sum / (low_sum + md_sum + high_sum + 1.0)
            neigh_pct_medium_tmp[hsh] = md_sum / (low_sum + md_sum + high_sum + 1.0)
            neigh_pct_high_tmp[hsh] = high_sum / (low_sum + md_sum + high_sum + 1.0) 
            neigh_count_tmp[hsh] = low_sum + md_sum + high_sum                
        for i in test_ind:
            x = train_df.iloc[i]
            manager_id = x['manager_id']      
            building_id = x['building_id']   
            round_lat = round(x['latitude'], 2)
            round_long = round(x['longitude'], 2)
            hsh =  str(round_lat)  + "#" + str(round_long)
            pct_low[i] = neigh_pct_low_tmp[hsh] if hsh in neigh_pct_low_tmp  else 0.6
            pct_medium[i] = neigh_pct_medium_tmp[hsh] if hsh in neigh_pct_medium_tmp else 0.3
            pct_high[i] =   neigh_pct_high_tmp[hsh] if hsh in neigh_pct_high_tmp else 0.1
            manager_score[i] = manager_sum[manager_id] / manager_count[manager_id] if manager_id in manager_count else avg
            manager_low[i] = manager_low_tmp[manager_id]  if manager_id in manager_low_tmp else low_total / manager_ct
            manager_medium[i] = manager_medium_tmp[manager_id] if manager_id in manager_medium_tmp else medium_total / manager_ct
            manager_high[i] = manager_high_tmp[manager_id] if manager_id in manager_high_tmp  else high_total / manager_ct
            manager_low_pct[i] = manager_low_tmp[manager_id] / manager_count[manager_id]  if manager_id in manager_low_tmp else low_total / ct
            manager_medium_pct[i] = manager_medium_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_medium_tmp else medium_total / ct            
            manager_high_pct[i] = manager_high_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_high_tmp else high_total / ct   
            neigh_count[i] =  neigh_count_tmp[hsh] if hsh in neigh_count_tmp else 0 
            building_score[i] = building_sum[building_id] / building_count[building_id] if building_id in building_count else avg
            building_low[i] = building_low_tmp[building_id] if building_id in building_low_tmp else low_total / building_ct
            building_medium[i] = building_medium_tmp[building_id] if building_id in building_medium_tmp else medium_total / building_ct
            building_high[i] = building_high_tmp[building_id] if building_id in building_high_tmp else high_total / building_ct
            building_low_pct[i] = building_low_tmp[building_id] / building_count[building_id] if building_id in building_low_tmp else 0.6
            building_medium_pct[i] = building_medium_tmp[building_id] / building_count[building_id] if building_id in building_medium_tmp else 0.3
            building_high_pct[i] = building_high_tmp[building_id] / building_count[building_id] if building_id in building_high_tmp else 0.1
    train_df['manager_score'] = manager_score 
    #train_df['manager_low'] = manager_low 
    #train_df['manager_medium'] = manager_medium
    #train_df['manager_high'] =  manager_high
    train_df['manager_low_pct'] = manager_low_pct
    train_df['manager_medium_pct'] = manager_medium_pct
    train_df['manager_high_pct'] = manager_high_pct
    train_df['neigh_low_pct'] = pct_low
    train_df['neigh_medium_pct'] = pct_medium
    train_df['neigh_high_pct'] = pct_high
    train_df['neigh_low_ct'] = np.array(pct_low) * np.array(neigh_count)
    train_df['neigh_medium_ct'] = np.array(pct_medium) * np.array(neigh_count)
    train_df['neigh_high_ct'] = np.array(pct_high) * np.array(neigh_count)
    train_df['building_score'] = building_score 
    #train_df['building_low'] = building_low 
    # train_df['building_medium'] = building_medium
    #train_df['building_high'] =  building_high
    train_df['building_low_pct'] = building_low_pct
    train_df['building_medium_pct'] = building_medium_pct
    train_df['building_high_pct'] = building_high_pct
    train_index = train_df.index
    test_index = test_df.index   
    manager_score = []
    manager_low = []
    manager_medium = []
    manager_high = []
    manager_low_pct = []
    manager_medium_pct = []
    manager_high_pct = []        
    building_score = []
    building_low = []
    building_medium = []
    building_high = []
    building_low_pct = []
    building_medium_pct = []
    building_high_pct = []
    pct_low =  []
    pct_medium =  []
    pct_high =  []
    neigh_count = []
    manager_sum = {}
    manager_high_tmp = {}
    manager_medium_tmp = {}
    manager_low_tmp = {}
    manager_count = {}
    building_sum = {}
    building_count = {}            
    building_sum = {}
    building_high_tmp = {}
    building_medium_tmp = {}
    building_low_tmp = {}
    building_count = {}
    high_total = 0.0
    medium_total = 0.0
    low_total = 0.0
    manager_ct = 0.0 
    sm = 0.0
    ct = 0.0        
    print 'cv statistics computed'
    for i in range(train_df.shape[0]):
        x = train_df.iloc[i]
        if x['manager_id'] not in manager_sum:
            manager_sum[x['manager_id']] = 0.0
            manager_count[x['manager_id']] = 0.0
            manager_ct += 1
        if x['building_id'] not in building_sum:
            building_sum[x['building_id']] = 0.0
            building_count[x['building_id']] = 0.0
            building_ct += 1
        building_sum[x['building_id']] += x['interest_level']
        manager_sum[x['manager_id']] += x['interest_level']
        if  x['interest_level'] == 0.0:
            if x['manager_id'] not in manager_low_tmp:
                manager_low_tmp[x['manager_id']] = 0.0
            if x['building_id'] not in building_low_tmp:
                building_low_tmp[x['building_id']] = 0.0
            manager_low_tmp[x['manager_id']] += 1
            building_low_tmp[x['building_id']] += 1
            low_total += 1.0
        if  x['interest_level'] == 1:
            if x['manager_id'] not in manager_medium_tmp:
                manager_medium_tmp[x['manager_id']] = 0.0
            if x['building_id'] not in building_medium_tmp:
                building_medium_tmp[x['building_id']] = 0.0
            manager_medium_tmp[x['manager_id']] += 1
            building_medium_tmp[x['building_id']] += 1
            medium_total += 1.0
        if  x['interest_level'] == 2:
            if x['manager_id'] not in manager_high_tmp:
                manager_high_tmp[x['manager_id']] = 0.0
            if x['building_id'] not in building_high_tmp:
                building_high_tmp[x['building_id']] = 0.0
            manager_high_tmp[x['manager_id']] += 1
            building_high_tmp[x['building_id']] += 1
            high_total += 1.0
        manager_count[x['manager_id']] += 1.0
        building_count[x['building_id']] += 1
        sm += x['interest_level']        
        ct += 1.0
    neigh_low = {}
    neigh_medium = {}
    neigh_high = {}
    for i in train_ind:
        x = train_df.iloc[i]
        round_lat = round(x['latitude'], 2)
        round_long = round(x['longitude'], 2)
        hsh = str(round_lat) + "#" + str(round_long)
        if x['interest_level'] == 0.0:
            if hsh not in neigh_low:
                neigh_low[hsh] = 0.0
            neigh_low[hsh] += 1
        if x['interest_level'] == 1:
            if hsh not in neigh_medium:
                neigh_medium[hsh] = 0.0
            neigh_medium[hsh] += 1
        if x['interest_level'] == 2:
            if hsh not in neigh_high:
                neigh_high[hsh] = 0.0
            neigh_high[hsh] += 1
    neigh_pct_low_tmp = {}
    neigh_pct_medium_tmp = {}
    neigh_pct_high_tmp = {}
    neigh_count_tmp = {}
    for i in train_ind:
        x = train_df.iloc[i]
        round_lat = round(x['latitude'], 2)
        round_long = round(x['longitude'], 2)
        lat_down = round_lat - 0.01
        lat_up = round_lat + 0.01
        long_down = round_long - 0.01
        long_up = round_long + 0.01        
        low_sum = 0.0    
        md_sum = 0.0
        high_sum = 0.0        
        pos = [str(lat_down) + "#" + str(long_down),  
               str(round_lat) + "#" + str(long_down),  
               str(lat_up) + "#" + str(long_down), 
               str(lat_down) + "#" + str(round_long),
               str(round_lat)  + "#" + str(round_long),
               str(lat_up) + "#" + str(round_long),
               str(lat_down)  + "#" + str(long_up), 
               str(round_lat)  + "#" + str(long_up),
               str(lat_up)+ "#" + str(long_up)]
        for ps in pos:
            if ps in neigh_low:    
                low_sum += neigh_low[ps]
            if ps in neigh_medium:
                md_sum += neigh_medium[ps]
            if ps in neigh_high:
                high_sum += neigh_high[ps]
        hsh =  str(round_lat)  + "#" + str(round_long)
        neigh_pct_low_tmp[hsh] = low_sum / (low_sum + md_sum + high_sum + 1.0)
        neigh_pct_medium_tmp[hsh] = md_sum / (low_sum + md_sum + high_sum + 1.0)
        neigh_pct_high_tmp[hsh] = high_sum / (low_sum + md_sum + high_sum + 1.0)
        neigh_count_tmp[hsh] = low_sum + md_sum + high_sum  
    for index, row in test_df.iterrows():
        x = row
        manager_id = row['manager_id']
        building_id = row['building_id']
        round_lat = round(x['latitude'], 2)
        round_long = round(x['longitude'], 2)
        hsh =  str(round_lat)  + "#" + str(round_long)
        manager_score.append(manager_sum[manager_id] / manager_count[manager_id] if manager_id in manager_count else avg)
        manager_low.append(manager_low_tmp[manager_id]  if manager_id in manager_low_tmp else low_total / manager_ct)
        manager_medium.append(manager_medium_tmp[manager_id] if manager_id in manager_medium_tmp else medium_total / manager_ct)
        manager_high.append(manager_high_tmp[manager_id] if manager_id  in manager_high_tmp  else high_total / manager_ct)
        manager_low_pct.append(manager_low_tmp[manager_id] / manager_count[manager_id]  if manager_id in manager_low_tmp else low_total / ct)
        manager_medium_pct.append(manager_medium_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_medium_tmp else medium_total / ct)
        manager_high_pct.append(manager_high_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_high_tmp else high_total / ct)  
        pct_low.append(neigh_pct_low_tmp[hsh] if hsh in neigh_pct_low_tmp else 0.6)
        pct_medium.append(neigh_pct_medium_tmp[hsh] if hsh in neigh_pct_medium_tmp else 0.3)
        pct_high.append(neigh_pct_high_tmp[hsh] if hsh in neigh_pct_high_tmp else 0.1)
        neigh_count.append(neigh_count_tmp[hsh] if hsh in neigh_count_tmp else 0)
        building_score.append(building_sum[building_id] / building_count[building_id] if building_id in building_count else avg)
        building_low.append(building_low_tmp[building_id] if building_id in building_low_tmp else low_total / building_ct)
        building_medium.append(building_medium_tmp[building_id] if building_id in building_medium_tmp else medium_total / building_ct)
        building_high.append(building_high_tmp[building_id] if building_id in building_high_tmp else high_total / building_ct)
        building_low_pct.append(building_low_tmp[building_id] / building_count[building_id] if building_id in building_low_tmp else 0.6)
        building_medium_pct.append(building_medium_tmp[building_id] / building_count[building_id] if building_id in building_medium_tmp else 0.3)
        building_high_pct.append(building_high_tmp[building_id] / building_count[building_id] if building_id in building_high_tmp else 0.1)
    test_df['manager_score'] = manager_score 
    #test_df['manager_low'] = manager_low 
    #test_df['manager_medium'] = manager_medium
    #test_df['manager_high'] =  manager_high
    test_df['manager_low_pct'] = manager_low_pct
    test_df['manager_medium_pct'] = manager_medium_pct
    test_df['manager_high_pct'] = manager_high_pct
    test_df['neigh_low_pct'] = pct_low
    test_df['neigh_medium_pct'] = pct_medium
    test_df['neigh_high_pct'] = pct_high
    test_df['neigh_low_ct'] = np.array(pct_low) * np.array(neigh_count)
    test_df['neigh_medium_ct'] = np.array(pct_medium) * np.array(neigh_count)
    test_df['neigh_high_ct'] = np.array(pct_high) * np.array(neigh_count)
    test_df['building_score'] = building_score 
    #test_df['building_low'] = building_low 
    #test_df['building_medium'] = building_medium
    #test_df['building_high'] =  building_high
    test_df['building_low_pct'] = building_low_pct
    test_df['building_medium_pct'] = building_medium_pct
    test_df['building_high_pct'] = building_high_pct
    data_df = pd.concat((train_df, test_df), axis=0)  
    manager_price = {}
    manager_count = {}
    for j in range(data_df.shape[0]):  
        x=data_df.iloc[j]
        if x['manager_id'] not in manager_price:
            manager_price[x['manager_id']] = 0.0
            manager_count[x['manager_id']] = 0.0
        manager_price[x['manager_id']] += x['price']
        manager_count[x['manager_id']] += 1
    data_df['manager_count'] = data_df['manager_id'].apply(lambda x: manager_count[x])
    data_df['avg_manager_price'] = data_df['manager_id'].apply(lambda x: manager_price[x] / manager_count[x])
    # add counting features 
    data_df['num_photos'] = data_df['photos'].apply(len)
    data_df['num_features'] = data_df['features'].apply(len)
    data_df['num_description'] = data_df['description'].apply(lambda x: len(x.split(' ')))
    data_df['num_display_address'] = data_df['display_address'].apply(lambda x: len(x.split(' ')))
    data_df['num_street_address'] = data_df['street_address'].apply(lambda x: len(x.split(' ')))
    data_df['photo_description_ratio'] =  data_df['num_photos'] * 1.0 / data_df['num_description']
    data_df.drop('photos', axis=1, inplace=True)
    # naive feature engineering
    data_df['room_difference'] = data_df['bedrooms'] - data_df['bathrooms']
    data_df['room_ratio'] = data_df['bedrooms'] * 1.0 / data_df['bathrooms']
    data_df['total_rooms'] = data_df['bedrooms'] + data_df['bathrooms']
    data_df['price_per_room'] = data_df['price'] / (data_df['total_rooms'] + 1)
    data_df['price_per_bedroom'] = data_df['price'] / (data_df['bedrooms'] + 1)
    data_df['price_per_bedroom'] = data_df['price'] / (data_df['bathrooms'] + 1)
    # add datetime features
    data_df['created'] = pd.to_datetime(data_df['created'])
    data_df['c_month'] = data_df['created'].dt.month
    data_df['c_day'] = data_df['created'].dt.day
    data_df['c_hour'] = data_df['created'].dt.hour
    data_df['c_dayofyear'] = data_df['created'].dt.dayofyear
    data_df['longitude'] = data_df['longitude'].apply(lambda x: round(x, 3))
    data_df['latitude'] = data_df['latitude'].apply(lambda x: round(x, 3))
    data_df.drop('created', axis=1, inplace=True)  
    # encode categorical features
    for col in ['display_address', 'street_address', 'manager_id', 'building_id']:
        data_df[col] = LabelEncoder().fit_transform(data_df[col])
    data_df.drop('description', axis=1, inplace=True)
    # get text features
    data_df['features'] = data_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    textcv = CountVectorizer(stop_words='english', max_features=200)
    text_features = pd.DataFrame(textcv.fit_transform(data_df['features']).toarray(),
                                                               columns=['f_' + format(x, '03d') for x in range(1, 201)], index=data_df.index)
    data_df = pd.concat(objs=(data_df, text_features), axis=1)
    data_df.drop('features', axis=1, inplace=True)
    feature_cols = [x for x in data_df.columns if x not in {'interest_level'}]
    del train_df, test_df
    return data_df.loc[train_index, feature_cols], data_df.loc[train_index, 'interest_level'],        data_df.loc[test_index, feature_cols]

train = pd.read_json(open("train.json", "r"))
test = pd.read_json(open("test.json", "r"))
train_X, train_y, test_df = preprocess(train, test)

train_X.drop('listing_id', axis=1, inplace=True)
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.02
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 321
param['nthread'] = 4
param['num_rounds'] = 2300

print 'training'
xgtrain = xgb.DMatrix(train_X, label=train_y)
#xgb.cv(param, xgtrain, 10000, nfold=3, verbose_eval = True, early_stopping_rounds=10)

model = xgb.train(param, xgtrain, 1100, verbose_eval = True)
listing_id = test_df['listing_id'].ravel()
test_df.drop('listing_id', axis=1, inplace=True)
xgtest = xgb.DMatrix(test_df)

preds = model.predict(xgtest)
sub = pd.DataFrame(data = {'listing_id': listing_id})
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
sub.to_csv("submission2.csv", index = False, header = True)


# we simply have to run the following code each time we modify the hyperparameters:
X = cross_validate_lgbm()


param['eta'] = 0.02
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.0.7
param['colsample_bytree'] = 0.0.7
param['seed'] = 321
param['nthread'] = 4
param['num_rounds'] = 2300
model1low = [np.nan]*len(train_X)
model2low = [np.nan]*len(train_X)
model3low = [np.nan]*len(train_X)
model4low = [np.nan]*len(train_X)
model1medium = [np.nan]*len(train_X)
model2medium = [np.nan]*len(train_X)
model3medium = [np.nan]*len(train_X)
model4medium = [np.nan]*len(train_X)
model1high = [np.nan]*len(train_X)
model2high = [np.nan]*len(train_X)
model3high = [np.nan]*len(train_X)
model4high = [np.nan]*len(train_X)
for j in range(5):
    print j 
    index=list(range(train_X.shape[0]))
    test_ind = index[int((j*train_X.shape[0])/5):int(((j+1)*train_X.shape[0])/5)]
    train_ind = list(set(index).difference(test_ind))
    train_Xfold = train_X.iloc[train_ind]
    train_YFold = train_y.iloc[train_ind]
    xgtrain = xgb.DMatrix(train_Xfold, label=train_YFold)
    model = xgb.train(param, xgtrain, 1150, verbose_eval = True)
    #param['max_depth'] = 5
    #model2 = xgb.train(param, xgtrain, 1700, verbose_eval = True)
    model3 = RandomForestClassifier(n_estimators=100)
    print 'model 2'
    model3.fit(train_Xfold, train_YFold)    
    model4 = KNeighborsClassifier(n_neighbors = 25)
    model4.fit(train_Xfold, train_YFold)    
    pred1 = model.predict(train_X.iloc[test_ind])
    #pred2 = model2.predict(train_X.iloc[test_ind])
    pred3 = model3.predict(train_X.iloc[test_ind])
    pred4 = model4.predict(train_X.iloc[test_ind])
    k = 0.0
    for i, row in test_ind.iterrows():
        x = pred1[k] 
        #x2 = pred2[k]
        x3 = pred3[k]
        x4 = pred4[k]   
        model1low[i] = x[0]
        #model2low[i] = x2[0]
        model3low[i] = x3[0]
        model4low[i] = x4[0]        
        model1medium[i] = x[1]
        #model2medium[i] = x2[1]
        model3medium[i] = x3[1]
        model4medium[i] = x4[1]        
        model1high[i] = x[1]
        #model2high[i] = x2[1]
        model3high[i] = x3[1]
        model4high[i] = x4[1]
        k += 1
train_X['model1low'] = model1low  
#train_X['model2low'] = model2low 
train_X['model3low'] = model3low 
train_X['model4low'] = model4low
train_X['model1medium'] = model1medium
#train_X['model2medium'] = model2medium
train_X['model3medium'] = model3medium 
train_X['model4medium'] = model4medium
train_X['model1high'] = model1high
#train_X['model2high'] = model2high
train_X['model3high'] = model3high
train_X['model4high'] = model4
highlm = LogisticRegression(multi_class='multinomial')
lm.fit(train_df, label=train_y)preds = lm.predict(xgtest)
sub = pd.DataFrame(data = {'listing_id': listing_id})
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
sub.to_csv("submission3.csv", index = False, header = True)
    
    

