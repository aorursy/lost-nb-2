#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import scikitplot as skplt
import numpy as np
import pandas as pd
import datetime

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, GRU, TimeDistributed, Input
from keras.optimizers import SGD

import xgboost as xgb
import lightgbm as lgbm

from sklearn import tree, neighbors, datasets, linear_model, svm, naive_bayes, ensemble, metrics, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate,GridSearchCV
from sklearn.utils.multiclass import unique_labels

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 999


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data_dir = '/kaggle/input/recruit-restaurant-visitor-forecasting/'
air_visit = pd.read_csv(data_dir + 'air_visit_data.csv.zip')
air_store_info = pd.read_csv(data_dir + 'air_store_info.csv.zip')
hpg_reserve = pd.read_csv(data_dir + 'hpg_reserve.csv.zip')
store_id_relation = pd.read_csv(data_dir + 'store_id_relation.csv.zip')
hpg_store_info = pd.read_csv(data_dir + 'hpg_store_info.csv.zip')
air_reserve = pd.read_csv(data_dir + 'air_reserve.csv.zip')
date_info = pd.read_csv(data_dir + 'date_info.csv.zip')


# This is the file that we submit with our results
submission = pd.read_csv(data_dir + 'sample_submission.csv.zip')
submission['air_store_id'] = submission['id'].str.slice(0, 20)
submission['visit_date'] = submission['id'].str.slice(21)
submission['is_test'] = True
submission['visitors'] = np.nan
submission['test_number'] = range(len(submission))


# In[3]:


air_visit['id'] = np.nan
air_visit['is_test'] = False
air_visit['test_number'] = np.nan
air_visit = air_visit[['id', 'visitors', 'air_store_id', 'visit_date', 'is_test','test_number']]
air_visit = pd.concat([air_visit, submission])

# We are combining the training data with the "submission" file
# There is no test data, just the NaN with where our prediction will go
air = pd.merge(air_store_info, store_id_relation, on='air_store_id', how='left')
air_visit = pd.merge(air, air_visit, on='air_store_id')
air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
air_visit = air_visit.drop(columns=['air_genre_name', 'hpg_store_id'])
air_visit.head(2)


# In[4]:


# Prior year mapping - previous year Monday to this year Monday
air_visit['prev_visitors'] = air_visit.groupby(
    [air_visit['visit_date'].dt.week,
     air_visit['visit_date'].dt.weekday])['visitors'].shift()


# In[5]:


# year / month / day_of_week
def seperate_date(data):
    data['dow'] = data['visit_date'].dt.dayofweek
    data['year'] = data['visit_date'].dt.year
    data['month'] = data['visit_date'].dt.month
    data['day'] = data['visit_date'].dt.day
    return data
air_visit = seperate_date(air_visit)
air_visit['Weekend'] = np.where(air_visit['dow'] == (0,1), 1, 0)


# Seasons
def seasonLabel(row):
    if row['month'] in [3,4,5]:
        return 'spring'
    if row['month'] in [6,7,8]:
        return 'summer'
    if row['month'] in [9,10,11]:
        return 'autumn'
    if row['month'] in [12,1,2]:
        return 'winter'
air_visit["season"] = air_visit.apply(lambda row:seasonLabel(row), axis=1) 
air_visit['summer_yes'] = np.where(air_visit['season'] == 'summer', 1, 0)


# In[6]:


# rename the columns to make the column name match other dataset
date_info.rename(columns={'holiday_flg': 'is_holiday', 'calendar_date': 'visit_date'}, inplace=True)

# previous days holiday flag. 1 means holiday, 0 means not
date_info['prev_day'] = date_info['is_holiday'].shift().fillna(0)

# following days holiday flag, 1 means holiday, 0 means not
date_info['next_day'] = date_info['is_holiday'].shift(-1).fillna(0)
date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
air_visit = pd.merge(air_visit, date_info, on='visit_date')


# In[7]:


# days since 25th
air_visit["dayofmonth"] = air_visit["visit_date"].dt.day    
air_visit["daysinPrevmonth"] = (air_visit["visit_date"] - pd.DateOffset(months=1)).dt.daysinmonth 

def daysToPrev25th(row):
    TARGET_DATE = 25
    if row['dayofmonth'] >= 25:
        return row['dayofmonth'] - TARGET_DATE
    else:
        return row['daysinPrevmonth'] - TARGET_DATE + row['dayofmonth']

air_visit["daysToPrev25th"] = air_visit.apply(lambda row:daysToPrev25th(row), axis=1)


# In[8]:


# Splitting up locations
air_df = pd.DataFrame(air_visit["air_area_name"].str.split(' ', expand=True))

# Naming the columns created from the split
air_df.columns = ['air_geo1','air_geo2','air_geo3','air_geo4','air_geo5']

# Only keeping relevant info from the split
air_df = air_df[['air_geo1','air_geo2']]

# Removing Japanese characters / symbols
air_df['geo1'] = air_df['air_geo1'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
air_df['geo2'] = air_df['air_geo2'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# Joining back with original df
air_visit = pd.concat([air_visit, air_df], axis=1)
air_visit = air_visit.drop(columns=['air_geo1', 'air_geo2'])


# In[9]:


# Population and Density

cities = ['Tokyo-to','Osaka-fu','Fukuoka-ken','Hyogo-ken','Hokkaido','Hiroshima-ken','Shizuoka-ken','Miyagi-ken','Niigata-ken','Osaka','Kanagawa-ken', 'Saitama-ken']
population = [2.7, 1.2, 8.6, 1.1, 1.0, 1.2, 1.5,3.7, 1.5, .8, 2.6, 1.6]
density = [11.9, 5.4, 13.9, 1.3, 1.3, 1.7, .5, 8.5,2.7, .7, 11.9, 4.4]

pop = pd.DataFrame(list(zip(cities, population, density)),
            columns=['cities','population', 'density'])

air_visit = pd.merge(air_visit, pop, left_on = 'geo1', right_on= 'cities', how='left')
air_visit = air_visit.drop(columns=['cities', 'air_area_name'])


# In[10]:


# =============================================================================
# Adding visitors related features
# =============================================================================

# Min
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(
    columns={'visitors': 'min_visitors'})
air_visit = pd.merge(air_visit, tmp, how='left', on=['air_store_id', 'dow'])

# Avg
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(
    columns={'visitors': 'mean_visitors'})
air_visit = pd.merge(air_visit, tmp, how='left', on=['air_store_id', 'dow'])

# Median
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(
     columns={'visitors': 'median_visitors'})
air_visit = pd.merge(air_visit, tmp, how='left', on=['air_store_id', 'dow'])

# Max
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(
     columns={'visitors': 'max_visitors'})
air_visit = pd.merge(air_visit, tmp, how='left', on=['air_store_id', 'dow'])

# Count of groups of people
tmp = air_visit.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(
    columns={'visitors': 'count_observations'})
air_visit = pd.merge(air_visit, tmp, how='left', on=['air_store_id', 'dow'])


# In[11]:


# https://github.com/MaxHalford/kaggle-recruit-restaurant/blob/master/Solution.ipynb

def find_outliers(series):
    return (series - series.mean()) > 2.4 * series.std()

def cap_values(series):
    outliers = find_outliers(series)
    max_val = series[~outliers].max()
    series[outliers] = max_val
    return series

# Identify outliers
stores = air_visit.groupby('air_store_id')
air_visit['is_outlier'] = stores.apply(lambda g: find_outliers(g['visitors'])).values
air_visit['visitors_capped'] = stores.apply(lambda g: cap_values(g['visitors'])).values
air_visit['visitors_capped_log1p'] = np.log1p(air_visit['visitors_capped'])


# In[12]:


# Split up
air_visit_train = air_visit[air_visit['is_test'] == False]
air_visit_test = air_visit[air_visit['is_test'] == True]

# Filter train
air_visit_train = air_visit_train[air_visit_train['is_outlier'] == False]
air_visit_train = air_visit_train[air_visit_train['visitors'] < 300]


# Bring them back together, drop extra columns
air_visit = pd.concat([air_visit_train, air_visit_test])
air_visit = air_visit.drop(columns = ['is_outlier', 'visitors_capped', 'visitors_capped_log1p', 'geo2'])
air_visit.head()


# In[ ]:





# In[13]:


air_visit['visitors'] = air_visit['visitors'].astype(float)
air_visit['visit_date'] = air_visit['visit_date'].astype(np.str)
air_visit = pd.get_dummies(air_visit, columns=['season', 'dow', 'geo1', 'day_of_week'])


# In[14]:


train = air_visit[air_visit.is_test == False]
test = air_visit[air_visit.is_test == True]

train.index = train[['visit_date']] # , 'air_store_id'
train_x = train.drop(columns = ['visitors'])
train_y = train[['visitors']]

test.index = test[['visit_date']] # , 'air_store_id'
test_x = test.drop(columns = ['visitors'])
test_y = test[['visitors']]

train_x = train_x.drop(columns=['air_store_id', 'id', 'visit_date'])
test_x = test_x.drop(columns=['air_store_id', 'id', 'visit_date'])

print(len(train_x.columns),len(test_x.columns))
train_x.head()


# In[15]:


#LGBM
np.random.seed(42)

model = lgbm.LGBMRegressor(
    objective='regression',
    max_depth=5,
    num_leaves=5 ** 2 - 1,
    learning_rate=0.007,
    n_estimators=30000,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=np.random.randint(10e6)
)

n_splits = 8
cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

val_scores = [0] * n_splits

sub = submission['id'].to_frame()
sub['visitors'] = 0

feature_importances = pd.DataFrame(index=train_x.columns)

for i, (fit_idx, val_idx) in enumerate(cv.split(train_x, train_y)):
    
    X_fit = train_x.iloc[fit_idx]
    y_fit = train_y.iloc[fit_idx]
    X_val = train_x.iloc[val_idx]
    y_val = train_y.iloc[val_idx]
    
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=('fit', 'val'),
        eval_metric='l2',
        early_stopping_rounds=200,
        feature_name=X_fit.columns.tolist(),
        verbose=False
    )
    
    val_scores[i] = np.sqrt(model.best_score_['val']['l2'])
    sub['visitors'] += model.predict(test_x, num_iteration=model.best_iteration_)
    feature_importances[i] = model.feature_importances_
    
    print('Fold {} RMSLE: {:.5f}'.format(i+1, val_scores[i]))
    
sub['visitors'] /= n_splits
sub['visitors'] = np.expm1(sub['visitors'])

val_mean = np.mean(val_scores)
val_std = np.std(val_scores)

print('Local RMSLE: {:.5f} (±{:.5f})'.format(val_mean, val_std))

feature_importances.sort_values(0, ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


# Basic tidying of the training and test table

# Drop unnecessary columns
train_df = train_df.drop(columns=[ 'population', 'reserve_visitors', 'days_diff', 'day', 'season'])

test = test.drop(columns=['population', 'reserve_visitors','days_diff', 'day', 'season'])
# Refine column names
train_df = train_df.rename({'visitors_x': 'visitors'}, axis = 1)
train_df = train_df.rename({'day_of_week_y': 'day_of_week'}, axis = 1)
train_df = train_df.rename({'month_y': 'month'}, axis = 1)
train_df = train_df.rename({'longitude_y': 'longitude'}, axis = 1)
train_df = train_df.rename({'latitude_y': 'latitude'}, axis = 1)
test = test.rename({'latitude_y': 'latitude'}, axis = 1)
test = test.rename({'longitude_y': 'longitude'}, axis = 1)
test = test.rename({'month_y': 'month'}, axis = 1)
test = test.rename({'day_of_week_y': 'day_of_week'}, axis = 1)

# Clean unnecessary columns
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test = test.loc[:, ~test.columns.str.contains('^Unnamed')]
# Fill the cells of missing values with -1
train_df = train_df.fillna(-1)
test = test.fillna(-1)


# In[17]:


# Encode categorical columns

# Weekday
le_weekday = LabelEncoder()
le_weekday.fit(train_df['day_of_week'])
train_df['day_of_week'] = le_weekday.transform(train_df['day_of_week'])
test['day_of_week'] = le_weekday.transform(test['day_of_week'])

# id
le_id = LabelEncoder()
le_id.fit(train_df['air_store_id'])
train_df['air_store_id'] = le_id.transform(train_df['air_store_id'])
test['air_store_id'] = le_id.transform(test['air_store_id'])

# food type
le_ftype = LabelEncoder()
le_ftype.fit(train_df['Food_Type'])
train_df['Food_Type'] = le_ftype.transform(train_df['Food_Type'])
test['Food_Type'] = le_ftype.transform(test['Food_Type'])


# In[18]:


# Simultaneous transformation of Train and test sets

# combine train and test sets
X_all = train_df.append(test)
# date table (includes all dates for training and test period)
dates = np.arange(np.datetime64(X_all.visit_date.min()),
                  np.datetime64(X_all.visit_date.max()) + 1,
                  datetime.timedelta(days=1))
ids = X_all['air_store_id'].unique()
dates_all = dates.tolist()*len(ids)
ids_all = np.repeat(ids, len(dates.tolist())).tolist()
df_all = pd.DataFrame({"air_store_id": ids_all, "visit_date": dates_all})
df_all['visit_date'] = df_all['visit_date'].copy().apply(lambda x: str(x)[:10])

# create copy of X_all with data relevant to 'visit_date'
X_dates = X_all[['visit_date', 'year','month','week',                 'is_holiday','next_day','prev_day',                 'daysToPrev25th','day_of_week','Consecutive_holidays']].copy()

# remove duplicates to avoid memory issues
X_dates = X_dates.drop_duplicates('visit_date')

# merge dataframe that represents all dates per each restaurant with information about each date
df_to_reshape = df_all.merge(X_dates,
                             how = "left",
                             left_on = 'visit_date',
                             right_on = 'visit_date')

# create copy of X_all with data relevant to 'air_store_id'
X_stores = X_all[['air_store_id', 'Food_Type', 'latitude','longitude']].copy()       

# remove duplicates to avoid memory issues
X_stores = X_stores.drop_duplicates('air_store_id')

# merge dataframe that represents all dates per each restaurant with information about each restaurant
df_to_reshape = df_to_reshape.merge(X_stores,
                                    how = "left",
                                    left_on = 'air_store_id',
                                    right_on = 'air_store_id')
# merge dataframe that represents all dates per each restaurant with inf. about each restaurant per specific date
df_to_reshape = df_to_reshape.merge(X_all[['air_store_id', 'visit_date',                                           'prev_visitors', 'mean_visitors',\ 
                                       'median_visitors', 'max_visitors',                                            'min_visitors','count_observations'                                           ,'visitors']],
                                    how = "left",
                                    left_on = ['air_store_id', 'visit_date'],
                                    right_on = ['air_store_id', 'visit_date'])

# separate 'visitors' into output array
Y_lstm_df = df_to_reshape[['visit_date', 'air_store_id', 'visitors']].copy().fillna(0)

# take log(y+1)
Y_lstm_df['visitors'] = np.log1p(Y_lstm_df['visitors'].values)

# add flag for days when a restaurant was closed
df_to_reshape['closed_flag'] = np.where(df_to_reshape['visitors'].isnull() &
                                       df_to_reshape['visit_date'].isin(train_df['visit_date']).values,1,0)

# drop 'visitors' and from dataset
df_to_reshape = df_to_reshape.drop(['visitors'], axis = 1)

# fill in NaN values
df_to_reshape = df_to_reshape.fillna(-1)

# list of df_to_reshape columns without 'air_store_id' and 'visit_date'
columns_list = [x for x in list(df_to_reshape.iloc[:,2:])]


# In[19]:


# Normalize

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df_to_reshape[columns_list])
df_to_reshape[columns_list] = scaler.transform(df_to_reshape[columns_list])


# In[20]:


# Reshape

# reshape X into (samples, timesteps, features)
X_all_lstm = df_to_reshape.values[:,2:].reshape(len(ids),
                                                len(dates),
                                                df_to_reshape.shape[1]-2)

# isolate output for train set and reshape it for time series
Y_lstm_df = Y_lstm_df.loc[Y_lstm_df['visit_date'].isin(train_df['visit_date'].values) &
                          Y_lstm_df['air_store_id'].isin(train_df['air_store_id'].values),]
Y_lstm = Y_lstm_df.values[:,2].reshape(len(train_df['air_store_id'].unique()),
                                       len(train_df['visit_date'].unique()),
                                       1)
# test dates
n_test_dates = len(test['visit_date'].unique())


# In[21]:


# Train test split

# make additional features for number of visitors in t-1, t-2, ... t-7
t_minus = np.ones([Y_lstm.shape[0],Y_lstm.shape[1],1])
for i in range(1,8):
    temp = Y_lstm.copy()
    temp[:, i:, :] = Y_lstm[:,0:-i,:].copy()
    t_minus = np.concatenate((t_minus[...], temp[...]), axis = 2)
t_minus = t_minus[:,:,1:]
print ("t_minus shape", t_minus.shape)


# split X_all into training and test data
X_lstm = X_all_lstm[:,:-n_test_dates,:]
X_lstm_test = X_all_lstm[:,-n_test_dates:,:]

# add t-1, t-2 ... t-7 visitors to feature vector
X_lstm = np.concatenate((X_lstm[...], t_minus[...]), axis = 2)

# split training set into train and validation sets
X_tr = X_lstm[:,39:-140,:]
Y_tr = Y_lstm[:,39:-140,:]

X_val = X_lstm[:,-140:,:]
Y_val = Y_lstm[:,-140:,:]


# In[22]:


# Model

# MODEL FOR ENCODER AND DECODER -------------------------------------------
num_encoder_tokens = X_lstm.shape[2]
latent_dim = 256 

# encoder training
encoder_inputs = Input(shape = (None, num_encoder_tokens))
encoder = LSTM(latent_dim, 
               batch_input_shape = (1, None, num_encoder_tokens),
               stateful = False,
               return_sequences = True,
               return_state = True,
               recurrent_initializer = 'glorot_uniform')

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c] # 'encoder_outputs' are ignored and only states are kept.

# Decoder training, using 'encoder_states' as initial state.
decoder_inputs = Input(shape=(None, num_encoder_tokens))

decoder_lstm_1 = LSTM(latent_dim,
                      batch_input_shape = (1, None, num_encoder_tokens),
                      stateful = False,
                      return_sequences = True,
                      return_state = False,
                      dropout = 0.4,
                      recurrent_dropout = 0.4) # True

decoder_lstm_2 = LSTM(128, 
                     stateful = False,
                     return_sequences = True,
                     return_state = True,
                     dropout = 0.4,
                     recurrent_dropout = 0.4)

decoder_outputs, _, _ = decoder_lstm_2(
    decoder_lstm_1(decoder_inputs, initial_state = encoder_states))

decoder_dense = TimeDistributed(Dense(Y_lstm.shape[2], activation = 'relu'))
decoder_outputs = decoder_dense(decoder_outputs)

# training model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# GENERATOR APPLIED TO FEED ENCODER AND DECODER ---------------------------
# generator that randomly creates times series of 39 consecutive days
# theses time series has following 3d shape: 829 restaurants * 39 days * num_features 
def dec_enc_n_days_gen(X_3d, Y_3d, length):
    while 1:
        decoder_boundary = X_3d.shape[1] - length - 1
        
        encoder_start = np.random.randint(0, decoder_boundary)
        encoder_end = encoder_start + length
        
        decoder_start = encoder_start + 1
        decoder_end = encoder_end + 1
        
        X_to_conc = X_3d[:, encoder_start:encoder_end, :]
        Y_to_conc = Y_3d[:, encoder_start:encoder_end, :]
        X_to_decode = X_3d[:, decoder_start:decoder_end, :]
        Y_decoder = Y_3d[:, decoder_start:decoder_end, :]
        
        yield([X_to_conc,
               X_to_decode],
               Y_decoder)
 


# In[23]:


# Generator
def dec_enc_n_days_gen(X_3d, Y_3d, length):
    while 1:
        decoder_boundary = X_3d.shape[1] - length - 1
        
        encoder_start = np.random.randint(0, decoder_boundary)
        encoder_end = encoder_start + length
        
        decoder_start = encoder_start + 1
        decoder_end = encoder_end + 1
        
        X_to_conc = X_3d[:, encoder_start:encoder_end, :]
        Y_to_conc = Y_3d[:, encoder_start:encoder_end, :]
        X_to_decode = X_3d[:, decoder_start:decoder_end, :]
        Y_decoder = Y_3d[:, decoder_start:decoder_end, :]
        
        yield([X_to_conc,
               X_to_decode],
               Y_decoder)


# In[24]:


'''
training_model.fit_generator(dec_enc_n_days_gen(X_tr, Y_tr, 39),
                             validation_data = dec_enc_n_days_gen(X_val, Y_val, 39),
                             steps_per_epoch = X_lstm.shape[0],
                             validation_steps = X_val.shape[0],
                             verbose = 1,
                             epochs = 1)
'''

# Training on full dataset
training_model.fit_generator(dec_enc_n_days_gen(X_lstm[:,:,:], Y_lstm[:,:,:], 39),
                            steps_per_epoch = X_lstm[:,:,:].shape[0],
                            verbose = 1,
                            epochs = 5)


# In[25]:


def predict_sequence(inf_enc, inf_dec, input_seq, Y_input_seq, target_seq):
    # state of input sequence produced by encoder
    state = inf_enc.predict(input_seq)
    
    # restrict target sequence to the same shape as X_lstm_test
    target_seq = target_seq[:,:, :X_lstm_test.shape[2]]
    
    
    # create vector that contains y for previous 7 days
    t_minus_seq = np.concatenate((Y_input_seq[:,-1:,:], input_seq[:,-1:, X_lstm_test.shape[2]:-1]), axis = 2)
    
    # current sequence that is going to be modified each iteration of the prediction loop
    current_seq = input_seq.copy()
    
    
    # predicting outputs
    output = np.ones([target_seq.shape[0],1,1])
    for i in range(target_seq.shape[1]):
        # add visitors for previous 7 days into features of a new day
        new_day_features = np.concatenate((target_seq[:,i:i+1,:], t_minus_seq[...]), axis = 2)
        
        # move prediction window one day forward
        current_seq = np.concatenate((current_seq[:,1:,:], new_day_features[:,]), axis = 1)
        
        
        # predict visitors amount
        pred = inf_dec.predict([current_seq] + state)
        
        # update t_minus_seq
        t_minus_seq = np.concatenate((pred[:,-1:,:], t_minus_seq[...]), axis = 2)
        t_minus_seq = t_minus_seq[:,:,:-1]        
        
        # update predicitons list
        output = np.concatenate((output[...], pred[:,-1:,:]), axis = 1)
        
        # update state
        state = inf_enc.predict(current_seq)
    
    return output[:,1:,:]

# inference encoder
encoder_model = Model(encoder_inputs, encoder_states)

# inference decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs,_,_ = decoder_lstm_2(decoder_lstm_1(decoder_inputs,
                                                    initial_state = decoder_states_inputs))
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs])

# Predicting test values
enc_dec_pred = predict_sequence(encoder_model,
                                decoder_model,
                                X_lstm[:,-X_lstm_test.shape[1]:,:],
                                Y_lstm[:,-X_lstm_test.shape[1]:,:],
                                X_lstm_test[:,:,:])


# In[ ]:





# In[26]:


# Manager intuition

# Using the previous day
df['prev_day_visitors'] = df['visitors_x'].shift(1)
df = df.groupby('air_store_id').apply(lambda group: group.iloc[1:, ])

# Using the same day of the previous week
df['prev_week_visitors'] = df.groupby([df['visit_date'].dt.weekday])['visitors_x'].shift()
df.groupby('air_store_id').apply(lambda group: group.iloc[7:, ])

# Error
df['difference_decimal'] = abs(
    df['visitors_x'] - df['prev_day_visitors']) / df['visitors_x']


# In[ ]:




