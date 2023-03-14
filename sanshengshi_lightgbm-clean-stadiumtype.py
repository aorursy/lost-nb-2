#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost
import gc
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# In[3]:


# You can only iterate through a result from `env.iter_test()` once
# so be careful not to lose it once you start iterating.
iter_test = env.iter_test()
type(iter_test)


# In[4]:


train_data=pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")


# In[5]:


train_data.columns


# In[6]:


train_data.head()


# In[7]:


# object dtype columns.
for c in train_data.columns:
    if train_data[c].dtype=="object":
        print(c, "is object dtype.","  lenght=",len(train_data[c].unique()))


# In[8]:


#StadiumType
train_data["StadiumType"].value_counts()


# In[9]:


# from https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win/output
# clean StadiumType
def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt=txt.lower()# lower case
    txt=txt.strip()# return a copy
    txt=txt.replace("outdoors","outdoor")
    txt=txt.replace("oudoor","outdoor")
    txt=txt.replace("ourdoor","outdoor")
    txt=txt.replace("outdor","outdoor")
    txt=txt.replace("outddors","outdoor")
    txt=txt.replace("outside","outdoor")
    txt=txt.replace("indoors","indoor")
    txt=txt.replace("retractable ","retr")
#     txt=txt.replace(" ","")
    return txt
train_data["StadiumType"]=train_data["StadiumType"].apply(clean_StadiumType)


# In[10]:


def transform_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0
    
    return np.nan
train_data["StadiumType"]=train_data["StadiumType"].apply(transform_StadiumType)


# In[11]:


# # author : ryancaldwell
# # Link : https://www.kaggle.com/ryancaldwell/location-eda
# def create_features(df, deploy=False):
#     def new_X(x_coordinate, play_direction):
#         if play_direction == 'left':
#             return 120.0 - x_coordinate
#         else:
#             return x_coordinate

#     def new_line(rush_team, field_position, yardline):
#         if rush_team == field_position:
#             # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
#             return 10.0 + yardline
#         else:
#             # half the field plus the yards between midfield and the line of scrimmage
#             return 60.0 + (50 - yardline)

#     def new_orientation(angle, play_direction):
#         if play_direction == 'left':
#             new_angle = 360.0 - angle
#             if new_angle == 360.0:
#                 new_angle = 0.0
#             return new_angle
#         else:
#             return angle

#     def euclidean_distance(x1,y1,x2,y2):
#         x_diff = (x1-x2)**2
#         y_diff = (y1-y2)**2

#         return np.sqrt(x_diff + y_diff)

#     def back_direction(orientation):
#         if orientation > 180.0:
#             return 1
#         else:
#             return 0

#     def update_yardline(df):
#         new_yardline = df[df['NflId'] == df['NflIdRusher']]
#         new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
#         new_yardline = new_yardline[['GameId','PlayId','YardLine']]

#         return new_yardline

#     def update_orientation(df, yardline):
#         df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
#         df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
#         df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

#         df = df.drop('YardLine', axis=1)
#         df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

#         return df

#     def back_features(df):
#         carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
#         carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
#         carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
#         carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
#         carriers = carriers.rename(columns={'X':'back_X',
#                                             'Y':'back_Y'})
#         carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

#         return carriers

#     def features_relative_to_back(df, carriers):
#         player_distance = df[['GameId','PlayId','NflId','X','Y']]
#         player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
#         player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
#         player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

#         player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])\
#                                          .agg({'dist_to_back':['min','max','mean','std']})\
#                                          .reset_index()
#         player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
#                                    'min_dist','max_dist','mean_dist','std_dist']

#         return player_distance

#     def defense_features(df):
#         rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
#         rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

#         defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
#         defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
#         defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

#         defense = defense.groupby(['GameId','PlayId'])\
#                          .agg({'def_dist_to_back':['min','max','mean','std']})\
#                          .reset_index()
#         defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

#         return defense

#     def static_features(df):
#         static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
#                                                             'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
#         static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

#         return static_features
    
#     def split_personnel(s):
#         splits = s.split(',')
#         for i in range(len(splits)):
#             splits[i] = splits[i].strip()

#         return splits

#     def defense_formation(l):
#         dl = 0
#         lb = 0
#         db = 0
#         other = 0

#         for position in l:
#             sub_string = position.split(' ')
#             if sub_string[1] == 'DL':
#                 dl += int(sub_string[0])
#             elif sub_string[1] in ['LB','OL']:
#                 lb += int(sub_string[0])
#             else:
#                 db += int(sub_string[0])

#         counts = (dl,lb,db,other)

#         return counts

#     def offense_formation(l):
#         qb = 0
#         rb = 0
#         wr = 0
#         te = 0
#         ol = 0

#         sub_total = 0
#         qb_listed = False
#         for position in l:
#             sub_string = position.split(' ')
#             pos = sub_string[1]
#             cnt = int(sub_string[0])

#             if pos == 'QB':
#                 qb += cnt
#                 sub_total += cnt
#                 qb_listed = True
#             # Assuming LB is a line backer lined up as full back
#             elif pos in ['RB','LB']:
#                 rb += cnt
#                 sub_total += cnt
#             # Assuming DB is a defensive back and lined up as WR
#             elif pos in ['WR','DB']:
#                 wr += cnt
#                 sub_total += cnt
#             elif pos == 'TE':
#                 te += cnt
#                 sub_total += cnt
#             # Assuming DL is a defensive lineman lined up as an additional line man
#             else:
#                 ol += cnt
#                 sub_total += cnt

#         # If not all 11 players were noted at given positions we need to make some assumptions
#         # I will assume if a QB is not listed then there was 1 QB on the play
#         # If a QB is listed then I'm going to assume the rest of the positions are at OL
#         # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
#         if sub_total < 11:
#             diff = 11 - sub_total
#             if not qb_listed:
#                 qb += 1
#                 diff -= 1
#             ol += diff

#         counts = (qb,rb,wr,te,ol)

#         return counts
    
#     def personnel_features(df):
#         personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
#         personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
#         personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
#         personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
#         personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
#         personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

#         personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
#         personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
#         personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
#         personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
#         personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
#         personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
#         personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

#         # Let's create some features to specify if the OL is covered
#         personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
#         personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
#         # Let's create a feature to specify if the defense is preventing the run
#         # Let's just assume 7 or more DL and LB is run prevention
#         personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

#         personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)
        
#         return personnel

#     def combine_features(relative_to_back, defense, static, personnel, deploy=deploy):
#         df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
#         df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')
#         df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')

#         if not deploy:
#             df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

#         return df
    
#     yardline = update_yardline(df)
#     df = update_orientation(df, yardline)
#     back_feats = back_features(df)
#     rel_back = features_relative_to_back(df, back_feats)
#     def_feats = defense_features(df)
#     static_feats = static_features(df)
#     personnel = personnel_features(df)
#     basetable = combine_features(rel_back, def_feats, static_feats, personnel, deploy=deploy)
#     return basetable
# outcomes = train_data[['GameId','PlayId','Yards']].drop_duplicates()
# train_data = create_features(train_data, False)


# In[12]:


# # from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
# # prove 0.002
# Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural',
#         'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 
#         'DD GrassMaster':'Artificial', 'Natural Grass':'Natural',
#         'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 
#         'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
#         'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 
#         'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 
#         'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'}
# train_data['Turf'] = train_data['Turf'].map(Turf)
# train_data['Turf'] = train_data['Turf'] == 'Natural'


# In[13]:


unused_columns = ["GameId","PlayId","Team","Yards","TimeHandoff","TimeSnap"]


# In[14]:


## Possession Team
# train_data[(train_data['PossessionTeam']!=train_data['HomeTeamAbbr']) & (train_data['PossessionTeam']!= \
#         train_data['VisitorTeamAbbr'])][['PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr']]


# In[15]:


unique_columns=[]
for c in train_data.columns:
    if c not in unused_columns+["PlayerBirthDate"] and len(set(train_data[c][:11]))!=1:
        unique_columns.append(c)
        print(c,"is unique!")


# In[16]:


all_columns=[]
for c in train_data.columns:
    if c not in unique_columns+unused_columns+["GameClock","DefensePersonnel","PlayerBirthDate"]:
        all_columns.append(c)
        
all_columns.extend(["DL","LB","DB","BirthY"])
for c in unique_columns:
    for i in range(22):
        all_columns.append(c+str(i))


# In[17]:


lbl_dict={}
for c in train_data.columns:
    if c=="DefensePersonnel":
        DL,LB,DB=[],[],[]
        for line in train_data[c]:
            features=line.split(", ")
            DL.append(int(features[0][0]))
            LB.append(int(features[1][0]))
            DB.append(int(features[2][0]))
        train_data["DL"],train_data["LB"],train_data["DB"]=DL,LB,DB
    elif c=="GameClock":
        ClockSecond=[]
        for line in train_data[c]:
            features=line.split(":")
            ClockSecond.append(features[0]*60*60+features[1]*60+features[2])
        train_data["GameClock"]=ClockSecond
    elif c=="PlayerBirthDate":
        BirthY=[]
        for line in train_data[c]:
            features=line.split("/")
            BirthY.append(int(features[-1]))
        train_data["BirthY"]=BirthY
    elif train_data[c].dtype=="object" and c not in unused_columns:
        lbl=LabelEncoder()
        lbl.fit(list(train_data[c].values))
        lbl_dict[c]=lbl
        train_data[c]=lbl.transform(list(train_data[c].values))


# In[18]:


ntrain=len(train_data.index)
Train_data=np.zeros(((ntrain-1)//22+1,len(all_columns)))
for ix in tqdm.tqdm(range(0,ntrain,22)):
    count=0
    for c in all_columns:
        if c in train_data.columns:
            Train_data[ix//22][count]=train_data[c][ix]
            count+=1
        if c in unique_columns:
            for j in range(22):
                Train_data[ix//22][count]=train_data[c][ix+j]
                count+=1     


# In[19]:


X_train=pd.DataFrame(data=Train_data,columns=all_columns)
y_train=np.array([train_data["Yards"][i] for i in range(0,ntrain,22)],dtype=np.int)


# In[20]:


data=[0]*199
for y in y_train:
    data[y]+=1
plt.figure()
plt.plot([ix-99 for ix in range(199)],data)
plt.show()


# In[21]:


Scaler=StandardScaler()
Scaler.fit(y_train.reshape(-1,1))
Y_train=Scaler.transform(y_train.reshape(-1,1)).flatten()


# In[22]:


X_train.shape,Y_train.shape


# In[23]:


folds=10
seed=22
kf=KFold(n_splits=folds,shuffle=True,random_state=seed)
y_val_pred=np.zeros(((ntrain-1)//22+1))
models=[]
for tr_idx,val_idx in kf.split(X_train,Y_train):
    x_tr,y_tr=X_train.iloc[tr_idx,:],Y_train[tr_idx]
    x_val,y_val=X_train.iloc[val_idx,:],Y_train[val_idx]
    clf = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.01)
#     clf=xgboost.XGBRegressor(n_estimators=100,learning_rate=0.1,objective='reg:squarederror',n_jobs=-1)
    clf.fit(x_tr,y_tr,eval_set=[(x_val,y_val)],
           early_stopping_rounds=20,verbose=False)
    y_val_pred[val_idx]+=clf.predict(x_val, num_iteration=clf.best_iteration_)
#     y_val_pred[val_idx]+=clf.predict(x_val)
    models.append(clf)
    
gc.collect()  


# In[24]:


Y_pred=np.zeros(((ntrain-1)//22+1,199))
Y_ans=np.zeros(((ntrain-1)//22+1,199))
for ix,p in enumerate(np.round(Scaler.inverse_transform(y_val_pred))):
    p+=99
    for j in range(199):
        if j>=(p+10):
            Y_pred[ix][j]=1.0
        elif j>=(p-10):
            Y_pred[ix][j]=(j+10-p)*0.05
            
for ix,p in enumerate(Scaler.inverse_transform(Y_train)):
    p+=99
    for j in range(199):
        if j>=p:
            Y_ans[ix][j]=1.0

print("validation score:",np.mean(np.power(Y_pred-Y_ans,2)))


# In[25]:


len(all_columns)


# In[26]:


#  test_df:DataFrame with player and game observations for the next rushing play.
#  sample_prediction_df: DataFrame with an example yardage prediction. 
#   Intended to be filled in and passed back to the predict function.
index=0
for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):
    for c in test_df.columns:
        if c=="DefensePersonnel":
            try:
                for ix,line in enumerate(test_df[c]):
                    features=line.split(", ")
                    test_df["DL"][ix]=int(features[0][0])
                    test_df["LB"][ix]=int(features[1][0])
                    test_df["DB"][ix]=int(features[2][0])
            except:
                test_df["DL"]=[np.nan for _ in range(22) ]
                test_df["LB"]=[np.nan for _ in range(22) ]
                test_df["DB"]=[np.nan for _ in range(22) ]
                    

        elif c=="GameClock":
            try:
                for ix,line in enumerate(test_df[c]):
                    features=line.split(":")
                    test_df["GameHour"][ix]=int(features[0]*60*60+features[1]*60+features[2])
            except:
                test_df["GameHour"]=[np.nan for _ in range(22) ]
        elif c=="PlayerBirthDate":
            try:
                for ix,line in enumerate(test_df[c]):
                    features=line.split("/")
                    test_df["BirthY"][ix]=int(features[-1])
            except:
                test_df["BirthY"]=[np.nan for _ in range(22) ]
        elif c in lbl_dict and test_df[c].dtype=="object" and c not in unused_columns            and not pd.isnull(test_df[c]).any():
            try:
                test_df[c]=lbl_dict[c].transform(list(test_df[c].values))
            except:
                test_df[c]=np.nan
    count=0
    test_data=np.zeros((len(all_columns)))
    for c in all_columns:
        if c in test_df.columns:
            try:
                test_data[count]=test_df[c][index]
            except:
                test_data[count]=np.nan
            count+=1
#     for c in unique_columns:
        if c in unique_columns:
            for j in range(22):
                try:
                    test_data[count]=test_df[c][index+j]
                except:
                    test_data[count]=[np.nan for _ in range(22)]
                count+=1
    Y_pred=np.zeros((199))
    Y_pred_p=np.sum(np.round(Scaler.inverse_transform([model.predict(test_data.reshape(1,-1))[0] for model in models])))/folds
    Y_pred_p+=99
    for j in range(199):
        if j>=Y_pred_p+10:
            Y_pred[j]=1.0
        elif j>=Y_pred_p-10:
            Y_pred[j]=(j+10-Y_pred_p)*0.05
    env.predict(pd.DataFrame(data=[Y_pred],columns=sample_prediction_df.columns))
    index+=22


# In[27]:


env.write_submission_file()


# In[ ]:




