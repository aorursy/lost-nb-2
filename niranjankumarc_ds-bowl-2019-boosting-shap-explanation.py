#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from scipy.stats import mode

#modeling
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

#LGB imports
import lightgbm as lgb


# In[2]:


get_ipython().run_cell_magic('time', '', '\n#read input files\n#ignoring the event data column - JSON data. Also reduces the Memory usage.\nwanted_cols = [\'event_id\', \'game_session\', \'installation_id\', \'type\', \'world\',\'timestamp\',\'event_count\', \'event_code\',\'title\' ,\'game_time\']\ntrain_df = pd.read_csv("../input/data-science-bowl-2019/train.csv", usecols = wanted_cols)\ntest_df = pd.read_csv("../input/data-science-bowl-2019/test.csv", usecols = wanted_cols)\ntrain_labels_df =pd.read_csv(\'../input/data-science-bowl-2019/train_labels.csv\')\nsubmission_df = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")\n\nspecs_df = pd.read_csv("../input/data-science-bowl-2019/specs.csv") ')


# In[3]:


def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


#reducing memory usage for traning and testing data

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
specs_df = reduce_mem_usage(specs_df)


# In[5]:


#peaking into the data
train_df.head()


# In[6]:


test_df.head()


# In[7]:


train_labels_df.head()


# In[8]:


#looking at the size of the data

train_df.shape, test_df.shape, train_labels_df.shape


# In[9]:


#checking for any missing values in training data

train_df.isna().sum()


# In[10]:


#missing values in train labels data

train_labels_df.isna().sum()


# In[11]:


#set the plotting style
plt.style.use("seaborn")


# In[12]:


#looking at the title from the train labels data

train_labels_df["title"].value_counts().plot(kind = "barh")
plt.title("Type of Assessment")
plt.show()


# In[13]:


#looking at the labels

train_labels_df["accuracy_group"].value_counts().plot(kind = "barh")
plt.ylabel("accuracy group")
plt.show()


# In[14]:


#we will look at the relation between title and accuracy group

plt.figure(figsize=(15,10))

sns.countplot(x = "title", hue = "accuracy_group", data = train_labels_df, orient = "h")
plt.title("Accuracy_group vs Title")
plt.show()


# In[15]:


#check the distribution of installation_id

sns.distplot(train_labels_df["installation_id"].value_counts().values)
plt.show()


# In[16]:


train_df.info()


# In[17]:


#looking at the Specs data

specs_df.head()


# In[18]:


#missing values in specs_df

specs_df.isna().sum()


# In[ ]:





# In[19]:


temp_df = train_df.groupby(["installation_id", "type"]).agg({'game_session': 'count'}).reset_index()
temp_pivot_df = temp_df.pivot(index = "installation_id", columns = "type", values = "game_session").reset_index()
temp_pivot_df.head()


# In[20]:


useful_installation_id = temp_pivot_df[~temp_pivot_df["Assessment"].isna()]["installation_id"].values

#filter out the data based on the useful_installation_id.
reduced_train_df = train_df[train_df["installation_id"].isin(useful_installation_id)]

#get the unique installation id from train labels data
train_labels_installation_ids = train_labels_df["installation_id"].unique()

#further reducing the data based on the unique installation id from train labels data.
reduced_train_df = reduced_train_df[reduced_train_df["installation_id"].isin(train_labels_installation_ids)].reset_index(drop = True)

reduced_train_df.shape


# In[21]:


reduced_train_df.head()


# In[22]:


#extracting the month, hour, year, day of the week.

def extract_time_info(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"]) #convert the variable to datetime format.
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["weekday_name"] = df["timestamp"].dt.weekday_name
    
    #If weekday is sunday then set boolean to 1 else 0.
    df["is_sunday"] = df["weekday_name"].apply(lambda x: 1 if x == "Sunday" else 0)
    
    #drop the column
    df = df.drop(["weekday_name", "timestamp"], axis = 1)
    
    return df


# In[23]:


time_features=['month','hour','day','minute','is_sunday']

def prepare_data(df):
    
    #extract time features
    df=extract_time_info(df)
    
    #generate dummies for event_code and perform groupby based on installation_id and game_session
    join_one=pd.get_dummies(df[['event_code','installation_id','game_session']], columns=['event_code']).groupby(['installation_id','game_session'],
                                                            as_index=False,sort=False).agg(sum)

    #define aggregation for columns
    agg={'event_count':sum,'game_time':['sum','mean'],'event_id':'count'}
    
    #group by installation_id and game_session and perform agg
    join_two=df.drop(time_features,axis=1).groupby(['installation_id','game_session'],as_index=False,sort=False).agg(agg)
    join_two.columns= [' '.join(col).strip() for col in join_two.columns.values]
    
    #fetch the first instance of the data points
    join_three=df[['installation_id','game_session','type','world','title']].groupby(['installation_id','game_session'],as_index=False,sort=False).first()
    
    join_four=df[time_features+['installation_id','game_session']].groupby(['installation_id','game_session'],as_index=False,sort=False).agg(mode)[time_features].applymap(lambda x: x.mode[0])
    
    join_one=join_one.join(join_four)
    
    #final data
    join_five=(join_one.join(join_two.drop(['installation_id','game_session'],axis=1))).join(join_three.drop(['installation_id','game_session'],axis=1))
    
    return join_five


# In[24]:


#create aggregate train df and change the data type

agg_train_df = prepare_data(reduced_train_df)
cols=agg_train_df.columns.to_list()[2:-3]
agg_train_df[cols]=agg_train_df[cols].astype('int16')

agg_train_df.head()


# In[25]:


#create aggregate test df and change the data type

agg_test_df = prepare_data(test_df)
cols=agg_test_df.columns.to_list()[2:-3]
agg_test_df[cols]=agg_test_df[cols].astype('int16')

agg_test_df.head()


# In[26]:


cols=agg_test_df.columns[2:-12].to_list()
cols.append('event_id count')
cols.append('installation_id')


# In[27]:


#group test data by installation_id

df=agg_test_df[['event_count sum','game_time mean','game_time sum','installation_id']].groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=agg_test_df[cols].groupby('installation_id',as_index=False,sort=False).agg('sum').drop('installation_id',axis=1)

df_three=agg_test_df[['title','type','world','installation_id']].groupby('installation_id',
         as_index=False,sort=False).last().drop('installation_id',axis=1)

df_four=agg_test_df[time_features+['installation_id']].groupby('installation_id',as_index=False,sort=False).         agg(mode)[time_features].applymap(lambda x : x.mode[0])


# In[28]:


#merge with train labels for accuracy_group

final_train_df=pd.merge(train_labels_df[['installation_id','game_session','accuracy_group']],agg_train_df,on=['installation_id','game_session'],how='left').drop(['game_session', "installation_id"],axis=1)
final_train_df.head()


# In[29]:


final_train_df.shape


# In[30]:


#creating final test df

final_test_df = (df.join(df_two)).join(df_three.join(df_four)).drop('installation_id',axis=1)
final_test_df.head()


# In[ ]:





# In[31]:


final_df = pd.concat([final_train_df,final_test_df])
encoding=['type','world','title']
for col in encoding:
    lb=LabelEncoder()
    lb.fit(final_df[col])
    final_df[col]=lb.transform(final_df[col])
    
final_train_df = final_df[:len(final_train_df)]
final_test_df = final_df[len(final_train_df):]


# In[32]:


#drop the accuracy group

final_test_df.drop("accuracy_group", axis = 1, inplace = True)


# In[33]:


X_train=final_train_df.drop('accuracy_group',axis=1)
y_train=final_train_df['accuracy_group']


# In[34]:


from numba import jit 

@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e


# In[35]:


#define the parameters for lgbm.

SEED = 42
N_FOLD = 10
params = {
    'min_child_weight': 10.0,
    'objective': 'multi:softprob',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.5,
    'num_class':4,
    'learning_rate':0.05,
    'n_estimators':2000,
    'eta': 0.025,
    'gamma': 0.65,
    'eval_metric':'mlogloss'
    }

features = [i for i in final_train_df.columns if i not in ['accuracy_group']]


# In[36]:


def model(train_X,train_Y, test, params, n_splits=N_FOLD):
    
    #define KFold Strategy
    folds = StratifiedKFold(n_splits=N_FOLD,shuffle=True, random_state=SEED)
    scores = []
    
    #out of the fold 
    y_pre = np.zeros((len(test),4), dtype=float)
    target = ["accuracy_group"]
    #print("done")
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_Y)):
        print("------------------------ fold {} -------------------------".format(fold_ + 1))
        
        X_train, X_valid = train_X.iloc[trn_idx], train_X.iloc[val_idx]
        y_train, y_valid = train_Y.iloc[trn_idx], train_Y.iloc[val_idx]
        
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        
        xgb_model = xgb.train(params,
                      d_train,
                      num_boost_round=1600,
                      evals=[(d_train, 'train'), (d_valid, 'val')],
                      verbose_eval=False,
                      early_stopping_rounds=70
                     )
        
        d_val = xgb.DMatrix(X_valid)
        pred_val = [np.argmax(x) for x in xgb_model.predict(d_val)]
        
        #calculate cohen kappa score
        score = cohen_kappa_score(pred_val,y_valid,weights='quadratic')
        scores.append(score)

        pred = xgb_model.predict(xgb.DMatrix(test))
        #save predictions
        y_pre += pred
        
        print(f'Fold: {fold_+1} quadratic weighted kappa score: {np.round(score,4)}')

    pred = np.asarray([np.argmax(line) for line in pred])
    print('Mean choen_kappa_score:',np.round(np.mean(scores),6))
    
    return xgb_model,pred


# In[37]:


xgb_model,pred = model(X_train,y_train,final_test_df,params)


# In[38]:


pred[1:10]


# In[39]:


sub=pd.DataFrame({'installation_id':submission_df.installation_id,'accuracy_group':pred})
sub.to_csv('submission.csv',index=False)


# In[40]:


sub.accuracy_group.value_counts()


# In[41]:


#Classic feature attributions
ax = xgb.plot_importance(xgb_model, title='Feature importance (Weight)', importance_type='weight')
ax.figure.set_size_inches(10,8)

plt.show()


# In[42]:


#Classic feature attributions
ax = xgb.plot_importance(xgb_model, title='Feature importance (Cover)', importance_type='cover')
ax.figure.set_size_inches(10,8)

plt.show()


# In[43]:


#Classic feature attributions
ax = xgb.plot_importance(xgb_model, title='Feature importance (Gain)', importance_type='gain')
ax.figure.set_size_inches(10,8)

plt.show()


# In[ ]:





# In[44]:


import shap


# In[45]:


explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train.values)


# In[46]:


shap.summary_plot(shap_values, X_train)


# In[ ]:




