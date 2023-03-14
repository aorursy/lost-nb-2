#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import warnings
from tabulate import tabulate
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train_users = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
sessions=pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')
train_users.head()


# In[3]:


train_users.dtypes


# In[4]:


from sklearn.model_selection import train_test_split

def get_split(df):
    X = df.drop(columns=['country_destination', 'id'])
    y = df['country_destination']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return(X_train,X_test,y_train,y_test)

X_train,X_test,y_train,y_test=get_split(train_users)


# In[5]:


#get columns by type
def get_coltypes(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    return numeric_features,categorical_features

numeric_features,categorical_features=get_coltypes(X_train)


# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#define transformers as pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[7]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
# classifiers = [
#     AdaBoostClassifier(),
#     ExtraTreesClassifier()
#     ]

def make_preds(classifier):
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])
    model=pipe.fit(X_train, y_train)
    print(classifier)
    print("model score: ",pipe.score(X_test, y_test))
    y_pred = pipe.predict(X_test)
    return model,y_pred

ada_model,ada_pred=make_preds(AdaBoostClassifier())
et_model,et_pred=make_preds(ExtraTreesClassifier())


# In[8]:


def compare_preds(y_pred):
    preds_df = pd.DataFrame(data = y_pred, columns = ['y_pred'], index = X_test.index.copy())
    df_out = pd.merge(y_test, preds_df, how = 'left', left_index = True, right_index = True)
    preds_summary=df_out.apply(pd.Series.value_counts).fillna(0)
    preds_summary['cdest_pct'] = preds_summary.country_destination / preds_summary.country_destination.sum()
    preds_summary['predicted_pct'] = preds_summary.y_pred / preds_summary.y_pred.sum()
    return preds_summary.reset_index().sort_values('country_destination',ascending=False)

ada_preds_df=compare_preds(ada_pred)
et_preds_df=compare_preds(et_pred)

ada_preds_df


# In[9]:


ada_plot=pd.melt(ada_preds_df,id_vars=['index'], value_vars=['country_destination','y_pred','cdest_pct','predicted_pct'])
ada_plot=ada_plot[ada_plot['variable'].str.contains("pct")]

et_plot=pd.melt(et_preds_df,id_vars=['index'], value_vars=['country_destination','y_pred','cdest_pct','predicted_pct'])
et_plot=et_plot[et_plot['variable'].str.contains("pct")]

ada_plot.head()


# In[10]:


from plotnine import *

(ggplot(ada_plot)+
    aes(x='index',y='value')+
    geom_col()+
    facet_wrap('variable')+
    xlab("country")+
    ylab("percent"))


# In[11]:


print(et_preds_df)
(ggplot(et_plot)+
    aes(x='index',y='value')+
    geom_col()+
    facet_wrap('variable')+
    xlab("country")+
    ylab("percent"))


# In[12]:


#base new features
sess_feat = sessions.loc[ : , ['user_id', 'secs_elapsed','action']]     .groupby('user_id')    . agg(total_secs=('secs_elapsed', 'sum'),
          total_actions=('action', 'count'))

sess_feat.head()


# In[13]:


#get top 10 actions
top_actions=sessions     .groupby('action')    .count().sort_values('user_id',ascending=False).nlargest(10,'action_type').reset_index()
print(top_actions['action'])


sessions=sessions.loc[sessions['action'].isin(top_actions['action'])]


# In[14]:


# gets count of each user,action pair and counts--> pivots to wide w/unstack
user_actions=sessions.groupby(['user_id', 'action'])         .size().unstack('action',fill_value=0).reset_index()
        

user_actions=user_actions.drop(columns=['index'],axis=1)

user_actions.head()


# In[15]:


user_actions=train_users[['id']].merge(user_actions,right_on="user_id",left_on="id",how="left").fillna(0)

user_actions=user_actions.drop('user_id',axis=1)

# add session features to user_actions
user_actions=user_actions.merge(sess_feat,left_on="id",right_on="user_id",how="left").fillna(0)

user_actions.head()


# In[16]:


assert user_actions['id'].nunique() == train_users['id'].nunique(), "Uh oh.."


# In[17]:


#join to train df
train_users=train_users     .merge(user_actions,on="id",how="left")


# In[18]:


#1
X_train,X_test,y_train,y_test=get_split(train_users)

#2
numeric_features,categorical_features=get_coltypes(X_train)

#3
ada_model,ada_pred=make_preds(AdaBoostClassifier(n_estimators=100))
et_model,et_pred=make_preds(ExtraTreesClassifier())

#preds summary df (see above)
ada_preds_df=compare_preds(ada_pred)
et_preds_df=compare_preds(et_pred)


# In[19]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ada_pred))
print(classification_report(y_test, et_pred))


# In[20]:


headers = ["name", "score"]
ada_values = sorted(zip(X_train.columns, ada_model['classifier'].feature_importances_), key=lambda x: x[1] * -1)
et_values=sorted(zip(X_train.columns, et_model['classifier'].feature_importances_), key=lambda x: x[1] * -1)

print(tabulate(ada_values, headers, tablefmt="plain"))
print(tabulate(et_values, headers, tablefmt="plain"))

