#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('../input/data.csv')


# In[6]:


df.shape


# In[334]:


df.head()


# In[335]:


df.info()


# In[336]:


#to show the null values in each column, as stated only shot_made_flag has 5000 Nan's
df.isnull().sum()


# In[337]:


sns.set(rc={'figure.figsize':(20,15)})

#plot when shots made flag
sns.lmplot('loc_x','loc_y',data=df[df.shot_made_flag == 1.0],hue = 'shot_type',fit_reg = False).set(ylim=(-70, 900))


# In[338]:


#plot when shots did not made flag
sns.lmplot('loc_x','loc_y',data=df[df.shot_made_flag == 0.0],hue='shot_type',fit_reg = False).set(ylim=(-70, 900))


# In[339]:


#The farthest point in the first graph, is the farthest distance from which he scored a basket in his entire caree.
# which is calculated as shown.
df[df.shot_made_flag == 1.0].shot_distance.max()


# In[340]:


sns.lmplot('loc_x','loc_y',data=df,hue='shot_zone_area',fit_reg = False)
#A simple visualization to show the different shot zones


# In[341]:


sns.lmplot('loc_x','loc_y',data=df,hue='shot_zone_basic',fit_reg = False)


# In[342]:


#Groups the dataframe according to the shot_distance and the sums the columns(only numerical)
df1 = df.groupby('shot_distance').sum()


# In[343]:


plt.plot(df1.index,df1.shot_made_flag)
plt.xlabel("shot distance in (ft)")
plt.ylabel("Number of times the shots made flag")
plt.title("Shot_distance vs the shots_made_flag")


# In[344]:


df1.head()


# In[345]:


#team id is 1610612747, so dividing it by df1['team_id'] gives the number of times the particular index has occured,(there may be other efficients methods, but this idea came to my mind).
#dividing the shot_made_flag with value calculated above gives the accuracy of the shotmadeflag.
#Little tricky, but you will understand.
df1['hit_rate'] = df1['shot_made_flag']/(df1['team_id']/1610612747)


# In[346]:


df1.head()


# In[347]:


plt.plot(df1.index,df1.hit_rate)
plt.xlabel("shot distance in (ft)")
plt.ylabel("Accuracy of the shots made flag")
plt.title("Shot_distance vs the accuracy of the shots made flag")


# In[348]:


df2 = df.groupby('shot_zone_area').sum()


# In[349]:


df2['shot_made_flag']


# In[350]:


# groups by season and sums the numerical columns.
df3 = df.groupby('season').sum()


# In[351]:


df3['hit_rate'] = df3['shot_made_flag']/(df3['team_id']/1610612747)


# In[352]:


df3.head()


# In[353]:


plt.plot(df3.index,df3.hit_rate)
plt.xlabel("Season(Time period)")
plt.ylabel("Accuracy of the shots made flag")
plt.title("Season vs the accuracy of the shots made flag")


# In[354]:


#Groups data frame by season and the averages the numerical columns.
df4 = df.groupby('season').mean()


# In[355]:


df4.head()


# In[356]:


plt.plot(df4.index,df4.shot_distance)
plt.xlabel("Season(Time period)")
plt.ylabel("Shot_distance")
plt.title("Season vs Shot_distance")


# In[357]:


sns.countplot('combined_shot_type',data = df, hue ='shot_made_flag')


# In[358]:


#Lets drop some useless columns.
sns.pairplot(df, vars =['lat','lon','loc_x','loc_y'])


# In[359]:


df.drop('lat',axis=1,inplace = True)
df.drop('lon',axis=1,inplace = True)


# In[360]:


df.shape


# In[361]:


df.iloc[:5,:12]


# In[362]:


df.iloc[:5,12:]


# In[363]:


len(df.game_id.unique())


# In[364]:


df['points'] = df.shot_type.apply(lambda x: x.split('P')[0]).astype(int)


# In[365]:


df['points_scored'] = df['shot_made_flag'] * df['points']
df.drop('points',axis =1, inplace = True)


# In[366]:


df.head()


# In[367]:


# groups by season and sums the numerical columns.
df5 = df.dropna()
df5 = df5.groupby('season').sum()


# In[368]:


plt.bar(df5.index,df5.points_scored)
plt.xlabel("Season(Time period)")
plt.ylabel("Points scored per season")
plt.title("Season vs Points scored in that season")


# In[369]:


df.drop('game_id',axis=1,inplace = True)
df.drop('game_event_id',axis=1,inplace = True)
df.drop('team_id',axis=1,inplace = True)
df.drop('team_name',axis=1,inplace = True)


# In[370]:


df.shape


# In[371]:


df.drop('matchup',axis =1, inplace = True)


# In[372]:


df.shape


# In[373]:


df.drop('game_date',axis=1,inplace=True)
df.drop('shot_id',axis=1,inplace=True)


# In[374]:


df.shape


# In[375]:


df.drop('points_scored',axis =1, inplace = True)


# In[376]:


df.shape


# In[377]:


df['total_seconds'] =  df['seconds_remaining'] + 60*df['minutes_remaining']


# In[378]:


df.head()


# In[379]:


df.drop('minutes_remaining',axis=1, inplace = True )


# In[380]:


df.drop('seconds_remaining',axis=1, inplace = True )


# In[381]:


df.head()


# In[382]:


df.drop('shot_zone_area',axis=1, inplace = True )
df.drop('shot_zone_basic',axis=1, inplace = True )


# In[383]:


df.shape


# In[384]:


df.head()


# In[385]:


df.info()


# In[386]:


dfalt = df


# In[387]:


dummy_columns = ['action_type','combined_shot_type','season','shot_type','shot_zone_range','opponent']


# In[388]:


for col in dummy_columns:
        dumcol = pd.get_dummies(df[col], prefix = col)
        df = df.join(dumcol)
        df.drop(col, axis =1, inplace = True)
        


# In[389]:


df.head()


# In[390]:


#Submission set with all the nan's and with no shot_flag_made column in it.
sub = df[pd.isnull(df.shot_made_flag)]


# In[391]:


#dataset is complete dataset without Nan's
dataset = df[~pd.isnull(df.shot_made_flag)]


# In[392]:


#Sub file shouldnt have the shot_made_flag columns
sub.drop('shot_made_flag',axis =1, inplace = True)


# In[393]:


y = dataset.shot_made_flag


# In[394]:


X = dataset.drop('shot_made_flag',axis=1)


# In[395]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


# In[396]:


#Splitting the training and testing dataset.
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =1)


# In[397]:


from sklearn.linear_model import LogisticRegression


# In[398]:


logreg = LogisticRegression()


# In[399]:


logreg.fit(X_train,y_train)


# In[400]:


y_pred_logreg = logreg.predict(X_test)


# In[401]:


print("The accuracy on the testing set using logistic Regression is",accuracy_score(y_test,y_pred_logreg))


# In[402]:


confusion_matrix(y_test,y_pred_logreg)


# In[1]:


print("The cross val score using logistic Regression is",cross_val_score(logreg, X, y, cv=5).mean())


# In[404]:


from sklearn.neighbors import KNeighborsClassifier


# In[405]:


knn = KNeighborsClassifier(n_neighbors = 5)


# In[406]:


knn.fit(X_train,y_train)


# In[407]:


y_pred_knn = knn.predict(X_test)


# In[408]:


print("The accuracy on the testing set using Kneighbors Classifier is",accuracy_score(y_test,y_pred_knn))


# In[409]:


confusion_matrix(y_test,y_pred_knn)


# In[410]:


print("The cross val score using Kneighbors Classifier is",cross_val_score(knn, X, y, cv=5).mean())


# In[434]:


df.shot_made_flag.value_counts()


# In[435]:


14232/(14232+11465)


# In[411]:


from sklearn.ensemble import RandomForestClassifier


# In[412]:


rfc = RandomForestClassifier(n_estimators = 10)


# In[413]:


rfc.fit(X_train,y_train)


# In[414]:


y_pred_rfc = rfc.predict(X_test)


# In[415]:


print("The accuracy on the testing set using Kneighbors Classifier is",accuracy_score(y_test,y_pred_rfc))


# In[416]:


confusion_matrix(y_test,y_pred_rfc)


# In[417]:


print("The cross val score using RandomForest Classifier is",cross_val_score(rfc, X, y, cv=5).mean())


# In[418]:


logreg_final = LogisticRegression()


# In[419]:


logreg_final.fit(X,y)


# In[420]:


output = logreg_final.predict_proba(sub)


# In[421]:


sub.index


# In[422]:


output


# In[430]:


submission_dataframe = pd.DataFrame({'shot_id':sub.index+1,'shot_made_flag':output[:,1]})


# In[431]:


submission_dataframe.to_csv("kobesubmission.csv",index= False)

