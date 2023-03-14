#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
# from pandas_profiling import ProfileReport


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


# train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
# train_profile


# In[6]:


# test_profile = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})
# test_profile


# In[7]:


from plotly.offline import iplot
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_dark"
py.init_notebook_mode(connected=True)


# In[8]:


temp = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()
temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')
temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5

fig = px.scatter_geo(temp, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country_Region", 
                     range_color=[1,100],
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Cases Over Time', color_continuous_scale="greens")
fig.show()


# In[9]:


grouped = train.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.line(grouped, x="Date", y="ConfirmedCases", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()

fig = px.line(grouped, x="Date", y="ConfirmedCases", 
              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 
              log_y=True)
fig.show()


# In[10]:


latest_grouped = train.groupby('Country_Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()


# In[11]:


fig = px.bar(latest_grouped.sort_values('ConfirmedCases', ascending=False)[:20][::-1], 
             x='ConfirmedCases', y='Country_Region',
             title='Confirmed Cases Worldwide', text='ConfirmedCases', height=1000, orientation='h')
fig.show()


# In[12]:


europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])
europe_grouped_latest = latest_grouped[latest_grouped['Country_Region'].isin(europe)]


# In[13]:


temp = train[train['Country_Region'].isin(europe)]
temp = temp.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()
temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')
temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5

fig = px.scatter_geo(temp, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country_Region", 
                     range_color=[1,100],scope='europe',
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Cases Over Time', color_continuous_scale='Cividis_r')
fig.show()


# In[14]:


fig = px.bar(europe_grouped_latest.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Country_Region', color_discrete_sequence=['#84DCC6'],
             title='Confirmed Cases in Europe', text='ConfirmedCases', orientation='h')
fig.show()


# In[15]:


usa = train[train['Country_Region'] == "US"]
usa_latest = usa[usa['Date'] == max(usa['Date'])]
usa_latest = usa_latest.groupby('Province_State')['ConfirmedCases', 'Fatalities'].max().reset_index()
fig = px.bar(usa_latest.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Province_State', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases in USA', text='ConfirmedCases', orientation='h')
fig.show()


# In[16]:


ch = train[train['Country_Region'] == "China"]
ch = ch[ch['Date'] == max(ch['Date'])]
ch = ch.groupby('Province_State')['ConfirmedCases', 'Fatalities'].max().reset_index()
fig = px.bar(ch.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Province_State', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases in china', text='ConfirmedCases', orientation='h')
fig.show()


# In[17]:


province_encoded = {state:index for index, state in enumerate(train['Province_State'].unique())}


# In[18]:


train['province_encoded'] = train['Province_State'].apply(lambda x: province_encoded[x])
train.head()


# In[19]:


country_encoded = dict(enumerate(train['Country_Region'].unique()))
country_encoded = dict(map(reversed, country_encoded.items()))


# In[20]:


train['country_encoded'] = train['Country_Region'].apply(lambda x: country_encoded[x])
train.head()


# In[21]:


from datetime import datetime
import time


# In[22]:


# date_encoded = {}
# for s in train['Date'].unique():
#     date_encoded[s] = time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())


# In[23]:


# train['date_encoded'] = train['Date'].apply(lambda x: date_encoded[x])
# train['date_encoded'] = (train['date_encoded'] - train['date_encoded'].mean()) / train['date_encoded'].std()
# train.head()


# In[24]:


train['Mon'] = train['Date'].apply(lambda x: int(x.split('-')[1]))
train['Day'] = train['Date'].apply(lambda x: int(x.split('-')[2]))


# In[25]:


train['serial'] = train['Mon'] * 30 + train['Day']
train.head()


# In[26]:


train['serial'] = train['serial'] - train['serial'].min()


# In[27]:


train.describe()


# In[28]:


gdp2020 = pd.read_csv('/kaggle/input/gdp2020/GDP2020.csv')
population2020 = pd.read_csv('/kaggle/input/population2020/population2020.csv')


# In[29]:


gdp2020 = gdp2020.rename(columns={"rank":"rank_gdp"})
gdp2020_numeric_list = [list(gdp2020)[0]] + list(gdp2020)[2:-1]
gdp2020.head()


# In[30]:


map_state = {'US':'United States', 
             'Korea, South':'South Korea',
             'Cote d\'Ivoire':'Ivory Coast',
             'Czechia':'Czech Republic',
             'Eswatini':'Swaziland',
             'Holy See':'Vatican City',
             'Jersey':'United Kingdom',
             'North Macedonia':'Macedonia',
             'Taiwan*':'Taiwan',
             'occupied Palestinian territory':'Palestine'
            }
map_state_rev = {v: k for k, v in map_state.items()}


# In[31]:


population2020['name'] = population2020['name'].apply(lambda x: map_state_rev[x] if x in map_state_rev else x)
gdp2020['country'] = gdp2020['country'].apply(lambda x: map_state_rev[x] if x in map_state_rev else x)


# In[32]:


set(train['Country_Region']) - set(population2020['name'])


# In[33]:


set(train['Country_Region']) - set(gdp2020['country'])


# In[34]:


population2020 = population2020.rename(columns={"rank":"rank_pop"})
population2020_numeric_list = [list(population2020)[0]] + list(gdp2020)[2:]
population2020.head()


# In[35]:


train = pd.merge(train, population2020, how='left', left_on = 'Country_Region', right_on = 'name')
train = pd.merge(train, gdp2020, how='left', left_on = 'Country_Region', right_on = 'country')


# In[36]:


train.isnull().sum()


# In[37]:


train = train.fillna(-1)


# In[38]:


# numeric_features_X = ['Lat','Long', 'province_encoded' ,'country_encoded','Mon','Day']
numeric_features_X = ['province_encoded' ,'country_encoded','Mon','Day'] + population2020_numeric_list + gdp2020_numeric_list
numeric_features_Y = ['ConfirmedCases', 'Fatalities']
train_numeric_X = train[numeric_features_X]
train_numeric_Y = train[numeric_features_Y]


# In[39]:


test['province_encoded'] = test['Province_State'].apply(lambda x: province_encoded[x] if x in province_encoded else max(province_encoded.values())+1)


# In[40]:


test['country_encoded'] = test['Country_Region'].apply(lambda x: country_encoded[x] if x in country_encoded else max(country_encoded.values())+1)


# In[41]:


test['Mon'] = test['Date'].apply(lambda x: int(x.split('-')[1]))
test['Day'] = test['Date'].apply(lambda x: int(x.split('-')[2]))


# In[42]:


test['serial'] = test['Mon'] * 30 + test['Day']
test['serial'] = test['serial'] - test['serial'].min()


# In[43]:


test = pd.merge(test, population2020, how='left', left_on = 'Country_Region', right_on = 'name')
test = pd.merge(test, gdp2020, how='left', left_on = 'Country_Region', right_on = 'country')


# In[44]:


# date_encoded = {}
# for s in test['Date'].unique():
#     date_encoded[s] = time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())
# test['date_encoded'] = test['Date'].apply(lambda x: date_encoded[x])
# test['date_encoded'] = (test['date_encoded'] - test['date_encoded'].mean()) / test['date_encoded'].std()
# test.head()


# In[45]:


# test.loc[:,'Lat'][test['Country/Region']=='Aruba'] = -69.9683
# test.loc[:,'Long'][test['Country/Region']=='Aruba'] = 12.5211


# In[46]:


test_numeric_X = test[numeric_features_X]
test_numeric_X.isnull().sum()


# In[47]:


test_numeric_X = test_numeric_X.fillna(-1)


# In[48]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[49]:


pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
pipeline.fit(train_numeric_X, train_numeric_Y)
predicted = pipeline.predict(test_numeric_X)


# In[50]:



fig = go.Figure()

# Make traces for graph
trace1 = go.Bar(x=train_numeric_X.columns, y=pipeline['lr'].coef_[0], xaxis='x2', yaxis='y2',
                marker=dict(color='#0099ff'),
                name='ConfirmedCases')
trace2 = go.Bar(x=train_numeric_X.columns, y=pipeline['lr'].coef_[1], xaxis='x2', yaxis='y2',
                marker=dict(color='#404040'),
                name='Fatalities')

# Add trace data to figure
fig.add_traces([trace1, trace2])

fig.update_layout(
    title_text='LR trainable weights', # title of plot
    xaxis_title_text='feature', # xaxis label
    yaxis_title_text='weight', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
# Plot!
fig.show()


# In[51]:


# submission = np.vstack((test['ForecastId'], predicted[:,0],predicted[:,1])).T
# submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('LR_submission.csv', index=False)


# In[52]:


from sklearn.svm import SVR


# In[53]:


pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])
pipeline.fit(train_numeric_X, train_numeric_Y.values[:,0])
pipeline2 = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])
pipeline2.fit(train_numeric_X, train_numeric_Y.values[:,1])
discovered, fatal = pipeline.predict(test_numeric_X), pipeline2.predict(test_numeric_X)


# In[54]:


# submission = np.vstack((test['ForecastId'], discovered, fatal)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('SVR_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[55]:


predicted_x1 =  pipeline.predict(train_numeric_X)
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_numeric_Y['ConfirmedCases'],
    histnorm='percent',
    name='actual discovered', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=-4.0,
        end=3.0,
        size=0.5
    ),
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=predicted_x1,
    histnorm='percent',
    name='predicted discovered',
    xbins=dict(
        start=-3.0,
        end=4,
        size=0.5
    ),
    opacity=0.75
))

fig.update_layout(
    title_text='SVR Histogram of ConfirmedCases', # title of plot
    xaxis_title_text='bins', # xaxis label
    yaxis_title_text='ConfirmedCases', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.show()


# In[56]:


predicted_x2 =  pipeline2.predict(train_numeric_X)
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_numeric_Y['Fatalities'],
    histnorm='percent',
    name='actual died', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=-4.0,
        end=3.0,
        size=0.5
    ),
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=predicted_x2,
    histnorm='percent',
    name='predicted died',
    xbins=dict(
        start=-3.0,
        end=4,
        size=0.5
    ),
    opacity=0.75
))

fig.update_layout(
    title_text='SVR Histogram of Fatalities', # title of plot
    xaxis_title_text='bins', # xaxis label
    yaxis_title_text='Fatalities', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.show()


# In[57]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
outcomes = []
    
fold = 0
for train_index, test_index in kf.split(train_numeric_X):
    fold += 1
    X_train, X_test = train_numeric_X.values[train_index], train_numeric_X.values[test_index]
    y_train, y_test = train_numeric_Y['ConfirmedCases'].values[train_index], train_numeric_Y['ConfirmedCases'].values[test_index]
    pipeline.fit(X_train, y_train)
    predictions = RF_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
    print("Fold {0} accuracy: {1}".format(fold, accuracy))     
mean_outcome = np.mean(outcomes)
print("\n\nMean Accuracy: {0}".format(mean_outcome)) 


# In[58]:


from sklearn.neighbors import KNeighborsClassifier


# In[59]:


pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', KNeighborsClassifier(n_jobs=4))])
pipeline.fit(train_numeric_X, train_numeric_Y)
predicted_x = pipeline.predict(train_numeric_X)


# In[60]:


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_numeric_Y['ConfirmedCases'],
    y=train_numeric_Y['Fatalities'],
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="actual",
))

fig.add_trace(go.Scatter(
    x=predicted_x[:,0],
    y=predicted_x[:,1],
    marker=dict(color="lightseagreen", size=8),
    mode="markers",
    name="predicted",
))

fig.update_layout(title="RF result",
                  xaxis_title="ConfirmedCases",
                  yaxis_title="Fatalities")

fig.show()


# In[61]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
outcomes = []
    
fold = 0
for train_index, test_index in kf.split(train_numeric_X):
    fold += 1
    X_train, X_test = train_numeric_X.values[train_index], train_numeric_X.values[test_index]
    y_train, y_test = train_numeric_Y['ConfirmedCases'].values[train_index], train_numeric_Y['ConfirmedCases'].values[test_index]
    pipeline.fit(X_train, y_train)
    predictions = RF_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
    print("Fold {0} accuracy: {1}".format(fold, accuracy))     
mean_outcome = np.mean(outcomes)
print("\n\nMean Accuracy: {0}".format(mean_outcome)) 


# In[62]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


RF_model = RandomForestClassifier(n_estimators=50, n_jobs=4, max_depth=5)
RF_model.fit(train_numeric_X, train_numeric_Y)
predicted = RF_model.predict(test_numeric_X)


# In[64]:


# submission = np.vstack((test['ForecastId'], predicted[:,0],predicted[:,1])).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('RF_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[65]:


predicted_x = RF_model.predict(train_numeric_X)


# In[66]:


# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=train_numeric_Y['ConfirmedCases'],
#     y=train_numeric_Y['Fatalities'],
#     marker=dict(color="crimson", size=12),
#     mode="markers",
#     name="actual",
# ))

# fig.add_trace(go.Scatter(
#     x=predicted_x[:,0],
#     y=predicted_x[:,1],
#     marker=dict(color="lightseagreen", size=8),
#     mode="markers",
#     name="predicted",
# ))

# fig.update_layout(title="RF result",
#                   xaxis_title="ConfirmedCases",
#                   yaxis_title="Fatalities")

# fig.show()


# In[67]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_numeric_Y['ConfirmedCases'],
    histnorm='percent',
    name='actual discovered', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=-4.0,
        end=3.0,
        size=0.5
    ),
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=predicted_x[:,0],
    histnorm='percent',
    name='predicted discovered',
    xbins=dict(
        start=-3.0,
        end=4,
        size=0.5
    ),
    opacity=0.75
))

fig.update_layout(
    title_text='RF Histogram of ConfirmedCases', # title of plot
    xaxis_title_text='bins', # xaxis label
    yaxis_title_text='ConfirmedCases', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.show()


# In[68]:


from sklearn.metrics import make_scorer, accuracy_score
accuracy_score(train_numeric_Y['ConfirmedCases'], predicted_x[:,0]), accuracy_score(train_numeric_Y['Fatalities'], predicted_x[:,1])


# In[69]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
outcomes = []
   
fold = 0
for train_index, test_index in kf.split(train_numeric_X):
   fold += 1
   X_train, X_test = train_numeric_X.values[train_index], train_numeric_X.values[test_index]
   y_train, y_test = train_numeric_Y['ConfirmedCases'].values[train_index], train_numeric_Y['ConfirmedCases'].values[test_index]
   RF_model.fit(X_train, y_train)
   predictions = RF_model.predict(X_test)
   accuracy = accuracy_score(y_test, predictions)
   outcomes.append(accuracy)
   print("Fold {0} accuracy: {1}".format(fold, accuracy))     
mean_outcome = np.mean(outcomes)
print("\n\nMean Accuracy: {0}".format(mean_outcome)) 


# In[70]:


from sklearn.ensemble import AdaBoostClassifier


# In[71]:


adaboost_model_for_ConfirmedCases = AdaBoostClassifier(n_estimators=5)
adaboost_model_for_ConfirmedCases.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[0]])
adaboost_model_for_Fatalities = AdaBoostClassifier(n_estimators=5)
adaboost_model_for_Fatalities.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[1]])


# In[72]:


# predicted = adaboost_model_for_ConfirmedCases.predict(test_numeric_X)
# predicted2 = adaboost_model_for_Fatalities.predict(test_numeric_X)
# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('Adaboost_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[73]:


predicted_x1 = adaboost_model_for_ConfirmedCases.predict(train_numeric_X)
predicted_x2 = adaboost_model_for_Fatalities.predict(train_numeric_X)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_numeric_Y['ConfirmedCases'],
    y=train_numeric_Y['Fatalities'],
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="actual",
))

fig.add_trace(go.Scatter(
    x=predicted_x1,
    y=predicted_x2,
    marker=dict(color="lightseagreen", size=8),
    mode="markers",
    name="predicted",
))

fig.update_layout(title="ADB result",
                  xaxis_title="ConfirmedCases",
                  yaxis_title="Fatalities")

fig.show()


# In[74]:


# from sklearn.ensemble import StackingClassifier


# In[75]:


# estimators = [('rf',RF_model ), ('ada', adaboost_model_for_ConfirmedCases)]
# stacking_model_for_ConfirmedCases = StackingClassifier(estimators=estimators, n_jobs=4)
# stacking_model_for_ConfirmedCases.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[0]])


# In[76]:


# stacking_model_for_Fatalities = StackingClassifier(estimators=estimators, n_jobs=4)
# stacking_model_for_Fatalities.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[1]])


# In[77]:


# predicted = stacking_model_for_ConfirmedCases.predict(test_numeric_X)
# predicted2 = stacking_model_for_Fatalities.predict(test_numeric_X)

# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)

# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('stacking_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[78]:


# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB 
# from sklearn.linear_model import LogisticRegression
# from sklearn import model_selection
# from mlxtend.classifier import StackingCVClassifier


# In[79]:


# clf1 = KNeighborsClassifier(n_neighbors=100)
# clf2 = RandomForestClassifier(n_estimators=5)
# clf3 = GaussianNB()
# # Logit will be used for stacking
# lr = LogisticRegression(solver='lbfgs')
# # sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, cv=3)
# sclf = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=lr, use_probas=True, cv=3)


# # Do CV
# for clf, label in zip([clf1, clf2, clf3, sclf], 
#                       ['KNN', 
#                        'Random Forest', 
#                        'Naive Bayes',
#                        'StackingClassifier']):

#     scores = model_selection.cross_val_score(clf, train_numeric_X.values, train_numeric_Y[numeric_features_Y[0]].values, cv=3, scoring='neg_mean_squared_log_error')
#     print("Avg_rmse: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[80]:


# clf1 = KNeighborsClassifier(n_neighbors=100)
# clf1.fit(train_numeric_X.values, train_numeric_Y[numeric_features_Y[1]])
# predicted2 = clf1.predict(test_numeric_X)

# clf2 = RandomForestClassifier(n_estimators=10)
# clf2.fit(train_numeric_X.values, train_numeric_Y[numeric_features_Y[0]])
# predicted = clf2.predict(test_numeric_X)

# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('opt_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[81]:


train_y_pred = RF_model.predict(train_numeric_X)


# In[82]:


# train_y_pred2 = clf2.predict(train_numeric_X)
# train_y_pred =  np.stack((train_y_pred, train_y_pred2), axis=-1)


# In[83]:


plt.figure(figsize=(12,8))
plt.hist([train_numeric_Y['ConfirmedCases'],train_y_pred[:,0]],bins=100, range=(1,100), label=['ConfirmedCases_actual','ConfirmedCases_pred'],alpha=0.75)
plt.title('ConfirmedCases Comparison',fontsize=20)
plt.xlabel('sample',fontsize=20)
plt.ylabel('match',fontsize=20)
plt.legend()
plt.show()


# In[84]:


plt.figure(figsize=(12,8))
plt.hist([train_numeric_Y['Fatalities'],train_y_pred[:,1]],bins=100, range=(1,100), label=['Fatalities_actual','Fatalities_pred'],alpha=0.75)
plt.title('Fatalities Comparison',fontsize=20)
plt.xlabel('sample',fontsize=20)
plt.ylabel('match',fontsize=20)
plt.legend()
plt.show()


# In[85]:


error = np.sqrt((train_y_pred - train_numeric_Y)**2)
error = error.cumsum()


# In[86]:


fig,ax = plt.subplots()
 
plt.xlabel('sample')
plt.ylabel('error')
plt.subplot(2, 1, 1)
plt.plot(range(len(error)), error['ConfirmedCases'], "x-",label="ConfirmedCases",color='orange')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(len(error)), error['Fatalities'], "+-", label="Fatalities")
plt.legend()

plt.show()


# In[87]:


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train_numeric_Y, train_y_pred , squared=False)
rmse


# In[88]:


corr = train[numeric_features_X+numeric_features_Y].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# In[89]:


corr = train[numeric_features_X+numeric_features_Y].corr(method='spearman')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# In[90]:


corr = train[numeric_features_X+numeric_features_Y].corr(method='kendall')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# In[91]:


RF_model.feature_importances_


# In[92]:


plt.bar(range(len(numeric_features_X)), RF_model.feature_importances_, tick_label=numeric_features_X)
plt.xlabel('feature')
plt.ylabel('weight')
plt.xticks(rotation=90)
plt.show()


# In[93]:


f,ax = plt.subplots()
ax.scatter(train_numeric_Y['ConfirmedCases'], train_y_pred[:,0])
ax.scatter(train_numeric_Y['Fatalities'], train_y_pred[:,1])

plt.show()


# In[94]:


# clf1 = RandomForestClassifier(n_estimators=1,n_jobs=4)
# clf3 = RandomForestClassifier(n_estimators=3,n_jobs=4)
# clf5 = RandomForestClassifier(n_estimators=5,n_jobs=4)
# clf10 = RandomForestClassifier(n_estimators=10,n_jobs=4)
# clf50 = RandomForestClassifier(n_estimators=50,n_jobs=4)


# In[95]:


# clf1.fit(train_numeric_X, train_numeric_Y)
# clf3.fit(train_numeric_X, train_numeric_Y)
# clf5.fit(train_numeric_X, train_numeric_Y)
# clf10.fit(train_numeric_X, train_numeric_Y)
# clf50.fit(train_numeric_X, train_numeric_Y)


# In[96]:


# predicted1 = clf1.predict(train_numeric_X)
# predicted3 = clf3.predict(train_numeric_X)
# predicted5 = clf5.predict(train_numeric_X)
# predicted10 = clf10.predict(train_numeric_X)
# predicted50 = clf50.predict(train_numeric_X)


# In[97]:


# a = np.sum((predicted1) - (train_numeric_Y))**2 / len(predicted1)
# b = np.sum((predicted3) - (train_numeric_Y))**2 / len(predicted3)
# c = np.sum((predicted5) - (train_numeric_Y))**2 / len(predicted5)
# d = np.sum((predicted10) - (train_numeric_Y))**2 / len(predicted10)
# e = np.sum((predicted50) - (train_numeric_Y))**2 / len(predicted50)


# In[98]:


# dt_nums = [1,3,5,10,50]
# plt.figure(figsize=(15,10))

# plt.subplot(221)
# plt.title('Decision Tree Number & MSE \n of ConfirmedCases',fontsize=20)
# plt.plot(range(len(dt_nums)), [a['ConfirmedCases'],b['ConfirmedCases'],c['ConfirmedCases'],d['ConfirmedCases'],e['ConfirmedCases']],
#          label='ConfirmedCases')
# plt.xlabel('decision tree numbers')
# plt.ylabel('mse')
# plt.xticks(range(len(dt_nums)),dt_nums)
# plt.legend()

# plt.subplot(222)
# plt.title('Decision Tree Number & MSE \n of Fatalities',fontsize=20)
# plt.plot(range(len(dt_nums)), [a['Fatalities'],b['Fatalities'],c['Fatalities'],d['Fatalities'],e['Fatalities']],
#          label='Fatalities',color='y')
# plt.xlabel('decision tree numbers')
# plt.ylabel('mse')
# plt.xticks(range(len(dt_nums)),dt_nums)
# plt.legend()

# plt.show()


# In[99]:


# clf1 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=1)
# clf2 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=2)
# clf3 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=3)
# clf4 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=4)
# clf5 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=5)
# clf10 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=10)


# In[100]:


# clf1.fit(train_numeric_X, train_numeric_Y)
# clf2.fit(train_numeric_X, train_numeric_Y)
# clf3.fit(train_numeric_X, train_numeric_Y)
# clf4.fit(train_numeric_X, train_numeric_Y)
# clf5.fit(train_numeric_X, train_numeric_Y)
# clf10.fit(train_numeric_X, train_numeric_Y)


# In[101]:


# predicted1 = clf1.predict(train_numeric_X)
# predicted2 = clf2.predict(train_numeric_X)
# predicted3 = clf3.predict(train_numeric_X)
# predicted4 = clf4.predict(train_numeric_X)
# predicted5 = clf5.predict(train_numeric_X)
# predicted10 = clf10.predict(train_numeric_X)


# In[102]:


# a = np.sum((predicted1) - (train_numeric_Y))**2 / len(predicted1)
# b = np.sum((predicted2) - (train_numeric_Y))**2 / len(predicted2)
# c = np.sum((predicted3) - (train_numeric_Y))**2 / len(predicted3)
# d = np.sum((predicted4) - (train_numeric_Y))**2 / len(predicted4)
# e = np.sum((predicted5) - (train_numeric_Y))**2 / len(predicted5)
# f = np.sum((predicted10) - (train_numeric_Y))**2 / len(predicted10)


# In[103]:


# dt_nums = [1,2,3,4,5,10]
# plt.figure(figsize=(15,10))

# plt.subplot(221)
# plt.title('Decision Tree Depth & MSE \n of ConfirmedCases',fontsize=20)
# plt.plot(range(len(dt_nums)), [a['ConfirmedCases'],b['ConfirmedCases'],c['ConfirmedCases'],d['ConfirmedCases'],e['ConfirmedCases'],f['ConfirmedCases']],
#          label='ConfirmedCases')
# plt.xlabel('decision tree depth')
# plt.ylabel('mse')
# plt.xticks(range(len(dt_nums)),dt_nums)
# plt.legend()

# plt.subplot(222)
# plt.title('Decision Tree Depth & MSE \n of Fatalities',fontsize=20)
# plt.plot(range(len(dt_nums)), [a['Fatalities'],b['Fatalities'],c['Fatalities'],d['Fatalities'],e['Fatalities'],f['ConfirmedCases']],
#          label='Fatalities',color='y')
# plt.xlabel('decision tree depth')
# plt.ylabel('mse')
# plt.xticks(range(len(dt_nums)),dt_nums)
# plt.legend()

# plt.show()


# In[ ]:





# In[ ]:




