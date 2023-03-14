#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


taxi_ride_train= pd.read_csv("../input/train.csv", sep=",", index_col="key", header=0, parse_dates=["pickup_datetime"], nrows=99999)
taxi_ride_test= pd.read_csv("../input/test.csv", sep=",", index_col="key", header=0, parse_dates=["pickup_datetime"])
taxi_ride_train.head()


# In[ ]:


print("The shape train data are {0}".format((taxi_ride_train.shape)))
print("The shape test data are {0}".format((taxi_ride_test.shape)))


# In[ ]:


taxi_ride_train.info()


# In[ ]:


taxi_ride_test.info()


# In[ ]:


taxi_ride_train.dtypes.value_counts().reset_index()


# In[ ]:


taxi_ride_train.isnull().sum().sum()


# In[ ]:


taxi_ride_test.isnull().sum().sum()


# In[ ]:


taxi_ride_train=taxi_ride_train.dropna(axis=0)
taxi_ride_test=taxi_ride_test.dropna(axis=0)
print(taxi_ride_train.isnull().sum().sum())
print(taxi_ride_test.isnull().sum().sum())


# In[ ]:


def calculate_distance(row):
    R = 6373.0 # approximate radius of earth in km
    lat1 = radians(row[0])
    lon1 = radians(row[1])
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    longitude_distance = lon2 - lon1
    latitude_distance = lat2 - lat1
    a = sin(latitude_distance / 2)**2 + cos(lat1) * cos(lat2) * sin(longitude_distance / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


# In[ ]:


taxi_ride_train['ride_distance_km']=taxi_ride_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(calculate_distance, axis=1)
taxi_ride_test['ride_distance_km']=taxi_ride_test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(calculate_distance, axis=1)


# In[ ]:


taxi_ride_train['ride_distance_km'].describe()


# In[ ]:


sns.boxplot(taxi_ride_train['ride_distance_km'])


# In[ ]:


IQR = taxi_ride_train.ride_distance_km.quantile(0.75) - taxi_ride_train.ride_distance_km.quantile(0.25)
Lower_fence = taxi_ride_train.ride_distance_km.quantile(0.25) - (IQR * 3)
Upper_fence = taxi_ride_train.ride_distance_km.quantile(0.75) + (IQR * 3)
print('Distance outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[ ]:


distance_outlier_train=len(taxi_ride_train[taxi_ride_train['ride_distance_km']>=30])
distance_outlier_test=len(taxi_ride_test[taxi_ride_test['ride_distance_km']>=30])
print("There are {0} trains rows and {1} test rows that have distance value more than 30km".format(distance_outlier_train,distance_outlier_test))


# In[ ]:


taxi_ride_train['ride_distance_km'] = np.where(taxi_ride_train['ride_distance_km'].astype("float64") <= 30.0, taxi_ride_train['ride_distance_km'], 30.0)
taxi_ride_train['ride_distance_km'] = np.where(taxi_ride_train['ride_distance_km'].astype("float64") >= 0.0 , taxi_ride_train['ride_distance_km'], 0.0)

taxi_ride_test['ride_distance_km'] = np.where(taxi_ride_test['ride_distance_km'].astype("float64") <= 30.0, taxi_ride_test['ride_distance_km'], 30.0)
taxi_ride_test['ride_distance_km'] = np.where(taxi_ride_test['ride_distance_km'].astype("float64") >= 0.0 , taxi_ride_test['ride_distance_km'], 0.0)


# In[ ]:


sns.boxplot(taxi_ride_train['ride_distance_km'])


# In[ ]:


sns.jointplot(x="ride_distance_km", y="fare_amount", data=taxi_ride_train);


# In[ ]:


pick_up_date_train = taxi_ride_train.ix[:,'pickup_datetime']
pick_up_date_test = taxi_ride_test.ix[:,'pickup_datetime']

temp_df_train=pd.DataFrame({"year": pick_up_date_train.dt.year,
              "month": pick_up_date_train.dt.month,
              "day": pick_up_date_train.dt.day,
              "hour": pick_up_date_train.dt.hour,
              "dayofyear": pick_up_date_train.dt.dayofyear,
              "week": pick_up_date_train.dt.week,
              "weekday": pick_up_date_train.dt.weekday,
              "quarter": pick_up_date_train.dt.quarter,
             })

temp_df_test=pd.DataFrame({"year": pick_up_date_test.dt.year,
              "month": pick_up_date_test.dt.month,
              "day": pick_up_date_test.dt.day,
              "hour": pick_up_date_test.dt.hour,
              "dayofyear": pick_up_date_test.dt.dayofyear,
              "week": pick_up_date_test.dt.week,
              "weekday": pick_up_date_test.dt.weekday,
              "quarter": pick_up_date_test.dt.quarter,
             })

taxi_ride_train= pd.concat([taxi_ride_train, temp_df_train], axis=1)
taxi_ride_test= pd.concat([taxi_ride_test, temp_df_test], axis=1)
taxi_ride_train.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_test.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_train.head()


# In[ ]:


taxi_ride_train.dtypes.value_counts().reset_index()


# In[ ]:


print("The new dataset contains {0} null entries ".format(taxi_ride_train.isnull().sum().sum()))


# In[ ]:


sns.distplot(taxi_ride_train['fare_amount'])


# In[ ]:


taxi_ride_train['fare_amount'].describe()


# In[ ]:


length_before=len(taxi_ride_train)
taxi_ride_train= taxi_ride_train[taxi_ride_train.fare_amount>=0.0]
length_after=len(taxi_ride_train)
print("No of rows removed {0}".format(length_before-length_after))


# In[ ]:


print("Skweness before transformation {0}".format( taxi_ride_train.fare_amount.skew()))
sns.distplot(np.log(taxi_ride_train['fare_amount']+1))
taxi_ride_train['fare_amount']=np.log(taxi_ride_train['fare_amount']+1)
print("Skweness after transformation {0}".format( taxi_ride_train.fare_amount.skew()))


# In[ ]:


Y_train=taxi_ride_train.fare_amount
X_train=taxi_ride_train.drop("fare_amount", axis=1)
X_test=taxi_ride_test
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
print("Shape of training set is {0}".format(X_train.shape))
print("Shape of Validation set is {0}".format(X_valid.shape))
print("Shape of testing set is {0}".format(X_test.shape))


# In[ ]:


discrete_col_list=[]
continous_col_list=[]
for col in X_train.columns.tolist():
    if(taxi_ride_train[col].value_counts().count()/len(taxi_ride_train)) < 0.1:
        discrete_col_list.append(col)
    else:
        continous_col_list.append(col)
print("The descrete column in our data are {0}".format(discrete_col_list))
print("The continous column in our data are {0}".format(continous_col_list))


# In[ ]:


# box plot and histogram of all continous variable.
for var in continous_col_list:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title('')
    
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)
 
    plt.show()


# In[ ]:


# Fix the range of latitude 
latitude_upper_range=90.0
latitude_lower_range=-90.0
for var in ['pickup_latitude','dropoff_latitude']:
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") <= latitude_upper_range, taxi_ride_train[var], latitude_upper_range)
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") >= latitude_lower_range , taxi_ride_train[var], latitude_lower_range)
    
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") <= latitude_upper_range, taxi_ride_test[var], latitude_upper_range)
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") >= latitude_lower_range , taxi_ride_test[var], latitude_lower_range)
    
# Fix the range of longitude 
longitude_upper_range=180.0
longitude_lower_range=-180.0
for var in ['pickup_latitude','dropoff_latitude']:
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") <= longitude_upper_range, taxi_ride_train[var], longitude_upper_range)
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") >= longitude_lower_range , taxi_ride_train[var], longitude_lower_range)
    
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") <= longitude_upper_range, taxi_ride_test[var], longitude_upper_range)
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") >= longitude_lower_range , taxi_ride_test[var], longitude_lower_range)


# In[ ]:


for var in continous_col_list:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title('')
    
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)


# In[ ]:


sns.distplot(np.sqrt(taxi_ride_train["ride_distance_km"]))
taxi_ride_train["ride_distance_km"]=np.sqrt(taxi_ride_train["ride_distance_km"])
taxi_ride_test["ride_distance_km"]=np.sqrt(taxi_ride_test["ride_distance_km"])


# In[ ]:


for i,var in enumerate(discrete_col_list):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    sns.countplot(taxi_ride_train[var], ax=ax)


# In[ ]:


#sns.pairplot(X_train[continous_col_list])
sns.pairplot(taxi_ride_train, x_vars=continous_col_list, y_vars='fare_amount', size=15, aspect=0.7, kind='reg')


# In[ ]:


sns.heatmap(X_train.corr())


# In[ ]:


taxi_ride_train.groupby("hour")['fare_amount'].sum().plot()


# In[ ]:


taxi_ride_train.groupby("weekday")['fare_amount'].sum().plot()


# In[ ]:


taxi_ride_train.groupby("passenger_count")['fare_amount'].sum().plot()


# In[ ]:


taxi_ride_train.groupby("month")['fare_amount'].sum().plot()


# In[ ]:


taxi_ride_train.groupby("year")['fare_amount'].sum().plot()


# In[ ]:


pd.crosstab(taxi_ride_train.quarter, len(taxi_ride_train.fare_amount), margins=True) # create a crosstab


# In[ ]:


constant_features = [
    feat for feat in taxi_ride_train.columns if taxi_ride_train[feat].std() == 0
]
print(constant_features)


# In[ ]:


sel_ = SelectFromModel(RandomForestRegressor(n_estimators=100))
sel_.fit(X_train, Y_train)
selected_feat = X_train.columns[(sel_.get_support())]
print("So the feature that holds highest importance are {0}".format(list(selected_feat)))


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print("The features that are corelated with each other are {0}".format(corr_features))
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_valid.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)


# In[ ]:


scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train) #  fit  the scaler to the train set and then transform it
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test) # transform (scale) the test set


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(X_train_scaled, Y_train)
Y_valid_pred = regr.predict(X_valid_scaled)
Y_test_pred = regr.predict(X_test_scaled)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_valid_pred))
print('Variance score: %.2f' % r2_score(Y_valid, Y_valid_pred))


# In[ ]:


# the function can be used to generate residual plots
def generate_residual_plot(label, prediction, type):
    plt.scatter(prediction, np.subtract(label, prediction))  # scatter plot
    title = 'Residual plot for predicting ' + type
    plt.title(title)  # set title
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.hlines(y=0, xmin=min(prediction), xmax=max(prediction), colors='orange', linewidth=3)  # plot ref line


# In[ ]:


# function that can be used to generate a scatter plot of actual vs prediction values
def generate_actual_vs_predicted_plot(label, prediction, type):
    plt.scatter(prediction, label, s=30, c='r', marker='+', zorder=10)  # scatter plot
    title = 'Actual vs Predicted values for ' + type
    plt.title(title)  # set title
    plt.xlabel("Predicted Values from model")  # set the xlabel
    plt.ylabel("Actual Values")  # set the ylabel
    plt.tight_layout()


# In[ ]:


generate_residual_plot(Y_valid, Y_valid_pred,
                       "Taxi fares")


# In[ ]:


generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred,
                       "Taxi fares")


# In[ ]:


params = {'n_estimators': 700, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train_scaled, Y_train)
mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
print("MSE: %.4f" % mse)
print('Variance score: %.2f' % r2_score(Y_valid, clf.predict(X_valid_scaled)))


# In[ ]:


test_pred=pd.DataFrame(clf.predict(X_test_scaled), index=X_test.index)
test_pred.columns=["fare_amount"]
test_pred['fare_amount']= np.exp(test_pred.fare_amount)
test_pred.to_csv("my_submission.csv")


# In[ ]:


# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


generate_actual_vs_predicted_plot(Y_valid, clf.predict(X_valid_scaled),
                       "Taxi fares")


# In[ ]:


generate_residual_plot(Y_valid, clf.predict(X_valid_scaled),
                       "Taxi fares")

