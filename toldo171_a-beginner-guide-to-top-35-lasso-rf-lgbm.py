# Data analysis and wrangling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
import sklearn as skl
import lightgbm as lgb
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import normalized_mutual_info_score
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# File handling
import os
import gc
gc.enable()
print(os.listdir("../input"))

training_df = pd.read_csv("../input/train_V2.csv")
testing_df = pd.read_csv("../input/test_V2.csv")

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

training_df.head()

training_df.info(verbose=True)
print('_'*40)
testing_df.info(verbose=True)


training_df.describe()

training_df = training_df.drop(['Id', 'longestKill', 'rankPoints', 'numGroups', 'matchType'], axis=1)
testing_df = testing_df.drop(['Id', 'longestKill', 'rankPoints', 'numGroups', 'matchType'], axis=1)

training_df['kills'].value_counts().tail(10)

training_df = training_df.drop(training_df[training_df.kills > 20].index)
training_df['kills'].value_counts()

training_df = training_df.drop(training_df[training_df.DBNOs > 15].index)
training_df['DBNOs'].value_counts()

training_df = training_df.drop(training_df[training_df.weaponsAcquired > 30].index)
training_df['weaponsAcquired'].value_counts()

training_df.info(verbose=True)

training_df.isnull().sum()
training_df = training_df.dropna(how='any',axis=0) 
training_df.winPlacePerc.isnull().sum()

features = list(training_df.columns)
features.remove("matchId")
features.remove("groupId")
features.remove("winPlacePerc")

#Getting match mean features
print("get match mean feature")
match_mean = training_df.groupby(['matchId'])[features].agg('mean').reset_index()
training_df = training_df.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
match_mean = testing_df.groupby(['matchId'])[features].agg('mean').reset_index()
testing_df = testing_df.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])

#Getting match size features
print("get match size feature")
match_size = training_df.groupby(['matchId']).size().reset_index(name='match_size')
training_df = training_df.merge(match_size, how='left', on=['matchId'])
match_size = testing_df.groupby(['matchId']).size().reset_index(name='match_size')
testing_df = testing_df.merge(match_size, how='left', on=['matchId'])

del match_mean, match_size
gc.collect()

print("get group size feature")
group_size = training_df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
training_df = training_df.merge(group_size, how='left', on=['matchId', 'groupId'])
group_size = testing_df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
testing_df = testing_df.merge(group_size, how='left', on=['matchId', 'groupId'])

#print("get group mean feature")
#group_mean = training_df.groupby(['matchId','groupId'])[features].agg('mean')
#group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
#training_df = training_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
#training_df = training_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])
#group_mean = testing_df.groupby(['matchId','groupId'])[features].agg('mean')
#group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
#testing_df = testing_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
#testing_df = testing_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])

print("get group max feature")
group_max = training_df.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
training_df = training_df.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
training_df = training_df.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])
group_max = testing_df.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
testing_df = testing_df.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
testing_df = testing_df.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])

print("get group min feature")
group_min = training_df.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
training_df = training_df.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
training_df = training_df.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])
group_min = testing_df.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
testing_df = testing_df.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
testing_df = testing_df.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])

del group_size, group_max, group_max_rank, group_min, group_min_rank
gc.collect()

#print("get group mean feature")
#group_mean = training_df.groupby(['matchId','groupId'])[features].agg('mean')
#group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
#training_df = training_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
#training_df = training_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])

#training_df = reduce_mem_usage(training_df)
#print('_'*40)
#testing_df = reduce_mem_usage(testing_df)

#group_mean = testing_df.groupby(['matchId','groupId'])[features].agg('mean')
#group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
#testing_df = testing_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
#testing_df = testing_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])

#We don't need matchId and groupId anymore
training_df.drop(["matchId", "groupId"], axis=1, inplace=True)
testing_df.drop(["matchId", "groupId"], axis=1, inplace=True)

training_df = reduce_mem_usage(training_df)
print('_'*40)
testing_df = reduce_mem_usage(testing_df)

training_df['headshotRate'] = training_df['headshotKills'] / training_df['kills']
training_df['headshotRate'].fillna(0, inplace=True)
training_df['headshotRate'].replace(np.inf, 0, inplace=True)
testing_df['headshotRate'] = testing_df['headshotKills'] / training_df['kills']
testing_df['headshotRate'].fillna(0, inplace=True)
testing_df['headshotRate'].replace(np.inf, 0, inplace=True)

training_df['totalDistance'] = training_df['rideDistance'] + training_df['swimDistance'] + training_df['walkDistance']
testing_df['totalDistance'] = testing_df['rideDistance'] + testing_df['swimDistance'] + testing_df['walkDistance']

training_df['items'] = training_df['heals'] + training_df['boosts']
testing_df['items'] = testing_df['heals'] + testing_df['boosts']

training_df['healsPerWalkDistance'] = training_df['heals'] / training_df['walkDistance']
training_df['healsPerWalkDistance'].fillna(0, inplace=True)
training_df['healsPerWalkDistance'].replace(np.inf, 0, inplace=True)
testing_df['healsPerWalkDistance'] = testing_df['heals'] / testing_df['walkDistance']
testing_df['healsPerWalkDistance'].fillna(0, inplace=True)
testing_df['healsPerWalkDistance'].replace(np.inf, 0, inplace=True)

training_df['killsPerWalkDistance'] = training_df['kills'] / training_df['walkDistance']
training_df['killsPerWalkDistance'].fillna(0, inplace=True)
training_df['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)
testing_df['killsPerWalkDistance'] = testing_df['kills'] / testing_df['walkDistance']
testing_df['killsPerWalkDistance'].fillna(0, inplace=True)
testing_df['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)

training_df.head()

# every feature with low number of discrete values (<100). 
feature_comparison = [
    'assists',
    'boosts',
    'DBNOs',
    'headshotKills',
    'heals',
    'killPlace',
    'kills',
    'killStreaks',
    'maxPlace',
    'revives',
    'roadKills',
    'teamKills',
    'vehicleDestroys',
    'weaponsAcquired',
    'items'
    ]

#We will store every comparison table in this list
table_comparison = []
row_axis = 0
column_axis = 0

#graph individual features
fig, saxis = plt.subplots(4, 4,figsize=(16,12))

#Creating the comparison dataframes with two columns : feature, and the mean of the winplace percentage
for feature in feature_comparison:
    table_comparison.append(training_df[[feature, 'winPlacePerc']].groupby([feature], as_index=False).mean().sort_values(by=feature, ascending=True))    

#Plotting the win place percentage as a function of each feature
for table in table_comparison: 
    sns.scatterplot(x = table.iloc[:,0], y = table.winPlacePerc, ax = saxis[row_axis,column_axis])
    row_axis += 1
    if row_axis > 3:
        row_axis = 0
        column_axis += 1

# every feature with continuous value. 
feature_comparison_2 = [
    'damageDealt',
    'killPoints',
    'rideDistance',
    'swimDistance',
    'walkDistance',
    'winPoints',
    'headshotRate',
    'totalDistance',
    'healsPerWalkDistance',
    'killsPerWalkDistance'
    ]

#We will store every comparison table in this list
table_comparison_2 = []
row_axis = 0
column_axis = 0

#graph individual features
fig, saxis = plt.subplots(4, 3,figsize=(16,12))

#Creating the comparison dataframes with two columns : feature, and the mean of the winplace percentage
for feature in feature_comparison_2:
    table_comparison_2.append(training_df[[feature, 'winPlacePerc']].groupby([feature], as_index=False).mean().sort_values(by=feature, ascending=True))  
    table_comparison_2[-1][feature + '_binned'] = pd.cut(table_comparison_2[-1][feature], bins = 100, labels=False)
    table_comparison_2[-1] = table_comparison_2[-1].groupby([feature + '_binned'], as_index=False).mean().sort_values(by=feature + '_binned', ascending=True)

#Plotting the win place percentage as a function of each feature
for table in table_comparison_2: 
    sns.scatterplot(x = table.iloc[:,1], y = table.winPlacePerc, ax = saxis[row_axis,column_axis])
    row_axis += 1
    if row_axis > 3:
        row_axis = 0
        column_axis += 1

feature_MI = [
    'assists',
    'boosts',
    'damageDealt',
    'DBNOs',
    'headshotKills',
    'heals',
    'killPlace',
    'killPoints',
    'kills',
    'killStreaks',
    'matchDuration',
    'maxPlace',
    'revives',
    'rideDistance',
    'roadKills',
    'swimDistance',
    'teamKills',
    'vehicleDestroys',
    'walkDistance',
    'weaponsAcquired',
    'winPoints',
    'winPlacePerc',
    'headshotRate',
    'totalDistance',
    'items',
    'healsPerWalkDistance',
    'killsPerWalkDistance'
    ]

mutual_info_df = training_df.truncate(after=-1)

#for feature in feature_MI:
    #mutual_info_df.loc[feature] = pd.Series([np.nan])

#for feature1 in feature_MI:
    #for feature2 in feature_MI:
        #mutual_info = normalized_mutual_info_score(training_df[feature1], training_df[feature2], average_method='arithmetic')
        #if mutual_info == 1:
            #print('OK')
        #mutual_info_df[feature1][feature2] = mutual_info

#plt.figure(figsize=(9,7))
#sns.heatmap(
    mutual_info_df,
    xticklabels=mutual_info_df.columns.values,
    yticklabels=mutual_info_df.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
#plt.show()

#mutual_info_target_df = abs(mutual_info_df[['winPlacePerc']])
#mutual_info_target_df = mutual_info_target_df.drop(['winPlacePerc'])
#mutual_info_target_df['feature'] = mutual_info_target_df.index

#plt.figure(figsize=(10, 6))
#sns.barplot(x='winPlacePerc', y='feature', data=mutual_info_target_df.sort_values(by="winPlacePerc", ascending=False))
#plt.title('Mutual Information between each feature and the target value')
#plt.tight_layout()

#corr_df = training_df.corr()

#plt.figure(figsize=(9,7))
#sns.heatmap(
    corr_df,
    xticklabels=corr_df.columns.values,
    yticklabels=corr_df.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
#plt.show()

#corr_target_df = abs(corr_df[['winPlacePerc']])
#corr_target_df = corr_target_df.drop(['winPlacePerc'])
#corr_target_df['feature'] = corr_target_df.index

#plt.figure(figsize=(10, 6))
#sns.barplot(x='winPlacePerc', y='feature', data=corr_target_df.sort_values(by="winPlacePerc", ascending=False))
#plt.title('Pearson Correlation between each feature and the target value')
#plt.tight_layout()

#training_df_truncated = training_df.truncate(before=50000,after=60000)
#X_train_truncated = np.asarray(training_df_truncated[['walkDistance','killPlace', 'damageDealt', 'boosts', 'weaponsAcquired']])
#X_train_truncated = np.float32(X_train_truncated)
#X_train_truncated = PolynomialFeatures(2, interaction_only=False).fit_transform(X_train_truncated)
#X_train_truncated[0:5]

#y_train_truncated = np.asarray(training_df_truncated[['winPlacePerc']])
#y_train_truncated[0:5]

#print ('Train set truncated:', X_train_truncated.shape,y_train_truncated.shape)

#Split the model
cross_validation_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = 0)
#Create dataframe to store results according to degree of polynomial features.
lasso_results = pd.DataFrame(data = {'degree': [], 'test_score_mean': [], 'fit_time_mean': []})
#lasso_results = pd.DataFrame(data = {'degree': [], 'test_score_mean': [], 'fit_time_mean': [], 'mean_absolute_error': []})

#Evaluate the model for different dataframes. Each step increases the degree of the PolynomialFeatures function and outputs the accuracy of the model. 
for degree in range (1,6):
    X_train_truncated = np.asarray(training_df_truncated[['walkDistance','killPlace', 'damageDealt', 'boosts', 'weaponsAcquired']])
    X_train_truncated = np.float32(X_train_truncated)
    X_train_truncated = PolynomialFeatures(degree, interaction_only=False).fit_transform(X_train_truncated)
    #Evaluate the model
    cross_validation_results = model_selection.cross_validate(Lasso(alpha = 0.00001, max_iter=10000, normalize=True), X_train_truncated, y_train_truncated, cv = cross_validation_split, return_train_score = True)
    #The line below is here if you want to see the effect of LASSO compared to a classic linear regression.
    #cross_validation_results = model_selection.cross_validate(LinearRegression(), X_train_truncated, y_train_truncated, cv = cross_validation_split, return_train_score = True)
    #Predicts the target value
    #y_hat_truncated = Lasso(alpha=0.00001, max_iter=10000, normalize=True).fit(X_train_truncated, y_train_truncated).predict(X_train_truncated)
    lasso_results = lasso_results.append({'degree' : degree,
                                          'test_score_mean' : cross_validation_results['test_score'].mean(), 
                                          'fit_time_mean' : cross_validation_results['fit_time'].mean()}, ignore_index=True) 
                                          #'mean_absolute_error' : mean_absolute_error(y_train_truncated, y_hat_truncated)}, ignore_index=True)
    print('OK degree ' + str(degree))
        
sns.pointplot(x = lasso_results.degree, y = lasso_results.test_score_mean)
#sns.pointplot(x = lasso_results.degree, y = lasso_results.fit_time_mean)

#This part was here to find a good value of alpha where the test_score converge.
#It showed that alpha = 0.00001 is a good value, in terms of convergence and fit time
#------------------------------------------------------------------------------------------------------
#lasso_results = pd.DataFrame(data = {'1 / alpha': [], 'test_score_mean': [], 'fit_time_mean': []})
#lasso_alpha = 1
#denominator = 1

#for i in range (1,7):
    #cross_validation_results = model_selection.cross_validate(Lasso(alpha = (lasso_alpha / denominator), max_iter=10000, normalize=True), X_train_truncated, y_train_truncated, cv = cross_validation_split, return_train_score = True)
    #lasso_results = lasso_results.append({'1 / alpha' : (denominator), 'test_score_mean' : cross_validation_results['test_score'].mean(), 'fit_time_mean' : cross_validation_results['fit_time'].mean()}, ignore_index=True)
    #i += 1
    #denominator *= 10
#------------------------------------------------------------------------------------------------------

feature_FS = [
    'assists',
    'DBNOs',
    'headshotKills',
    'heals',
    'killPoints',
    'kills',
    'killStreaks',
    'matchDuration',
    'maxPlace',
    'revives',
    'rideDistance',
    'roadKills',
    'swimDistance',
    'teamKills',
    'vehicleDestroys',
    'winPoints',
    'headshotRate',
    'totalDistance',
    'items',
    'healsPerWalkDistance',
    'killsPerWalkDistance'
    ]

#Split the model
cross_validation_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = 0)
#Create dataframe to store results according to degree of polynomial features.
FS_results = pd.DataFrame(data = {'feature': [], 'test_score_mean': [], 'fit_time_mean': [], 'mean_absolute_error': []})
#Create X_train_truncated. The best new features will be appended to this array
X_train_truncated = np.asarray(training_df_truncated[['walkDistance','killPlace', 
                                                      'damageDealt', 'boosts', 
                                                      'weaponsAcquired']])
X_train_truncated = np.float32(X_train_truncated)
#Number of feature we want to add in the model
features_to_add = 6

#Loop for adding new feature into the model
#for i in range(1,features_to_add + 1):
    #Loops through each feature and computes cross_validation test score with LASSO regression model.
    #for feature in feature_FS:
        #Creates a temporary array
        X_temp = X_train_truncated
        #Add a new feature to the temporary array, and apply PolynomialFeatures function
        added_feat = np.asarray(training_df_truncated[[feature]])
        X_temp = np.append(X_temp, added_feat, axis = 1)
        X_temp = PolynomialFeatures(3, interaction_only=False).fit_transform(X_temp)
        #Evaluate the model
        cross_validation_results = model_selection.cross_validate(Lasso(alpha = 0.00001, max_iter=10000, normalize=True), X_temp, y_train_truncated, cv = cross_validation_split, return_train_score = True)
        #Predicts the target value
        y_hat_truncated = Lasso(alpha=0.00001, max_iter=10000, normalize=True).fit(X_temp, y_train_truncated).predict(X_temp)
        FS_results = FS_results.append({'feature' : feature, 
                                        'test_score_mean' : cross_validation_results['test_score'].mean(), 
                                        'fit_time_mean' : cross_validation_results['fit_time'].mean(), 
                                        'mean_absolute_error' : mean_absolute_error(y_train_truncated, y_hat_truncated)}, ignore_index=True)
        print('OK for ' + feature)
    
    #Store the results into a dataframe, sort it, and choose the best feature to add to the model.
    FS_results = FS_results.sort_values(by='mean_absolute_error', ascending=True)
    new_feat = FS_results.feature.iloc[0]
    new_score = FS_results.test_score_mean.iloc[0]
    new_MAE = FS_results.mean_absolute_error.iloc[0]
    new_fit_time = FS_results.fit_time_mean.iloc[0]
    X_train_truncated = np.append(X_train_truncated, np.asarray(training_df_truncated[[new_feat]]), axis = 1)
    print(new_feat + ' feature has been added to the model. Test score mean is now ' + str(new_score) + '. Mean absolute error is now ' + str(new_MAE) + '. Fit time mean is now ' + str(new_fit_time) + '.')
    i += 1

training_df_truncated = training_df.truncate(after=100000)
X_train_truncated = np.asarray(training_df_truncated.drop(['winPlacePerc'], axis = 1))
X_train_truncated[0:1]

#X_train = np.asarray(training_df.drop(['winPlacePerc'], axis = 1))
#X_train[0:1]

y_train_truncated = np.asarray(training_df_truncated[['winPlacePerc']])
y_train_truncated[0:5]

#y_train = np.asarray(training_df[['winPlacePerc']])
#y_train[0:5]

del training_df
gc.collect()

X_test = np.asarray(testing_df)
X_test[0:1]

del testing_df
gc.collect()

print ('Train set truncated:', X_train_truncated.shape,y_train_truncated.shape)
print ('Train set:', X_train.shape,y_train.shape)
print ('Test set:', X_test.shape)

models = [ 
    RandomForestRegressor(n_estimators=10, criterion = 'mse', oob_score = True, random_state = 1)
    ]

model_results = pd.DataFrame(data = {'test_score_mean': [], 'fit_time_mean': [], 'mean_absolute_error': []})

# Spliting the model
cross_validation_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = 0 )
# Performing shufflesplit cross validation, with the whole training set (the cross_validate function coupled with ShuffleSplit take care of spliting the training set) 
#for model in models:
    #cross_validation_results = model_selection.cross_validate(model, X_train_truncated, y_train_truncated, 
                                                              cv= cross_validation_split, return_train_score=True)
    #Predicts the target value on the whole training set
    y_hat = model.fit(X_train_truncated, y_train_truncated).predict(X_train)    
    # Checking the mean of test scores for each iteration of the validation    
    model_results = model_results.append({'test_score_mean' : cross_validation_results['test_score'].mean(), 
                                          'fit_time_mean' : cross_validation_results['fit_time'].mean(), 
                                          'mean_absolute_error' : mean_absolute_error(y_train, y_hat)}, ignore_index=True) 
 
model_results

#A first iteration (see below) gave these results: 
#0.9099475158875073
#{'n_estimators': 30, 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_depth': 30}
#--------------------------------------------------------------------------------------------------
#RFR = RandomForestRegressor(criterion = 'mse', oob_score = True, random_state = 1)
#param_grid = {'min_samples_leaf' : [1, 10, 50, 100, 500, 1000], 
              #'min_samples_split' : [2, 20, 100, 200, 1000, 2000], 
              #'max_depth': [10, 20, 30, 40, 50, None],
              #'n_estimators': [10, 20, 30]}

#RS = RandomizedSearchCV(estimator = RFR, 
                        #param_distributions = param_grid, 
                        #n_iter = 100, 
                        #cv = cross_validation_split, verbose = 5, random_state = 0, n_jobs = -1)

#RS = RS.fit(X_train_truncated, y_train_truncated)

#print(RS.best_score_)
#print(RS.best_params_)
#--------------------------------------------------------------------------------------------------

RFR = RandomForestRegressor(criterion = 'mse', oob_score = True, random_state = 1)
param_grid = {'min_samples_leaf' : [5, 10, 20, 40, 70, 100], 
              'min_samples_split' : [10, 20, 40, 60, 80, 100], 
              'max_depth': [10, 20, 30, 40, 50, None],
              'n_estimators': [30, 40, 50]}

#RS = RandomizedSearchCV(estimator = RFR, 
                        param_distributions = param_grid, 
                        n_iter = 100, 
                        cv = cross_validation_split, verbose = 5, random_state = 0, n_jobs = -1)

#RS = RS.fit(X_train_truncated, y_train_truncated)

print(RS.best_score_)
print(RS.best_params_)

param_grid = {'min_samples_leaf' : [5, 10, 15], 
              'min_samples_split' : [10, 15, 20], 
              'max_depth': [30, 35, 40, None],
              'n_estimators': [40]}

#GS = GridSearchCV(estimator = RFR, param_grid = param_grid, cv = cross_validation_split, verbose = 5, n_jobs = -1)
#GS = GS.fit(X_train_truncated, y_train_truncated)

print(GS.best_score_)
print(GS.best_params_)

#best_model = RandomForestRegressor(n_estimators=40, 
                                    oob_score = True,
                                    min_samples_leaf = 5,
                                    min_samples_split = 15,
                                    max_depth = 30,
                                    random_state = 1).fit(X_train_truncated,y_train_truncated)
#yhat = best_model.predict(X_train)
print("%.4f" % best_model.oob_score_)
print ("%.4f" % mean_absolute_error(y_train, y_hat))
importance_df = pd.concat((pd.DataFrame(training_df_truncated.drop(['winPlacePerc'], axis=1).columns, columns = ['variable']), 
           pd.DataFrame(best_model.feature_importances_, columns = ['importance'])), 
           axis = 1).sort_values(by='importance', ascending = False)
importance_df

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='variable', data=importance_df.sort_values(by="importance", ascending=False))
plt.title('Feature Importance')
plt.tight_layout()

#training_df_truncated = training_df.truncate(after=10000)
#X_train_truncated = np.asarray(training_df_truncated.drop(['winPlacePerc'], axis = 1))
#X_train_truncated = np.float32(X_train_truncated)
#X_train_truncated = preprocessing.StandardScaler().fit(X_train_truncated).transform(X_train_truncated)
#X_train_truncated[0:1]

#y_train_truncated = np.asarray(training_df_truncated[['winPlacePerc']])
#y_train_truncated[0:5]

print ('Train set truncated:', X_train_truncated.shape,y_train_truncated.shape)
print ('Train set:', X_train.shape,y_train.shape)
print ('Test set:', X_test.shape)

models = [ 
    #lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=1000, learning_rate=0.05, bagging_fraction = 0.9, max_bin = 127, metric = 'mae', n_jobs=-1, 
                      #max_depth=-1, num_leaves=200, min_data_in_leaf = 100),
    #lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=50, learning_rate=0.003, metric = 'mae', n_jobs=-1, 
                      #max_depth=-1, num_leaves=200, min_data_in_leaf = 100),
    #lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=100, learning_rate=0.003, metric = 'mae', n_jobs=-1, 
                      #max_depth=-1, num_leaves=200, min_data_in_leaf = 100)
    ]

model_results = pd.DataFrame(data = {'test_score_mean': [], 'fit_time_mean': [], 'mean_absolute_error': []})

# Spliting the model
cross_validation_split = model_selection.ShuffleSplit(n_splits = 4, test_size = .3, train_size = .6, random_state = 0 )
# Performing shufflesplit cross validation, with the whole training set (the cross_validate function coupled with ShuffleSplit take care of spliting the training set) 
for model in models:
    cross_validation_results = model_selection.cross_validate(model, X_train_truncated, y_train_truncated, cv= cross_validation_split, 
                                                              scoring = 'neg_mean_absolute_error', return_train_score=True)
    #Predicts the target value on the whole training set
    y_hat = model.fit(X_train_truncated, y_train_truncated).predict(X_train)    
    # Checking the mean of test scores for each iteration of the validation    
    model_results = model_results.append({'test_score_mean' : cross_validation_results['test_score'].mean(), 
                                          'fit_time_mean' : cross_validation_results['fit_time'].mean(), 
                                          'mean_absolute_error' : mean_absolute_error(y_train, y_hat)}, ignore_index=True) 
 
model_results

LGBM = lgb.LGBMRegressor(learning_rate=0.003, metric = 'mae', n_estimators = 100, n_jobs=-1)
#early_stopping_rounds = 100, 
param_grid = {'boosting_type' : ['gbdt', 'dart', 'goss'],
              'max_depth' : [10, 20, 30, -1],
              'min_data_in_leaf' : [10, 50, 100, 500, 1000],
              'num_leaves' : [50, 100, 200, 500, 1000]
             }

#RS = RandomizedSearchCV(estimator = LGBM, param_distributions = param_grid, 
                        n_iter = 50, scoring = 'neg_mean_absolute_error',
                        cv = cross_validation_split, verbose = 10, random_state = 0, n_jobs = -1)

#RS = RS.fit(X_train_truncated, y_train_truncated)

print(RS.best_score_)
print(RS.best_params_)

param_grid = {'boosting_type' : ['goss'],
              'max_depth' : [20, 30, 40, -1],
              'min_data_in_leaf' : [10, 20, 50, 100],
              'num_leaves' : [400, 500, 600]}

#GS = GridSearchCV(estimator = LGBM, param_grid = param_grid, cv = cross_validation_split, verbose = 10, scoring = 'neg_mean_absolute_error', n_jobs = -1)
#GS = GS.fit(X_train_truncated, y_train_truncated)

print(GS.best_score_)
print(GS.best_params_)

#best_model = lgb.LGBMRegressor(learning_rate=0.003, metric = 'mae', n_estimators = 2000, n_jobs=-1,
                               boosting_type = 'gbdt',
                               max_depth = 30,
                               min_data_in_leaf = 10,
                               num_leaves = 500).fit(X_train_truncated,y_train_truncated)

#y_hat = best_model.predict(X_train)
print ("%.4f" % mean_absolute_error(y_train, y_hat))
importance_df = pd.concat((pd.DataFrame(training_df_truncated.drop(['winPlacePerc'], axis=1).columns, columns = ['variable']), 
           pd.DataFrame(best_model.feature_importances_, columns = ['importance'])), 
           axis = 1).sort_values(by='importance', ascending = False)
importance_df

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='variable', data=importance_df.sort_values(by="importance", ascending=False))
plt.title('Feature Importance')
plt.tight_layout()

# Predicting the results of the testing set with the model
#yhat_test = lgb.LGBMRegressor(learning_rate=0.05, bagging_fraction = 0.9, max_bin = 127, metric = 'mae', n_estimators = 1000, n_jobs=-1,
                              boosting_type = 'gbdt',
                              max_depth = 30,
                              min_data_in_leaf = 10,
                              num_leaves = 200).fit(X_train_truncated, y_train_truncated).predict(X_test)
# Submitting
testing_df = pd.read_csv("../input/test_V2.csv")
submission = testing_df.copy()
submission['winPlacePerc'] = yhat_test
submission.to_csv('submission.csv', columns=['Id', 'winPlacePerc'], index=False)
submission[['Id', 'winPlacePerc']].head()
