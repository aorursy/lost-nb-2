#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warning from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')


# In[2]:


# Read in bureau
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau.head(10)


# In[3]:


# Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns={'SK_ID_BUREAU':'previous_loan_counts'})
previous_loan_counts.head()


# In[4]:


# Join to the training dataframe
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
# 'SK_ID_CURR' 컬럼을 기준으로 머지 시켜준다.
train = train.merge(previous_loan_counts, on='SK_ID_CURR', how='left')

# Fill the missing values with 0
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()


# In[5]:


# Plots the distribution of a variable colored by value of the target
def kde_target(var_name, df):
    # Calculate the correlation coefficient between the new variable and the target.
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    # 중간값을 구하는 듯.
    avg_repaid = df.ix[df['TARGET']==0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET']==1, var_name].median()
    
    plt.figure(figsize=(12,6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET']==0, var_name], label='TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET']==1, var_name], label='TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # print out median values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid = %0.4f' % avg_repaid)
    
    


# In[6]:


kde_target('EXT_SOURCE_3', train)


# In[7]:


kde_target('previous_loan_counts', train)


# In[8]:


bureau_agg = bureau.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index=False).agg(['count','mean','max','min','sum']).reset_index()
bureau_agg.head()


# In[9]:


bureau_agg.columns.levels[0]


# In[10]:


bureau_agg.columns.levels[1] # 맨끝에 아무것도 없는 요소가 있다.


# In[11]:


bureau_agg.columns.levels[1][:-1]


# In[12]:


# List of column names
columns = ['SK_ID_CURR']

# Iterate through the variables names
for var in bureau_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        # Iterate through the stat names
        for stat in bureau_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_%s_%s' % (var, stat))


# In[13]:


# Assign the list of columns names as the dataframe column names
bureau_agg.columns = columns
bureau_agg.head()


# In[14]:


# Merge with the training data
train  = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
train.head()


# In[15]:


# List of new correlations
new_corrs = []

# Iterate through the columns
for col in columns:
    # Calculate correlations with the target
    corr = train['TARGET'].corr(train[col])
    
    # Append the list as a tuple
    # (str, float) 형태의 튜플 데이터가 new_corrs로 들어감.
    new_corrs.append((col, corr))


# In[16]:


new_corrs[:15]


# In[17]:


# Sort the correlations by the absolute value, key옵션을 써서 튜플의 1번째 인덱스 값의 절대값 기준으로 정렬할 것을 할당해줌
# Make sure to reverse to put the largest values at the front of list

new_corrs = sorted(new_corrs, key=lambda x: abs(x[1]), reverse=True)
new_corrs[:15]


# In[18]:


kde_target('bureau_DAYS_CREDIT_mean', train)


# In[19]:


def agg_numeric(df, group_var, df_name):
    """
    Aggregates the numeric values in a dataframe. This can be used to create features for 
    each instance of the grouping variable.
    
    Parameters
    ----------
        df(dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns
            
    Return
    ----------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, max, min, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created.
    """
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df.drop(columns=col)
    
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count','mean','max','min','sum']).reset_index()
    
    # Need to create new column names
    columns = [group_var]
    
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    return agg
    
    


# In[20]:


bureau_agg_new = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='bureau')
bureau_agg_new.head()


# In[21]:


bureau_agg.head()


# In[22]:


# Function to calculate correlation with the target for a dataframe

def target_corrs(df):
    # List of correalations, 빈 리스트
    corrs = []
    
    # Interate through the columns
    for col in df.columns:
        print(col)
        # Skip the target column:
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df['col'])
            
            # Append the list as a tuple
            corrs.append((col, corr))
            
    # Sort by absolute magnitude of correlation
    corrs = sorted(corrs, key = lambda x : abs(x[1]), reverse=True)
    
    return corrs


# In[23]:


categorical = pd.get_dummies(bureau.select_dtypes('object'))
# SK_ID_CURR은 숫자로 이루어진 컬럼이라 포함되어 있지 않았기 때문에, 아래와 같이 추가시켜 동기화? 시켜줌
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical.head()


# In[24]:


# SK_ID_CURR 컬럼으로 그룹화 해서 agg 특성 컬럼을 추가시켜줌
categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum','mean'])
categorical_grouped.head()


# In[25]:


categorical_grouped.columns.levels[0][:10] # 전체 23개 컬럼으로 이루어졌음


# In[26]:


categorical_grouped.columns.levels[1]


# In[27]:


group_var = 'SK_ID_CURR'

# Need to create new column names
# 여기서 아래의 리스트에 'SK_ID_CURR'를 초기값으로 넣지 않는 이유는,
# 위에서 categorical_grouped데이터프레임을 만들때, as_index를 지정하지 않아서 디폴트로 True롤 설정되어 버림.
# 즉, 'SK_ID_CURR'컬럼이 인덱스로 지정되어서 컬럼 이름에 초기값으로 설정하지 않은 것 같음.
columns = []

# Iterate through the variables names
for var in categorical_grouped.columns.levels[0]:
    # Skip the grouping variable
    if var != group_var:
        for stat in ['count', 'count_norm']:
            # Make a new column name for the variable and stat
            columns.append('%s_%s' % (var, stat))

# Rename the columns
categorical_grouped.columns = columns

categorical_grouped.head()
            


# In[28]:


train = train.merge(categorical_grouped, left_on='SK_ID_CURR', right_index = True, how='left')
train.head()


# In[29]:


train.shape


# In[30]:


train.iloc[:10,123:]


# In[31]:


def count_categorical(df, group_var, df_name):
    """
    Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    ----------
    df: dataframe
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns
        
    Return
    ------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum','mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1, with new name!
        for stat in ['count','count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    # 컬럼 이름 다시 할당해주기.
    
    categorical.columns = column_names
    
    return categorical     
    
    
    


# In[32]:


bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')
bureau_counts.head()


# In[33]:


# Read in bureau balance
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
bureau_balance.head()


# In[34]:


# Counts of each type of status for each previous loan
bureau_balance_counts = count_categorical(bureau_balance,group_var='SK_ID_BUREAU', df_name='bureau_balance')
bureau_balance_counts.head()


# In[35]:


# Calculate value count statistics for each `SK_ID_CURR`
# 숫자 컬럼에 대해서만 작업해주기 때문에 이걸 사용함
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()


# In[36]:


# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(
    bureau_balance_counts, 
    right_index=True, 
    left_on = 'SK_ID_BUREAU', 
    how='outer')
bureau_by_loan.head()


# In[37]:


# Merge to include the SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(
    bureau[['SK_ID_BUREAU','SK_ID_CURR']], 
    on='SK_ID_BUREAU', 
    how='left')
bureau_by_loan.head()


# In[38]:


bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='client')

bureau_balance_by_client.head()


# In[39]:


# Free up memory by deleting old objects
import gc # garbage collector?
gc.enable()
del train, bureau, bureau_balance, bureau_agg, bureau_agg_new, bureau_balance_agg,bureau_balance_counts,bureau_balance_by_client, bureau_by_loan, bureau_counts
gc.collect()


# In[40]:


# Read in new copies of all the dataframes
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')


# In[41]:


bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')
bureau_counts.head()
#dtype=='object' 인 컬럼만 처리됨.


# In[42]:


# dtype=='number' 인 컬럼만 가지고 와서 처리함
bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='bureau' )
bureau_agg.head()


# In[43]:


bureau_balance_counts = count_categorical(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
bureau_balance_counts.head()


# In[44]:


# 수치데이터에 대해서 수치작업 진행!
bureau_balance_agg = agg_numeric(bureau_balance, group_var='SK_ID_BUREAU',df_name='bureau_balance')
bureau_balance_agg.head()


# In[45]:


# Dataframe grouped by the loan
# 위에서 작성한 대출아이디에 대해 통계화 한 수치적 데이터와 카테고리 데이터를 하나로 머지 시켜줌
bureau_by_loan = bureau_balance_agg.merge(
    bureau_balance_counts, 
    right_index=True,
    left_on='SK_ID_BUREAU',
    how='outer'
)

# Merge to include the SK_ID_CURR
# SK_ID_BUREAU, SK_ID_CURR를 서로 맞춰 주기 위해서 
# 원래 데이터 프레임에서 두 컬럼 데이터를 가지고와서 머지에 사용함
bureau_by_loan = bureau[['SK_ID_BUREAU','SK_ID_CURR']].merge(
    bureau_by_loan,
    on='SK_ID_BUREAU',
    how='left'
)


# In[46]:


bureau_by_loan.head()


# In[47]:


# Aggregate the stats for each client
bureau_balance_by_client = agg_numeric(
    bureau_by_loan.drop(columns=['SK_ID_BUREAU']),
    group_var='SK_ID_CURR',
    df_name='client'
)


# In[48]:


bureau_balance_by_client.head()


# In[49]:


original_features = list(train.columns)
print('Original Number of Features : ', len(original_features))


# In[50]:


# Merge with the value counts of bureau
train = train.merge(bureau_counts, on = 'SK_ID_CURR', how='left')

# Merge with the stats of bureau
train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Merge with the monthly information grouped by client
train = train.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')


# In[51]:


new_features = list(train.columns)
print('Number of features using previous loans from other instituation data : ',
     len(new_features))


# In[52]:


# Function to calculate missing values by column 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'}   
    )
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print('Your selected dataframe has ' + str(df.shape[1]) + ' columns.\n'
          'There are ' + str(mis_val_table_ren_columns.shape[0]) + 
          ' columns that have missing values.')
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# In[53]:


missing_train = missing_values_table(train)
missing_train.head(10)


# In[54]:


missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])
len(missing_train_vars)


# In[55]:


# Read in the test dataframe
test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
test.head()


# In[56]:


# Merge with the value counts of bureau
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how='left')

# Merge with the stats of bureau
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how='left')

# Merge with the value counts of bureau balance
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')


# In[57]:


print('Shape of Training Data : ', train.shape)
print('Shape of Testing Data: ', test.shape)


# In[58]:


train_labels = train['TARGET']

# Align the dataframes, this will remove the 'TARGET' column
train, test = train.align(test, join = 'inner', axis = 1)

train['TARGET'] = train_labels


# In[59]:


print('Training Data Shape : ', train.shape)
print('Testing Data Shape : ', test.shape)


# In[60]:


train.head(5)


# In[61]:


test.head(5)


# In[62]:


missing_test = missing_values_table(test)
missing_test.head(10)


# In[63]:


missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
len(missing_test_vars)


# In[64]:


# set을 쓰는 이유는, 중복되는 요소를 없애주기 위함. 근데 여긴 어차피 두 변수 아무 요소도 없음.
# 개념적으로, 구성하는 방법만 참고.
missing_columns = list(set(missing_test_vars + missing_train_vars))
print('There are %d columns with more than 90%% missing in either the training or testing data.'
     % len(missing_columns))


# In[65]:


# set사용법 참고
set([1,2,3] + [3,10,11])


# In[66]:


# Drop the missing columns
# 의미 없는 dropping
train = train.drop(columns=missing_columns)
test = test.drop(columns=missing_columns)


# In[67]:


train.to_csv('train_bureau_raw.csv', index=False)
test.to_csv('test_bureau_raw.csv', index=False)


# In[68]:


# Calculate all correlations in dataframe
corrs = train.corr()


# In[69]:


corrs = corrs.sort_values('TARGET', ascending = False)

# Ten most positive correlations
pd.DataFrame(corrs['TARGET'].head(10))


# In[70]:


# Ten most negative correlations
pd.DataFrame(corrs['TARGET'].dropna().tail(10))


# In[71]:


kde_target(var_name='client_bureau_balance_MONTHS_BALANCE_count_mean', df=train)


# In[72]:


kde_target(var_name='bureau_CREDIT_ACTIVE_Active_count_norm', df=train)


# In[73]:


# Set the threshold
threshold = 0.8

# Empty dictionary to hold correlated variables
above_threshold_vars = {}

# For each column, record the variables that are above the threshold
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])


# In[74]:


above_threshold_vars


# In[75]:


# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
# 사전형 뒤에 .items() 로 루프를 돌리면, key,value 두개를 같은 짝으로 뿌려줄수 있음.

for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)

cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove : ', len(cols_to_remove))


# In[76]:


train_corrs_removed = train.drop(columns=cols_to_remove)
test_corrs_removed = test.drop(columns=cols_to_remove)


# In[77]:


print('Training Corrs Removed Shape : ', train_corrs_removed.shape)
print('Testing Corrs Removed Shape : ', test_corrs_removed.shape)


# In[78]:


train_corrs_removed.to_csv('train_bureau_corrs_removed.csv', index=False)
test_corrs_removed.to_csv('test_bureau_corrs_removed.csv', index=False)


# In[79]:


import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc #garbage collector

import matplotlib.pyplot as plt


# In[80]:


# 이전 노트북에서 정의한 함수 그대로 가져옴
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    """
    Train and Test a light gradient boosting model using cross validation.
    
    Parameters:
    -----------
        features(pd.DataFrame):
            dataframe of training feature to use
            for training a model. Must include Target column.
        test_features(pd.DataFrame):
            dataframe of testing feature to use
            for making predictions with the model.
        encoding(str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for interger label encoding
        n_folds(int, default = 5): number of folds to use for cross validation
        
    Return:
    -------
        submission(pd.DataFrame):
            dataframe with 'SK_ID_CURR' and 'TARGET' probabilities
            predicted by the model.
        feature_importances(pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics(pd.DataFrame):
            dataframe with training and validation metrics(ROC AUC) for each fold and overall.
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


# In[81]:


#영향계수? 를 플랏하는 함수를 작성!
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of 
    feature importance provided that higher importance is better.
    
    Args:
        df(dataframe): feature importances. Must have the features in a column
        called 'features' and the importances in a column called 'importance'
        
    Reutrns:
        shows a plot of the 15 most importance features
        
        df(dataframe): feature importances sorted by importance(hightest to lowest)
        with a column for normalized importance
    """
    
    # Sort features according to importance
    # reset_index()를 하면, sorting 후 내림차순이 된 데이터의 row에 대해서 다시 0부터 인덱스를 매겨준다!
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    # 전체 영향계수? 를 더하면 1이 되므로, 그 값으로 각각의 영향 계수를 나눠서 일반화 해줌.(0,1)사이의 값을 갖는다.
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10,6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    # 여기서 인덱스를 리버스 하면서 앞에 list를 겹처서 붙여주는 이유는, 
    # 상속되는 대상이 객체(object) 타입의? 데이터라서, 리스트 화 시켜주기 위함이다.
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align='center', 
            edgecolor='k')
    
    # Set the yticks and labels, y축 값 범위 정해주고, y축 데이터 라벨 이름 할당해주기
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()
    
    return df


# In[82]:


train_control = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
test_control = pd.read_csv('../input/home-credit-default-risk/application_test.csv')


# In[83]:


submission, fi, metrics = model(train_control, test_control)


# In[84]:


metrics


# In[85]:


fi.head()


# In[86]:


fi_sorted = plot_feature_importances(fi)


# In[87]:


submission.to_csv('control.csv', index=False)


# In[88]:


submission_raw, fi_raw, metrics_raw = model(train, test)


# In[89]:


metrics_raw


# In[90]:


# feature importance 확인
fi_raw_sorted = plot_feature_importances(fi_raw)


# In[91]:


# top 100 개만 추려서 변수에 할당
top_100 = list(fi_raw_sorted['feature'])[:100]
# 추려낸 100개의 feature 중에서 fi_raw 에만 있는 feature 들을 new_feature로 따로 분류해주는 작엄, 즉 추려내면 100개가 아닐거임.
new_feature = [x for x in top_100 if x not in list(fi['feature'])]

print('%% of Top 100 Features created from the bureau data = %d.00' % len(new_feature))


# In[92]:


submission_raw.to_csv('test_one.csv', index=False)


# In[93]:


submission_corrs, fi_corrs, metrics_corrs = model(train_corrs_removed, test_corrs_removed)


# In[94]:


metrics_corrs


# In[95]:


fi_corrs_sorted = plot_feature_importances(fi_corrs)


# In[96]:


submission_corrs.to_csv('test_two.csv', index=False)


# In[ ]:




