#!/usr/bin/env python
# coding: utf-8

# In[1]:


# porto_seguro_insur.py
#  Assumes python vers. 3.6
# __author__ = 'mizio'

import csv as csv
import numpy as np
import pandas as pd
import pylab as plt
from fancyimpute import MICE
import random
from sklearn.model_selection import cross_val_score
import datetime
import seaborn as sns
import tensorflow as tf


# In[2]:


class PortoSeguroInsur:
    def __init__(self):
        self.df = PortoSeguroInsur.df
        self.df_test = PortoSeguroInsur.df_test
        self.df_submission = PortoSeguroInsur.df_submission
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')


    # Load data into Pandas DataFrame
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('../input/train.csv', header=0)
    df_test = pd.read_csv('../input/test.csv', header=0)
    df_submission = pd.read_csv('../input/sample_submission.csv', header=0)

    @staticmethod
    def features_with_null_logical(df, axis=1):
        row_length = len(df._get_axis(0))
        # Axis to count non null values in. aggregate_axis=0 implies counting for every feature
        aggregate_axis = 1 - axis
        features_non_null_series = df.count(axis=aggregate_axis)
        # Whenever count() differs from row_length it implies a null value exists in feature column and a False in mask
        mask = row_length == features_non_null_series
        return mask

    def missing_values_in_dataframe(self, df):
        mask = self.features_with_null_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print('\n')

    @staticmethod
    def extract_numerical_features(df):
        df = df.copy()
        df = df.copy()
        non_numerical_feature_names = df.columns[np.where(PortoSeguroInsur.numerical_feature_logical_incl_hidden_num(
            df) == 0)]
        return non_numerical_feature_names

    @staticmethod
    def extract_non_numerical_features(df):
        df = df.copy()
        non_numerical_feature_names = df.columns[np.where(PortoSeguroInsur.numerical_feature_logical_incl_hidden_num(
            df))]
        return non_numerical_feature_names

    @staticmethod
    def numerical_feature_logical_incl_hidden_num(df):
        logical_of_non_numeric_features = np.zeros(df.columns.shape[0], dtype=int)
        for ite in np.arange(0, df.columns.shape[0]):
            try:
                str(df[df.columns[ite]][0]) + df[df.columns[ite]][0]
                logical_of_non_numeric_features[ite] = True
            except TypeError:
                hej = 'Oops'
        return logical_of_non_numeric_features

    def clean_data(self, df, is_train_data=1):
        df = df.copy()
        if df.isnull().sum().sum() > 0:
            if is_train_data:
                df = df.dropna()
            else:
                df = df.dropna(1)
        return df

    def reformat_data(self, labels, num_labels):
        # Map labels/target value to one-hot-encoded frame. None is same as implying newaxis() just replicating array
        # if num_labels > 2:
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float64)
        return labels

    def accuracy(self, predictions, labels):
        # Sum the number of cases where the predictions are correct and divide by the number of predictions
        number_of_correct_predictions = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        return 100*number_of_correct_predictions/predictions.shape[0]

    def linear_model(self, input_vector, weight_matrix, bias_vector):
        # f(x) = Wx + b
        # W is the weight matrix with elements w_ij
        # x is the input vector
        # b is the bias vector
        # In the machine learning literature f(x) is called an activation
        return tf.matmul(input_vector, weight_matrix) + bias_vector

    def activation_out(self, logit):
        return self.activation(logit, switch_var=0)

    def activation_hidden(self, logit):
        return self.activation(logit, switch_var=0)

    def activation(self, logit, switch_var=0):
        # Also called the activation function
        if switch_var == 0:
            # Logistic sigmoid function.
            # sigma(a) = 1/(1+exp(-a))
            return tf.nn.sigmoid(logit)
        elif switch_var == 1:
            # Using Rectifier as activation function. Rectified linear unit (ReLU). Compared to sigmoid or other
            # activation functions it allows for faster and effective training of neural architectures.
            # f(x) = max(x,0)
            return tf.nn.relu(logit)
        else:
            # Softmax function.
            # S(y_i) = e^y_i/(Sum_j e^y_j)
            return tf.nn.softmax(logit)

    def missing_values_in_dataframe(self, df):
        mask = self.features_with_null_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print('\n')
        
    @staticmethod
    def extract_numerical_features(df):
        df = df.copy()
        # Identify numerical columns which are of type object
        numerical_features = pd.Series(data=False, index=df.columns, dtype=bool)

        for feature in df.columns:
            if any(tuple(df[feature].apply(lambda x: type(x)) == int)) or                             any(tuple(df[feature].apply(lambda x: type(x)) == float)) &                             (not any(tuple(df[feature].apply(lambda x: type(x)) == str))):
                numerical_features[feature] = 1
        return numerical_features[numerical_features == 1].index


# In[3]:


porto_seguro_insur = PortoSeguroInsur()
df = porto_seguro_insur.df.copy()
df_test = porto_seguro_insur.df_test.copy()
df_submission = porto_seguro_insur.df_submission.copy()

df = df.replace(-1, np.NaN)
df_test = df_test.replace(-1, np.NaN)

print('All df set missing values')
porto_seguro_insur.missing_values_in_dataframe(df)

# Train Data: numeric feature columns with none or nan in test data
print('\nColumns in train data with none/nan values:')
print('\nTraining set numerical features\' missing values')
df_numerical_features = porto_seguro_insur.extract_numerical_features(df)
print('\nNumber of numerical features df: %s' % df_numerical_features.shape[0])
porto_seguro_insur.missing_values_in_dataframe(df[df_numerical_features])

# Test Data: Print numeric feature columns with none/nan in test data
print('\nColumns in test data with none/nan values:')
print('\nTest set numerical features\' missing values')
df_test_numerical_features = porto_seguro_insur.extract_numerical_features(df_test)
print('\nNumber of numerical features df_test: %s' % df_test_numerical_features.shape[0])
porto_seguro_insur.missing_values_in_dataframe(df_test[df_test_numerical_features])

print(df.shape)
print(df_test.shape)
# Clean data for NaN
df = porto_seguro_insur.clean_data(df)
df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)
print('df_test.shape: %s' % str(df_test.shape))  # (892816, 46)
# df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)
id_df_test = df_test['id']  # Submission column
print("After dropping NaN")
print(df.shape)
print(df_test.shape)


# In[4]:


is_explore_data = 1
if is_explore_data:
    # Overview of train data
    print('\n TRAINING DATA:----------------------------------------------- \n')
    print(df.head(3))
    print('\n')
    print(df.info())
    print('\n')
    print(df.describe())
    print('\n')
    print(df.dtypes)
    print(df.get_dtype_counts())

    # missing_values
    print('All df set missing values')
    porto_seguro_insur.missing_values_in_dataframe(df)

    print('Uniques')
    uniques_in_id = np.unique(df.id.values).shape[0]
    print(uniques_in_id)
    print('uniques_in_id == df.shape[0]')
    print(uniques_in_id == df.shape[0])

    # Overview of sample_submission format
    print('\n sample_submission \n')
    print(df_submission.head(3))
    print('\n')
    print(df_submission.info())
    print('\n')


# In[5]:


# Categorical plot with seaborn
is_categorical_plot = 1
if is_categorical_plot:
    # sns.countplot(y='MSZoning', hue='MSSubClass', data=df, palette='Greens_d')
    # plt.show()
    # sns.stripplot(x='SalePrice', y='MSZoning', data=df, jitter=True, hue='LandContour')
    # plt.show()
    # sns.boxplot(x='SalePrice', y='MSZoning', data=df, hue='MSSubClass')
    # plt.show()

    # Heatmap of feature correlations
    print('\nCorrelations in training data')
    plt.figure(figsize=(10, 8))
    correlations_train = porto_seguro_insur.df.corr()
    sns.heatmap(correlations_train, vmax=0.8, square=True)
    plt.show()
    
    # Heatmap of feature correlations
    print('\nCorrelations in test data')
    plt.figure(figsize=(10, 8))
    correlations_test = porto_seguro_insur.df_test.corr()
    sns.heatmap(correlations_test, vmax=0.8, square=True)
    plt.show()


# In[6]:


# Zoom of heatmap with coefficients
plt.figure(figsize=(20, 12))
top_features = 10
columns = correlations_train.nlargest(top_features, 'target')['target'].index
correlation_coeff = np.corrcoef(porto_seguro_insur.df[columns].values.T)
sns.set(font_scale=1.20)
coeff_heatmap = sns.heatmap(correlation_coeff, annot=True, cmap='YlGn', cbar=True, 
                            square=True, fmt='.2f', annot_kws={'size': 10}, 
                            yticklabels=columns.values, xticklabels=columns.values)
plt.show()


# In[7]:


# Check output space for each feature. Expect 58 uniques i.e. one for every feature.
ser_with_uniques = pd.Series()
for ite in df.columns:
    ser_with_uniques[ite] = df[ite].unique().shape[0]
print(ser_with_uniques)


# In[8]:


# Check if two-value features are binaries
indices_of_two_value_feats = ser_with_uniques == 2
print(indices_of_two_value_feats)


# In[9]:


feats_with_two_value = ser_with_uniques[indices_of_two_value_feats]
print(feats_with_two_value.axes[0])
print(type(feats_with_two_value.axes))


# In[10]:


ser_with_max_of_uniques = pd.Series()
for ite in feats_with_two_value.axes[0]:
    ser_with_max_of_uniques[ite] = df[ite].unique()
print(ser_with_max_of_uniques)


# In[11]:




