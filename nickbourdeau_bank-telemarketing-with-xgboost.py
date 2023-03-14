#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import pandas as pd
df = pd.read_csv('bank-train.csv.zip')


# In[ ]:


data = df
df = df.drop(['duration', 'id'], 1)
df.head()


# In[ ]:


# Total Yes and Nos
df.y.value_counts()


# In[ ]:


# Transforming categorical data to labels
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

lbl_enc = preprocessing.LabelEncoder()
df['job'] = lbl_enc.fit_transform(df.job.values)
df['marital'] = lbl_enc.fit_transform(df.marital.values)
df['education'] = lbl_enc.fit_transform(df.education.values)
df['default'] = lbl_enc.fit_transform(df.default.values)
df['housing'] = lbl_enc.fit_transform(df.housing.values)
df['loan'] = lbl_enc.fit_transform(df.loan.values)
df['contact'] = lbl_enc.fit_transform(df.contact.values)
df['month'] = lbl_enc.fit_transform(df.month.values)
df['day_of_week'] = lbl_enc.fit_transform(df.day_of_week.values)
df['poutcome'] = lbl_enc.fit_transform(df.poutcome.values)


# In[ ]:


# Scaling numerical data to enhance machine learning algorithms
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
df[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']] = min_max_scaler.fit_transform(df[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']])
df


# In[ ]:


df.isnull().any(axis=1).sum()


# In[ ]:


# Subsetting training and testing data
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, LeaveOneOut
x = df.drop(['y'], 1)
y = df['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# In[ ]:


# Random Forest Classifier Model
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion = 'entropy')
forest.fit(x_train, y_train)
forest_predictions = forest.predict(x_test)
forest.score(x_test, y_test)  


# In[ ]:


# Random Forest Feature Importance
importance = pd.DataFrame({'Importance': forest.feature_importances_}, index = X_train.columns).sort_values('Importance', ascending = False)
importance


# In[ ]:


# XGBoost Model
import xgboost as xgb
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(x_train, y_train)


# In[ ]:


clf.score(x_test, y_test)


# In[ ]:


# Recursive Feature Elimination on XGBooost model
from sklearn.feature_selection import RFECV

rfe1 = RFECV(clf, 4, scoring = 'f1_weighted')
rfe1.fit(x_train, y_train)
rfe1.score(x_test, y_test)


# In[ ]:


# XGBoost Feature Importance
important = pd.DataFrame({'Importance': clf.feature_importances_}, index = x_train.columns).sort_values('Importance', ascending = False)
important


# In[ ]:


# Feature Selection with XGBoost
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs = SFS(clf,
           k_features=11,
           forward=True,
           floating=False,
           scoring = 'f1',
           cv = 0)
sfs.fit(x_train, y_train)


# In[ ]:


sfs.k_feature_names_
sffs.score(x_test, y_test)


# In[ ]:


# Grid Search for XBG Classifier
# Create XGB Classifier object
xgb_clf = xgb.XGBClassifier(tree_method = "exact", predictor = "gpu_predictor",
                            objective = "binary:logistic")

# Create parameter grid
parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 250, 500, 1000]}

from sklearn.model_selection import RandomizedSearchCV
# Create RandomizedSearchCV Object
xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions = parameters, scoring = "f1_micro",
                             cv = 10, verbose = 3, random_state = 40 )

# Fit the model
model_xgboost = xgb_rscv.fit(x_train, y_train)


# In[ ]:


# Best XGBoost Model
grid_clf = model_xgboost.best_estimator_
model_xgboost.best_estimator_


# In[ ]:


import xgboost as xgb
idea = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0.01,
              learning_rate=0.001, max_delta_step=0, max_depth=10,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic',
              predictor='cpu_predictor', random_state=0, reg_alpha=1,
              reg_lambda=3, scale_pos_weight=1, seed=None, silent=None,
              subsample=0.7, tree_method='exact', verbosity=1)
idea.fit(x_train, y_train)
idea.score(x_test, y_test)


# In[ ]:


# Best XGBoost Score
grid_clf.fit(x_train, y_train)
grid_clf.score(x_test, y_test)


# In[ ]:


from xgboost import plot_importance
grid_clf.feature_importances_


# In[ ]:


# Visualizing Feature Importance for XGBoost
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.metrics import accuracy_score

thresholds = sort(grid_clf.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model_xgboost, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train)
	# train model
	selection_model = xgb.XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(x_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
  print(select_X_train.)


# In[ ]:


#Recusive Feature Elimination on Best XGBoost Model
from sklearn.feature_selection import RFECV

rfe = RFECV(grid_clf, 4, scoring = 'f1')
rfe.fit(x_train, y_train)
rfe.score(x_test, y_test)


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)


# In[ ]:


# Create a Keras Learning Model

from keras.models import Sequential
from keras.layers import Dense

deep_model = Sequential()
deep_model.add(Dense(12, input_dim=19, activation='relu'))
deep_model.add(Dense(8, activation='relu'))
deep_model.add(Dense(1, activation='sigmoid'))

deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


deep_model.fit(train_x, train_y, epochs=150, batch_size=10)


# In[ ]:


deep_model.evaluate(train_x, train_y, verbose=0)


# In[ ]:


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)


# In[ ]:


# Create New Keras Model with more Layers

def create_baseline():
	# create model
	dmodel = Sequential()
	dmodel.add(Dense(19, input_dim=19, activation='relu'))
	dmodel.add(Dense(9, activation='relu'))
	dmodel.add(Dense(1, activation='sigmoid'))
	# Compile model
	dmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return dmodel


# In[ ]:


estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


test = pd.read_csv('bank-test.csv')
test


# In[ ]:


lbl_enc = preprocessing.LabelEncoder()
test['job'] = lbl_enc.fit_transform(test.job.values)
test['marital'] = lbl_enc.fit_transform(test.marital.values)
test['education'] = lbl_enc.fit_transform(test.education.values)
test['default'] = lbl_enc.fit_transform(test.default.values)
test['housing'] = lbl_enc.fit_transform(test.housing.values)
test['loan'] = lbl_enc.fit_transform(test.loan.values)
test['contact'] = lbl_enc.fit_transform(test.contact.values)
test['month'] = lbl_enc.fit_transform(test.month.values)
test['day_of_week'] = lbl_enc.fit_transform(test.day_of_week.values)
test['poutcome'] = lbl_enc.fit_transform(test.poutcome.values)


# In[ ]:


min_max_scaler = MinMaxScaler()
test[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']] = min_max_scaler.fit_transform(test[['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']])
test


# In[ ]:


bank_test = test[['euribor3m', 'pdays', 'nr.employed', 'month', 'emp.var.rate', 'contact', 'cons.price.idx', 'poutcome', 'cons.conf.idx', 'default', 'day_of_week', 'campaign']]
bank_test


# In[ ]:


test1 = pd.read_csv('bank-test.csv')
ID = test1['id']
dataframe1 = pd.DataFrame(ID, columns=['id']) 
dataframe1


# In[ ]:


btest = test.drop(['id', 'duration'], 1)
predictions1 = rfe1.predict(btest)


# In[ ]:


dataframe=pd.DataFrame(predictions1, columns=['Predicted'])
DF3 = pd.concat([dataframe1, dataframe], axis = 1)
DF3.to_csv('nb_sumbission4.csv', index=False)

