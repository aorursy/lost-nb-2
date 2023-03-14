#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

# set seed
np.random.seed(42)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_cell_magic('time', '', "\ntrainData = pd.read_table('../input/train.tsv')\ntestData = pd.read_table('../input/test.tsv')\n\nprint(trainData.shape, testData.shape)")


# In[3]:


trainData = trainData.drop(trainData[(trainData.price < 3.0)].index)
trainData.shape


# In[4]:


get_ipython().run_cell_magic('time', '', '# get name and description lengths\ndef wordCount(text):\n    try:\n        if text == \'No description yet\':\n            return 0\n        else:\n            text = text.lower()\n            words = [w for w in text.split(" ")]\n            return len(words)\n    except: \n        return 0\ntrainData[\'descLen\'] = trainData[\'item_description\'].apply(lambda x: wordCount(x))\ntestData[\'descLen\'] = testData[\'item_description\'].apply(lambda x: wordCount(x))\ntrainData[\'nameLen\'] = trainData[\'name\'].apply(lambda x: wordCount(x))\ntestData[\'nameLen\'] = testData[\'name\'].apply(lambda x: wordCount(x))\ntrainData.head()')


# In[5]:


get_ipython().run_cell_magic('time', '', '# split category name into 3 parts\ndef split_cat(text):\n    try: return text.split("/")\n    except: return ("No Label", "No Label", "No Label")\ntrainData[\'subcat_1\'], trainData[\'subcat_2\'], trainData[\'subcat_3\'] = \\\nzip(*trainData[\'category_name\'].apply(lambda x: split_cat(x)))\ntestData[\'subcat_1\'], testData[\'subcat_2\'], testData[\'subcat_3\'] = \\\nzip(*testData[\'category_name\'].apply(lambda x: split_cat(x)))')


# In[6]:


get_ipython().run_cell_magic('time', '', 'fullData = pd.concat([trainData,testData])\nbrands = set(fullData[\'brand_name\'].values)\ntrainData.brand_name.fillna(value="missing", inplace=True)\ntestData.brand_name.fillna(value="missing", inplace=True)\n\nmissing = len(trainData.loc[trainData[\'brand_name\'] == \'missing\'])\ndef brandfinder(line):\n    brand = line[0]\n    name = line[1]\n    namesplit = name.split(\' \')\n    if brand == \'missing\':\n        for x in namesplit:\n            if x in brands:\n                return name\n    if name in brands:\n        return name\n    return brand\ntrainData[\'brand_name\'] = trainData[[\'brand_name\',\'name\']].apply(brandfinder, axis = 1)\ntestData[\'brand_name\'] = testData[[\'brand_name\',\'name\']].apply(brandfinder, axis = 1)\nfound = missing-len(trainData.loc[trainData[\'brand_name\'] == \'missing\'])\nprint(found)')


# In[7]:


get_ipython().run_cell_magic('time', '', '# Scale target variable to log.\n# trainData["target"] = np.log1p(trainData.price)\n\n# Split training examples into train/dev examples.\ntrainData, devData = train_test_split(trainData, random_state=42, train_size=0.9)\n\n# Calculate number of train/dev/test examples.\nn_trains = trainData.shape[0]\nn_devs = devData.shape[0]\nn_tests = testData.shape[0]\nprint("Training on", n_trains, "examples")\nprint("Validating on", n_devs, "examples")\nprint("Testing on", n_tests, "examples")\n\n# Concatenate train - dev - test data for easy to handle\nfullData = pd.concat([trainData, devData, testData])')


# In[8]:


get_ipython().run_cell_magic('time', '', '\n# Filling missing values\ndef fill_missing_values(df):\n    df.category_name.fillna(value="missing", inplace=True)\n    df.brand_name.fillna(value="missing", inplace=True)\n    df.item_description.fillna(value="missing", inplace=True)\n    df.item_description.replace(\'No description yet\',"missing", inplace=True)\n    return df\n\nprint("Filling missing data ...")\nfullData = fill_missing_values(fullData)\nprint(fullData.category_name[1])')


# In[9]:


get_ipython().run_cell_magic('time', '', '\nprint("Processing categorical data...")\nle = LabelEncoder()\n\nle.fit(fullData.category_name)\nfullData[\'category\'] = le.transform(fullData.category_name)\n\nle.fit(fullData.brand_name)\nfullData.brand_name = le.transform(fullData.brand_name)\n\nle.fit(fullData.subcat_1)\nfullData.subcat_1 = le.transform(fullData.subcat_1)\n\nle.fit(fullData.subcat_2)\nfullData.subcat_2 = le.transform(fullData.subcat_2)\n\nle.fit(fullData.subcat_3)\nfullData.subcat_3 = le.transform(fullData.subcat_3)\n\ndel le')


# In[10]:


get_ipython().run_cell_magic('time', '', '\nprint("Handling missing values...")\nfullData[\'category_name\'] = fullData[\'category_name\'].fillna(\'missing\').astype(str)\nfullData[\'subcat_1\'] = fullData[\'subcat_1\'].astype(str)\nfullData[\'subcat_2\'] = fullData[\'subcat_2\'].astype(str)\nfullData[\'subcat_3\'] = fullData[\'subcat_3\'].astype(str)\nfullData[\'brand_name\'] = fullData[\'brand_name\'].fillna(\'missing\').astype(str)\nfullData[\'shipping\'] = fullData[\'shipping\'].astype(str)\nfullData[\'item_condition_id\'] = fullData[\'item_condition_id\'].astype(str)\nfullData[\'descLen\'] = fullData[\'descLen\'].astype(str)\nfullData[\'nameLen\'] = fullData[\'nameLen\'].astype(str)\nfullData[\'item_description\'] = fullData[\'item_description\'].fillna(\'No description yet\').astype(str)')


# In[11]:


get_ipython().run_cell_magic('time', '', '\nprint("Vectorizing data...")\ndefault_preprocessor = CountVectorizer().build_preprocessor()\ndef build_preprocessor(field):\n    field_idx = list(fullData.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\nvectorizer = FeatureUnion([\n    (\'name\', CountVectorizer(\n        ngram_range=(1, 2),\n        max_features=100000,\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'name\'))),\n    (\'category_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        max_features=20000,\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'category_name\'))),\n    (\'subcat_1\', CountVectorizer(\n        token_pattern=\'.+\',\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'subcat_1\'))),\n    (\'subcat_2\', CountVectorizer(\n        token_pattern=\'.+\',\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'subcat_2\'))),\n    (\'subcat_3\', CountVectorizer(\n        token_pattern=\'.+\',\n        stop_words=\'english\',\n        max_features=20000,\n        preprocessor=build_preprocessor(\'subcat_3\'))),\n    (\'brand_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'brand_name\'))),\n    (\'shipping\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'shipping\'))),\n    (\'item_condition_id\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'item_condition_id\'))),\n    (\'item_description\', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=20000,\n        stop_words=\'english\',\n        preprocessor=build_preprocessor(\'item_description\'))),\n])\n\nX = vectorizer.fit_transform(fullData.values)')


# In[12]:


X = sparse.hstack((X, fullData[['nameLen', 'descLen']].astype(float).as_matrix()), format = 'csr')

trainData["target"] = np.log1p(trainData.price)
devData["target"] = np.log1p(devData.price)

X_train = X[:n_trains]
Y_train = trainData.target.values.reshape(-1, 1)

X_dev = X[n_trains:n_trains+n_devs]
Y_dev = devData.target.values.reshape(-1, 1)

X_test = X[n_trains+n_devs:]

print(X.shape, X_train.shape, X_dev.shape, X_test.shape)


# In[13]:


# del trainData
# del testData
# del fullData


# In[14]:


get_ipython().run_cell_magic('time', '', "%env JOBLIB_TEMP_FOLDER=/tmp\n\nxgb_model = xgb.XGBRegressor()\n\nxgb_parameters = {'n_estimators': [100],\n              'subsample': [0.5],\n              'colsample_bytree': [0.1],\n              'colsample_bylevel': [0.1],\n              'reg_lambda': [0.7],\n              'reg_alpha': [0.3],\n              'seed': [42]}\n\n\nxgb_clf = GridSearchCV(xgb_model, xgb_parameters, n_jobs=-1, cv=3, \n                   scoring='neg_mean_squared_error')\n\nxgb_clf.fit(X_train, Y_train)\n\nprint('XGBoost training score: ', mean_squared_error(Y_train, xgb_clf.predict(X_train)))\nprint('XGBoost validation score: ', mean_squared_error(Y_dev, xgb_clf.predict(X_dev)))")


# In[15]:


# xgb_pred_test = np.expm1(xgb_clf.predict(X_test))

# submissionData = pd.DataFrame({
#         "test_id": testData.test_id,
#         "price": xgb_pred_test.reshape(-1),
# })

# submissionData.to_csv("./xgb_submission_first.csv", index=False)


# In[16]:


get_ipython().run_cell_magic('time', '', "ridge_model = Ridge(\n    fit_intercept=True, alpha=[10.0],\n    normalize=False, solver='sag', tol=0.05, random_state=42)\n\nridge_model.fit(X_train, Y_train)\n\nprint('Ridge training score: ', mean_squared_error(Y_train, ridge_model.predict(X_train)))\nprint('Ridge validation score: ', mean_squared_error(Y_dev, ridge_model.predict(X_dev)))")


# In[17]:


# ridge_pred_test = np.expm1(ridge_model.predict(X_test))

# submissionData = pd.DataFrame({
#         "test_id": testData.test_id,
#         "price": ridge_pred_test.reshape(-1),
# })

# submissionData.to_csv("./ridge_submission_first.csv", index=False)


# In[18]:


get_ipython().run_cell_magic('time', '', '\nxgb_pred_dev = np.expm1(xgb_clf.predict(X_dev))\nridge_pred_dev = np.expm1(ridge_model.predict(X_dev))\n\nxgb_pred_test = np.expm1(xgb_clf.predict(X_test))\nridge_pred_test = np.expm1(ridge_model.predict(X_test))\n\ndef aggregate_predicts2(Y1, Y2,ratio):\n    assert Y1.shape == Y2.shape\n    return Y1 * ratio + Y2 * (1.0 - ratio)\n\n#ratio optimum finder\nbest = 0\nlowest = 0.99\nfor i in range(100):\n    r = i*0.01\n    Y_dev_preds = aggregate_predicts2(xgb_pred_dev, ridge_pred_dev, r)\n    fpred = mean_squared_error(Y_dev, Y_dev_preds)\n    if fpred < lowest:\n        best = r\n        lowest = fpred\n    print(str(r) + " - score for XGBoost + Ridge on dev set:", fpred)')


# In[19]:


weighted_preds = aggregate_predicts2(xgb_pred_test, ridge_pred_test, best)

submissionData = pd.DataFrame({
        "test_id": testData.test_id,
        "price": weighted_preds.reshape(-1),
})

submissionData.to_csv("./ridge_xgb_weighted_submission.csv", index=False)

