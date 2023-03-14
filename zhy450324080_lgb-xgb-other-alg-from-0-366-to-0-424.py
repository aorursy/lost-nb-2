#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


pd.set_option('display.width', 1000) 

pd.set_option('display.max_rows', 200) 

pd.set_option('display.max_columns', 200) 


# In[ ]:


train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')


# In[ ]:



sum_id = test['Id']
del test['Id']

Y = train.Target.values.astype(int)

del train['Target']

all_data = pd.concat((train.loc[:,'v2a1':'agesq'],
                      test.loc[:,'v2a1':'agesq']))
del all_data['idhogar']


# In[ ]:


#------------ fill NaNs --------------

all_data.isnull().any()

all_data["v2a1"].fillna(all_data["v2a1"].median(), inplace=True)
all_data["v18q1"].fillna(0, inplace=True)
all_data["rez_esc"].fillna(0, inplace=True)
all_data["meaneduc"].fillna(all_data["meaneduc"].median(), inplace=True)
all_data["SQBmeaned"].fillna(all_data["SQBmeaned"].median(), inplace=True)


# In[ ]:


#------------- digitalizing -----------

all_data.loc[all_data["dependency"]=="yes","dependency"]=0.25      
all_data.loc[all_data["dependency"]=="no","dependency"]=8
all_data.loc[all_data["edjefe"]=="yes","edjefe"]=1
all_data.loc[all_data["edjefe"]=="no","edjefe"]=0      
all_data.loc[all_data["edjefa"]=="yes","edjefa"]=1
all_data.loc[all_data["edjefa"]=="no","edjefa"]=0  

all_data['dependency'] = all_data['dependency'].astype('float')
all_data['edjefe'] = all_data['edjefe'].astype('float')
all_data['edjefa'] = all_data['edjefa'].astype('float')


train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]


# In[ ]:


#------------------ Predicting ------------------------------------------------


#---------- 1 RandomForest ----------------- Score: 0.366


from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train, Y)


X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42) 
print('trainning accuracy：\n',random_forest.score(X_train, y_train))
print('validation accuracy：\n',random_forest.score(X_val, y_val))

print('RandomForest Accuracy：\n',random_forest.score(train, Y))

pred_RF = random_forest.predict(test)

sol_RF = pd.DataFrame({'Id':sum_id.values, 'Target':pred_RF}) 

sol_RF.to_csv('pred_RF.csv',index=None) 


# In[ ]:


#---------- 2 DecisionTree ----------------- Score: 0.352

from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()

DT.fit(train,Y)
X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42)

print('Accuracy on training：\n',DT.score(X_train, y_train))
print('Accuracy on validation：\n',DT.score(X_val, y_val))
print('DecisionTree Accuracy：\n',DT.score(train, Y))

pred_DT = (DT.predict(test))

sol_DT = pd.DataFrame({'Id':sum_id.values, 'Target':pred_DT}) 

sol_DT.to_csv('pred_DT.csv',index=None) 


# In[ ]:


#---------- 3 LogisticRegression ----------------- Score: 0.253

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(train, Y)
X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42) 
print('Accuracy on training：\n',LR.score(X_train, y_train)) 
print('Accuracy on validation：\n',LR.score(X_val, y_val))
print('LogisticRegression Accuracy：\n',LR.score(train, Y))

pred = LR.predict(test)

pred = pd.DataFrame({'Id':sum_id.values, 'Target':pred}) 

pred.to_csv('pred_LR.csv',index=None) 


# In[ ]:


#---------- 4 kNN ----------------- Score: 0.308

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(train, Y)
X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42) 

print('Accuracy on training：\n',knn.score(X_train, y_train)) 
print('Accuracy on validation：\n',knn.score(X_val, y_val))
print('kNN Accuracy：\n',knn.score(train, Y))

pred = knn.predict(test)

pred = pd.DataFrame({'Id':sum_id.values, 'Target':pred}) 

pred.to_csv('pred_kNN.csv',index=None)


# In[ ]:


#---------- 5 NaiveBayes Gaussian ----------------- Score: 0.373
 
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(train, Y)
X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42) 

print('Accuracy on training：\n',gaussian.score(X_train, y_train)) 
print('Accuracy on validation：\n',gaussian.score(X_val, y_val))
print('gaussian Accuracy：\n',gaussian.score(train, Y))

pred_NB = gaussian.predict(test)

sol_NB = pd.DataFrame({'Id':sum_id.values, 'Target':pred_NB})

sol_NB.to_csv('pred_NaiveBayes.csv',index=None) 


# In[ ]:


#---------- 6 LinearRegression ----------------- Grade: 0.346
# doesn't output int
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(train, Y)
X_train, X_val, y_train,y_val = train_test_split(train,Y,test_size=0.3, random_state=42) 

print('Accuracy on training：\n',LR.score(X_train, y_train)) 
print('Accuracy on validation：\n',LR.score(X_val, y_val))
print('LinearRegression Accuracy：\n',LR.score(train, Y))

pred = LR.predict(test)
  
pred = pd.DataFrame({'Id':sum_id.values, 'Target':pred}) 

pred.loc[pred["Target"] < 1.5,"Target"] = 1
pred.loc[(1.5 <= pred["Target"]) & (pred["Target"] < 2.5),"Target"] = 2
pred.loc[(2.5 <= pred["Target"]) & ( pred["Target"] < 3.5),"Target"] = 3
pred.loc[3.5 <= pred["Target"],"Target"] = 4
pred['Target'] = pred['Target'].astype('int')
pred.to_csv('pred_Linear.csv',index=None) 


# In[ ]:


#========================== lasso ridge xgb ============================== # Score: 0.363

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

print(cv_ridge.min())

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train, Y)

print(rmse_cv(model_lasso).mean())

coef = pd.Series(model_lasso.coef_, index = train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(train), "true":Y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")

#-------------- xgboosting -------------

import xgboost as xgb

dtrain = xgb.DMatrix(train, label = Y)
dtest = xgb.DMatrix(test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(train, Y)

xgb_preds = model_xgb.predict(test)
lasso_preds = model_lasso.predict(test)

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.3*lasso_preds + 0.7*xgb_preds

solution = pd.DataFrame({"Id":sum_id.values, "Target":preds})
solution.loc[solution["Target"] < 1.5,"Target"] = 1
solution.loc[(1.5 <= solution["Target"]) & (solution["Target"] < 2.5),"Target"] = 2
solution.loc[(2.5 <= solution["Target"]) & ( solution["Target"] < 3.5),"Target"] = 3
solution.loc[3.5 <= solution["Target"],"Target"] = 4
solution['Target'] = solution['Target'].astype('int')
solution.to_csv("ridge_sol.csv", index = False) 


# In[ ]:


#===================== Lightgbm ========================================= Score: 0.424

import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

clf = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)
kf = StratifiedKFold(n_splits=5, shuffle=True)
# partially based on https://www.kaggle.com/c0conuts/xgb-k-folds-fastai-pca
Y = pd.Series(Y)
predicts = []

for train_index, test_index in kf.split(train, Y):
    print("###")
    X_train, X_val = train.iloc[train_index], train.iloc[test_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            early_stopping_rounds=20)
    predicts.append(clf.predict(test))
    
predict = pd.DataFrame(np.array(sum_id),
                             columns=['Id'],
                             index=test.index)
predict['Target'] = np.array(predicts).mean(axis=0).round().astype(int)
predict.to_csv('predict.csv', index = False)

