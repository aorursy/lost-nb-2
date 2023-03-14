#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

import matplotlib
matplotlib.rc("font", family = "AppleGothic")
matplotlib.rc("axes", unicode_minus = False)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")


# In[3]:


train = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv",  index_col = "id")

print(train.shape)
train.head()


# In[4]:


test = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv",  index_col = "id")

print(test.shape)
test.head()


# In[5]:


get_ipython().system('conda install -c conda-forge -y lightgbm')


# In[6]:


# train 확인결과 이는 feature_name을 선별적으로 선택하는 것이 아니라 전체를 종합적으로 사용한 뒤 modeling해야하는 문제이다
# 따라 데이터분석본다는 정확한 예측모형을 만드는게 더 중요하다고 볼수 있음


# In[7]:


# 절대적인 수치는 class_2가 제일 많고 그다음 class_6이 많다 
# 가장 많은 데이터가 무엇인지 파악하는 차원에서 보는 것이고 이것이 label_name을 결정하는데 큰 영향을 주지는 않는다

sns.countplot(x = "target", data = train)


# In[8]:


# 가장 수가 많은 class_2를 일부 뽑아서 어떤 feat이 가장 많은지 확인해본다
# 결과 밑처럼 feat_14 / feat_40 / feat_15 순으로 숫자가 많은 것을 알 수 있다

# 하지만 여기서는 이런식으로 독립적인 분석을 진행하면 안된다
# feat_14가 합산을 했을 때 가장 큰 숫자이긴 하지만, 이외에 feat_9의 경우 일정 큰 수가 있는 경우 이것도 class_2의 결과로 이어지는 모습을 보여준다
# 이는 일정수의 feat별로 연관성을 가져 독립성이 아닌, 연속성을 띈 데이터라고 봐야한다 

# 아마 0,1,2,3,4,5와 같은 single product를 어느쪽에서는 feat1로 , 어느쪽은 feat2로 보았다는 이야기이고 
# 따라서 이러한 상관관계를 연결할 수 있는 모델을 찾아야함을 보여준다 

train_feat_2 = train[train["target"] == "Class_2"]
train_feat_2 = train_feat_2.drop("target", axis = 1)
train_feat_2 = train_feat_2.T
train_feat_2["Total"] = train_feat_2.sum(axis = 1)
train_feat_2

train_feat_2.groupby(train_feat_2.index)["Total"].sum().reset_index().sort_values(by = "Total", ascending = False).head()


# In[9]:


# 각 feat별로 상관관계를 파악하여 어느정도 연관성이 있는지 추측해본다. 
# 여기서 알 수 있는 사실은 feature_name을 진행할 때 전 feat를 사용해야 한다는 것이다 
# 각 feat가 어느정도 연관성을 가지고 서로에게 영향을 주며 움직이고 있음을 알 수 있고 확고하게 영향을 미치는 것은 적다는 점을 알 수 있기에 
# 이 문제가 log_loss를 통해 문제를 풀어야하는 이유를 설명해줄수 있다고 본다

# 주목할 점은 feat_4와 feat_19
# 보통 correspond는 -1~1사이의 값을 나타내는데 이들은 각기 다른 feat와의 상관관계가 거의 다 이를 넘어선다
# 이 이야기는 이 두가지 feat는 다른 feat에 영향을 주거나 받지 않고 독자적으로 움직일 가능성이 높다는 이야기가 된다
# 아마 이 둘은 독자적으로 class를 찾아갈 가능성이 있다
# feature_name 구할때는 필요한 정보가 될 수 있음으로(독립성이 있어서 확실한 class를 보장할 가능성이 크다) 그대로 담는다

train_correlation = train.corr()
train_correlation


# In[10]:


train_correlation[["feat_4"]].sort_values(by = "feat_4", ascending = False).head()#.value_counts()


# In[11]:


# label_name을 제외한 전부의 컬럼을 feature_name화 한다

label_name = "target"
feature_names = train.columns.difference([label_name])

x_train = train[feature_names]
y_train = train[label_name]
x_test = test[feature_names]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[12]:


# 적용 전 hyperparameter에서 어떠한 상관관계가 있는지 분석해본다 
# tree를 어느정도 쳐야하는지, 숫자는 대략적으로 얼마나 되는지 확인하여 후에 적용할 수 있도록 한다 


# In[13]:


# holdout validation을 활용하여 데이터를 분산해서 확인한다 

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

x_train_holdout, x_test_holdout, y_train_holdout, y_test_holdout = train_test_split(x_train, y_train, test_size = 0.3, random_state = 42)

print(x_train_holdout.shape)
print(x_test_holdout.shape)
print(y_train_holdout.shape)
print(y_test_holdout.shape)


# In[14]:


# DecisionTreeClassifier로 대략적으로 조사를 했을때 70%정도의 정확성을 보여준다 

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 42)
model.fit(x_train_holdout, y_train_holdout)

y_train_predict = model.predict(x_train_holdout)
y_test_predict = model.predict(x_test_holdout)

train_accuracy = (y_train_predict == y_train_holdout).mean()
test_accuracy = (y_test_predict == y_test_holdout).mean()

train_accuracy, test_accuracy


# In[15]:


# max_depth을 조사했을때 대부분의 model과 마찬가지로 max_depth가 일정수가 증가할때 score가 좋아진다 
# 이를 통해 후에 적용시 max_depth이 어느정도의 크기가 되어있어야 한다는 것을 알 수 있다

max_depth_list = range(30,51)
hyperparameter = []

for max_depth in max_depth_list:
   
    model = DecisionTreeClassifier(random_state = 42, max_depth = max_depth)
    model.fit(x_train_holdout, y_train_holdout)

    y_train_predict = model.predict(x_train_holdout)
    y_test_predict = model.predict(x_test_holdout)

    train_accuracy = (y_train_predict == y_train_holdout).mean()
    test_accuracy = (y_test_predict == y_test_holdout).mean()
    
    print(f"max_depth = {max_depth}, train = {train_accuracy :.6f}, test = {test_accuracy:.6f}") 
    # hyperparameter.append({"max_depth" : max_depth, "train" : train_accuracy, "test" :test_accuracy})


# In[16]:


# model을 적합성을 평가하기 위해 log_loss를 적용해야 한다.(문제의 조건임)
# log_loss의 최적조건을 맞출 수 있는 model을 선정해야함으로 이를 구하기 위해 
# LGBMClassifier / RandomForestClassifer 두가지를 hold-out으로 비교하여 점수가 더 좋은 모델을 선정하는 작업을 진행한다 


# In[17]:


from lightgbm import LGBMClassifier
model_first = LGBMClassifier(boosting_type='gbdt')
model_first


# In[18]:


from sklearn.ensemble import RandomForestClassifier
model_second = RandomForestClassifier()
model_second


# In[19]:


# LGBMClassifier로 log_loss 측정시 prediction은 0.51로 측정되고
# RandomForestClassifier로 log_loss 측정시 prediction은 1.49~ 1.51로 측정이 된다
# log_loss는 0 ~ 1 사이의 값을 주로 나타내고 이 범위를 벗어나면 무한대로 측정, 즉 오차가 심한결과를 가져오는데 
# RandomForestClassifier의 값이 많이 어긋난다는 것을 보여준다.
# 즉 이 문제에서는 Gradient Boosting을 통해 bias를 조절하는 작업을 진행해야 함을 보여준다 


# LGBMClassifier로 log_loss 측정
model_first.fit(x_train_holdout, y_train_holdout)
y_test_predict = model_first.predict_proba(x_test_holdout)

prediction_LGBM = log_loss(y_test_holdout, y_test_predict)


# RandomForestClassifier로 log_loss 측정
model_second.fit(x_train_holdout, y_train_holdout)
y_test_predict = model_second.predict_proba(x_test_holdout)

prediction_Random = log_loss(y_test_holdout, y_test_predict)



# 최종결과
print("LGBMClassifier is :", prediction_LGBM)
print("RandomForest is :", prediction_Random)


# In[20]:


# Gradient Boosting안에 있는 hyperparameters들을 조절하는 작업을 진행한다 
# boosting_type을 dart, gbdt 둘중 한개를 선택해야 하는데 
# 보통은 dart가 더 좋다고 하나 여기서는 gbdt의 점수가 더 높다(default가정)
# 따라서 gbdt로 진행한다

from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type = "gbdt")
model


# In[21]:


# hyperparameters 중 중요한 몇개들을 별도 세팅을 진행해준다 
# n_estimators / learning_rate가 제일 중요한 것이니 별도 세팅 진행해주고
# 구역을 나누어주는 max_bin, 가지수 결정하는 num_leaves
# random하게 서치하는데 필요한 colsample_bytree, subsample, subsample_freq, min_child_samples 정리한다

# 대략 몇개의 숫자를 넣고 돌려본다 맞는지 확인체크

random_search = 10

for number in range(random_search):
    
    n_estimators = np.random.randint(1, 10)
    learning_rate = 10 ** -np.random.uniform(0, 1)
    max_bin = np.random.randint(2, 500)
    num_leaves = np.random.randint(10, 300)
    min_child_samples = np.random.randint(2, 300)
    colsample_bytree = np.random.uniform(0.1, 1)
    subsample = np.random.uniform(0.4, 1)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate,
                           max_bin = max_bin,
                           num_leaves = num_leaves,
                           min_child_samples = min_child_samples,
                           colsample_bytree = colsample_bytree,
                           subsample = subsample,
                           subsample_freq = 1,
                           n_jobs = -1,
                           random_state = 42)
    
    model.fit(x_train_holdout, y_train_holdout)
    y_test_predict = model.predict_proba(x_test_holdout) 
    score = log_loss(y_test_holdout, y_test_predict)  
    
    print(number, score)


# In[22]:


# [random_search]
# basic = 100 / final = 10 

# [n_estimators]
# 이 둘의 상관관계를 파악하기 위해 n_estimators를 1000,2000 해본다

# [learning_rate]
# learning_rate = 10 ** -np.random.uniform(1, 10)
# final learning_rate = 10 ** -np.random.uniform(0.9, 3) 

from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

random_search = 10
hyperparameter_list= []
early_stopping_rounds = 10

for loop in range(random_search):
    
    # n_estimators = np.random.randint(1000, 3000)
    n_estimators = np.random.randint(10, 30)
    # learning_rate = 10 ** -np.random.uniform(1, 10)
    learning_rate = 10 ** -np.random.uniform(0.9, 3)
    max_bin = np.random.randint(2, 500)
    num_leaves = np.random.randint(10, 300)
    min_child_samples = np.random.randint(2, 300)
    colsample_bytree = np.random.uniform(0.1, 1)
    subsample = np.random.uniform(0.4, 1)
    reg_alpha = 10 ** -np.random.uniform(1, 10)
    reg_lambda = 10 ** -np.random.uniform(1, 15)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate,
                           max_bin = max_bin,
                           num_leaves = num_leaves,
                           min_child_samples = min_child_samples,
                           colsample_bytree = colsample_bytree,
                           subsample = subsample,
                           subsample_freq = 1,
                           n_jobs = -1,
                           random_state = 42)
    
    
    model.fit(x_train_holdout, y_train_holdout, eval_set = [(x_test_holdout, y_test_holdout)],
              early_stopping_rounds = early_stopping_rounds, verbose = 0)
    y_test_predict = model.predict_proba(x_test_holdout)
    
          # model.best_score_["valid_0"]['multi_logloss']
    score = log_loss(y_test_holdout, y_test_predict)
    
    hyperparameter = {"score" : score, "learning_rate" : learning_rate,
                     "max_bin" : max_bin,
                     "num_leaves" : num_leaves,
                     "min_child_samples" : min_child_samples,
                     "colsample_bytree" : colsample_bytree,
                     "subsample" : subsample,
                     "min_child_samples" : min_child_samples,
                     "reg_alpha" : reg_alpha,
                     "reg_lambda" : reg_lambda}
    
    hyperparameter_list.append(hyperparameter)
    
    print(f"score = {score:.6f}, n_estimators = {n_estimators},learning = {learning_rate:.6f},    max_bin = {max_bin}, num_leaves = {num_leaves}, subsample = {subsample:.6f},    colsample_bytree = {colsample_bytree}, min_child_samples = {min_child_samples}")


# In[23]:


final_list = pd.DataFrame.from_dict(hyperparameter_list)
final_list = final_list.sort_values(by = "score", ascending = True)
final_list.head(10)


# In[24]:


model.best_score_["valid_0"]['multi_logloss']


# In[25]:


# 결과(result) 

# 1. <n_estimators = 1,000인 경우 다음과 같이  score가 낮은 모습을 보인다>
# score = 0.448931, n_estimators = 1000,learning = 0.021010,    max_bin = 38, num_leaves = 235, subsample = 0.560212
# score = 0.453496, n_estimators = 1000,learning = 0.022231,    max_bin = 118, num_leaves = 291, subsample = 0.963156
# score = 0.455559, n_estimators = 1000,learning = 0.032189,    max_bin = 382, num_leaves = 70, subsample = 0.561532
# score = 0.457469, n_estimators = 1000,learning = 0.018955,    max_bin = 94, num_leaves = 205, subsample = 0.733559

# 2. <n_estimators = 2,000인 경우 다음과 같이  score가 낮은 모습을 보인다>
# score = 0.452683, n_estimators = 2000,learning = 0.011920,    max_bin = 387, num_leaves = 46, subsample = 0.652579,
# score = 0.454024, n_estimators = 2000,learning = 0.013580,    max_bin = 281, num_leaves = 270, subsample = 0.757202
# score = 0.455261, n_estimators = 2000,learning = 0.011020,    max_bin = 492, num_leaves = 179, subsample = 0.676981,
# score = 0.455836, n_estimators = 2000,learning = 0.011370,    max_bin = 9, num_leaves = 281, subsample = 0.716674,

# n_estimators가 1000인 경우 0.02XXX , n_estimators가 2000인 경우 0.01XXX에서 강한 점수를 보였다 
# 이는 n_estimators별로 특정위치의 learning_rate를 가질때 최대의 효과를 가진다고 봐야한다. 

# 3. <n_estimators = 1,000~3,000인 경우 다음과 같이  score가 낮은 모습을 보인다>
# score = 0.447990, n_estimators = 1917,learning = 0.008747,    max_bin = 257, num_leaves = 138, subsample = 0.683348,
# score = 0.450463, n_estimators = 1850,learning = 0.013366,    max_bin = 355, num_leaves = 269, subsample = 0.638961,
# score = 0.451037, n_estimators = 2196,learning = 0.009747,    max_bin = 17, num_leaves = 265, subsample = 0.900356,
# score = 0.453210, n_estimators = 2800,learning = 0.005284,    max_bin = 61, num_leaves = 242, subsample = 0.491945,

#  3번의 score = 0.447990


# In[26]:


from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type = "gbdt",
                       n_estimators = 1917,
                       learning_rate = 0.008747,
                       max_bin = 257,
                       min_child_samples = 118,
                       colsample_bytree = 0.784344330044348,
                       num_leaves = 138, 
                       subsample = 0.683348,
                       subsample_freq = 1,
                       n_jobs = -1, 
                       random_state = 42)
model


# In[27]:


# model.fit(x_train, y_train)

