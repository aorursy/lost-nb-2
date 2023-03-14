#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("../input/train.csv")
print(df.info())


# In[2]:


import plotnine as ggp

p = ggp.ggplot(ggp.aes(x="budget", y="revenue"), df) +ggp.geom_point() +ggp.geom_smooth() +ggp.xlab("Budget") +ggp.ylab("") +ggp.ggtitle("Revenue as a function of budget") +ggp.theme(legend_position='none')
print(p)


# In[3]:


train = pd.DataFrame(df["budget"])
label = df["revenue"]


# In[4]:


import numpy as np

def rmlse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# In[5]:


from sklearn.metrics.scorer import make_scorer

rmsle_scorer = make_scorer(rmlse, greater_is_better=False)


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model  = LinearRegression()
scores = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)
print(-np.mean(scores))                           


# In[7]:


p = ggp.ggplot(ggp.aes(x="popularity", y="revenue"), df) +ggp.geom_point() +ggp.geom_smooth() +ggp.xlab("Popularity") +ggp.ylab("") +ggp.ggtitle("Revenue as a function of popularity") +ggp.theme(legend_position='none')
print(p)


# In[8]:


train = pd.DataFrame(df["popularity"])
label = df["revenue"]

model  =   LinearRegression()
scores = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)
print(-np.mean(scores))   


# In[9]:


train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]

model  =   LinearRegression()
scores = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)
print(scores)   


# In[10]:


def rmlse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(np.clip(y0, 0, None)), 2)))

rmsle_scorer = make_scorer(rmlse, greater_is_better=False)

train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]

model  =   LinearRegression()
scores = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)
print(-np.mean(scores)) 


# In[11]:


model  =   LinearRegression()
train = pd.DataFrame(df[["budget"]])
label = df["revenue"]
scores_1 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["popularity"]])
label = df["revenue"]
scores_2 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]
scores_3 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

performance = pd.DataFrame({
    "modelname": ["1"]*10 + ["2"]*10 + ["3"]*10,
    "scores": list(-scores_1) + list(-scores_2) + list(-scores_3)
})

p = ggp.ggplot(ggp.aes(x="modelname", y="scores"), performance) +ggp.geom_boxplot(ggp.aes(fill = "factor(modelname)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Scores of the models") +ggp.theme(legend_position='none')
print(p)


# In[12]:


p = ggp.ggplot(ggp.aes(x="runtime", y="revenue"), df) +ggp.geom_point() +ggp.geom_smooth() +ggp.xlab("Budget") +ggp.ylab("") +ggp.ggtitle("Revenue as a function of runtime") +ggp.theme(legend_position='none')
print(p)


# In[13]:


temp = df[["runtime"]].dropna()
imputation = np.median(temp.runtime)

p = ggp.ggplot(ggp.aes(x="runtime"), temp) +ggp.geom_histogram() +ggp.xlab("Runtime") +ggp.ylab("") +ggp.geom_vline(xintercept=imputation, color="red") +ggp.ggtitle("Revenue as a function of runtime") +ggp.theme(legend_position='none')
print(p)


# In[14]:


df.runtime = df.runtime.fillna(imputation)


# In[15]:


model  =   LinearRegression()
train = pd.DataFrame(df[["budget"]])
label = df["revenue"]
scores_1 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["popularity"]])
label = df["revenue"]
scores_2 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]
scores_3 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["runtime"]])
label = df["revenue"]
scores_4 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime"]])
label = df["revenue"]
scores_5 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

performance = pd.DataFrame({
    "modelname": ["1"]*10 + ["2"]*10 + ["3"]*10 + ["4"]*10 + ["5"]*10,
    "scores": list(-scores_1) + list(-scores_2) + list(-scores_3) + list(-scores_4) + list(-scores_5)
})

p = ggp.ggplot(ggp.aes(x="modelname", y="scores"), performance) +ggp.geom_boxplot(ggp.aes(fill = "factor(modelname)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Scores of the models") +ggp.theme(legend_position='none')
print(p)


# In[16]:


df["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

p = ggp.ggplot(ggp.aes(x="belongs_to_collection_missing", y="revenue"), df) +ggp.geom_boxplot(ggp.aes(fill = "factor(belongs_to_collection_missing)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Distribution of revenues for missing belongs_to_collection") +ggp.theme(legend_position='none')
print(p)


# In[17]:


model  =   LinearRegression()
train = pd.DataFrame(df[["budget"]])
label = df["revenue"]
scores_1 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["popularity"]])
label = df["revenue"]
scores_2 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]
scores_3 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["runtime"]])
label = df["revenue"]
scores_4 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime"]])
label = df["revenue"]
scores_5 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["belongs_to_collection_missing"]])
label = df["revenue"]
scores_6 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime", "belongs_to_collection_missing"]])
label = df["revenue"]
scores_7 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

performance = pd.DataFrame({
    "modelname": ["1"]*10 + ["2"]*10 + ["3"]*10 + ["4"]*10 + ["5"]*10 + ["6"]*10 + ["7"]*10,
    "scores": list(-scores_1) + list(-scores_2) + list(-scores_3) + list(-scores_4) + 
    list(-scores_5) + list(-scores_6) + list(-scores_7)
})

p = ggp.ggplot(ggp.aes(x="modelname", y="scores"), performance) +ggp.geom_boxplot(ggp.aes(fill = "factor(modelname)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Scores of the models") +ggp.theme(legend_position='none')
print(p)


# In[18]:


df["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

p = ggp.ggplot(ggp.aes(x="homepage_missing", y="revenue"), df) +ggp.geom_boxplot(ggp.aes(fill = "factor(homepage_missing)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Distribution of revenues for missing homepage") +ggp.theme(legend_position='none')
print(p)


# In[19]:


model  =   LinearRegression()
train = pd.DataFrame(df[["budget"]])
label = df["revenue"]
scores_1 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["popularity"]])
label = df["revenue"]
scores_2 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity"]])
label = df["revenue"]
scores_3 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["runtime"]])
label = df["revenue"]
scores_4 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime"]])
label = df["revenue"]
scores_5 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["belongs_to_collection_missing"]])
label = df["revenue"]
scores_6 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime", "belongs_to_collection_missing"]])
label = df["revenue"]
scores_7 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["homepage_missing"]])
label = df["revenue"]
scores_8 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

model  =   LinearRegression()
train = pd.DataFrame(df[["budget", "popularity", "runtime", "belongs_to_collection_missing", "homepage_missing"]])
label = df["revenue"]
scores_9 = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

performance = pd.DataFrame({
    "modelname": ["1"]*10 + ["2"]*10 + ["3"]*10 + ["4"]*10 + ["5"]*10 + ["6"]*10 + ["7"]*10 + ["8"]*10 + ["9"]*10,
    "scores": list(-scores_1) + list(-scores_2) + list(-scores_3) + list(-scores_4) + 
    list(-scores_5) + list(-scores_6) + list(-scores_7) + list(-scores_8) + list(-scores_9)
})

p = ggp.ggplot(ggp.aes(x="modelname", y="scores"), performance) +ggp.geom_boxplot(ggp.aes(fill = "factor(modelname)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Scores of the models") +ggp.theme(legend_position='none')
print(p)


# In[20]:


train = pd.DataFrame(df[["budget", "popularity", "runtime", "belongs_to_collection_missing", "homepage_missing"]])
label = df["revenue"]

model  = LinearRegression()
scores_linear = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from sklearn.ensemble import AdaBoostRegressor
model  = AdaBoostRegressor()
scores_adaboost = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from sklearn.ensemble import RandomForestRegressor
model  = RandomForestRegressor(n_estimators=100)
scores_randomforest = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from sklearn.ensemble import GradientBoostingRegressor
model  = GradientBoostingRegressor()
scores_gradientboosting = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from xgboost import XGBRegressor
model  = XGBRegressor()
scores_xgboost = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from lightgbm import LGBMRegressor
model  = LGBMRegressor()
scores_lightgbm = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from lightgbm import LGBMRegressor
model  = LGBMRegressor()
scores_lightgbm = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

from catboost import CatBoostRegressor
model  = CatBoostRegressor(verbose=False)
scores_catboost = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

performance = pd.DataFrame({
    "modelname": ["Linear"]*10 + ["Adaboost"]*10 + ["Random Forest"]*10  + ["Boosting"]*10 
    + ["Xgboost"]*10 + ["Lightgbm"]*10 + ["Catboost"]*10,
    "scores": list(-scores_linear) + list(-scores_adaboost) + list(-scores_randomforest) +
    list(-scores_gradientboosting) + list(-scores_xgboost) + list(-scores_lightgbm) + 
    list(-scores_catboost)
})

p = ggp.ggplot(ggp.aes(x="modelname", y="scores"), performance) +ggp.geom_boxplot(ggp.aes(fill = "factor(modelname)")) +ggp.xlab("") +ggp.ylab("") +ggp.ggtitle("Scores of the models") +ggp.theme(legend_position='none')
print(p)


# In[21]:


train = pd.DataFrame(df[["budget", "popularity", "runtime"]])
train.runtime = train.runtime.fillna(imputation)
train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)
train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)
label = df["revenue"]

model  = RandomForestRegressor(verbose=False)
model.fit(train, label)


# In[22]:


test = pd.read_csv("../input/test.csv")
dfte = pd.DataFrame(test[["budget", "popularity", "runtime"]])
dfte["homepage_missing"] = np.array(test.homepage.isna(), dtype=int)
dfte["belongs_to_collection_missing"] = np.array(test.belongs_to_collection.isna(), dtype=int)
dfte.runtime = dfte.runtime.fillna(imputation)

predictions = model.predict(dfte)
predictions = np.clip(predictions, 0, None)
submission = pd.DataFrame({
    "id" : test.id,
    "revenue": predictions
})
submission.to_csv("submission.csv", index=False)


# In[23]:


print(scores_randomforest)
print(-np.mean(scores_randomforest))

