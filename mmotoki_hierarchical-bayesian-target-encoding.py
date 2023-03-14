#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd 
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error


# In[2]:


PRIOR_PRECISION = 10


# In[3]:


class GaussianTargetEncoder():
        
    def __init__(self, group_cols, target_col="target", prior_cols=None):
        self.group_cols = group_cols
        self.target_col = target_col
        self.prior_cols = prior_cols

    def _get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(1)
        return prior
                    
    def fit(self, df):
        self.stats = df.assign(mu_prior=self._get_prior(df), y=df[self.target_col])
        self.stats = self.stats.groupby(self.group_cols).agg(
            n        = ("y", "count"),
            mu_mle   = ("y", np.mean),
            sig2_mle = ("y", np.var),
            mu_prior = ("mu_prior", np.mean),
        )        
    
    def transform(self, df, prior_precision=1000, stat_type="mean"):
        
        precision = prior_precision + self.stats.n/self.stats.sig2_mle
        
        if stat_type == "mean":
            numer = prior_precision*self.stats.mu_prior                    + self.stats.n/self.stats.sig2_mle*self.stats.mu_mle
            denom = precision
        elif stat_type == "var":
            numer = 1.0
            denom = precision
        elif stat_type == "precision":
            numer = precision
            denom = 1.0
        else: 
            raise ValueError(f"stat_type={stat_type} not recognized.")
        
        mapper = dict(zip(self.stats.index, numer / denom))
        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])
        
        values = np.array([mapper.get(k) for k in keys]).astype(float)
        
        prior = self._get_prior(df)
        values[~np.isfinite(values)] = prior[~np.isfinite(values)]
        
        return values
    
    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)


# In[4]:


def rmsle(x,y):
    x = np.log1p(x)
    y = np.log1p(y)
    return np.sqrt(mean_squared_error(x, y))


# In[5]:


# load data
train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")


# In[6]:


# sample for kernel
train = train.sample(int(2.5e5)).reset_index(drop=True)


# In[7]:


# create target
train["target"] = np.log1p(train.meter_reading)
test["target"] = train.target.mean()


# In[8]:


# create time features
def add_time_features(df):
    df.timestamp = pd.to_datetime(df.timestamp)    
    df["hour"]    = df.timestamp.dt.hour
    df["weekday"] = df.timestamp.dt.weekday
    df["month"]   = df.timestamp.dt.month

add_time_features(train)
add_time_features(test)


# In[9]:


# define groupings and corresponding priors
groups_and_priors = {
    
    # singe encodings
    ("hour",):        None,
    ("weekday",):     None,
    ("month",):       None,
    ("building_id",): None,
    ("meter",):       None,
    
    # second-order interactions
    ("meter", "hour"):        ["gte_meter", "gte_hour"],
    ("meter", "weekday"):     ["gte_meter", "gte_weekday"],
    ("meter", "month"):       ["gte_meter", "gte_month"],
    ("meter", "building_id"): ["gte_meter", "gte_building_id"],
        
    # higher-order interactions
    ("meter", "building_id", "hour"):    ["gte_meter_building_id", "gte_meter_hour"],
    ("meter", "building_id", "weekday"): ["gte_meter_building_id", "gte_meter_weekday"],
    ("meter", "building_id", "month"):   ["gte_meter_building_id", "gte_meter_month"],
}


# In[10]:


features = []
for group_cols, prior_cols in groups_and_priors.items():
    features.append(f"gte_{'_'.join(group_cols)}")
    gte = GaussianTargetEncoder(list(group_cols), "target", prior_cols)    
    train[features[-1]] = gte.fit_transform(train, PRIOR_PRECISION)
    test[features[-1]]  = gte.transform(test,  PRIOR_PRECISION)


# In[11]:


# clean up
drop_cols = ["hour", "weekday", "month", "building_id"]
train.drop(drop_cols, 1, inplace=True)
test.drop(drop_cols, 1, inplace=True)
del  gte
gc.collect()


# In[12]:


train[features + ["target"]].head()


# In[13]:


test[features].head()


# In[14]:


train_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for m in range(4):
    
    print(f"Meter {m}", end="") 
    
    # instantiate model
    model = RidgeCV(
        alphas=np.logspace(-10, 1, 25), 
        normalize=True,
    )    
    
    # fit model
    model.fit(
        X=train.loc[train.meter==m, features].values, 
        y=train.loc[train.meter==m, "target"].values
    )

    # make predictions 
    train_preds[train.meter==m] = model.predict(train.loc[train.meter==m, features].values)
    test_preds[test.meter==m]   = model.predict(test.loc[test.meter==m, features].values)
    
    # transform predictions
    train_preds[train_preds < 0] = 0
    train_preds[train.meter==m] = np.expm1(train_preds[train.meter==m])
    
    test_preds[test_preds < 0] = 0 
    test_preds[test.meter==m] = np.expm1(test_preds[test.meter==m])
    
    # evaluate model
    meter_rmsle = rmsle(
        train_preds[train.meter==m],
        train.loc[train.meter==m, "meter_reading"].values
    )
    
    print(f", rmsle={meter_rmsle:0.5f}")

print(f"Overall rmsle={rmsle(train_preds, train.meter_reading.values):0.5f}")
del train, train_preds, test
gc.collect()


# In[15]:


# create submission
subm  = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")
subm["meter_reading"] = test_preds
subm.to_csv(f"submission.csv", index=False)

