#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.utils import shuffle


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        if 'train' in filename:
            train = pd.read_csv(path)
        elif 'test' in filename:
            test = pd.read_csv(path)

# Scale X
scaler = StandardScaler()
            


# In[3]:


train.sample(frac=1).head(5)
test.sample(frac=1).head(5)


# In[4]:


# Rename columns
train.rename(columns={'Country_Region':'Country', 
                         'Province_State': 'State'},
                inplace=True)
test.rename(columns={'Country_Region':'Country', 
                         'Province_State': 'State'},
                inplace=True)

# Convert date
train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
submission = pd.DataFrame(columns=['ForecastId','ConfirmedCases','Fatalities'], dtype='int32')

# Concat both DFs
df = pd.concat([train,test],sort=False)

# Fill N/A for one which missing
df['Country'] = df.Country.fillna("None")
df['State'] = df.State.fillna("None")


# In[5]:


def plot_data(train, test=None, submission=None):
    past_n_day = 30
    top_ctry_num = 10

    if test is not None:
        train_n = len(train); test_n = len(test)
        # Merge test + submission
        df = pd.merge(test, submission, on='ForecastId')
        # Adjust ID
        df['Id'] = pd.Series(range(train_n+1, train_n+test_n+1)).astype(int)
        # Concat. train + (test + subm.)
        df = pd.concat([train,df],sort=True)
        df = df.drop_duplicates(subset=['Date','Country','State'],keep='first')
    else:
        df = train

    # First/Last date in the DF
    first_date = train.Date.min(); last_date = train.Date.max()
    _,first_ctry = list(df.groupby('Country'))[0]
    last_idx = np.where(first_ctry['Date'] == last_date)[0][0]

    # Find countries with most fatalities
    ctry = df.groupby(['Country','Date']).sum()
    top_ctry = ctry.groupby(['Country'])                      .sum()                      .sort_values('Fatalities', ascending=False)[:top_ctry_num]['Fatalities']

    # Plot settings
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    fig.suptitle("\nFatality Growth")
    plt.xlabel("Day Since {}".format(str(first_date).split(' ')[0]))
    plt.ylabel("Number of Fatalities",labelpad=30)
    ax2.set_yscale("log")
    ax1.grid(which="both", alpha=0.75, linestyle='dashed', linewidth=0.5)
    ax2.grid(which="both", alpha=0.75, linestyle='dashed', linewidth=0.5)

    labels = {}
    for idx,(name, country) in enumerate(ctry.groupby('Country')):
        #print(country)
        if name not in top_ctry.index.values:
            continue
        y = country.Fatalities.values
        x = range(0,len(y))
        #print(y)
        l = ax1.plot(x,y,'-o',label=name,
                 linewidth=2, markersize=3,markevery=7)
        ax2.plot(x,y,'-o',label=name,
                 linewidth=2, markersize=3,markevery=7)
        labels[name] = l[0]

    # Plot settings
    fig.legend(list(labels.values()), list(labels.keys()),
               loc='center right')
    plt.subplots_adjust(right=0.77)
    #print(last_date)
    if test is not None:
        ax1.axvline(x=last_idx, color='k',linewidth='1',linestyle='--')
        ax2.axvline(x=last_idx, color='k',linewidth='1',linestyle='--')
        
    fig = plt.gcf()
    fig.savefig("result.png")
    plt.show()


# In[6]:


plot_data(train)


# In[7]:



class BellModel():
    def __init__(self,streak_threshold=7,power=10):
        self.streak_threshold = streak_threshold
        self.power = power
    def __str__(self):
        return "BellModel(streak_threshold={},power={},\npopt={})".format(
            self.streak_threshold,
            self.power,
            self.popt
        )

    def func(self,X,a,b,c):
        result = a*np.exp(-(((X-b)**2) / (2*(c**2+0.1))))
        return result.flatten()

    def _find_streak(self, arr):
        pos = np.clip(arr, 0, 1).astype(bool).cumsum()
        neg = np.clip(arr, -1, 0).astype(bool).cumsum()
        streaks = np.where(arr >= 0,
                           pos-np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                           -neg+np.maximum.accumulate(np.where(arr >= 0, neg, 0)))
        return streaks

    def fit(self,x,y):
        self.streak_threshold = -self.streak_threshold
        # First non-zero index
        x = scaler.inverse_transform(x)
        y = y.T.flatten(); x = x.T.flatten()
        # Sort array
        c = np.argsort(x[:]); y = y[c]
        x = np.sort(x)

        # Find the first index of population > 0
        nonzero_index, *_ = np.where(np.sign(y).cumsum() == 1)

        if not isinstance(nonzero_index,np.ndarray) and not nonzero_index:
            # If no real value, a = 0.
            self.popt = [0,0,0]; return
        else:
            try:
                nonzero_index = nonzero_index.item(0)
            except IndexError:
                self.popt = [0,0,0]; return
                
        zeros, y_sp = y[:nonzero_index],y[nonzero_index:]

        a = b = d = 0; c = 1
        a_min = b_min = 0; c_min = 1

        # The peak of the curve
        a_min = np.max(y_sp)
        a_max = a_min*self.power+1
        # Slide length
        d = len(zeros)

        # First Derivative
        y1 = np.diff(y_sp)
        y1_streaks = self._find_streak(y1)
        if self.streak_threshold in y1_streaks:
            # The peak is over!
            a = np.max(y_sp)
        else:
            b_min = len(y_sp) + d

        # Second Derivative
        y2 = np.diff(y1)
        y2_streaks = self._find_streak(y2)
        if self.streak_threshold in y2_streaks:
            # Near half of the peak
            b = np.where(y2_streaks==self.streak_threshold)[0][0] * 2
            c = b/3
        else:
            # Not even near the half of the peak
            # b_min: already assigned
            c_min = len(y_sp)*2/3
        a = a_min if a < a_min else a
        b = 300 if b < b_min else b
        c = 100 if c < c_min else c
        import traceback
        try: 
            self.popt, self.pconv = curve_fit(
                self.func, x, y,
                maxfev=100000,
                check_finite=False,
                p0=[a,b,c],
                bounds=([a_min,b_min,c_min],[a_max,300.,100.])
            )
        except Exception as e:
            traceback.print_exc()
            self.popt = [0,0,0]
        return self

    def get_params(self, deep=False):
            return { 
                'streak_threshold': self.streak_threshold,
                'power': self.power,
            }
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_estimated_params(self):
        return [self.streak_threshold, self.power]

    def set_curve_params(self, streak_threshold, power):
        self.streak_threshold = streak_threshold
        self.power = power
        return self

    def predict(self,x):
        x = scaler.inverse_transform(x).T.flatten()
        return self.func(x,*self.popt)


# In[8]:


##################################################
#
# Fit Model
#
##################################################

def model_fit(X_train, Y_train, X_test):
    """Create a model, find the best fit, predict, and ensemble to predict the best result."""
    X_train = scaler.fit_transform(X_train.reshape(-1,1))
    X_test = scaler.transform(X_test.reshape(-1,1))
    X_train_sh, Y_train_sh = shuffle(X_train,Y_train, random_state=0)
    remove_neg = lambda x: np.rint(x).astype(int).clip(min=0)
    metric = "neg_mean_squared_error"

    ##################################################
    # Bell-Shaped
    ##################################################

    param_grid = {
        #'streak_threshold': [3,4,5,6,7],
        'streak_threshold': [5,6,7,14,21],
        'power': [2,3,5,10,20,30,50]
    }
    bell_grid = GridSearchCV(BellModel(),
                             param_grid,
                             cv=3,
                             scoring=metric,
                             verbose=0
    )
    bell_grid.fit(X_train_sh, Y_train_sh)
    bell_model = bell_grid.best_estimator_
    ##################################################
    # Ridge
    ##################################################
    param_grid = {
        'polynomialfeatures__degree': np.arange(1,5),
        'ridge__alpha':[1e2, 1e3,1e4],
        'ridge__fit_intercept': [True, False],
        'ridge__normalize': [True, False],
        'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    }
    def PolynomialRidgeRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), Ridge(**kwargs))
    
    ridge_grid = GridSearchCV(PolynomialRidgeRegression(),
                                 param_grid,
                                 cv=3,
                                 scoring=metric,
                                 verbose=0 )
    ridge_grid.fit(X_train_sh, Y_train_sh)
    ridge_model = ridge_grid.best_estimator_
    ##################################################
    # SVR
    ##################################################
    param_grid = {
        'polynomialfeatures__degree': np.arange(1,5),
        'linearsvr__C' : np.logspace(0,1,5),
        'linearsvr__epsilon' : np.logspace(-1,1,5),
        'linearsvr__fit_intercept': [True, False],
    }

    def PolynomialSVRRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearSVR(**kwargs,max_iter=10000))
    
    svr_grid = GridSearchCV(PolynomialSVRRegression(),
                                 param_grid,
                                 cv=3,
                                 scoring=metric,
                                 verbose=0 )
    svr_grid.fit(X_train_sh, Y_train_sh)
    svr_model = svr_grid.best_estimator_

    ##################################################
    # second feature matrix
    X_train2 = pd.DataFrame( {'Bell': remove_neg(bell_model.predict(X_train)),
                              'Ridge': remove_neg(ridge_model.predict(X_train)),
                              'SVR': remove_neg(svr_model.predict(X_train)),
    })
    X_test2 = pd.DataFrame( { 'Bell': remove_neg(bell_model.predict(X_test)),
                              'Ridge': remove_neg(ridge_model.predict(X_test)),
                              'SVR': remove_neg(svr_model.predict(X_test)),
    })


    # second-feature modeling using linear regression
    reg = LinearRegression()
    reg.fit(X_train2, Y_train)

    Y_test = reg.predict(X_test2)
    Y_test = remove_neg(Y_test)

    return Y_test


# In[9]:


##################################################
# Iterate along Country/State
##################################################

for name,state in df.groupby(['Country','State']):
    # Save Train/Test overrap
    mask = state.duplicated(subset=['Date'],keep='first')
    df_tmp = state.loc[~mask]
    df_tmp_dropped = state.loc[mask]

    # Duplicate number
    drop_num = len(state) - len(df_tmp)

    try:
        df_tmp.insert(0,'Index', range(1,len(df_tmp)+1))
    except:
        pass

    # Training Data
    tmp_train = df_tmp.dropna(subset=['ConfirmedCases'])
    X_tr = tmp_train['Index'].values

    # Testing Data
    tmp_test = df_tmp[ df_tmp['ConfirmedCases'].isna() ]
    X_te = tmp_test.Index.values

    for cat in ('ConfirmedCases', 'Fatalities'):
        # Training Data
        Y_tr = tmp_train[cat].values
        Y_te = model_fit(X_tr, Y_tr, X_te)

        # Save to DF
        tmp_test.loc[:,cat] = Y_te
        
    # Merge train(last 13) + test
    tmp_train = pd.merge(tmp_train.tail(drop_num)[['Date','ConfirmedCases','Fatalities']],
                     df_tmp_dropped[['Date','ForecastId']],
                     on='Date')
    col = ['ForecastId','ConfirmedCases','Fatalities']
    tmp_train = tmp_train[col]
    tmp_test = tmp_test[col]
    tmp = pd.concat([tmp_train,tmp_test]) 
    submission = pd.concat([submission, tmp])

os.chdir("/kaggle/working/")
submission = submission.fillna(0)
submission = submission.astype(int)
submission.to_csv('submission.csv', index=False)
plot_data(train,test,submission) 
 

