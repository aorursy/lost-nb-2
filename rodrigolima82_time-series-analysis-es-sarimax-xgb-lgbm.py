#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando bibliotecas que serao utilizadas neste projeto
import pandas as pd
import numpy as np

from itertools import product
from multiprocessing import Pool
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
import scipy.stats as scs
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

# Stats
from scipy import stats
from scipy.stats import skew, norm
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import datetime
from dateutil.relativedelta import relativedelta 
import time
import gc
import os
from tqdm import tqdm_notebook

# Ignorar warnings
import warnings
warnings.filterwarnings(action="ignore")

# Seta algumas opções no Jupyter para exibição dos datasets
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# Variavel para controlar o treinamento no Kaggle
TRAIN_OFFLINE = False

# Variavel para indicar o path local
LOCAL_DATA_FOLDER  = 'data/'
KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy/'


# In[2]:


# Importando bibliotecas do sklearn
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression

# lib de modelos de machine learning
import xgboost as xgb
import lightgbm as lgb


# In[3]:


# Funcao para reducao da memoria utilizada
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# Funcao para realizar a leitura dos arquivos LOCAL ou do KAGGLE
def read_data():
    
    # Se for local
    if TRAIN_OFFLINE:

        calendar               = pd.read_csv(os.path.join(LOCAL_DATA_FOLDER, 'calendar.csv'))
        sell_prices            = pd.read_csv(os.path.join(LOCAL_DATA_FOLDER, 'sell_prices.csv'))
        sales_train_validation = pd.read_csv(os.path.join(LOCAL_DATA_FOLDER, 'sales_train_validation.csv'))
        submission             = pd.read_csv(os.path.join(LOCAL_DATA_FOLDER, 'sample_submission.csv'))

    # Se estiver no ambiente do Kaggle
    else:
        
        calendar               = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'calendar.csv'))
        sell_prices            = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sell_prices.csv'))
        sales_train_validation = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sales_train_validation.csv'))
        submission             = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sample_submission.csv'))

    calendar               = reduce_mem_usage(calendar)
    sell_prices            = reduce_mem_usage(sell_prices)
    sales_train_validation = reduce_mem_usage(sales_train_validation)
    submission             = reduce_mem_usage(submission)
        
    return calendar, sell_prices, sales_train_validation, submission


# In[4]:


# Leitura dos dados e aplicando redução de memória
calendar, sell_prices, sales_train_validation, submission = read_data()


# In[5]:


# Funcao para realizar o merge dos datasets retornando apenas um dataframe
def reshape_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows=55000000, merge=False):
    
    # realizando o reshape dos dados de venda usando melt
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    print('Melted sales train validation tem {} linhas e {} colunas'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    #sales_trian_validation = reduce_mem_usage(sales_train_validation)
    
    # separando os registros de teste e validacao
    test_rows = [row for row in submission['id'] if 'validation' in row]
    val_rows = [row for row in submission['id'] if 'evaluation' in row]
    
    test = submission[submission['id'].isin(test_rows)]
    val = submission[submission['id'].isin(val_rows)]
    
    # renomeando as colunas
    test.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 
                    'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 
                    'd_1931', 'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 
                    'd_1940', 'd_1941']
    val.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 
                   'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 
                   'd_1959', 'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 
                   'd_1968', 'd_1969']
    
    # obtendo somente dados do produto e removendo registros duplicados 
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    # realizando merge com a tabela de produto
    test = test.merge(product, how = 'left', on = 'id')
    val = val.merge(product, how = 'left', on = 'id')
    
    # realizando o reshape dos dados de test e validacao
    test = pd.melt(test, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    val = pd.melt(val, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
    # criando uma nova coluna para definir dados de treino, teste e validacao
    sales_train_validation['part'] = 'train'
    test['part'] = 'test'
    val['part'] = 'val'
    
    # criando um so dataset com a juncao de todos os registros de treino, validacao e teste
    data = pd.concat([sales_train_validation, test, val], axis = 0)
    
    # removendo datasets anteriores
    del sales_train_validation, test, val
    
    # selecionando somente alguns registros para treinamento
    data = data.loc[nrows:]
    
    # removendo os dados de validacao
    data = data[data['part'] != 'val']
    
    # realizando o merge com calendario e preco
    if merge:
        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        data.drop(['d', 'day', 'weekday'], inplace = True, axis = 1)
        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        print('Dataset final para treino tem {} linhas e {} colunas'.format(data.shape[0], data.shape[1]))
    else: 
        pass
    
    return data

# Funcao para tratamento valores missing transformacao das features categoricas e numericas
def transform(data):
    
    # realizando tratamento em valores missing nas features categoricas
    nan_features_cat = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features_cat:
        data[feature].fillna('unknown', inplace = True)
    
    # realizando tratamento em valores missing na feature sell_price
    data['sell_price'].fillna(0, inplace = True)
        
    # transformando features categorias em numericas para realizar as previsoes
    encoder = preprocessing.LabelEncoder()
    data['id_encode'] = encoder.fit_transform(data['id'])
    
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Realizando o reshape e o merge dos datasets\ndata = reshape_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows=45000000, merge=True)\n\n# Chamando as funcoes de transformacao dos dados\ndata = transform(data)\n\n# Visualizando o cabecalho do dataset final\ndata.head()\n\n# Limpando dados da memória\ngc.collect()')


# In[7]:


# Verificando a data de inicio e fim do dataset
print(min(data['date']), max(data['date']))


# In[8]:


data[data['id'] == 'FOODS_3_634_WI_2_validation'].head()


# In[9]:


data[(data['id'] == 'FOODS_3_634_WI_2_validation') & (data['demand'] > 0) & (data['part'] == 'train')]


# In[10]:


# Selecionar apenas os dados de treino e validação para as analises
# Selecionando somente 1 item para testes: FOODS_3_634_WI_2
df = data[(data['date'] <= '2016-04-24') & (data['id'] == 'FOODS_3_634_WI_2_validation') & (data['demand'] > 0) & (data['demand'] <= 15)]

# Selecionando apenas algumas colunas para a analise e treinamento
df = df[['date','demand','dept_id','cat_id','store_id','state_id','event_name_1','event_type_1','snap_WI','sell_price']]

# Transformando a data como index 
df = df.set_index('date')

# Visualizando o resultado do dataset
df.head()


# In[11]:


plt.figure(figsize=(24, 7))
plt.plot(df['demand'])
plt.title('Soma das Vendas por dia')
plt.grid(True)
plt.xticks(rotation=90)
plt.show()


# In[12]:


# Visualizando informações de distribuicao da variavel "demand"
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(12, 8))

# Fit a normal distribution
mu, std = norm.fit(df['demand'])

# Verificando a distribuicao de frequencia da variavel "demand"
sns.distplot(df['demand'], color="b", fit = stats.norm)
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="Demand")
ax.set(title="Demand distribution: mu = %.2f,  std = %.2f" % (mu, std))
sns.despine(trim=True, left=True)

# Adicionando Skewness e Kurtosis
ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % df['demand'].skew(),        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',        backgroundcolor='white', color='xkcd:poo brown')
ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % df['demand'].kurt(),        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',        backgroundcolor='white', color='xkcd:dried blood')

plt.show()


# In[13]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[14]:


def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

#  realizando as previsões dos últimos 28 dias
moving_average(df, 28)


# In[15]:


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


# In[16]:


#  Vamos suavizar usando uma janela de 7 dias
plotMovingAverage(df['demand'], 7)


# In[17]:


#  Vamos suavizar usando uma janela de 28 dias
plotMovingAverage(df['demand'], 28)


# In[18]:


plotMovingAverage(df['demand'], 28, plot_intervals=True)


# In[19]:


def weighted_average(series, weights):
    """
        Calculate weighter average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)


# In[20]:


weighted_average(df['demand'], [0.6, 0.3, 0.1])


# In[21]:


def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


# In[22]:


def plotExponentialSmoothing(series, alphas):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);


# In[23]:


plotExponentialSmoothing(df['demand'], [0.3, 0.05])


# In[24]:


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


# In[25]:


def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)


# In[26]:


plotDoubleExponentialSmoothing(df['demand'], alphas=[0.9, 0.02], betas=[0.9, 0.02])


# In[27]:


class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


# In[28]:


def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=28):
    """
        Returns error on CV  
        
        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []
    
    values = series.values
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=2) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], 
                            slen=slen, 
                            alpha=alpha, 
                            beta=beta, 
                            gamma=gamma, 
                            n_preds=len(test))
        
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))


# In[29]:


get_ipython().run_cell_magic('time', '', '\nnew_data = df[\'demand\']\n\n# initializing model parameters alpha, beta and gamma\nx = [0, 0, 0] \n\n# Minimizing the loss function \nopt = minimize(timeseriesCVscore, \n               x0=x, \n               args=(new_data, mean_squared_error), \n               method="TNC", \n               bounds = ((0, 1), (0, 1), (0, 1))\n              )\n\n# Take optimal values...\nalpha_final, beta_final, gamma_final = opt.x\nprint(alpha_final, beta_final, gamma_final)\n\n# ...and train the model with them, forecasting for the next 28 days\nmodel = HoltWinters(new_data, \n                    slen = 28, \n                    alpha = alpha_final, \n                    beta = beta_final, \n                    gamma = gamma_final, \n                    n_preds = 28, \n                    scaling_factor = 3)\n\nmodel.triple_exponential_smoothing()')


# In[30]:


def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
    """
        series - dataset with timeseries
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    plt.figure(figsize=(20, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(series.values, label = "Actual")
    
    #error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    #plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    error = rmse(series.values, model.result[:len(series)])
    plt.title("RMSE: {0:.2f}".format(error))
    
    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values<model.LowerBond[:len(series)]] =             series.values[series.values<model.LowerBond[:len(series)]]
        anomalies[series.values>model.UpperBond[:len(series)]] =             series.values[series.values>model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                         y2=model.LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-60, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);


# In[31]:


plotHoltWinters(df['demand'])


# In[32]:


plotHoltWinters(df['demand'], plot_intervals=True, plot_anomalies=True)


# In[33]:


plt.figure(figsize=(25, 5))
plt.plot(model.PredictedDeviation)
plt.grid(True)
plt.axis('tight')
plt.title("Brutlag's predicted deviation");


# In[34]:


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[35]:


tsplot(df['demand'], lags=60)


# In[36]:


df_diff = df['demand'] - df['demand'].shift(28)
tsplot(df_diff[28:], lags=60)


# In[37]:


tsplot(df_diff[28+1:], lags=60)


# In[38]:


# setting initial values and some bounds for them
ps = range(2, 3)
d=1 
qs = range(2, 3)
Ps = range(0, 2)
D=1 
Qs = range(0, 2)
s = 28 # season length is still 28

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# In[39]:


def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(df['demand'], 
                                            order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


# In[40]:


get_ipython().run_cell_magic('time', '', 'result_table = optimizeSARIMA(parameters_list, d, D, s)')


# In[41]:


result_table.head()


# In[42]:


# set the parameters that give the lowest AIC
p, q, P, Q = result_table.parameters[0]

best_model=sm.tsa.statespace.SARIMAX(df['demand'], 
                                     order=(p, d, q),
                                     seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())


# In[43]:


tsplot(best_model.resid[28+1:], lags=60)


# In[44]:


def plotSARIMA(series, model, n_steps):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    dfCopy = series.copy()
    
    dfCopy['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    dfCopy['arima_model'][:s+d] = np.NaN    
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = dfCopy.shape[0], end = dfCopy.shape[0]+n_steps)
    forecast = dfCopy['arima_model'].append(forecast)

    # calculate error, again having shifted on s+d steps from the beginning
    error = rmse(dfCopy['demand'][s+d:], dfCopy['arima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("RMSE: {0:.2f}".format(error))
    #plt.plot(forecast, color='r', label="model")
    #plt.axvspan(dfCopy.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(dfCopy['demand'], label="actual")
    plt.legend()
    plt.grid(True);


# In[45]:


plotSARIMA(df, best_model, 28)


# In[46]:


# Criar uma copia do dataset original
new_df = df.copy()

# Visualizando o dataset
new_df.head()


# In[47]:


# Adicionando features considerando o atraso da demanda de 7 a 28 dias
for i in range(7, 29):
    new_df["lag_{}".format(i)] = new_df['demand'].shift(i)


# In[48]:


# Visualizando o resultado do dataset
new_df.tail()


# In[49]:


# para o cross-validation da serie temporal
tscv = TimeSeriesSplit(n_splits=5)


# In[50]:


def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test  = X.iloc[test_index:]
    y_test  = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


# In[51]:


y = new_df.dropna()['demand']
X = new_df.dropna().drop(['demand'], axis=1)


# In[52]:


# split com 10% para dados de teste
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.1)


# In[53]:


# machine learning em 2 linhas
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[54]:


def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = rmse(y_test, prediction)
    plt.title("RMSE: {0:.2f}".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');


# In[55]:


plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)


# In[56]:


new_df.index = pd.to_datetime(new_df.index)
new_df["day"] = new_df.index.day
new_df["weekday"] = new_df.index.weekday
new_df['is_weekend'] = new_df.weekday.isin([5,6])*1
new_df.tail()


# In[57]:


plt.figure(figsize=(16, 5))
plt.title("Encoded features")
new_df['day'].plot()
new_df['weekday'].plot()
new_df['is_weekend'].plot()
plt.grid(True);


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[59]:


X = new_df.dropna().drop(['demand'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)


# In[60]:


def code_mean(data, cat_feature, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())


# In[61]:


average_day = code_mean(new_df, 'day', "demand")
plt.figure(figsize=(12, 5))
plt.title("Médias diárias")
pd.DataFrame.from_dict(average_day, orient='index')[0].plot()
plt.grid(True);


# In[62]:


def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
    """
        series: pd.DataFrame
            dataframe with timeseries

        lag_start: int
            initial step back in time to slice target variable 
            example - lag_start = 1 means that the model 
                      will see yesterday's values to predict today

        lag_end: int
            final step back in time to slice target variable
            example - lag_end = 4 means that the model 
                      will see up to 4 days back in time to predict today

        test_size: float
            size of the test dataset after train/test split as percentage of dataset

        target_encoding: boolean
            if True - add target averages to the dataset
        
    """
    
    # copy of the initial dataset
    new_df = df.copy()

    # lags of series
    for i in range(7, 29):
        new_df["lag_{}".format(i)] = new_df['demand'].shift(i)

    # datetime features
    new_df.index = pd.to_datetime(new_df.index)
    new_df["day"] = new_df.index.day
    new_df["weekday"] = new_df.index.weekday
    new_df['is_weekend'] = new_df.weekday.isin([5,6])*1

    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(new_df.dropna())*(1-test_size))
        new_df['weekday_average'] = list(map(code_mean(new_df[:test_index], 'weekday', "demand").get, new_df['weekday']))
        new_df["day_average"] = list(map(code_mean(new_df[:test_index], 'day', "demand").get, new_df['day']))

        # frop encoded variables 
        new_df.drop(["day", "weekday"], axis=1, inplace=True)
    
    # train-test split
    y = new_df.dropna()['demand']
    X = new_df.dropna().drop(['demand'], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


# In[63]:


X_train, X_test, y_train, y_test =prepareData(df, lag_start=1, lag_end=29, test_size=0.1, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lr)


# In[64]:


X_train, X_test, y_train, y_test =prepareData(df, lag_start=1, lag_end=29, test_size=0.1, target_encoding=False)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[65]:


plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr());


# In[66]:


from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, 
                 plot_anomalies=True)
plotCoefficients(ridge)


# In[67]:


lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)

plotModelResults(lasso, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, 
                 plot_anomalies=True)
plotCoefficients(lasso)


# In[68]:


from xgboost import XGBRegressor 

xgb = XGBRegressor()
xgb.fit(X_train_scaled, y_train)


# In[69]:


plotModelResults(xgb, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, 
                 plot_anomalies=True)


# In[70]:


from lightgbm import LGBMRegressor 

lgb = LGBMRegressor()
lgb.fit(X_train_scaled, y_train)


# In[71]:


plotModelResults(lgb, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, 
                 plot_anomalies=True)


# In[ ]:




