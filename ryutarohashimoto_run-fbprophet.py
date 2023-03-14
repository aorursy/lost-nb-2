#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 100)
import pickle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# pystan & prophet

import pystan
from fbprophet import Prophet



# In[ ]:


# set which items are predicted by adjusting n, start, end.
# Currentry this model predicts first 10 items
# If you predict all items, run from n = 1 to n = 32
n = 1
start = 0
# start = 0 + 1000*(n-1)
end = 10
# end = 1000 + 1000*(n-1)
../input/dataset


# In[ ]:


# read data
df = pd.read_pickle('../input/dataset/dataset/simple_matrix.pkl') ## df is main y matrix
date_df = pd.read_pickle('../input/dataset/dataset/calendar.pkl')
price_train = pd.read_pickle('../input/df-price/df_price_train.pkl').set_index("d")
price_predict = pd.read_pickle('../input/dataset/dataset/df_price_predict.pkl')
df_calendar = pd.read_pickle('../input/dataset/dataset/df_calendar.pkl')


# In[ ]:


def linear_part(price_train, price_predict, calender, item_name):
        train = price_train[price_train["store_id"] == item_name]
        predict = price_predict.tail(28)[item_name]
        seg= pd.concat([train["sell_price"], predict])
        seg = pd.DataFrame(seg)
        seg.columns = ["price"]
        res =  pd.merge(calender, seg, left_on="d", right_on="d", how = "left")
        
        return res

# Change the date column to be readable on prophet
def day_format_converter(df, name_id, calendar):
    # name_id is the column name of date id in the dataframe
    calendar_temp = calendar[["date", "d"]]
    calendar_temp = calendar_temp.rename(columns={'date': 'ds'})
    df = pd.merge(df, calendar_temp, left_on=name_id, right_on='d')
    df = df.drop([name_id, "d"], axis=1)

    return df


# In[ ]:


# Create event dataframe
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2011-01-08',
                        '2013-01-12',
                        '2014-01-12', 
                        '2014-01-19',
                        '2014-02-02',
                        '2015-01-11', 
                        '2016-01-17',
                        '2016-01-24', 
                        '2016-02-07']),
  'lower_window': 0, # Seemingly affecting on the past
  'upper_window': 1, # Seemingly affecting on the future
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2011-02-06',
                        '2012-02-05',
                        '2013-02-03',
                        '2014-02-02', 
                        '2015-02-01',
                        '2016-02-07',]),
  'lower_window': 0,
  'upper_window': 1,
})


# Sum up two events
holidays = pd.concat((playoffs, superbowls))


# In[ ]:


df_forecast = pd.DataFrame()
for i in range(start, end):
    item_name = df.columns[i]

    # preprocessing
    df_y = df.rename({item_name: "y"}, axis=1)
    df_y['floor'] = 0

    reggresors = linear_part(price_train, price_predict, df_calendar, item_name)
    # 欠損値以外の値が最初に出てくるインデックスを検索(販売開始時点)
    start_point = reggresors.reset_index()[reggresors["d"] == reggresors[-pd.isnull(reggresors["price"])].iloc[1, 0]].index[0]

    # 販売期間のデータフレーム を作成
    df_y = df_y.iloc[start_point-1: , :]
    temp = reggresors.iloc[start_point-1: len(df.iloc[start_point-1: , :]) + start_point -1, :].set_index("d")
    temp = temp.fillna(reggresors["price"].max() *3)
    reggresors_list = temp.columns
    df_y = pd.concat([df_y, temp], axis = 1)

    # Execute the function
    df_y = day_format_converter(df_y.reset_index(), "d", date_df)

    # Construct model
    if df_y["y"].mean() >= 20:
        model = Prophet(yearly_seasonality = True, 
                        weekly_seasonality = True, 
                        daily_seasonality = False,
                        changepoint_prior_scale=0.05,
                        holidays=holidays,
                        seasonality_mode='multiplicative',
                        )
        model.add_country_holidays(country_name='US')

    else:
        model = Prophet(yearly_seasonality = True, 
                        weekly_seasonality = True, 
                        daily_seasonality = False,
                        changepoint_prior_scale=0.05,
                        holidays=holidays,
                        seasonality_mode='additive',
                        )
        model.add_country_holidays(country_name='US')

    for j in reggresors_list:
        model.add_regressor(j, standardize='auto')

    forecast = model.fit(df_y)

    # 予測期間のデータフレームを作成
    future = model.make_future_dataframe(periods=28, freq='D')
    future = pd.concat([future.set_index("ds"), 
                        day_format_converter(reggresors.iloc[start_point-1: , :], "d", date_df).set_index("ds")], axis = 1)
    future = future.reset_index()
    future['floor'] = 0
    # predict
    forecast = model.predict(future)
    predict = forecast.tail(56)["yhat"]
    predict = predict.reset_index()["yhat"]
    predict.name = str(i)
    df_forecast = pd.concat([df_forecast, predict], axis = 1)


# In[ ]:


path = "output/" + str(n) + ".pkl"
df_forecast.to_pickle(path)

