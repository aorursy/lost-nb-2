#!/usr/bin/env python
# coding: utf-8

# In[1]:


# crawling data from 2017-09-01 to 2017-09-07
import urllib
import pandas as pd
import numpy as np
import multiprocessing
import warnings
import json
warnings.filterwarnings("ignore")

def get_views(web_info):
    global date
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'     '{0}/{1}/{2}/{3}/daily/{4}/{5}'    .format(web_info[1], web_info[2], web_info[3], web_info[0], date[0], date[-1])
    res = np.array([np.nan for i in date])
    try:
        url = urllib.request.urlopen(url)
        api_res = json.loads(url.read().decode())['items']
    except:
        return res
    
    for i in api_res:
        time = i['timestamp'][0:-2]
        res[date.index(time)] = i['views']
    
    return res

def get_views_main(input_page):
    pool_size = multiprocessing.cpu_count()*2
    pool = multiprocessing.Pool(processes=pool_size)
    res = pool.map(get_views, input_page)
    pool.close()
    pool.join()
    return res

date = [
    '20170901', 
    '20170902', 
    '20170903',
    '20170904',
    '20170905',
    '20170906',
    '20170907'
]

pages = pd.read_csv("../input/train_2.csv", usecols=['Page'], nrows=2)
page_details = pd.DataFrame([i.split("_")[-4:] for i in pages["Page"]],
                            columns=["name", "project", "access", "agent"])

page_web_traffic = np.array(get_views_main(page_details.values))
# results:
# array([[ 19.,  33.,  33.,  18.,  16.,  27.,  29.],
#        [ 32.,  30.,  11.,  19.,  54.,  25.,  26.]])

