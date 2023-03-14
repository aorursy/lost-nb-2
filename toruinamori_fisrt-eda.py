#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from logging import getLogger


# In[ ]:


TRAIN_DATA = "../input/train.csv"
TEST_DATA = "../input/test.csv"
SUMPLE_SUBMIT_FILE = "../input/sample_submission.csv"

logger = getLogger(__name__)

def read_csv(path):
    logger.debug("enter")
    df = pd.read_csv(path)
    logger.debug("exit")
    return df

def load_train_data():
    logger.debug("enter")
    df = read_csv(TRAIN_DATA)
    logger.debug("exit")
    return df

def load_test_data():
    logger.debug("enter")
    df = read_csv(TEST_DATA)
    logger.debug("exit")
    return df


# In[ ]:


print(load_train_data().head())
print(load_test_data().head())


# In[ ]:


from sklearn.linear_model import LogisticRegression

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)

DIR = "./"

if __name__ == "__main__":
    log_fmt = Formatter("%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ")
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + "train.py.log", "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)

    logger.info("start")
    
    df = load_train_data()
    
    x_train = df.drop("target", axis=1)
    y_train = df["target"].values
    
    use_cols = x_train.columns.values
    
    logger.info("tarin columns: {} {}".format(use_cols.shape, use_cols))
    
    logger.info("data preparation end {}".format(x_train.shape))
    
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    
    logger.info("train end")
    
    df = load_test_data()
    
    x_test = df[use_cols].sort_values("id")
    
    logger.info("test data load end {}".format(x_test.shape))
    pred_test = clf.predict_proba(x_test)
    
    df_submit = pd.read_csv(SUMPLE_SUBMIT_FILE)
    df_submit["target"] = pred_test
    
    df_submit.to_csv(DIR + "submit.csv", index=False)


# In[ ]:


less ./submit.csv

