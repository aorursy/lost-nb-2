import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from __future__ import division
%matplotlib inline



test = pd.read_csv("../input/test_users.csv")
train = pd.read_csv("../input/train_users_2.csv")
session = pd.read_csv("../input/sessions.csv")
users = pd.concat([test,train],axis=0)
#有缺失值的变量有：
#age, country_destination, date_first_booking,first_affiliate_tracked
users.info()

users.age = [(2014 - x) if x > 160 else x for x in users.age]
users.loc[users.age>100, "age"] = np.nan
users.loc[users.age<12, "age"] = np.nan
#"NA percentage:'
print sum(pd.isnull(users.age))/users.shape[0]*100

sns.distplot(users.age.dropna())

users.country_destination.value_counts().plot.barh()
xlabel('Number of Users')
title('Count of Users by Dest Country')

users.gender.value_counts().plot.barh()

users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['timestamp_first_active'] = pd.to_datetime(test['timestamp_first_active'], format='%Y%m%d%H%M%S')

users.affiliate_channel.value_counts().plot.barh()

users.affiliate_provider.value_counts().plot.barh()

users.first_affiliate_tracked.value_counts().plot.barh()

users.date_account_created.value_counts().sort_index().plot()

bkts = pd.read_csv("../input/age_gender_bkts.csv")
sns.factorplot(x="age_bucket", y="population_in_thousands", hue="gender",col="country_destination",data=bkts,kind="bar",size=6)

session.apply(lambda x: x.nunique(),axis=0)
grpby = session.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']
session.groupby(['action_type'])['user_id'].nunique().reset_index()


