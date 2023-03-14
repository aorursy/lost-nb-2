%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from ggplot import *
# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv",low_memory = False)
data.head()

## Columns on the input dataset
print(data.columns.values)
types_cols = [(col,str(type(data[col][0]))) for col in data.columns.values]
## Column variable types:
print("Data types of the Column Variables: ")
print(types_cols) 

type(data['count'])

p = ggplot(data,aes(x = 'temp',y = 'count')) + geom_point() + facet_wrap('weather')

print(p)

# Make a copy of the data to do some analysis.
plotdata = data.copy()
plotdata = plotdata[plotdata['weather'] != 4]

import math
plotdata['log_count'] = np.log(plotdata['count']) + 1
ggplot(plotdata,aes(x = 'humidity',y = 'log_count')) + geom_point(color = 'blue') + facet_wrap('weather')

ggplot(plotdata,aes(x = 'temp',y = 'log_count')) + geom_point(color = 'blue') + facet_wrap('weather')

plotdata['datetime'] = pd.to_datetime(plotdata['datetime'])

plotdata['season'] = plotdata['season'].astype('category')

p1 = ggplot(plotdata,aes(x = 'count',y = 'season')) + geom_point(color = 'green') + geom_boxplot()
print(p1)

plotdata['month'] = plotdata['datetime'].apply(lambda x:x.month)

print(plotdata.head())

plot1 = ggplot(plotdata,aes(x = '',y = 'casual')) + geom_bar(fill = 'green')
#print(plot1)
print(plot1 + geom_bar(aes(x = 'month',y = 'registered'),fill = 'blue'))

print(plot1)

ggplot(plotdata,aes(x = 'datetime',y = 'casual')) + geom_point(color = "blue") + stat_smooth(color = "lightblue")
+ geom_point(aes(x = 'datetime',y = 'registered'),color = "darkgreen") + stat_smooth(color = "lightgreen") +
ylab("Bike Riders") + xlab("DateValue")






