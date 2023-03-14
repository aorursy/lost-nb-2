#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


def corr_plot(dataframe, top_n, target, fig_x, fig_y):
    corrmat = dataframe.corr()
    #top_n - top n correlations +1 since price is included
    top_n = top_n + 1 
    cols = corrmat.nlargest(top_n, target)[target].index
    cm = np.corrcoef(train[cols].values.T)
    return cols,cm
# matrice correlation 

train = pd.read_csv("./input/train.csv")


train2 = pd.read_csv("./input/macro.csv")

train = pd.merge(train, train2, on='timestamp')


del train["timestamp"]
# fusion du train et du macro 

train3 = train


total = train.isnull().sum().sort_values(ascending=False)
 #calculer le total des valeurs manquantes
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
 #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

train = train.drop((missing_data[missing_data['Total'] > 10000]).index,1)
# supprimer les variables avec des valeurs manquantes sup a 10000

train = train.dropna(thresh=train.shape[1])
# supprimer l'observation si une observation est manquante 
print train.shape

dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

# identifier le type des variables

tab = []

for x in range(0,dtype_df.shape[0]):
	if(dtype_df["Column Type"][x] == "object"):
		tab.append(dtype_df["Count"][x])
# nombre de variable de chaque type
for x in range(0,15):
	train[tab[x]] = pd.factorize(train[tab[x]])[0]

# transforme des variables object en variables de type entier , quali en quanti 



corr_20,cm = corr_plot(train, 150, 'price_doc', 10,10)

for y in range(1,25):
	for x in range(y+1,cm.shape[0]):
		if(cm[y][x] > 0.75):
			del train[corr_20[x]]
	corr_20,cm = corr_plot(train, 150, 'price_doc', 10,10)
	print train.shape

corr_20 = corr_20[0:25]

print corr_20

# si deux variable sont corrélé on elimine un des deux 



train3 = train3[corr_20].copy()

# on retient 25 variables

#meme algo sur le test 

test = pd.read_csv("./input/test.csv")

train2 = pd.read_csv("./input/macro.csv")

test = pd.merge(test, train2, on='timestamp')

del test["timestamp"]
print test.shape
test = test[corr_20[1:corr_20.shape[0]]].copy()



total = train3.isnull().sum().sort_values(ascending=False) #calculer le total des valeurs manquantes
percent = (train3.isnull().sum()/train3.isnull().count()).sort_values(ascending=False) #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



train3 = train3.drop((missing_data[missing_data['Total'] > 10000]).index,1)


train3 = train3.dropna(thresh=train3.shape[1])
dtype_df = train3.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

tab = []

for x in range(0,dtype_df.shape[0]):
    if(dtype_df["Column Type"][x] == "object"):
        tab.append(dtype_df["Count"][x])
        #print dtype_df["Count"][x]

for x in range(0,len(tab)):
    train3[tab[x]] = pd.factorize(train3[tab[x]])[0]
    test[tab[x]] = pd.factorize(test[tab[x]])[0]


test  = test.fillna(test.mean())

# test ready 

# traitement des outlyers 
train3 = train3[train3.price_doc > 100]
train3 = train3[train3.price_doc < 0.2e8]
train3 = train3[train3.full_sq < 130]
train3 = train3[train3.full_sq > 4]
train3 = train3[train3.max_floor < 80]




price = train3.price_doc
del train3["price_doc"]
train3 = train3[corr_20[1:corr_20.shape[0]]].copy()



modeleReg=LinearRegression()



modeleReg.fit(train3,price) #effectuer la regression lineaire
y_predicted = modeleReg.predict(test)

id_test = range(30474,38136)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predicted})
output.head()
output.to_csv('submission.csv', index=False)

