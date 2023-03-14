#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#carrega o csv para o dataframe
df = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
total_nrows_train = len(df)


# In[3]:


#a função isnull retorna uma matriz, com true para todos os valores que sao nulos e false para os que nao sao nulos
#aí , nós somamos os valores true por coluna , e com isso conseguimos perceber as colunas que tem mais dados faltantes
df.isnull().sum(axis = 0)/len(df)


# In[4]:


df.isnull().sum(axis = 1)/len(df.columns)


# In[5]:


#aqui estamos verificando se cada row possui pelo menos um valor NaN
#e obtendo a razão de rows que possuem pelo menos um valor NaN
(df.isnull().sum(axis = 1) != 0).sum(axis = 0)/len(df)


# In[6]:


limiteNan = 0.05
dataframe = df

listaColunasRemovidas = []
listaColunasNaoRemovidas = []

for i in df.columns:
    if(dataframe[i].isnull().sum(axis = 0)/len(dataframe) > limiteNan):
        #insere na lista de colunas a serem removidas
        listaColunasRemovidas.insert(0,i)
    else:
        #insere na lista de colunas a não serem removidas
        listaColunasNaoRemovidas.insert(0,i)

dataframe = dataframe.drop(columns = listaColunasRemovidas)

df = dataframe


# In[7]:


#aqui calculamos para todo o dataframe, o percentual de rows que possuem algum valor NaN
df.drop(df[(df.isnull().sum(axis = 1) != 0)].index).count()/len(df)
                                                                                      
#agora que , no nosso dataframe garantimos que temos menos de 5% de rows com algum valor NaN
#podemos excluir as rows que tem algum valor NaN
df.drop(df[(df.isnull().sum(axis = 1) != 0)].index,inplace = True)

print(df)


# In[8]:


(total_nrows_train * 0.95 < len(df))


# In[9]:


#calcula a quantidade de valores nulos na tabela de treino
((df.isnull().sum(axis = 1) != 0).sum(axis = 0) == 0)


# In[10]:


df2 = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

total_nrows_test = len(df2)

df2 = df2.drop(columns = listaColunasRemovidas)

#agora que , no nosso dataframe garantimos que temos menos de 5% de rows com algum valor NaN
#podemos excluir as rows que tem algum valor NaN
df2.drop(df2[(df2.isnull().sum(axis = 1) != 0)].index,inplace = True)


print(df2)


# In[11]:


((df2.isnull().sum(axis = 1) != 0).sum(axis = 0) == 0)


# In[12]:


(total_nrows_train * 0.95 < len(df))


# In[13]:


df.to_csv('trainID.csv')
df2.to_csv('testID.csv')

