#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# Bibliotecas gráficas
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Carregar os dados
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

df.shape, test.shape


# In[4]:


test.info()


# In[5]:


# Fazendo um cópia do dataframe
df_raw = df.copy()


# In[6]:


df.head()


# In[7]:


# Concatenar os dataframes
df = df.append(test, sort=False)


# In[8]:


df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('%',' ')


# In[9]:


df[df['comissionados_por_servidor'] == '#DIV/0!'].head()


# In[10]:



# Corrigindo a coluna comissionados_por_servidor
df['comissionados_por_servidor'] = np.where(df['comissionados_por_servidor']=='#DIV/0!',0,df['comissionados_por_servidor'])


# In[11]:


df[df['capital'] == 1]


# In[12]:


df[df['pib'] == df['pib'].max()]


# In[13]:


df.info()


# In[14]:


# Corrigindo as colunas com dados faltando
df['densidade_dem'].fillna(-1,inplace=True) 
df['participacao_transf_receita'].fillna(-1,inplace=True)
df['servidores'].fillna(-1,inplace=True) 
df['perc_pop_econ_ativa'].fillna(-1,inplace=True) 
df['gasto_pc_saude'].fillna(-1,inplace=True)  
df['hab_p_medico'].fillna(-1,inplace=True)   
df['exp_vida'].fillna(-1,inplace=True) 
df['gasto_pc_educacao'].fillna(-1,inplace=True)
df['exp_anos_estudo'].fillna(-1,inplace=True)


# In[15]:


f,ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)


# In[16]:


plt.figure(figsize=(20,8))
sns.boxplot(x="regiao", y="exp_anos_estudo", data=df, hue='porte')


# In[17]:


plt.figure(figsize=(20,8))
sns.swarmplot(x="regiao", y="exp_anos_estudo", data=df, hue='porte', color=".25")


# In[18]:


plt.figure(figsize=(12,8))
sns.countplot(x='estado', data=df, hue='regiao' )


# In[19]:


plt.figure(figsize=(12,8))
sns.countplot(x='regiao', data=df, hue='porte' )


# In[20]:


# Variáveis contínuas
sns.pairplot(x_vars='exp_vida', y_vars='exp_anos_estudo', data=df, size=7)


# In[21]:


# Dummificar Variáveis 
dm_estado = pd.get_dummies(df['estado'], prefix='es')
df = pd.concat([df, dm_estado], axis = 1)


# In[22]:


dm_regiao = pd.get_dummies(df['regiao'], prefix='r')
df = pd.concat([df, dm_regiao], axis = 1)


# In[23]:


df.columns


# In[24]:


df.head().T


# In[25]:


# Transformando dados categorias em números
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes


# In[26]:


df.info()


# In[27]:


df.head()


# In[28]:


# Importandoa função
from sklearn.model_selection import train_test_split


# In[29]:


# Separando os dataframes
test = df[df['nota_mat'].isnull()]
df = df[~df['nota_mat'].isnull()]


# In[30]:


test.shape, df.shape


# In[31]:


# Separando o dataframe em treino e validação
train, valid = train_test_split(df,random_state=13)


# In[32]:


train.shape, valid.shape


# In[33]:


# Usar o modelo de RandomForest
# Importar o modelo
from sklearn.ensemble import RandomForestClassifier


# In[34]:


# analisar as previsões com base na metrica
# Importar a metrica
from sklearn.metrics import accuracy_score


# In[35]:


train.columns


# In[36]:


# Selecionar as colunas a serem usadas no treinamento e validação
# Lista das colunas não usadas
removed_cols = ['nota_mat','municipio','codigo_mun','capital',
                'porte','participacao_transf_receita', 'comissionados'
                
                ]
# Separando as colunas a serem usadas no treino
feats = [c for c in train.columns if c not in removed_cols]


# In[37]:


# Instanciar o modelo
rf = RandomForestClassifier(random_state=13, n_jobs=-1, n_estimators=150, min_samples_split=3)


# In[38]:


# Treinar o modelo
rf.fit(train[feats],train['nota_mat'])


# In[39]:


# Fazer as previsões
preds = rf.predict(valid[feats])


# In[40]:


# Validar as previsões
accuracy_score(valid['nota_mat'],preds)


# In[41]:


def cv(df, test, feats, y_name, k=5):
        score, preds, fis = [], [], []
        chunk = df.shape[0] // k
        
        for i in range(k):
            if i+1 < k:
                valid = df.iloc[i*chunk: (i+1)*chunk]
                train = df.iloc[:i*chunk].append(df.iloc[(i+1)*chunk:])
            else:
                valid: df.iloc[i*chunk:]
                train: df.iloc[:i*chunk]
                    
            rf = RandomForestClassifier(random_state=13, n_jobs=-1, n_estimators=150, min_samples_split=3)
            rf.fit(train[feats],train[y_name])
            
            score.append(accuracy_score(valid[y_name],rf.predict(valid[feats])))
            
            preds.append(rf.predict(test[feats]))
            
            fis.append(rf.feature_importances_)
            
            print(i, 'OK')
        return score, preds, fis


# In[42]:


train.columns


# In[43]:


# Selecionar as colunas a serem usadas no treinamento e validação
# Lista das colunas não usadas
removed_cols = ['nota_mat' 
                ,'codigo_mun'
                ,'municipio'
                , 'porte'
                ,'capital'
                ,'participacao_transf_receita' 
                ,'comissionados'
                
                                 
               ]
# Separando as colunas a serem usadas no treino
feats = [c for c in train.columns if c not in removed_cols]


# In[44]:


score, preds, fis = cv(df, test, feats, 'nota_mat')


# In[45]:


pd.Series(score).mean()


# In[46]:


test['nota_mat'] = pd.DataFrame(preds).mean()


# In[47]:


test['nota_mat'] = np.where(test['nota_mat']>=0.5,1.0,0.0)


# In[48]:


test[['codigo_mun','nota_mat']].to_csv('rf.csv', index=False)


# In[49]:


plt.figure(figsize=(12,8))
pd.DataFrame(fis, columns=feats).mean().sort_values().plot.barh()


# In[50]:




