#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn


# In[ ]:


xTrain = pd.read_csv("../input/train.csv",
           sep=r'\s*,\s*',engine='python'
           )
print(xTrain.head)


# In[ ]:


xTrain = xTrain.drop(columns=["v2a1", "v18q1", "rez_esc"], index =1)


# In[ ]:


xTrain = xTrain.dropna()
xTrain.shape


# In[ ]:


yTrain = xTrain.Target


# In[ ]:


xTrain = xTrain.drop(columns=["Id","Target"])


# In[ ]:


from sklearn import preprocessing


# In[ ]:


xTrain = xTrain.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[ ]:


k_scores = []
k_values = []
k=80
while(k <= 130):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xTrain, yTrain, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    k_values.append(k)
    k = k + 5


# In[ ]:


plt.plot(k_values,k_scores)
plt.xlabel("Numero de vizinhos(k)")
plt.ylabel("Precisão Média por Validação Cruzada")


# In[ ]:


k = 116
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(xTrain, yTrain)


# In[ ]:


xTest = pd.read_csv("../input/test.csv",
           sep=r'\s*,\s*',engine='python')
Pred = pd.DataFrame(columns=["Id","Target"])
Pred["Id"]= xTest.Id


# In[ ]:


xTest = xTest.fillna(xTest.mean())
xTest = xTest.drop(columns=["Id", "v2a1", "v18q1", "rez_esc"])
xTest = xTest.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


yPred = knn.predict(xTest)
Pred["Target"] = yPred


# In[ ]:


Pred.to_csv("sample_submission.csv", index=False)
Pred


# In[ ]:


kaggle competitions submit -c costa-rican-household-poverty-prediction -f submission.csv -m "Message"

