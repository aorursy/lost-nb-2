#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


#Als erstes lesen wir die .arff-Datei ein, die unsere Trainingsdaten enthält
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")


# In[ ]:


import numpy as np
import pandas as pd
df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})

df_data.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X = df_data[["x","y"]].values
Y = df_data["Category"].values
colors = {-1:'red',1:'blue'}

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.2)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_Train)

X_Train = scaler.transform(X_Train)
X_Test = scaler.transform(X_Test)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,01],c=df_data["Category"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()

