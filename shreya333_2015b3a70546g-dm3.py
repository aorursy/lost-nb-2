#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np


# In[2]:


df = read_csv("../input/opcode_frequency_benign.csv")


# In[3]:


# df.head()


# In[4]:


df_mal = read_csv("../input/opcode_frequency_malware.csv")


# In[5]:


# df_mal.head()


# In[6]:


# df.drop('FileName', axis=1)
# df_mal.drop('FileName', axis=1)


# In[7]:


# df


# In[8]:


# import pandas as pd
dff = pd.concat([df, df_mal])


# In[9]:


# df.shape[0]
# # df_mal.shape[0]
# # dff.shape[0]


# In[10]:


# dff['2'][0]


# In[11]:


# df_final = dff


# In[12]:


dff = dff.reset_index()


# In[13]:


Y = [0]*1400
Y = Y + [1]*1999


# In[14]:


len(Y)


# In[15]:


dff = dff.drop('FileName', axis=1)


# In[16]:


dff = dff.drop('index', axis=1)


# In[17]:


# dff
X = dff.values


# In[18]:


model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)


# In[19]:


arr = np.array(model.feature_importances_)
inds = np.argsort(arr)
ar = arr
np.sort(ar)


# In[20]:


# arr[0]


# In[21]:


from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier


# In[22]:


# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)


# In[23]:


# arr[926]


# In[24]:


# inds[-404:]


# In[25]:


# dff = df_final
inds_int = inds[-404:]
# inds_int


# In[26]:


# dff_copy = dff


# In[27]:


# dff


# In[28]:


# dff = dff_copy
for x in range(1,1809):
    if x not in inds_int:
        dff = dff.drop(str(x), axis=1)


# In[29]:


# dff


# In[30]:


X = dff.values
# len(X[0])


# In[31]:


# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)


# In[32]:


# print(cv_scores)
# print(np.mean(cv_scores))


# In[33]:


# from sklearn.model_selection import GridSearchCV

# knn2 = KNeighborsClassifier()

# param_grid = {'n_neighbors': np.arange(1, 25)}

# knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# knn_gscv.fit(X, Y)


# In[34]:


# knn_gscv.best_params_


# In[35]:


test_df = read_csv('../input/Test_data.csv')
# test_df
# file = df['FileName']


# In[36]:


test_df = test_df.drop('Unnamed: 1809', axis=1)


# In[37]:


test_df = test_df.drop('FileName', axis=1)


# In[38]:


# test_df


# In[39]:


# %matplotlib inline
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# pca = PCA().fit(dff)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');


# In[40]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier()


# In[41]:


# len(X[0])
ranfor = RandomForestClassifier()
ranfor.fit(X, Y)


# In[42]:


for x in range(1,1809):
    if x not in inds_int:
        test_df = test_df.drop(str(x), axis=1)


# In[43]:


# test_df


# In[44]:


X_val = test_df.values
predicted = ranfor.predict(X_val)


# In[45]:


# predicted[0:18]


# In[46]:


sample = read_csv('../input/sample_submission.csv')
sample['Class'] = predicted
# sample


# In[47]:


sample = sample.set_index('FileName')


# In[48]:


sample.to_csv('submission.csv')


# In[49]:





# In[49]:


# from sklearn.linear_model import LogisticRegression


# In[50]:


# logreg = LogisticRegression()
# cv_scores = cross_val_score(logreg, X, Y, cv=5)


# In[51]:


# print(cv_scores)
# print(np.mean(cv_scores))


# In[52]:


# X = dff_404.values


# In[53]:


# from sklearn.ensemble import RandomForestClassifier
# ranfor = RandomForestClassifier()
# cv_scores = cross_val_score(ranfor, X, Y, cv=5)


# In[54]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html)
create_download_link(sample)

