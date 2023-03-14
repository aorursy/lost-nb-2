#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[2]:


data_o=pd.read_csv("../input/dataset.csv")
data=data_o


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data


# In[6]:


data.dtypes


# In[7]:


# from sklearn.ensemble import ExtraTreesClassifier
# X=data_o.drop(['id','Class'],axis=1)[:175]
# y=data_o['Class'][:175]
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()


# In[8]:


data['Account1'].unique()


# In[9]:


x=float('nan')
data.replace('?',x,inplace=True)


# In[10]:


datar['Account1'].unique()


# In[11]:


for col in data.columns:
    data[col].fillna(data[col].mode()[0],inplace=True)


# In[12]:


data['Account1'].unique()


# In[13]:





# In[13]:


data = pd.get_dummies(data, columns=['Account1'], prefix = ['Account1'])
data.head()


# In[14]:


data['History'].unique()


# In[15]:


data = pd.get_dummies(data, columns=['History'], prefix = ['History'])
data.head()


# In[16]:


# data = pd.get_dummies(data, columns=['Motive'], prefix = ['Motive'])
# data.head()


# In[17]:


data=data.drop(['Motive'],axis=1)


# In[18]:


data['Account2'].unique()


# In[19]:


data['Account2'].replace({'Sacc4':'sacc4'},inplace=True);


# In[20]:


data = pd.get_dummies(data, columns=['Account2'], prefix = ['Account2'])
data.head()


# In[21]:


data['Employment Period'].unique()


# In[22]:


data = pd.get_dummies(data, columns=['Employment Period'], prefix = ['Emp Per'])
data.head()


# In[23]:


data['Gender&Type'].unique()


# In[24]:


data = pd.get_dummies(data, columns=['Gender&Type'], prefix = ['G&T'])
data.head()


# In[25]:


data['Sponsors'].unique()


# In[26]:


data['Sponsors'].replace({'g1':'G1'},inplace=True);


# In[27]:


data['Sponsors'].unique()


# In[28]:


data = pd.get_dummies(data, columns=['Sponsors'], prefix = ['Sponsors'])
data.head()


# In[29]:


data['Plotsize'].unique()


# In[30]:


data['Plotsize'].replace({
    'sm':0,
    'SM':0,
    'me':1,
    'ME':1,
    'M.E.':1,
    'la':2,
    'LA':2,
    'XL':3
},inplace=True);


# In[31]:


data.head()


# In[32]:


data['Housing'].unique()


# In[33]:


data = pd.get_dummies(data, columns=['Housing'], prefix = ['Housing'])
data.head()


# In[34]:


data['Plan'].unique()


# In[35]:


data = pd.get_dummies(data, columns=['Plan'], prefix = ['Plan'])
data.head()


# In[36]:


data['Post'].unique()


# In[37]:


data = pd.get_dummies(data, columns=['Post'], prefix = ['Post'])
data.head()


# In[38]:


# data = pd.get_dummies(data, columns=['Expatriate'], prefix = ['Expatriate'])
# data.head()


# In[39]:


# data = pd.get_dummies(data, columns=['Phone'], prefix = ['Phone'])
# data.head()


# In[40]:





# In[40]:


data['InstallmentCredit']=pd.to_numeric(data['InstallmentCredit'],errors='coerce')


# In[41]:


data['Yearly Period']=pd.to_numeric(data['Yearly Period'],errors='coerce')


# In[42]:


data.dtypes


# In[43]:


data['Monthly Period']=pd.to_numeric(data['Monthly Period'],errors='coerce')
data['Credit1']=pd.to_numeric(data['Credit1'],errors='coerce')
data['InstallmentRate']=pd.to_numeric(data['InstallmentRate'],errors='coerce')
data['Tenancy Period']=pd.to_numeric(data['Tenancy Period'],errors='coerce')
data['Age']=pd.to_numeric(data['Age'],errors='coerce')


# In[44]:


data.dtypes


# In[45]:


d1=data.drop(['#Credits','#Authorities','Expatriate','Phone'],axis=1)


# In[46]:


d1=d1.drop(['id','Class'],axis=1)
len(d1)


# In[47]:


# from sklearn.ensemble import ExtraTreesClassifier
# X=d1[:175]
# y=data_o['Class'][:175]
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(26).plot(kind='barh')
# plt.show()


# In[48]:


d1.dtypes


# In[49]:


# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features


# In[50]:


# dx = pd.DataFrame()
# for x in featureScores.nlargest(10, 'Score')['Specs']:
#     dx[x] = data[x]
# dx


# In[51]:


# feat_importances.nlargest(24)


# In[52]:


do = pd.DataFrame()
for x in ['Post_Jb2','InstallmentCredit', 'Tenancy Period', 'Sponsors_G3', 'Sponsors_G1', 'Account2_sacc5', 'G&T_M1', 'Emp Per_time5', 'Plan_PL3','InstallmentRate','History_c4','Housing_H3', 'Age', 'Plotsize', 'History_c3', 'Yearly Period', 'Housing_H2']:
    do[x] = d1[x]
do.head()


# In[53]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 30))
corr = d1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
plt.show()


# In[54]:


d1=d1.drop(['Monthly Period'],axis=1)


# In[55]:


d1=d1.drop(['Credit1'],axis=1)


# In[56]:


d1=d1.drop(['InstallmentCredit'],axis=1)


# In[57]:





# In[57]:


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[58]:


scaler=MinMaxScaler()
scaled_data=scaler.fit(d1).transform(d1)
d1=pd.DataFrame(scaled_data,columns=d1.columns)
d1.tail()


# In[59]:


from sklearn.cluster import KMeans
wcss = []
for i in range(2, 50):
    kmean = KMeans(n_clusters = i, random_state = 42)
    kmean.fit(d1)
    wcss.append(kmean.inertia_)
plt.plot(range(2,50),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[60]:


from sklearn.decomposition import PCA


# In[61]:


pca = PCA().fit(d1)


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()


# In[62]:



pca1 = PCA(n_components=41)
pca1.fit(d1)
T1 = pca1.transform(d1)


# In[63]:


kmean = KMeans(n_clusters = 3, random_state = 42)
kmean.fit(d1)
pred = kmean.predict(d1)


# In[64]:



plt.scatter(T1[:, 0], T1[:, 1], c=pred)


# In[65]:


centroids = kmean.cluster_centers_
centroids = pca1.transform(centroids)
plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)


# In[66]:


plt.show()


# In[67]:


pred


# In[68]:


pred[:200]


# In[69]:


data_o['Class'][:200]


# In[70]:


dat = [x for x in data_o['Class'] if x==2.0]
len(dat)


# In[71]:


from sklearn.cluster import AgglomerativeClustering as AC
aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')
y_aggclus= aggclus.fit_predict(d1)
plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
plt.show()


# In[72]:


y_aggclus[:200]


# In[73]:


final1=pd.DataFrame()


# In[74]:


final2=pd.DataFrame()


# In[75]:


final2['id']=data['id']
final2['Class']=pred


# In[76]:


final1['id']=data['id']
final1['Class']=y_aggclus


# In[77]:


final1[:200]


# In[78]:


final1[175:].to_csv('20150546g.csv',index=False)


# In[79]:


len(final1[175:])


# In[80]:


for i 


# In[81]:


final2[175:].to_csv('20150546.csv',index=False)


# In[82]:


for x in range(0, 175):
    if fina['Class'][x]==0:
        fina['Class'][x]=3
    if fina['Class'][x]==1:
        fina['Class'][x]=4
for x in range(0, 175):
    if fina['Class'][x]==3:
        fina['Class'][x]=1
    if fina['Class'][x]==4:
        fina['Class'][x]=0


# In[83]:


count = 0
for x in range(0, 175):
    if y_aggclus[x] == data_o['Class'][x]:
        count+=1
print(count/175)


# In[84]:


fina2[175:].to_csv('20150546G.csv',index=False)


# In[85]:


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

create_download_link(final1)

