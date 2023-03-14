#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data analysis
import pandas as pd
import numpy as np

#visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier


# In[2]:


#read file
family=pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
family


# In[3]:


x=family.iloc[:,0:11].values
x


# In[4]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2  , n_init =10 , random_state =0 )


# In[5]:


kmeans.fit(x)
kmeans.predict(x)


# In[6]:


from sklearn.cluster import KMeans
wss= []
for i in range(1,10) :
    kmeans = KMeans(n_clusters = i, n_init =10 ,random_state =0 )
    kmeans.fit(x)
    wss.append(kmeans.inertia_)  #Inertia: Sum of distances of samples to their closest cluster center 
    print (i, kmeans.inertia_)
    
#find the values where change slowing down .   


# In[7]:


## Plotting the Within Sum of Squares
plt.plot(range(9),wss)
plt.title("The Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.show()


# In[8]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5 , n_init =10 , random_state =0 )
kmeans.fit_predict(x)


# In[9]:


family['assigned_day']=kmeans.fit_predict(x)
family


# In[10]:


family['assigned_day'].unique()


# In[11]:


f1=family.groupby('family_id')['n_people'].sum().sort_values(ascending=False)
f1


# In[12]:


f2=family.groupby('family_id')['assigned_day'].sum().sort_values(ascending=False)
f2


# In[13]:


family.tail()


# In[14]:


family.describe()


# In[15]:


family[['n_people', 'family_id']].groupby(['n_people'], as_index=False).mean().sort_values(by='family_id', ascending=False)


# In[16]:


a = sns.FacetGrid(family, col='n_people')
a.map(plt.hist, 'family_id', bins=30)


# In[17]:


sns.FacetGrid(family, hue="n_people", size=6)    .map(sns.kdeplot, "family_id")    .add_legend()


# In[18]:


sns.violinplot(x="n_people", y="choice_3", data=family, size=8)


# In[19]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="n_people", y="choice_0", data=family)


# In[20]:


g = sns.distplot(family['n_people'], color="b", label="number of days : %.2f"%(family["n_people"].skew()))
g = g.legend(loc="best")


# In[21]:


g = sns.barplot(x="n_people",y="family_id",data=family)
g = g.set_ylabel("santa tour")


# In[22]:


family[["n_people","choice_5"]].groupby('n_people').mean()


# In[23]:


family[["n_people","choice_6"]].groupby('n_people').mean()


# In[24]:


g= sns.factorplot(x="n_people",y="family_id",data=family,kind="bar", size = 6 , 
palette="muted")
g.despine(left=True)
g = g.set_ylabels("number of families")


# In[25]:


g = sns.countplot(x="n_people",data=family)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[26]:


x['clustor']:kmeans.fit_predict(x)
    


# In[27]:


from scipy.fftpack import fft, ifft
x = np.arange(5)
np.allclose(fft(ifft(x)), x)  # within numerical accuracy.


# In[28]:


x=family.iloc[:,1:13]
y=family.iloc[:,12]
y


# In[29]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.10,random_state=0)
train_y


# In[30]:


from sklearn.preprocessing import MinMaxScaler      
mms = MinMaxScaler()
train_x.np = mms.fit_transform (train_x) 
test_x.np = mms.transform (test_x) 


print(train_x.np)

print(test_x.np)


# In[31]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(train_x, train_y)

prediction = gnb.predict(test_x)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_y, prediction))



# In[ ]:




