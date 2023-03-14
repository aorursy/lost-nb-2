#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
import operator
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# In[2]:


path_input = "../Data/"
path_output = "../Output/"
datasetname = "dataset.csv"
finalcsv = "Submission.csv"


# In[3]:


data_orig = pd.read_csv(path_input + datasetname)
data = data_orig


# In[4]:


#splitting data into features (X) and target values (y)
y = pd.DataFrame(data['Class'], columns = ['Class'])
X = data.loc[:, data.columns != 'Class']


# In[5]:


#drop the id values from the features
#it is required only for identification purposes
ids = X['id']
X = X.drop(columns=['id'])


# In[6]:


#Handling multiple representations of unique values in columns
X['Sponsors'] = X['Sponsors'].replace('g1', 'G1')
X['Plotsize'] = X['Plotsize'].replace('la', 'LA')
X['Plotsize'] = X['Plotsize'].replace('M.E.', 'ME')
X['Plotsize'] = X['Plotsize'].replace('me', 'ME')
X['Plotsize'] = X['Plotsize'].replace('sm', 'SM')


# In[7]:


#label encoding discrete value features
le = LabelEncoder()
encode_cols = ['Account1', 'History', 'Motive', 'Account2', 'Employment Period', 'InstallmentRate', 'Gender&Type', 'Sponsors', 
               'Plotsize', 'Plan', 'Housing', 'Post', 'Phone', 'Expatriate', 'Tenancy Period', '#Credits', '#Authorities']
for col in encode_cols:
    X[col] = le.fit_transform(X[col].astype(str))


# In[8]:


#Removing nan values
X = X.replace({'?': np.nan})
null_columns = X.columns[X.isna().any()]
mean_cols = ['Monthly Period', 'Credit1', 'Age', 'InstallmentCredit', 'Yearly Period']
mode_cols = ['Account1', 'History', 'Motive', 'Tenancy Period', 'InstallmentRate']
#For continuous values
for col in mean_cols:
    X[col] = (X[col].astype(float))
    X[col] = X[col].fillna((X[col].mean()))
#For discrete values
for col in mode_cols:
    X[col] = X[col].fillna(X[col].mode())


# In[9]:


#Plotting the correlation matrix for the continuous variables
continuous_vars = ['Monthly Period', 'Credit1', 'Age', 'InstallmentCredit', 'Yearly Period']
f, ax = plt.subplots(figsize=(10, 8))
corr = (X[continuous_vars]).corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[10]:


#Storing columns to drop based on high correlation
todrop = ['Monthly Period', 'Credit1', 'InstallmentCredit']


# In[11]:


#Using Gradient Boosting Classifier to find feature importance
gbc = GradientBoostingClassifier()
gbc.fit(X.iloc[:175,:], y.iloc[:175, 0])
fimp = {}
for i, col in enumerate(X.columns):
    fimp[col] = gbc.feature_importances_[i]
#sorting features according to their importance
sfimp = sorted(fimp.items(), key=operator.itemgetter(1),reverse = True)
#storing feature importance in a dataframe
dfimp = pd.DataFrame(sfimp, columns = ['Features', 'Importance'])

#print(dfimp)


# In[12]:


#adding columns to drop based on feature importance
todrop.extend(['InstallmentRate', 'Plan', 'Sponsors', 'Account1', 'Motive', 'Gender&Type',
          'Tenancy Period','#Authorities', 'Account2','Post', '#Credits', 'Employment Period',
          'Expatriate', 'Phone'])


# In[13]:


#Performing one hot encoding on necessary columns
toencode = list(set(X.columns) - set(continuous_vars) - set(todrop))
data1 = pd.get_dummies(X, columns = toencode)
X = data1


# In[14]:


#Adding one hot encoded columns to drop based on if they have less than 5% of ones
for col in list(set(X.columns) - set(continuous_vars) - set(todrop)):
    if sum(X[col]) < 0.05*X.shape[0]:
        todrop.append(col)      


# In[15]:


#Dropping all columns not required
X = X.drop(columns=todrop)


# In[16]:


#Scaling Continuous variable columns
toscale = ['Monthly Period', 'Credit1', 'Age', 'InstallmentCredit', 
           'Yearly Period']
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X[list(set(toscale) - set(todrop))])
X_scaled = pd.DataFrame(np_scaled, columns=list(set(toscale) - set(todrop)))
X = X.drop(columns=list(set(toscale) - set(todrop)))
X = pd.concat((X, X_scaled), axis = 1)


# In[17]:


#pca = PCA()
#pca.fit(X)
#pca.explained_variance_ratio_


# In[18]:


#Performing PCA to plot the clustering
pca1 = PCA(n_components=2)
pca1.fit(X)
T1 = pca1.transform(X)


# In[19]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, accuracy = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Code taken from sklearn documentation
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    if accuracy == True:
        print('Accuracy: {}'.format(((float)(cm[0][0] + cm[1][1] + cm[2][2])/cm.sum())*100))
    return ax


# In[20]:


def runmodel(model, X, y, title, T1):
    '''
    This function runs the clustering algorithm and plots the confusion matrix based on predicted values
    '''
    #predict cluster values
    pred = model.fit(X).labels_
    
    #plot clusters
    plt.title(title)
    fig = plt.scatter(T1[:175,0], T1[:175,1], c=pred[:175])

    #plot confusion matrix
    plot_confusion_matrix(pred[:175], y.iloc[:175,0], [0,1,2], title= title)
    
    return pred


# In[21]:


def replace_calculate(newlabel0, newlabel1, newlabel2, pred,y, title):
    '''
    This function replaces labels in our predicted value with the correct label value. 
    It also calculates accuracy.
    '''
    #replace labels
    replace_labels = {0:newlabel0, 1:newlabel1, 2:newlabel2}
    p = pd.DataFrame(np.array([pred]).reshape(1031,1), columns=['Class'])
    p = p.replace(replace_labels)
    
    #plot new confusion matrix with accuracy
    plot_confusion_matrix(p[:175], y.iloc[:175,0], [0,1,2], title = title, accuracy = True)
    return p


# In[22]:


pred = runmodel(AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage = "ward"), X, y, 'Agglomerative Clustering', T1)


# In[23]:


p = replace_calculate(2,1,0, pred, y, 'Agglomerative Clustering')


# In[24]:


#pred = runmodel(KMeans(n_clusters=3, n_init=30), X, y, 'KMeans', T1)


# In[25]:


#p = replace_calculate(0,1,2, pred, y, 'KMeans')


# In[26]:


#pred = runmodel(DBSCAN(), X, y, 'DBSCAN', T1)


# In[27]:


#pred = runmodel(Birch(n_clusters=3, branching_factor=10, threshold=0.5), X, y, 'Birch', T1)


# In[28]:


final = pd.concat((ids, p), axis = 1)
final = final.iloc[175:,:] 


# In[29]:


#Writing result to Submission.csv in output folder
final.to_csv(path_output + finalcsv, index=False)


# In[30]:


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

create_download_link(<final>)

