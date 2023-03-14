#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# creating train and test data from csv files
train_data = pd.read_csv('../input/santander-customer-satisfaction/train.csv')
test_data = pd.read_csv('../input/santander-customer-satisfaction/test.csv')


# In[3]:


# checking the shape of the train dataset
print(train_data.shape,test_data.shape)


# In[4]:


# checking the 5 values from top in the dataset 
train_data.head()


# In[5]:


# It will describe the dataset by giving mean,standard deviation, min,max, top 25,50,75 percent values for each column
train_data.describe()


# In[6]:


# It will get column and its data type
train_data.dtypes


# In[7]:


train_data.info()


# In[8]:


# To check the distribution of data in the dataset
plt.figure(figsize = (10, 8))
sns.countplot(x = 'TARGET', data = train_data) # from the ouput we can say that it is an unbalanced data 
plt.show()
print('Percentage of happy customers: ',len(train_data[train_data['TARGET']==0])/len(train_data['TARGET'])*100,"%")
print('Percentage of unhappy customers: ',len(train_data[train_data['TARGET']==1])/len(train_data['TARGET'])*100,"%")


# In[9]:


train_data.isnull().sum()


# In[10]:


train_data['var3'].describe()


# In[11]:


train_data['var3'].hist(bins=14)


# In[12]:


train_data['var15'].describe()


# In[13]:


train_data['var15'].hist(by=train_data['TARGET'],bins=50)
plt.xlabel('age')
plt.ylabel('no of customers')


# In[14]:


sns.FacetGrid(train_data, hue="TARGET", size=6)    .map(plt.hist, "var15",edgecolor='w')    .add_legend()
plt.xlabel("var15(Age)")
plt.ylabel("No of Customers")
plt.title('var 15 impact on Target value')
plt.show()


# In[15]:


train_data['var38'].describe()


# In[16]:


train_data['var38'].hist(by=train_data['TARGET'],bins=25)


# In[17]:


import plotly.express as px
from matplotlib import rcParams
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[18]:


fig = px.bar(train_data, x="var15", y="var38", color="TARGET", barmode="group",labels={0,1},log_y=True,color_discrete_sequence=["green","red"])
fig.show()


# In[19]:


unhappy = train_data.loc[train_data['TARGET']==1]["var38"]
happy = train_data.loc[train_data['TARGET']==0]["var38"]
hist_data=[unhappy,happy]
fig = ff.create_distplot(hist_data,group_labels=['unhappy','happy'],show_hist=False,show_rug=False)
fig['layout'].update(title='Santander Customer Satisfaction Time Density Plot',xaxis=dict(title='amount',range=[5000,2000000]))
iplot(fig,filename='dist_only')


# In[20]:


# Removing the data with constant features (i.e zero variance where std=0)
constant_features = [
    features for features in train_data.columns if train_data[features].std() == 0
]
len(constant_features)
train_data[constant_features]


# In[21]:


# Drop the constant features from traina and test dataset
train_data.drop(labels = constant_features, axis = 1, inplace=True)
test_data.drop(labels = constant_features, axis = 1, inplace = True)


# In[22]:


# checking the shape of the dataset after dropping the constant features
print(train_data.shape,test_data.shape)


# In[23]:


# Find the columns where most of the values are equal( approx to 99.9% )
approx_constants = []
for feature in train_data.columns:
    approx_value = (train_data[feature].value_counts()/ np.float(
        len(train_data))).sort_values(ascending=False).values[0]
    if approx_value > 0.999:
      approx_constants.append(feature)
len(approx_constants)
train_data[approx_constants]


# In[24]:


# Temporary dataframe for approx_constants
train_data_ac=train_data.copy()
test_data_ac=test_data.copy()


# In[25]:


# Drop the approximate constant features from traina and test dataset
train_data_ac.drop(labels = approx_constants, axis = 1, inplace=True)
test_data_ac.drop(labels = approx_constants, axis = 1, inplace = True)


# In[26]:


print(train_data.shape,train_data_ac.shape,test_data.shape,test_data_ac.shape)


# In[27]:


# Remove duplicate data, columns having same values
duplicate_features=[]
# Applying modified sort algorithm , instead of sorting we are creating a features list which mets condition, ignoring other columns
for i in range(0, len(train_data.columns)):
    col_1 = train_data.columns[i]
    for col_2 in train_data.columns[i + 1:]:
        if train_data[col_1].equals(train_data[col_2]):
            duplicate_features.append(col_2)
len(duplicate_features)
train_data[duplicate_features]


# In[28]:


# Dropping the duplicate features from the dataset as their contribute towards prediction of target is neligible 
train_data.drop(labels = duplicate_features, axis = 1, inplace=True)
test_data.drop(labels = duplicate_features, axis = 1, inplace = True)


# In[29]:


# Dropping the duplicate features from the dataset as their contribute towards prediction of target is neligible 
for feature in duplicate_features:
  if feature in train_data_ac and feature in test_data_ac:
    train_data_ac.drop(labels = feature, axis = 1, inplace=True)
    test_data_ac.drop(labels = feature, axis = 1, inplace = True)


# In[30]:



print(train_data.shape,test_data.shape,train_data_ac.shape,test_data_ac.shape)


# In[31]:


from sklearn.feature_selection import VarianceThreshold
def features_wo_low_variance(data):
  threshold_n=0.98  
  sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
  sel_var=sel.fit_transform(data)
  return data.columns[sel.get_support(indices=True)] 


# In[32]:


(train_data.var() < 0.02).value_counts()


# In[33]:


train_data_hv = train_data[features_wo_low_variance(train_data)].copy()
test_data_hv = test_data[features_wo_low_variance(train_data)[:-1]].copy()


# In[34]:


print(train_data_hv.shape,test_data_hv.shape)


# In[35]:


# Dataset without accurate constants
train_data_ac_hv = train_data_ac[features_wo_low_variance(train_data_ac)].copy()
test_data_ac_hv = test_data_ac[features_wo_low_variance(train_data_ac)[:-1]].copy()


# In[36]:


print(train_data_ac_hv.shape,test_data_ac_hv.shape)


# In[37]:


def find_correlated_features(data):
  # Create correlation matrix
  corr_matrix = data.corr().abs()

  # Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Find features with correlation greater than 0.95
  to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

  print(len(to_drop))
  return to_drop


# In[38]:


plt.figure(figsize = (20,25))
sns.heatmap(train_data_ac_hv.corr())


# In[39]:


# Drop features having correlation greater than 0.95
train_data_hv_co= train_data_hv.drop(find_correlated_features(train_data_hv), axis=1)
test_data_hv_co= test_data_hv.drop(find_correlated_features(train_data_hv), axis=1)


# In[40]:


# shape of the train_data after dropping high correlated features
print(train_data_hv_co.shape,test_data_hv_co.shape)


# In[41]:


# Drop features having correlation greater than 0.95
train_data_ac_hv_co= train_data_ac_hv.drop(find_correlated_features(train_data_ac_hv), axis=1)
test_data_ac_hv_co= test_data_ac_hv.drop(find_correlated_features(train_data_ac_hv), axis=1)


# In[42]:


# shape of the train_data after dropping high correlated features
print(train_data_ac_hv_co.shape,test_data_ac_hv_co.shape)


# In[43]:


for feature in train_data[train_data_ac_hv_co.var().sort_values(ascending=False).index[0:10]]:
  plt.figure(figsize = (12, 8))
  data = train_data.copy()
  if 0 in data[feature].unique():
    pass
  else:
    data[feature]=np.log(data[feature])
    sns.boxplot(y = feature, x = 'TARGET', data = data)
    plt.ylabel(feature)
    plt.title(feature)
    plt.yticks()
    plt.show()


# In[44]:


# Calculating IQR
Q1 = train_data_ac_hv_co.quantile(0.25)
Q3 = train_data_ac_hv_co.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[45]:


# Finding outliers in the dataset
print(train_data_ac_hv_co < (Q1 - 1.5 * IQR)) or (train_data_ac_hv_co > (Q3 + 1.5 * IQR))


# In[46]:


# Removing the outliers
train_data_ol=train_data_ac_hv_co.copy()
train_data_out = train_data_ol[((train_data_ol >= (Q1 - 1.5 * IQR)) & (train_data_ol <= (Q3 + 1.5 * IQR))).all(axis=1)]


# In[47]:


train_data_out.shape


# In[48]:


train_data_out["TARGET"].value_counts()


# In[49]:


# Replacing value "-999999" in var3 column with most occuring value(75%) 2
train_data.var3 = train_data.var3.replace(-999999,2)


# In[50]:


test_data.var3 = test_data.var3.replace(-999999,2)


# In[51]:


test_data_hv_co.var3 = test_data_hv_co.var3.replace(-999999,2)
train_data_hv_co.var3 = train_data_hv_co.var3.replace(-999999,2)


# In[52]:


test_data_ac.var3 = test_data_ac.var3.replace(-999999,2)
train_data_ac.var3 = train_data_ac.var3.replace(-999999,2)


# In[53]:


test_data_ac_hv.var3 = test_data_ac_hv.var3.replace(-999999,2)
train_data_ac_hv.var3 = train_data_ac_hv.var3.replace(-999999,2)


# In[54]:


test_data_ac_hv_co.var3 = test_data_ac_hv_co.var3.replace(-999999,2)
train_data_ac_hv_co.var3 = train_data_ac_hv_co.var3.replace(-999999,2)


# In[55]:


# Removing Target and ID columns to scale the data across all columns between -1 to 1
train_data_scaled2= train_data_hv_co.drop(["ID","TARGET"],axis=1)
train_data_scaled1= train_data_ac_hv_co.drop(["ID","TARGET"],axis=1)
train_data_scaled= train_data.drop(["ID","TARGET"],axis=1)


# In[56]:


print(train_data_scaled.shape,train_data_scaled1.shape,train_data_scaled2.shape)


# In[57]:


from sklearn.decomposition import PCA
def find_pca_components(data):
  pca = PCA().fit(data)
  plt.rcParams["figure.figsize"] = (12,6)

  fig, ax = plt.subplots()
  xi = np.arange(1, data.shape[1]+1, step=1)
  y = np.cumsum(pca.explained_variance_ratio_)

  plt.ylim(0.0,1.1)
  plt.plot(xi/2, y, marker='o', linestyle='--', color='b')

  plt.xlabel('Number of Components')
  plt.xticks(np.arange(0, data.shape[1]/2, step=2)) #change from 0-based array index to 1-based human-readable label
  plt.ylabel('Cumulative variance (%)')
  plt.title('The number of components needed to explain variance')
  plt.axhline(y=0.98, color='r', linestyle='-')
  plt.text(0.7, 0.85, '98% cut-off threshold', color = 'red', fontsize=16)
  print("Pca component prediciton:")
  ax.grid(axis='x')
  plt.show()


# In[58]:


from sklearn.preprocessing import StandardScaler
for data in [train_data_scaled,train_data_scaled1,train_data_scaled2]:
  scaler = StandardScaler()
  find_pca_components(scaler.fit_transform(data))


# In[59]:


def pca_analysis(n_co,data):
  pca = PCA(n_components=n_co)
  data_transformed = pca.fit_transform(data)
  print("PCA Analysis for %s pca components"%(n_co))
  print("Eigen vector for each principal component : ",pca.components_)
  print("Amount of variance by each PCA : ", pca.explained_variance_)
  print("Percentage of variance by each PCA : ", pca.explained_variance_ratio_)
  print("number of features in training data : ", pca.n_features_)
  print("number of samples in training data: ", pca.n_samples_)
  print("noise variance of the data : ",pca.noise_variance_)
  return pd.DataFrame(data_transformed)


# In[60]:


data_transformed=[]
for no, data in [(48,train_data_scaled),(44,train_data_scaled1),(60,train_data_scaled2)]:
  scaler = StandardScaler()
  data_transformed.append(pca_analysis(no,scaler.fit_transform(data)))


# In[61]:


from sklearn.utils import resample
def upsampling_dataset(data):
  data_majority=data[data.TARGET==0] 
  data_minority=data[data.TARGET==1]  

  data_minority_upsampled=resample(data_minority,replace=True,n_samples=73012)
  data_upsampled=pd.concat([data_minority_upsampled,data_majority])

  data_upsampled.info()
  print(data_upsampled['TARGET'].value_counts())
  return data_upsampled


# In[62]:


datasets = [train_data_ac,train_data_ac_hv_co,train_data_hv_co]


# In[63]:


test_datasets = [test_data_ac,test_data_ac_hv_co,test_data_hv_co]


# In[64]:


upsampled_data = []
for data in datasets:
  upsampled_data.append(upsampling_dataset(data))


# In[65]:


if 'ID' not in data_transformed[2]:
    data_transformed[2].insert(1,'ID',train_data['ID'])
x_pca = data_transformed[2]
y_pca = train_data["TARGET"]


# In[66]:


x_upsample = upsampled_data[2].drop("TARGET",axis=1)
y_upsample = upsampled_data[2]["TARGET"]


# In[67]:


x = datasets[2].drop("TARGET",axis=1)
y = datasets[2]["TARGET"]


# In[68]:


print(x_pca.shape,y_pca.shape,x_upsample.shape,y_upsample.shape,x.shape,y.shape)


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=44)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(x_pca, y_pca, test_size=0.30, random_state=44)
X_upsample_train, X_upsample_test, y_upsample_train, y_upsample_test = train_test_split(x_upsample, y_upsample, test_size=0.30, random_state=44)


# In[70]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[71]:


sel_ = SelectFromModel(RandomForestClassifier(n_estimators=200))
sel_.fit(X_train.fillna(0), y_train)


# In[72]:


selected_features = X_train.columns[(sel_.get_support())]
len(selected_features)


# In[73]:


train_data[selected_features]


# In[74]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,  BaggingClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


# In[75]:


# Model preparation
models = []
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('Bagging Classifier',BaggingClassifier()))
models.append(('KNN', KNeighborsClassifier(weights='distance')))
models.append(('RandomForest', RandomForestClassifier(class_weight='balanced')))
models.append(('DecisionTree', DecisionTreeClassifier(class_weight='balanced')))
models.append(('GradientBoosting', GradientBoostingClassifier()))
models.append(('xgb', XGBClassifier(missing=np.nan, max_depth=6, 
n_estimators=350, learning_rate=0.025, nthread=4, subsample=0.95,
colsample_bytree=0.85, seed=4242)))


# In[76]:


from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import f1_score,roc_curve,roc_auc_score,precision_score,recall_score,accuracy_score


# In[77]:


def model_comparison_plot(model_metrics):
  plt.figure(figsize = (12,4))
  sns.heatmap(model_metrics, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"),linewidth=2)
  plt.title('Metrics comparison for diff models')
  plt.show()


# In[78]:


def plot_roc_curve(y_test, prob_dict):
  sns.set_style('whitegrid')
  plt.figure()
  i=0
  fig, ax = plt.subplots(4,2,figsize=(16,30))
  for key,prob in prob_dict.items():
    fpr, tpr, thresholds = metrics.roc_curve( y_test, prob,
                                                  drop_intermediate = False )
    roc_auc = metrics.roc_auc_score( y_test, prob)
    i+= 1
    plt.subplot(4,2,i)
    plt.plot( fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.axis('tight')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(key)
  plt.show()


# In[79]:


def model_analysis(title,x_train,y_train,x_test,y_test):
  df_scores=pd.DataFrame()
  pred_dict={}
  for name,model in models:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    pred_dict[name] = y_pred
    confusion = confusion_matrix(y_test,y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    error = 1-accuracy
    sensitivity = TP / float(FN + TP)
    specificity = TN / (TN + FP)
    False_positive_rate = 1-specificity
    precision = TP / float(TP + FP)
    bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    Null_accuracy = max(y_test.mean(), (1 - y_test.mean()))
    f1 = metrics.f1_score(y_test,y_pred)
    auc_score = metrics.roc_auc_score(y_test,y_pred)
    clf_score = pd.DataFrame(
        {name: [accuracy, bal_acc, Null_accuracy,precision,sensitivity,f1,error,specificity,auc_score]},
        index=['Accuracy', 'Balanced accuracy','Null_accuracy','precision','recall','f1 score','error','specificity','auc_score']
    )
   
    df_scores = pd.concat([df_scores, clf_score], axis=1).round(decimals=3)
  print("Roc_curve for all models")
  plot_roc_curve(y_test,pred_dict)
  print(title,end='\n\n')
  print(df_scores.to_markdown(),end='\n\n')
  model_comparison_plot(df_scores)


# In[80]:


model_analysis("Model with normal data",X_train,y_train,X_test,y_test)


# In[81]:


model_analysis("Model with upsampled data",X_upsample_train,y_upsample_train,X_upsample_test,y_upsample_test)


# In[82]:


model_analysis("Model with pca data",X_pca_train,y_pca_train,X_pca_test,y_pca_test)


# In[83]:


final_model = RandomForestClassifier(class_weight='balanced',random_state=42)


# In[84]:


final_model.fit(X_upsample_train,y_upsample_train)


# In[85]:


probs = final_model.predict_proba(test_data_hv_co)


# In[86]:


submission = pd.DataFrame({"ID":test_data_hv_co.ID, "TARGET": probs[:,1]})


# In[87]:


submission.to_csv("santander_solution.csv")


# In[88]:


submission


# In[89]:


from sklearn.calibration import CalibratedClassifierCV
xgb_classifier = XGBClassifier(missing=np.nan, max_depth=6, 
n_estimators=350, learning_rate=0.025, nthread=4, subsample=0.95,
colsample_bytree=0.85, seed=4242)
xgb_mdl = CalibratedClassifierCV(xgb_classifier, method='isotonic', cv=10)
xgb_mdl.fit(X_upsample_train,y_upsample_train)


# In[90]:


probs_xgb = xgb_mdl.predict_proba(test_data_hv_co)


# In[91]:


submission1 = pd.DataFrame({"ID":test_data_hv_co.ID, "TARGET": probs_xgb[:,1]})


# In[92]:


submission1.to_csv("submission_xgb.csv",index=False)


# In[93]:


submission1.to_csv("/kaggle/working/submission.csv",index=False)


# In[ ]:




