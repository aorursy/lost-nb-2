#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from seaborn import countplot,lineplot, barplot
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Load the datasets
train_df = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
test_df = pd.read_csv('../input/X_test.csv')


# In[3]:


train_df.describe()


# In[4]:


test_df.describe()


# In[5]:


y_train.describe()


# In[6]:


# Check for missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())
print(y_train.isnull().sum())


# In[7]:


# Count of target classes 
sns.set(style='whitegrid')
sns.countplot(y = 'surface',
              data = y_train,
              order = y_train['surface'].value_counts().index)
plt.show()


# In[8]:


# Count of classes based on group id 
plt.figure(figsize=(23,5)) 
sns.despine()
countplot(x="group_id", data=y_train, order = y_train['group_id'].value_counts().index)
plt.show()


# In[9]:


series1 = train_df.head(128)
series1.head()


# In[10]:


# Plots of series 1
plt.figure(figsize=(26, 16))
for i, col in enumerate(series1.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(series1[col])
    plt.title(col)


# In[11]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(train_df.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap="YlGnBu")


# In[12]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(test_df.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap="YlGnBu")


# In[13]:


def plot_feature_distribution(classes,target, features,a=5,b=2):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(a,b,figsize=(16,24))

    for f in features:
        i += 1
        plt.subplot(a,b,i)
        for c in classes:
            ttc = target[target['surface']==c]
            sns.kdeplot(ttc[f], bw=0.5,label=c)
        plt.xlabel(f, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();


# In[14]:


classes = (y_train['surface'].value_counts()).index
target = train_df.merge(y_train, on='series_id', how='inner')
features = train_df.columns.values[3:]
plot_feature_distribution(classes, target, features)


# In[15]:


# Conversion of quaternion to Euler angles
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


# In[16]:


def fe_step0 (actual):
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual


# In[17]:


train_df = fe_step0(train_df)
test_df = fe_step0(test_df)
train_df.head()


# In[18]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))

ax1.set_title('quaternion X')
sns.kdeplot(train_df['norm_X'], ax=ax1, label="train")
sns.kdeplot(test_df['norm_X'], ax=ax1, label="test")

ax2.set_title('quaternion Y')
sns.kdeplot(train_df['norm_Y'], ax=ax2, label="train")
sns.kdeplot(test_df['norm_Y'], ax=ax2, label="test")

ax3.set_title('quaternion Z')
sns.kdeplot(train_df['norm_Z'], ax=ax3, label="train")
sns.kdeplot(test_df['norm_Z'], ax=ax3, label="test")

ax4.set_title('quaternion W')
sns.kdeplot(train_df['norm_W'], ax=ax4, label="train")
sns.kdeplot(test_df['norm_W'], ax=ax4, label="test")

plt.show()


# In[19]:


# Quaternions to Euler Angles
def fe_step1 (actual):    
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual


# In[20]:


train_df = fe_step1(train_df)
test_df = fe_step1(test_df)
train_df.head()


# In[21]:


# Feature engineering
def feat_eng(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


# In[22]:


train_df = feat_eng(train_df)
test_df = feat_eng(test_df)


# In[23]:


# Fill the missing values
train_df.fillna(0,inplace=True)
test_df.fillna(0,inplace=True)
train_df.replace(-np.inf,0,inplace=True)
train_df.replace(np.inf,0,inplace=True)
test_df.replace(-np.inf,0,inplace=True)
test_df.replace(np.inf,0,inplace=True)


# In[24]:


# Label encoding
y_train['surface'] = le.fit_transform(y_train['surface'])


# In[25]:


y_train


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(train_df, y_train["surface"], test_size = 0.3, random_state = 42, stratify = y_train["surface"], shuffle = True)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[27]:


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True,                  print_cm=True, cm_cmap=plt.cm.Greens):
    
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    print(X_train.shape, X_test.shape)
    y_pred = model.predict(X_test)
    # prediction = model.predict(test_data)
    #y_pred = np.argmax(y_pred, axis=1)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
   
    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    
    
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    #plt.figure(figsize=(8,8))
    #plt.grid(b=False)
    #plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    #plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results
    


# In[29]:


def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('--------------------------')
    print('|     Best parameters     |')
    print('--------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


    #  number of cross validation splits
    print('---------------------------------')
    print('|   No of CrossValidation sets   |')
    print('--------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))


# In[30]:


labels = ['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc', 'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood']


# In[31]:


# Select the best parameters for logistic regression

parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
log_reg = linear_model.LogisticRegression(penalty = 'l2', C = 20,random_state = 0)
# log_reg_grid = GridSearchCV(log_reg, param_grid = parameters, cv = 5, verbose = 1, n_jobs = -1)
log_reg_grid_results =  perform_model(log_reg, X_train, y_train, X_test, y_test, class_labels = labels)


# In[32]:


rfc=RandomForestClassifier(n_estimators=200, max_depth=5,
                             random_state=0)
rfc_results =  perform_model(rfc, X_train, y_train, X_test, y_test, class_labels = labels)

