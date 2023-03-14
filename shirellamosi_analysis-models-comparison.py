#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from colorama import Fore
import matplotlib.pyplot as plt
import numpy as np
import re
import scikitplot as skplt 

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb

from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


def report(y_true, y_pred, labels):
    f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average='macro', pos_label=1)
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    cm = pd.DataFrame(confusion_matrix(y_true=y_true, 
                                       y_pred=y_pred), 
                      index=labels, 
                      columns=labels)
    rep = classification_report(y_true=y_true, 
                                y_pred=y_pred)
    
    return (f'F1 Macro = {f1:.4f}\n\nAccuracy = {acc:.4f}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n{rep}')


# In[3]:


data = pd.read_csv('../input/train.csv', index_col='idhogar')
data = data.rename(columns = {'v2a1':'rent','hacdor':'crowd_bedroom', 'hacapo': 'crowd_room', 'v14a': 'bathroom' })
print(data.shape)
data.tail(7)


# In[4]:


data.info(verbose=False)


# In[5]:


data.describe()


# In[6]:


data = data.reset_index()
data.Target.value_counts(normalize=True)


# In[7]:


print((data['agesq'] == data['SQBage']).all())
print((data['tamhog'] == data['hhsize']).all())
print((data['hhsize'] == data['hogar_total']).all())
data.drop(columns=['agesq', 'tamhog','hhsize'], inplace=True)
data.shape


# In[8]:


issues_list = ['epared', 'etecho', 'elimbasu', 'eviv', 'tipovivi', 'area'] #good, normal, bad etc.

for issue in issues_list:

    col_list = list(data.columns[data.columns.str.startswith(issue)])
    df = data[col_list]
    
    s = pd.Series(df.columns[np.where(df!=0)[1]])
    s = s.apply(lambda x: x[-1]).astype('int64')
    s = s.rename(issue+'_cat', copy=True)
    
    data = pd.concat([data, s], axis=1)
    data = data.drop(columns=col_list)
data.head()


# In[9]:


data.shape


# In[10]:


#educational level

def instlevel_cat(row):
    col_list = list(data.columns[data.columns.str.startswith('instlevel')])
    
    for i in col_list:
        if row[i] == 1:
            return (int(i[-1]))


# In[11]:


# water provision

def water_cat(row):
    if row.abastaguadentro == 1:
        return 1
    if row.abastaguafuera == 1:
        return 2
    if row.abastaguano == 1:
        return 3

# source of energy used for cooking
def energcocinar_cat(row):
    if row.energcocinar1 == 1:
        return 1
    if row.energcocinar4 == 1:
        return 2
    if row.energcocinar3 == 1:
        return 3
    if row.energcocinar2 == 1:
        return 4


# In[12]:


data['water_cat'] = data.apply(water_cat, axis=1)
data['energcocinar_cat'] = data.apply(energcocinar_cat, axis=1)
data['instlevel_cat'] = data.apply(instlevel_cat, axis=1)

data.shape


# In[13]:


print('nulls: ', data['instlevel_cat'].isnull().sum())


# In[14]:


data[data['instlevel_cat'].isnull()]['escolari']


# In[15]:


data.instlevel_cat.fillna(1, inplace=True)


# In[16]:


import re

columns_list = list(data.columns)
col_drop = []

for i in columns_list:
    if re.match("instlevel\d", i) or re.match("abastagua", i) or re.match("energcocinar\d", i):
        col_drop.append(i)
print(col_drop)


# In[17]:


print('Before:',data.shape)
data.drop(columns=col_drop, inplace=True)
print('After:',data.shape)


# In[18]:


data[['female', 'male']].head(3)


# In[19]:


data = data.rename(columns={'female': 'sex'}).drop(columns='male')
data.shape


# In[20]:


data[['dependency','edjefe','edjefa']].head(5)


# In[21]:


for col in ['dependency','edjefe','edjefa']:
    print(col.capitalize(),':\n', data[col].unique(),'\n')
    


# In[22]:


for col in ['dependency','edjefe','edjefa']:
    data[col].replace({'yes': 1, 'no': 0 }, inplace=True)
    data[col] = data[col].astype('float64')
    if data[col].dtype == 'float64':
        print(col,': fixed')


# In[23]:


columns_list = list(data.columns)
print(len(columns_list))
print(columns_list)


# In[24]:


print('Before:',data.shape)

col_drop = []

for i in columns_list:
    if re.match("SQB", i):
        col_drop.append(i)
print(col_drop)

data.drop(columns=col_drop, inplace=True)

print('After:',data.shape)


# In[25]:


data.isnull().sum().sort_values(ascending=False).head(7)


# In[26]:


data.rez_esc.value_counts()


# In[27]:


ages = list(data[data['rez_esc'].isnull()]['age'].unique())
print(sorted(ages))


# In[28]:


data[['rez_esc', 'age']].head()


# In[29]:


data['rez_esc'] = data['rez_esc'].fillna(0)


# In[30]:


data.isnull().sum().sort_values(ascending=False).head()


# In[31]:


data.v18q1.describe(np.arange(0,1.1,0.2))


# In[32]:


data.v18q1.value_counts()


# In[33]:


data[['v18q1', 'v18q']].head()


# In[34]:


print(data[data['v18q1'].isnull()]['v18q'].sum())
data['v18q1'] = data['v18q1'].fillna(0)


# In[35]:


data.isnull().sum().sort_values(ascending=False).head()


# In[36]:


data[['rent', 'tipovivi_cat']].head() #own the house or no house to pay rent for


# In[37]:


data[data['rent'].isnull()]['tipovivi_cat'].unique()


# In[38]:


data['rent'] = data['rent'].fillna(0)


# In[39]:


data.isnull().sum().sort_values(ascending=False).head()


# In[40]:


meaneduc_df = data.groupby(['idhogar', 'Id'])['age', 'escolari'].sum()
meaneduc_df.head()


# In[41]:


def fix_meaneduc(df): 
    counter=0
    escolari=0
    
    list_age = list(df.age) 
    list_escolari = list(df.escolari) 
    
    for i in range(len(list_age)):
        if (list_age[i] >= 18):
            counter +=1
            escolari += list_escolari[i]
    if counter==0:
        return 0
    return (escolari/counter)
    


# In[42]:


meaneduc_new = data.groupby('idhogar').apply(fix_meaneduc)
meaneduc_new.isnull().any()


# In[43]:


meaneduc_new.head() #series, each row is for idhogar


# In[44]:


print(meaneduc_new.shape) 
print((data.parentesco1==1).sum()) 


# In[45]:


data[data['meaneduc'].isnull()]


# In[46]:


# family example:
print(data.shape)
data[data['idhogar'] == '2b58d945f']


# In[47]:


data = data.drop(columns=['meaneduc','mobilephone', 'v18q'])
data.shape


# In[48]:


# Houseowners Example:
owners_data = data[data['parentesco1']==1]
print(owners_data.shape)
owners_data.head()


# In[49]:


train, test = split(owners_data, test_size=0.3, random_state=44, stratify=owners_data.Target)
train.shape, test.shape


# In[50]:


train.Target.value_counts(normalize=True)


# In[51]:


rf = RandomForestClassifier(class_weight='balanced', n_estimators=300, 
                            max_depth=30, min_samples_leaf=10, max_features=30, random_state=111)

X = train.drop(columns=['Target','Id','idhogar'])
y = train.Target

params = {'max_depth': range(25,30),
          'min_samples_leaf': range(7,10)}

gs = GridSearchCV(rf, params, cv=5, scoring='f1_macro')
gs.fit(X,y)


# In[52]:


model1 = gs.best_estimator_
gs.best_params_


# In[53]:


X_test = test.drop(columns=['Target','Id','idhogar'])
y_test = test.Target

print(Fore.BLUE+report(y_test, model1.predict(X_test), model1.classes_))


# In[54]:


my_list = list(zip(model1.feature_importances_ ,X.columns))
my_list.sort(key=lambda tup: tup[0],reverse=True)
# for item in my_list:
#     if item[0]> 0.01:
#         print(item)


# In[55]:


importances = pd.DataFrame(data=None,columns=['importance', 'feature'])

importance_l = []
feature_l = []

for t in my_list:
    importance_l.append(t[0])
    feature_l.append(t[1])
    
importances['importance'] = importance_l
importances['feature'] = feature_l


# In[56]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(x="feature", y="importance", data=importances)


# In[57]:


owners_data_only = owners_data.copy()


# In[58]:


# adding meaneduc_new
owners_data = owners_data.merge(meaneduc_new.to_frame(), left_on='idhogar', right_index=True)                        .rename(index=str, columns={0: "meaneduc_new"})

owners_data.head()


# In[59]:


print(list(data.columns))


# In[60]:


# Individual Features:

family_data = data[['idhogar', 'Id','escolari', 'rez_esc','dis','instlevel_cat','age',
                           'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                           'estadocivil5', 'estadocivil6', 'estadocivil7', 
                           'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 
                           'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 
                           'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']]
print(owners_data.shape, family_data.shape)
family_data.head()


# In[61]:


family_data[family_data.idhogar == '2b58d945f']


# In[62]:


print(f'There are {(len(family_data.idhogar.unique()))-(len(owners_data.idhogar.unique()))} family members with no related homeowner')


# In[63]:


wrong_idhogars = []
for id_ in list(family_data.idhogar.unique()):
    if id_ not in owners_data.idhogar.unique():
        wrong_idhogars.append(id_)
print(wrong_idhogars)


# In[64]:


print(family_data.shape)
for id_ in wrong_idhogars:
    family_data = family_data[family_data['idhogar'] != id_]
print('Number of family members in the dataset (rows): ',family_data.shape)


# In[65]:


print('Number of families in the dataset: ',owners_data.shape)

owners_data = owners_data.sort_values(by='idhogar')
family_data = family_data.sort_values(by='idhogar')

df = family_data.groupby('idhogar')


# In[66]:


#school min, max and mean per family

owners_data['escolari_max_fam'] = np.array(df.escolari.max())
owners_data['escolari_min_fam'] = np.array(df.escolari.min())
owners_data['escolari_mean_fam'] = np.array(df.escolari.mean())

owners_data.shape


# In[67]:


#age min, max and mean per family

owners_data['age_max_fam'] = np.array(df.age.max())
owners_data['age_min_fam'] = np.array(df.age.min())
owners_data['age_mean_fam'] = np.array(df.age.mean())
owners_data['age_median_fam'] = np.array(df.age.median())

owners_data.shape


# In[68]:


#rez_esc max and mean per family

owners_data['rez_esc_max'] = np.array(df.rez_esc.max())
owners_data['rez_esc_mean'] = np.array(df.rez_esc.mean())

owners_data.shape


# In[69]:


# sum of disabled in the family
owners_data['dis_fam'] = np.array(df.rez_esc.sum())
owners_data.shape


# In[70]:


#mean of family members that studied in each education level:
owners_data['instlevel_mean_fam'] = np.array(df.instlevel_cat.mean())
owners_data.shape


# In[71]:


col_names = pd.Series(owners_data.columns)
print(list(col_names))


# In[72]:


#sum of family members that are married, divorced etc.:

col_list = owners_data.columns[col_names.str.startswith('estadocivil')]

for col in col_list:
    owners_data[col+'_fam'] = np.array(df[col].sum())

print(owners_data.shape)
owners_data.head()


# In[73]:


#sum of family members in each family category (spouse/partner, son/doughter, mother/father etc.):

col_names = pd.Series(owners_data.columns)
#col_names
col_list = owners_data.columns[col_names.str.startswith('parentesco')]
for col in col_list:
    owners_data[col+'_fam'] = np.array(df[col].sum())

print(owners_data.shape)
owners_data.head()


# In[74]:


print(list(owners_data.columns))


# In[75]:


print('Before:',owners_data.shape)

owners_data.drop(columns=['Id','rez_esc',
                   'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                   'estadocivil5', 'estadocivil6', 'estadocivil7', 
                   'parentesco1','parentesco1_fam','parentesco2', 'parentesco3', 'parentesco4', 'parentesco5',
                   'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
                   'parentesco11', 'parentesco12' ], inplace=True)

print('After:',owners_data.shape)


# In[76]:


owners_data = owners_data.set_index('idhogar')
print(owners_data.shape)
owners_data.head()


# In[77]:


owners_data_all_features = owners_data.copy()
owners_data.tail()


# In[78]:


owners_data['instlevel_cat*age_mean_fam'] = owners_data['instlevel_cat']*owners_data['age_mean_fam']
owners_data['instlevel_cat*escolari_mean_fam'] = owners_data['instlevel_cat']*owners_data['escolari_mean_fam']
owners_data['v18q1/r4t2'] = owners_data['v18q1']/owners_data['r4t2'] #tablets/people
owners_data['qmobilephone/r4t2'] = owners_data['qmobilephone'] /  owners_data['r4t2'] #phones/people

owners_data['SQescolari_mean_fam'] = np.square(owners_data['escolari_mean_fam'])
owners_data['SQdependency'] = np.square(owners_data['dependency'])
owners_data['SQmeaneduc_new'] =  np.square(owners_data['meaneduc_new'])
owners_data['SQescolari_mean_fam'] = np.square(owners_data.escolari_mean_fam)
owners_data['SQescolari'] = np.square(owners_data['escolari'])
owners_data['SQage'] = np.square(owners_data.age)
owners_data['SQhogar_total'] = np.square(owners_data['hogar_total'])
owners_data.shape


# In[79]:


plt.figure(figsize=(25,7))
owners_data.nunique().sort_values(ascending=False).plot(kind='bar')


# In[80]:


corrmat = owners_data.corr()
cols = corrmat.nlargest(30, 'Target')['Target'].index
cm = np.corrcoef(owners_data[cols].values.T)
sns.set(font_scale=1.5)
plt.figure(figsize=(20, 20))

hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# In[81]:


plt.figure(figsize=(12,7))
plt.title('Mean Years of Education per Family by Poverty Level')
ax = sns.boxplot(x="Target", y="escolari_mean_fam", data=owners_data)


# In[82]:


plt.figure(figsize=(12,7))
plt.title('Median age per Family by Poverty Level')
ax = sns.boxplot(x="Target", y="age_median_fam", data=owners_data)


# In[83]:


colors = {1: 'b', 2:'orange', 3: 'g', 4:'r'}
plt.figure(figsize = (12, 3))

# overcrowding - persons per room
# meaneduc_new - mean years of education of adults (>=18)

for i, col in enumerate(['overcrowding', 'meaneduc_new']):
    ax = plt.subplot(2, 1, i + 1)
    
    for poverty_level, color in colors.items():
        sns.kdeplot(owners_data.loc[owners_data['Target'] == poverty_level, col], 
                     ax = ax, color = color, label = poverty_level)
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')
plt.subplots_adjust(top = 4)


# In[84]:


# from target, how many appeared in each instlevel

map_df = pd.crosstab(columns=owners_data.Target, 
                          index=owners_data.instlevel_cat, 
                          normalize='columns')

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(ax=ax, data=map_df, cmap='coolwarm')


# In[85]:


X = owners_data.drop(columns='Target')
y = owners_data.Target


# In[86]:


X_train, X_test, y_train, y_test = split(X,y, random_state=555, test_size=0.3, stratify = y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[87]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[88]:


train_scores = []
# test_scores = []
for k in range(1, X.shape[1]):
    scaled_pca_transformer = PCA(n_components=k).fit(X_train_scaled)
        
    X_train_scaled_pca = scaled_pca_transformer.transform(X_train_scaled)
        
    clf = KNeighborsClassifier().fit(X_train_scaled_pca, y_train)
    
    X_test_scaled_pca = scaled_pca_transformer.transform(X_test_scaled)
    
    train_scores.append(clf.score(X_train_scaled_pca, y_train))
#    test_scores.append(clf.score(X_test_scaled_pca, y_test))


# In[89]:


plt.figure(figsize=(20,7))
# plt.plot(list(zip(train_scores, test_scores)), linewidth=5)
plt.plot(train_scores, linewidth=5)
plt.xticks(ticks=np.arange(0, len(X.columns)+1, 2.0))
plt.title('Model score vs. number of components')
plt.xlabel('n_components')
plt.ylabel('Score (accuracy)')
plt.legend(['train score', 'test score'], loc='best')


# In[90]:


pca_transformer = PCA(n_components=2).fit(X_train_scaled)
X_train_scaled_pca = pca_transformer.transform(X_train_scaled)
X_test_scaled_pca = pca_transformer.transform(X_test_scaled)
X_train_scaled_pca[:1]


# In[91]:


plt.figure(figsize=(15,7))
sns.scatterplot(x=X_train_scaled_pca[:, 0], 
                y=X_train_scaled_pca[:, 1], 
                hue=y_train, 
                sizes=100,
                palette="Accent") 


# In[92]:


X = owners_data.drop(columns='Target')
y = owners_data.Target


# In[93]:


X_train, X_test, y_train, y_test = split(X, y, random_state=555, test_size=0.3, stratify=y)


# In[94]:


print('train:')
print(y_train.value_counts(normalize=True))
print('test:')
print(y_test.value_counts(normalize=True))


# In[95]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,index=X_test.index, columns=X_test.columns)
X_train_scaled.head()


# In[96]:


selector = VarianceThreshold(0.008)
selector.fit(X_train_scaled)


# In[97]:


print(f'There are {len(X_train_scaled.columns[selector.get_support()])} columns left')


# In[98]:


dropped_features = X_train_scaled.columns[~selector.get_support()]
dropped_features


# In[99]:


selected_data_train = selector.transform(X_train_scaled)
selected_data_test = selector.transform(X_test_scaled)


selected_data_train = pd.DataFrame(selected_data_train, 
                             columns=X_train_scaled.columns[selector.get_support()])
selected_data_test = pd.DataFrame(selected_data_test, 
                             columns=X_test_scaled.columns[selector.get_support()])

selected_data_train.sample(5)


# In[100]:


pca_transformer = PCA(n_components=50).fit(selected_data_train)
X_train_scaled_pca = pca_transformer.transform(selected_data_train)
X_test_scaled_pca = pca_transformer.transform(selected_data_test)


# In[101]:


classifiers = [('LR', LogisticRegression(solver='lbfgs', multi_class='auto', 
                            max_iter=1000, class_weight='balanced',
                            random_state=555)), 
               ('KNC', KNeighborsClassifier(n_neighbors=5, metric='manhattan'))] #no class_weights


model2 = VotingClassifier(estimators=classifiers, voting='soft')
model2.fit(X_train_scaled_pca, y_train)


# In[102]:


print(Fore.BLUE+report(y_test, model2.predict(X_test_scaled_pca), model2.classes_))


# In[103]:


classifiers = [('LR', LogisticRegression(solver='lbfgs', multi_class='auto', 
                            max_iter=1000, class_weight='balanced',
                            random_state=555)), 
               ('KNC', KNeighborsClassifier(n_neighbors=5, metric='manhattan'))] #no class_weights


model4 = VotingClassifier(estimators=classifiers, voting='soft')
model4.fit(selected_data_train, y_train)


# In[104]:


print(Fore.BLUE+report(y_test, model4.predict(selected_data_test), model4.classes_))


# In[105]:


dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=555 )

model8 = AdaBoostClassifier(base_estimator=dt, learning_rate=0.01, n_estimators=100, random_state=222)
model8.fit(selected_data_train ,y_train)


# In[106]:


print(Fore.BLUE+report(y_test, model8.predict(selected_data_test), model8.classes_))


# In[107]:


svc = SVC(class_weight='balanced',random_state=222, kernel='rbf', C= 1.0, gamma='auto')

params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
          'C': [0.1, 1.0, 10]}

gs = GridSearchCV(svc, params, cv=3, return_train_score=False, scoring='f1_macro')
gs.fit(selected_data_train, y_train)


# In[108]:


print(gs.best_params_)
svc_best = gs.best_estimator_


# In[109]:


model10 = BaggingClassifier(base_estimator=svc_best, n_estimators=100, n_jobs=4, max_features=30, random_state=555)

model10.fit(selected_data_train, y_train)


# In[110]:


print(Fore.BLUE+report(y_test, model10.predict(selected_data_test), model10.classes_))


# In[111]:


lr = LogisticRegression(solver='lbfgs', multi_class='auto', 
                            max_iter=10000, class_weight='balanced',
                            random_state=555)
params = {'C': [0.1, 1.0, 10]}

gs = GridSearchCV(lr, params, scoring='f1_macro', cv=5, return_train_score=False)
gs.fit(selected_data_train, y_train)


# In[112]:


gs.best_params_


# In[113]:


model5 = gs.best_estimator_
print(Fore.BLUE+report(y_test, model5.predict(selected_data_test), model5.classes_))


# In[114]:


rf = RandomForestClassifier(class_weight='balanced', n_estimators=50000, #max_depth=30, 
                            min_samples_leaf=5, max_features=30, n_jobs=4, random_state=222)

params = {#'max_depth': range(25,30),
          'min_samples_leaf': range(5,7)}

gs = GridSearchCV(rf, params, cv=5, return_train_score=False, scoring='f1_macro')
gs.fit(selected_data_train,y_train)


# In[115]:


print(gs.best_params_)
model6 = gs.best_estimator_


# In[116]:


print(Fore.RED+report(y_train, model6.predict(selected_data_train), model6.classes_))


# In[117]:


print(Fore.BLUE+report(y_test, model6.predict(selected_data_test), model6.classes_))


# In[118]:


grid_results = pd.DataFrame(gs.cv_results_)
grid_results.sort_values(by='rank_test_score').head(4)


# In[119]:


df_scores = grid_results.sort_values(by='rank_test_score')[['params', 'mean_test_score', 'std_test_score' ]]

df_scores[['mean_test_score',  'std_test_score']].plot(kind='scatter', x='mean_test_score', y='std_test_score', 
                                                       color='lightblue', figsize=(10,5))

P = [df_scores.iloc[0,1] , df_scores.iloc[0,2]]
plt.plot(P[0], P[1], marker='o', markersize=5, color="darkblue")


# In[120]:


my_list = list(zip(model6.feature_importances_ ,selected_data_test.columns))
my_list.sort(key=lambda tup: tup[0],reverse=True)
# for item in my_list:
#     if item[0]> 0.009:
#         print(item)


# In[121]:


importances = pd.DataFrame(data=None,columns=['importance', 'feature'])

importance_l = []
feature_l = []

for t in my_list:
    importance_l.append(t[0])
    feature_l.append(t[1])
    
importances['importance'] = importance_l
importances['feature'] = feature_l


# In[122]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('Feature Importances')
sns.barplot(x="feature", y="importance", data=importances)


# In[123]:


model13 = KMeans(n_clusters=4).fit(selected_data_train)
y_pred = model13.predict(selected_data_train)
print(f1_score(y_true=y_train, y_pred=y_pred, average='macro',labels=np.unique(y_pred)))


# In[124]:


print(f1_score(y_true=y_test, 
               y_pred=model13.predict(selected_data_test), 
               average='macro', 
               labels=np.unique(model13.predict(selected_data_test))))


# In[125]:


selected_data_train1 = selected_data_train.copy()
selected_data_test1= selected_data_test.copy()

model13 = KMeans(n_clusters=4).fit(selected_data_train1)

selected_data_train1['clustering'] = model13.predict(selected_data_train1)
selected_data_test1['clustering'] = model13.predict(selected_data_test1)


# In[126]:


rf = RandomForestClassifier(class_weight='balanced', n_estimators=50000, 
                            min_samples_leaf=5, max_features=30, n_jobs=4, random_state=222)

params = {#'max_depth': range(25,30),
          'min_samples_leaf': range(5,7)}

gs = GridSearchCV(rf, params, cv=5, return_train_score=False, scoring='f1_macro')
gs.fit(selected_data_train1,y_train)


# In[127]:


print(gs.best_params_)
model14 = gs.best_estimator_


# In[128]:


print(Fore.BLUE+report(y_test, model14.predict(selected_data_test1), model14.classes_))

