#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import scipy.stats as stats


# In[3]:


df_train = pd.read_csv('../input/train.csv')
df_train.shape


# In[4]:


df_test = pd.read_csv('../input/test.csv')
df_test.shape


# In[5]:


df = pd.concat((df_train, df_test), ignore_index= True, sort = False)
df.shape


# In[6]:


unique_hh_heads = (df_train['parentesco1'] == 1).sum()
unique_hh = len(df_train['idhogar'].unique())

print ('There are {} unique households and the dataset contain {} records of household heads'.format(unique_hh, unique_hh_heads))
print ('There is {} households without household head'.format(unique_hh - unique_hh_heads))


# In[7]:


unique_hh_heads = (df_test['parentesco1'] == 1).sum()
unique_hh = len(df_test['idhogar'].unique())

print ('There are {} unique households and the dataset contain {} records of household heads'.format(unique_hh, unique_hh_heads))
print ('There is {} households without household head'.format(unique_hh - unique_hh_heads))


# In[8]:


df.info(verbose=False)


# In[9]:


df.columns[df.dtypes == object]


# In[10]:


df.columns[df.dtypes == float]


# In[11]:


# check columns for null values
df.isnull().sum()[df.isnull().sum() > 0]


# In[12]:


df['Target'].value_counts(normalize = True)


# In[13]:


sns.countplot(x = 'Target', data = df)


# In[14]:


hh_heads = set(df['idhogar'][df['parentesco1'] == 1])
households = set(df['idhogar'])


# In[15]:


'''
missing_hh =  households.difference(hh_heads)
rows_to_delete = df[df['idhogar'].isin(missing_hh)].index
df.drop(index= rows_to_delete, inplace = True) '''


# In[16]:


# Number of records with no rent amount
df['v2a1'].isnull().sum()


# In[17]:


# 
col = [i for i in df.columns if i.startswith('tipovivi')]
df.loc[:, col].sum()


# In[18]:


# Create temporary column to identify the home ownership type.
df['temp_tipovivi'] = df[col].idxmax(axis = 1)


# In[19]:


## Identify the home ownership status of the hh with zero rent
df['temp_tipovivi'][df['v2a1'].isnull()].value_counts()


# In[20]:


df['v2a1'].fillna(value = 0, inplace = True)


# In[21]:


df['temp_tipovivi'][df['v2a1'] == 0].value_counts()


# In[22]:


# Change the homeownership type to  be consistant with the rent amount.
tipovivi2 = (df['v2a1'] == 0)&(df['tipovivi2'] == 1)
tipovivi3 = (df['v2a1'] == 0)&(df['tipovivi3'] == 1)


# In[23]:


df.loc[tipovivi2,'tipovivi1'] = 1
df.loc[tipovivi3,'tipovivi1'] = 1

df.loc[tipovivi2,'tipovivi2'] = 0
df.loc[tipovivi3,'tipovivi3'] = 0


# In[24]:


## Update temp_tipovivi to reflect change
df['temp_tipovivi'] = df[col].idxmax(axis = 1)


# In[25]:


df[col][df['v2a1'] == 0].sum()


# In[26]:


sns.distplot(df['v2a1'],fit = stats.norm)


# In[27]:


## Seperate out the records where the households does not pay rent
df['RentPaying'] = (df['v2a1'] > 0)*1
## Log transfrom to make distribution normal
df['v2a1'] = np.log1p(df['v2a1'])
sns.distplot(df['v2a1'][df['RentPaying'] == 1],fit = stats.norm)


# In[28]:





# In[28]:


df.pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_tipovivi', aggfunc= 'count', margins= True)


# In[29]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_tipovivi', aggfunc= 'count')
cat = df['temp_tipovivi'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[30]:


df['v18q'][df['v18q1'].isnull()].value_counts()


# In[31]:


df['v18q1'].fillna(value = 0, inplace = True)


# In[32]:


df[df['parentesco1'] == 1].groupby('Target')['v18q'].mean()


# In[33]:


temp = df[(df['parentesco1'] == 1)&(df['Target'].notnull())].pivot_table(index = 'Target', columns = 'v18q1', values = 'idhogar', aggfunc='count')
cat = df['v18q1'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[34]:


#### Years behind in school
df['rez_esc'].value_counts(dropna = False)


# In[35]:


df['age'][df['rez_esc'].notna()].value_counts().sort_index()


# In[36]:


df['rez_esc'].fillna(value = 0, inplace = True)


# In[37]:


## Fixing the large age behind in school value
df.loc[df['rez_esc'] > 50, 'rez_esc'] = 0


# In[38]:


## Obtain a list of households where the average years schooled is NA
na_mean_households = df['idhogar'][df['meaneduc'].isna()].unique()


# In[39]:


## Checking if there are 18+ persons in households 
df[df['meaneduc'].isna()].groupby('idhogar')['age'].max()


# In[40]:


## recompute meaneduc for households.
mapper = df[df['age'] >= 18].groupby('idhogar')['escolari'].mean().to_dict()


# In[41]:


df['meaneduc'] = df.apply(lambda x: mapper.get('idhogar', 0) if np.isnan(x['meaneduc']) else x['meaneduc'], axis = 1)


# In[42]:


df['meaneduc'].isna().sum()


# In[43]:


df['SQBmeaned'] = df['meaneduc']**2


# In[44]:


sns.countplot(x = 'Target', data = df[df['hacdor'] == 1])


# In[45]:


df[df['parentesco1'] == 1].groupby('Target')['hacdor'].mean()


# In[46]:


## Since overcrowing occures only ~3% of the time this is a possible candidates for deletion
## df.drop(column = ['hacdor', 'hacapo'])


# In[47]:





# In[47]:


### IGNORE !!!
### For the time being we'll calculate the likelihood based on the household head only.
temp = df[df['Target'].notnull()].pivot_table(index = 'Target', columns = 'v18q1', values = 'idhogar', aggfunc='count')
cat = df['v18q1'][df['Target'].notnull()].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[48]:


df[['tamhog', 'tamviv', 'r4t3', 'hhsize', 'hogar_total']][df['r4t3'] != df['hhsize']].head()


# In[49]:


df.drop(columns= ['tamhog', 'hogar_total', 'r4t3'], inplace = True)


# In[50]:


(df['v14a'][df['parentesco1']== 1]).mean()


# In[51]:


df[df['parentesco1'] == 1].groupby('Target')['v14a'].mean()


# In[52]:


(df['refrig'][df['parentesco1']== 1]).mean()


# In[53]:


df[df['parentesco1'] == 1].groupby('Target')['refrig'].mean()


# In[54]:


col = [i for i in df.columns if i.startswith('pared')]
df.loc[:, col].sum()


# In[55]:


df['temp_pared'] = df[col].idxmax(axis = 1)


# In[56]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_pared', aggfunc= 'count')
cat = df['temp_pared'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[57]:





# In[57]:


col = [i for i in df.columns if i.startswith('piso')]
df.loc[:, col].sum()


# In[58]:


df['temp_piso'] = df[col].idxmax(axis = 1)


# In[59]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_piso', aggfunc= 'count')
cat = df['temp_piso'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[60]:





# In[60]:


col = [i for i in df.columns if i.startswith('techo')]
df.loc[:, col].sum()


# In[61]:


df['temp_techo'] = df[col].idxmax(axis = 1)


# In[62]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_techo', aggfunc= 'count')
cat = df['temp_techo'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[63]:





# In[63]:


df[df['parentesco1'] == 1].groupby('Target')['cielorazo'].mean()


# In[64]:


col = [i for i in df.columns if i.startswith('abastagua')]
df.loc[:, col].sum()


# In[65]:


df['temp_abastagua'] = df[col].idxmax(axis = 1)


# In[66]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_abastagua', aggfunc= 'count')
cat = df['temp_abastagua'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[67]:





# In[67]:


df['temp_electricity'] = df[['public', 'planpri', 'noelec', 'coopele']].idxmax(axis = 1)


# In[68]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_electricity', aggfunc= 'count')
cat = df['temp_electricity'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[69]:





# In[69]:


col = [i for i in df.columns if i.startswith('sanit')]
df.loc[:, col].sum()


# In[70]:


df['temp_sanitario'] = df[col].idxmax(axis = 1)


# In[71]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_sanitario', aggfunc= 'count')
cat = df['temp_sanitario'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[72]:





# In[72]:


col = [i for i in df.columns if i.startswith('energcocinar')]
df.loc[:, col].sum()


# In[73]:


df['temp_energcocinar'] = df[col].idxmax(axis = 1)


# In[74]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_energcocinar', aggfunc= 'count')
cat = df['temp_energcocinar'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[75]:


df['cooking_lowEng'] = ((df['energcocinar1'] == 1)|(df['energcocinar4'] == 1))*1


# In[76]:


col = [i for i in df.columns if i.startswith('elimbasu')]
df.loc[:, col].sum()


# In[77]:


df['temp_elimbasu'] = df[col].idxmax(axis = 1)


# In[78]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_elimbasu', aggfunc= 'count')
cat = df['temp_elimbasu'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[79]:





# In[79]:


col = [i for i in df.columns if i.startswith('epared')]
df.loc[:, col].sum()


# In[80]:


df['temp_epared'] = df[col].idxmax(axis = 1)


# In[81]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_epared', aggfunc= 'count')
cat = df['temp_epared'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[82]:





# In[82]:


col = [i for i in df.columns if i.startswith('etecho')]
df.loc[:, col].sum()


# In[83]:


df['temp_etecho'] = df[col].idxmax(axis = 1)


# In[84]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_etecho', aggfunc= 'count')
cat = df['temp_etecho'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[85]:





# In[85]:


col = [i for i in df.columns if i.startswith('eviv')]
df.loc[:, col].sum()


# In[86]:


df['temp_eviv'] = df[col].idxmax(axis = 1)


# In[87]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_eviv', aggfunc= 'count')
cat = df['temp_eviv'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[88]:





# In[88]:


df[df['parentesco1'] == 1].groupby('Target')['dis'].mean()


# In[89]:


df[df['parentesco1'] == 1].groupby('Target')['male'].mean()


# In[90]:


df.drop(columns= 'female', inplace = True)


# In[91]:





# In[91]:


col = [i for i in df.columns if i.startswith('estadocivil')]
df.loc[:, col].sum()


# In[92]:


df['temp_estadocivil'] = df[col].idxmax(axis = 1)


# In[93]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_estadocivil', aggfunc= 'count')
cat = df['temp_estadocivil'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[94]:





# In[94]:





# In[94]:


col = [i for i in df.columns if i.startswith('instlevel')]
df.loc[:, col].sum()


# In[95]:


df['temp_instlevel'] = df[col].idxmax(axis = 1)


# In[96]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_instlevel', aggfunc= 'count')
cat = df['temp_instlevel'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/(cat.T), vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[97]:





# In[97]:


col = [i for i in df.columns if i.startswith('lugar')]
df.loc[:, col].sum()


# In[98]:


df['temp_lugar'] = df[col].idxmax(axis = 1)


# In[99]:


temp = df[df['parentesco1'] == 1 ].pivot_table(values = 'idhogar' , index = 'Target', columns = 'temp_lugar', aggfunc= 'count')
cat = df['temp_lugar'][(df['Target'].notnull())&(df['parentesco1'] == 1)].value_counts()

##np.divide(temp, cat.values)
sns.heatmap(temp/cat.T, vmin= 0, vmax= 1, cmap = 'viridis', annot= True)


# In[100]:





# In[100]:


df[df['parentesco1'] == 1].groupby('Target')['area1'].mean()


# In[101]:


df.drop(columns= 'area2', inplace = True)


# In[102]:


df['hogar_workingAge'] = df['hogar_adul'] - df['hogar_mayor']
df['hogar_dependent'] = df['hogar_nin'] + df['hogar_mayor']


# In[103]:


## df[['hogar_nin', 'hogar_adul','hogar_mayor', 'hogar_workingAge', 'hogar_dependent','dependency']][df['dependency'] == 'no']

df[['hogar_nin', 'hogar_adul','hogar_mayor', 'hogar_workingAge', 'hogar_dependent','dependency']][df['dependency'] == '8']


# In[104]:


df['dependency'] = df['dependency'].replace({'yes': 1, 'no': 0}).astype(float)


# In[105]:





# In[105]:





# In[105]:


df['edjefe'] = df['edjefe'].replace({'no': 0, 'yes': 1})
df['edjefa'] = df['edjefa'].replace({'no': 0, 'yes': 1})


# In[106]:


df['edjefe'] = df['edjefe'].astype(int)
df['edjefa'] = df['edjefa'].astype(int)


# In[107]:


df['median_schooling'] = df['escolari'].groupby(df['idhogar']).transform('median')
df['max_schooling'] = df['escolari'].groupby(df['idhogar']).transform('max')


# In[108]:


df['eduForHeadofHH'] = 0
df.loc[(df['parentesco1']== 1), 'eduForHeadofHH'] = df['escolari']


# In[109]:


df['eduForHeadofHH'] = df['eduForHeadofHH'].groupby(df['idhogar']).transform('max')


# In[110]:


df['SecondaryEduLess'] = ((df[['instlevel1','instlevel2', 'instlevel3', 'instlevel4']] == 1).any(axis = 1)&(df['age'] > 19))*1
df['SecondaryEduMore'] = ((df[['instlevel5','instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']] == 1).any(axis = 1)&(df['age'] > 19))*1


# In[111]:


df['MembersWithSecEdu']  = df['SecondaryEduMore'].groupby(df['idhogar']).transform('sum')
df['MembersWithPrimEdu']  = df['SecondaryEduLess'].groupby(df['idhogar']).transform('sum')


# In[112]:


df['Educated_Gap'] = (df['MembersWithSecEdu'] - df['MembersWithPrimEdu'])


# In[113]:





# In[113]:





# In[113]:





# In[113]:





# In[113]:


df['marital_status'] = (((df['estadocivil3'] ==1)|(df['estadocivil4'] == 1))&(df['parentesco1'] == 1))*1

df['marital_status'] = df['marital_status'].groupby(df['idhogar']).transform('max')


# In[114]:


df['FemaleHousehold'] = ((df['male'] == 0)&(df['parentesco1'] == 1))*1
df['FemaleHousehold'] = df['FemaleHousehold'].groupby(df['idhogar']).transform('max')


# In[115]:


df['phones_percap'] = df['qmobilephone'] / df['tamviv']
df['tablets_percap'] = df['v18q1'] / df['tamviv']
df['rooms_percap'] = df['rooms'] / df['tamviv']
df['rent_percap'] = df['v2a1'] / df['tamviv']


# In[116]:


df['minors_ratio'] = df['hogar_nin']/df['tamviv']
df['elder_ratio'] = df['hogar_mayor']/df['tamviv']


# In[117]:


df['child_ratio'] = df['r4t1']/ df['tamviv']
df['malefemale_ratio'] = df['r4h3'] -  df['r4m3'] 


# In[118]:


df['ismale_only'] =  (df['r4m3'] == 0)*1
df['isfemale_only'] = (df['r4h3'] == 0)*1
df['no_adultmale'] = (df['r4h2'] == 0)*1
df['no_adultfemale'] = (df['r4m2'] == 0)*1


# In[119]:


df['rent_per_room'] = df['v2a1']/df['rooms']
df['bedroom_per_room'] = df['bedrooms']/df['rooms']


# In[120]:


df['rent_per_room'] = df['v2a1'] / df['rooms']


# In[121]:


df['total_disabled'] = df.groupby('idhogar')['dis'].transform(lambda x: x.sum())


# In[122]:


df['average_age'] = df.groupby('idhogar')['age'].transform(lambda x: x.mean())


# In[123]:


df['disable_ratio'] = df['total_disabled']/df['tamviv']


# In[124]:


df['info_accessibility'] = df[['mobilephone', 'television', 'computer', 'v18q']].any(axis = 1)


# In[125]:


df.select_dtypes(include = 'number').columns


# In[126]:


df.to_csv('./processed.csv')


# In[127]:


df.drop(columns=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace = True)


# In[128]:


training_df = df.select_dtypes(include = 'number')[df['Target'].notnull()]
test_df = df.select_dtypes(include = 'number')[df['Target'].isnull()]


# In[129]:


training_df.shape


# In[130]:


features = [col for col in training_df.columns if col != 'Target']
X, y = training_df[features], training_df['Target']


# In[131]:


test_df.drop(columns = 'Target', inplace = True)


# In[132]:


from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


# In[133]:


tree = DecisionTreeClassifier(max_features= 75, class_weight='balanced')
tree.fit(X,y)


# In[134]:





# In[134]:


model = RandomForestClassifier(n_estimators=100, random_state=10, max_features= 75, n_jobs = -1 ,class_weight= 'balanced')
cv_score = cross_val_score(model, X, y, cv = 10, scoring = 'f1_macro')
cv_score.mean()


# In[135]:


from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier


# In[136]:


ets = []
for i in range(10):
    rf = RandomForestClassifier(random_state=217+i, n_jobs=4, n_estimators=700, min_impurity_decrease=1e-3, min_samples_leaf=2, verbose=0, class_weight= 'balanced')
    ets.append(('rf{}'.format(i), rf)) 


# In[137]:


vclf = VotingClassifier(ets, voting= 'soft')


# In[138]:


### Score CV results
cv_score = cross_val_score(vclf, X, y, cv= 5, scoring = 'f1_macro')


# In[139]:


cv_score.mean()


# In[140]:


cv_predict = cross_val_predict(vclf, X, y, cv = 5)


# In[141]:


confusion_matrix(y, cv_predict)


# In[142]:


f1_score()


# In[143]:





# In[143]:





# In[143]:


vclf = VotingClassifier(ets, voting= 'hard')
cv_score = cross_val_score(vclf, X, y, cv = 5, scoring = 'f1_macro')
cv_score


# In[144]:


cv_score.mean()


# In[145]:


vclf.fit(X,y)
vclf_hardvoting = vclf.predict(test_df)


# In[146]:


len(vclf_hardvoting)


# In[147]:


import lightgbm as lgb


# In[148]:


##clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                                 random_state=None, silent=True, metric='None', 
                                 n_jobs=4, n_estimators=500, class_weight='balanced',
                                 colsample_bytree =  0.89, min_child_samples = 90, num_leaves = 56, subsample = 0.96)


# In[149]:


## cv_score = cross_val_score(clf, X, y, cv = 3, scoring = 'f1_macro')
## cv_score


# In[150]:





# In[150]:


### prediction = [model].predict(test_df)


# In[151]:


submit=pd.DataFrame({'Id': df['Id'][df['Target'].isna()] , 'Target': vclf_hardvoting.astype(int)})


# In[152]:


submit['Target'].value_counts(normalize = True)


# In[153]:


submit.to_csv('./submission.csv', index= False)


# In[154]:


##training_df_hhO = df.select_dtypes(include = 'number')[(df['Target'].notnull())&(df['parentesco1'] == 1)]
##test_df_hhO = df.select_dtypes(include = 'number')[(df['Target'].isnull())&(df['parentesco1'] == 1)]


# In[155]:


##features = [col for col in training_df.columns if col != 'Target']
##X_hhO, y_hhO = training_df_hhO[features], training_df_hhO['Target']


# In[156]:


##cv_score = cross_val_score(vclf, X_hhO, y_hhO, cv = 5, scoring = 'f1_macro')


# In[157]:


##cv_score

array([0.4596173 , 0.42617731, 0.38660971, 0.35653499, 0.37714824])  
array([0.46729109, 0.42841538, 0.38395209, 0.35942352, 0.37480922])
# In[158]:


## np.array([0.4596173 , 0.42617731, 0.38660971, 0.35653499, 0.37714824]).mean()


# In[159]:




