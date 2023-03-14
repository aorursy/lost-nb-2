#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install lightgbm


# In[2]:


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 150

#for machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb



# In[3]:


train = pd.read_csv('../input/train.csv')
train.info()
train.head()


# In[4]:


test = pd.read_csv('../input/test.csv')
test.info()
test.head()


# In[5]:


train.describe()
test.describe()


# In[6]:


print(test.loc[:,"rez_esc"].describe())
test.loc[test.loc[:,"rez_esc"]==99,"rez_esc"]


# In[7]:


test.loc[test.loc[:,"rez_esc"]==99,"rez_esc"]=5
test.loc[:,"rez_esc"].describe()


# In[8]:


train_na= pd.DataFrame((train.isnull().sum().values),index=train.columns, columns=['isNA']).sort_values(by=['isNA'],ascending=False)
if train_na.loc[train_na.loc[:,'isNA']>0,:].shape[0]>1 :
    print("Train Dataset")
    print(train_na.loc[train_na.loc[:,'isNA']> 0,])
else:
    print('no NA in train set')

test_na= pd.DataFrame((test.isnull().sum().values),index=test.columns, columns=['isNA']).sort_values(by=['isNA'],ascending=False)
if train_na.loc[train_na.loc[:,'isNA']>0,:].shape[0]>1 :
    print("")
    print("Test Dataset")
    print(test_na.loc[test_na.loc[:,'isNA']> 0,])
else:
    print('no NA in test set')


# In[9]:


rez_esc_age=train.loc[train['rez_esc'].isnull()==False, 'age']

plt.hist(x=rez_esc_age,)
plt.xticks(np.arange(min(rez_esc_age), max(rez_esc_age)+1, 1.0),rotation = 60),
plt.ylabel('frequence of rez_esc')
plt.xlabel('Age')
plt.title('Non-null rez_esc Frequency according to age')
plt.show()


# In[10]:


tipos=[x for x in train if x.startswith('tipo')]
rentNA_status=train.loc[train['v2a1'].isnull(), tipos].sum()
plt.bar(tipos,rentNA_status,align='center')
plt.xticks([0,1,2,3,4],['Owns and Paid off','Owns and Paying', 'Renting','Precarious','Other'],rotation = 60),
plt.ylabel('Frequency')
plt.title("Missing Rental 'v2a1' according to Home Ownership Status")
plt.show()


# In[11]:


Tablet_status=train.loc[train['v18q1'].isnull(), 'v18q']
plt.hist(x=Tablet_status)
plt.xticks([0,1,2],['Do not Own a Table','Owns a Tablet'],rotation=60),
plt.ylabel('Frequency missing value on v18q1')
plt.xlabel('Individual Tablet Ownership (v18q)')
plt.title('Missing value on household tablet ownership vs individual tablet ownership')
plt.show()


# In[12]:


plt.figure(figsize=(10,5))
plt.hist(x=train['meaneduc'],bins=int(train['meaneduc'].max())+1)
plt.xticks(np.arange(min(train['meaneduc']), max(train['meaneduc'])),rotation=60),
plt.ylabel('Frequency')
plt.xlabel('average years of education for adults (18+)')
plt.title('Histogram for meaneduc')
plt.show()


# In[13]:


train.loc[:,"meaneduc"].mode()
#train: mode for meaneduc is 6 replace NA with 6, replace SQBmeaned NA to 36
train.loc[train.loc[:,"meaneduc"].isnull()==True,"meaneduc"] = 6
train.loc[train.loc[:,"SQBmeaned"].isnull()==True,"SQBmeaned"] = 36

test.loc[:,"meaneduc"].mode()
#test: mode for meaneduc is 6 replace NA with 6, replace SQBmeaned NA to 36
test.loc[test.loc[:,"meaneduc"].isnull()==True,"meaneduc"] = 6
test.loc[test.loc[:,"SQBmeaned"].isnull()==True,"SQBmeaned"] = 36


#Replace all NA values for remaining 3 attributes with 0no
train.loc[train.loc[:,"rez_esc"].isnull()==True,"rez_esc"] = 0
train.loc[train.loc[:,"v18q1"].isnull()==True,"v18q1"] = 0
train.loc[train.loc[:,"v2a1"].isnull()==True,"v2a1"] = 0

test.loc[test.loc[:,"rez_esc"].isnull()==True,"rez_esc"] = 0
test.loc[test.loc[:,"v18q1"].isnull()==True,"v18q1"] = 0
test.loc[test.loc[:,"v2a1"].isnull()==True,"v2a1"] = 0


# In[14]:


#Check for missing values again:
train_na= pd.DataFrame((train.isnull().sum().values),index=train.columns, columns=['isNA']).sort_values(by=['isNA'],ascending=False)
if train_na.loc[train_na.loc[:,'isNA']>0,:].shape[0]>1 :
    train_na.loc[train_na.loc[:,'isNA']> 0,]

else:
    print('No NA in train set')

test_na= pd.DataFrame((test.isnull().sum().values),index=test.columns, columns=['isNA']).sort_values(by=['isNA'],ascending=False)
if train_na.loc[train_na.loc[:,'isNA']>0,:].shape[0]>1 :
    test_na.loc[test_na.loc[:,'isNA']> 0,]
else:
    print('No NA in test set')


# In[15]:


## Check for Data Inconsistency


# In[16]:


target_Discrepancy=(train.groupby('idhogar')['Target'].nunique()>1)
num_unique_households = train["idhogar"].unique().shape[0]
print('There are',target_Discrepancy.sum(),'households with contradicting targets, out of', num_unique_households, 'households in the train dataset.')


# In[17]:


Discrepancy_Index=(train.groupby('idhogar')['Target'].transform('nunique')>1)
HHID_Discrepancy=train.loc[Discrepancy_Index,'idhogar'].unique()
#household with contradicting target
train.loc[train['idhogar'].isin(HHID_Discrepancy),['idhogar','parentesco1','Target']].head()


# In[18]:



for HH in HHID_Discrepancy:
    Targets= (train.loc[train['idhogar']==HH,'Target'])

    if Targets.mode().shape[0] >1:
        for i in Targets.index:
            if train.loc[i,'parentesco1']==1:
                HeadTarget= train.loc[i,"Target"]    
        for i in Targets.index:
            train.loc[i,'Target']=HeadTarget
    elif Targets.mode().shape[0]==1:
        for i in Targets.index:
            TrueTarget=int(Targets.mode())
            train.loc[i,'Target']=TrueTarget
        


# In[19]:


target_Discrepancy=(train.groupby('idhogar')['Target'].nunique()>1)

print('There are ',target_Discrepancy.sum(),'households with contradicting targets, out of 2988 households in the train dataset')

train.head()
train.shape


# In[20]:


train=train.drop(columns=train.columns[133:142],axis=1)
test=test.drop(columns=test.columns[133:142],axis=1)


# In[21]:


print(train.shape)
print(test.shape)


# In[22]:


#These are Household Numerical columns. Attribute "Dependency" is not included since it has some yes and no elements
numColumnsHH= ["v2a1","rooms","v18q1","qmobilephone","r4h3","r4m3","r4t3","tamhog","hhsize","hogar_nin","hogar_adul",
               "hogar_mayor","hogar_total","meaneduc","bedrooms","overcrowding"]


# In[23]:


fig, ax =plt.subplots(6,3, figsize=(18,15))
plt.subplots_adjust(top=2.0)
x=0
y=0
for i in range(0,len(numColumnsHH)):
    if (i>0) & (i %3==0):
        x=x+1
        y=0
    sns.boxplot(x="Target",y=numColumnsHH[i], data=train, ax=ax[x,y])
    ax[x,y].title.set_text(numColumnsHH[i])
    y=y+1


# In[24]:


cond_data = train.query("parentesco1==1").copy()
cond_data.shape


# In[25]:


wall = ["paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother"]
floor = ["pisomoscer","pisocemento","pisoother","pisonatur","pisonotiene","pisomadera"]
roof = ['techozinc','techoentrepiso','techocane','techootro', 'cielorazo']
water = ["abastaguadentro","abastaguafuera","abastaguano"]
electric = ["public","planpri","noelec","coopele"]
toilet = ['sanitario1','sanitario2','sanitario3','sanitario5','sanitario6']
energy = ['energcocinar1','energcocinar2','energcocinar3','energcocinar4']
rubbish = ['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6']
wall_cond = ['epared1','epared2','epared3']
roof_cond = ['etecho1','etecho2','etecho3']
floor_cond = ['eviv1','eviv2','eviv3']

col_types = [wall, floor, roof, water, electric, toilet, energy, rubbish, wall_cond, roof_cond, floor_cond]
titles = ['wall', 'floor', 'roof', 'water', 'electric', 'toilet', 'energy', 'rubbish', 'wall condition', 'roof condition', 'floor condition']

fig, ax =plt.subplots(len(col_types)//3 + 1, 3, figsize=(30,10))
plt.subplots_adjust(top=2.0)
x=0
y=0
for i, cols in enumerate(col_types):
    if (i>0) & (i % 3==0):
        x=x+1
        y=0
    title = titles[i]
    plot_df = pd.melt(cond_data, id_vars=['Target'], value_vars=cols).groupby(["Target","variable"]).apply(lambda x: np.mean(x))
    plot_df["variable"] = plot_df.index.get_level_values(1)
    plot_df = plot_df.rename(columns = dict(value="Percentage Frequency", variable = "Attribute"))
    sns.barplot(x="Attribute", y = "Percentage Frequency", hue="Target", data=plot_df, ax=ax[x,y])
    ax[x,y].title.set_text(title)
    y=y+1


# In[26]:


region = ["lugar1","lugar2","lugar3","lugar4","lugar5","lugar6"]
area= ["area1","area2"]

col_types = [region, area]
titles = ['region', 'area']

fig, ax =plt.subplots(1,2, figsize=(15,5))
plt.subplots_adjust(top=1.0)
x=0
y=0
for i, cols in enumerate(col_types):
    if (i>0) & (i % 3==0):
        x=x+1
        y=0
    title = titles[i]
    plot_df = pd.melt(cond_data, id_vars=['Target'], value_vars=cols).groupby(["Target","variable"]).apply(lambda x: np.mean(x))
    plot_df["variable"] = plot_df.index.get_level_values(1)
    plot_df = plot_df.rename(columns = dict(value="Percentage Frequency", variable = "Attribute"))
    sns.barplot(x="Attribute", y = "Percentage Frequency", hue="Target", data=plot_df, ax=ax[y])
    ax[y].title.set_text(title)
    y=y+1


# In[27]:


geoTrain= train.loc[train['parentesco1'] == 1, :].copy()
geoTrain['lugar'] = np.argmax(np.array(cond_data[["lugar1","lugar2","lugar3","lugar4","lugar5","lugar6"]]),
                           axis = 1)
geoTrain['lugar']=geoTrain['lugar'].replace({5:6,4:5,3:4,2:3,1:2,0:1})


# In[28]:


wall = ["paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother"]
floor = ["pisomoscer","pisocemento","pisoother","pisonatur","pisonotiene","pisomadera"]
roof = ['techozinc','techoentrepiso','techocane','techootro', 'cielorazo']
water = ["abastaguadentro","abastaguafuera","abastaguano"]
electric = ["public","planpri","noelec","coopele"]
toilet = ['sanitario1','sanitario2','sanitario3','sanitario5','sanitario6']
energy = ['energcocinar1','energcocinar2','energcocinar3','energcocinar4']
rubbish = ['elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6']
wall_cond = ['epared1','epared2','epared3']
roof_cond = ['etecho1','etecho2','etecho3']
floor_cond = ['eviv1','eviv2','eviv3']

col_types = [wall, floor, roof, water, electric, toilet, energy, rubbish, wall_cond, roof_cond, floor_cond]
titles = ['wall', 'floor', 'roof', 'water', 'electric', 'toilet', 'energy', 'rubbish', 'wall condition', 'roof condition', 'floor condition']

fig, ax =plt.subplots(len(col_types)//3 + 1, 3, figsize=(30,10))
plt.subplots_adjust(top=2.0)
x=0
y=0
for i, cols in enumerate(col_types):
    if (i>0) & (i % 3==0):
        x=x+1
        y=0
    title = titles[i]
    plot_df = pd.melt(geoTrain, id_vars=['lugar'], value_vars=cols).groupby(["lugar","variable"]).apply(lambda x: np.mean(x))
    plot_df["variable"] = plot_df.index.get_level_values(1)
    plot_df = plot_df.rename(columns = dict(value="Percentage Frequency", variable = "Attribute"))
    sns.barplot(x="Attribute", y = "Percentage Frequency", hue="lugar", data=plot_df, ax=ax[x,y])
    ax[x,y].title.set_text(title)
    y=y+1


# In[29]:


fig, ax =plt.subplots(6,3, figsize=(18,15))
plt.subplots_adjust(top=2.0)
x=0
y=0
for i in range(0,len(numColumnsHH)):
    if (i>0) & (i %3==0):
        x=x+1
        y=0
    sns.boxplot(x="lugar",y=numColumnsHH[i], data=geoTrain, ax=ax[x,y])
    ax[x,y].title.set_text(numColumnsHH[i])
    y=y+1


# In[30]:


#Setting new features for household specific in train data

#Number of Adults not including seniors >65
train['Adults']=train['hogar_adul']-train['hogar_mayor']
#Number of children < 19yo and seniors>65
train['Dependents']=train['hogar_nin']+train['hogar_mayor']
#Number of teenager from 12 to 19
train['Teenagers']=train['hogar_nin']-train['r4t1']
#Dependency is number of dependents per adults. This replaces the original dependency data from dataset.
train['dependency']=train['Dependents']/train['Adults']
#Percentage of Adults in household
train['P_Adults']=train['Adults']/train['hogar_total']
#Percentage of Male Adults in household
train['P_Adults_Male']=train['r4h3']/train['hogar_total']
#Percentage Female Adults in household
train['P_Adults_Female']=train['r4m3']/train['hogar_total']
#Percentage Children <19yo in household
train['P_Children']=train['hogar_nin']/train['hogar_total']
#Percentage of Seniors in household
train['P_Seniors']=train['hogar_mayor']/train['hogar_total']
#Percentage of Teenagers in household
train['P_Teenagers']=train['Teenagers']/train['hogar_total']
#Rent per person in household
train['RentHH']=train['v2a1']/train['hogar_total']
#Rent per Adult in household
train['RentAdults'] = train['v2a1']/train['Adults']
train['RentAdults'] = train['RentAdults'].fillna(train['v2a1']) # Replace NA value with the Rent value itself (Assume Adults = 0 as Adults = 1, there has to be 1 adult to pay the Rent amount.)

#Tablet per person in household
train['Tablet_PP']=train['v18q1']/train['hogar_total']
#Mobile Phone per person in household
train['Phone_PP']=train['qmobilephone']/train['hogar_total']
#Bedroom per person in household
train['Bedroom_PP']=train['bedrooms']/train['hogar_total']
#Appliance scoring. Higher the better
train['Appliances']=train['refrig']+train['computer']+train['television']
#Number of Appliances per person
train['Appliances_PP']=train['Appliances']/test['hogar_total']

#Household size Difference
train['HHS_Diff']=train['tamviv']-train['hhsize']

#Number of Adults per room
train["AdultsPerRoom"] = train['Adults']/train['rooms']
#Number of Dependents per room
train["DependentsPerRoom"] = train['Dependents']/train['rooms']
#Number of Teenagers per room
train["TeenagersPerRoom"] = train['Teenagers']/train['rooms']

#Number of Males per room
train["MalesPerRoom"] = train['r4h3']/train['rooms']
#Number of Females per room
train["FemalesPerRoom"] = train['r4m3']/train['rooms']

#Percentage of rooms that are bedrooms
train["BedroomPerRoom"] =  train['bedrooms']/train['rooms']
train["RentPerRoom"] =  train['v2a1']/train['rooms']

#Years of Schooling over Age
train["Schooling_Age"]  = train["escolari"]/train["age"]

#Years behind schooling vs Years of schooling proportion
train['SchoolingProp'] = train['rez_esc']/train['escolari']


#New Scoring For Education Level
train["EduLevel"] = 0 
train.loc[train["instlevel9"] == 1,"EduLevel"] = 6
train.loc[train["instlevel8"] == 1,"EduLevel"] = 5 #higher scoring for completing tertiary education
train.loc[train["instlevel7"] == 1,"EduLevel"] = 3
train.loc[train["instlevel5"] == 1,"EduLevel"] = 2
train.loc[(train[["instlevel4","instlevel3","instlevel6"]].sum(axis = 1) > 0),"EduLevel"] = 1

train.head()

#We replicate the same for test data since we need the same features for prediction

test['Adults']=test['hogar_adul']-test['hogar_mayor']
test['Dependents']=test['hogar_nin']+test['hogar_mayor']
test['Teenagers']=test['hogar_nin']-test['r4t1']
test['dependency']=test['Dependents']/test['Adults']
test['P_Adults']=test['Adults']/test['hogar_total']
test['P_Adults_Male']=test['r4h3']/test['hogar_total']
test['P_Adults_Female']=test['r4m3']/test['hogar_total']
test['P_Children']=test['hogar_nin']/test['hogar_total']
test['P_Seniors']=test['hogar_mayor']/test['hogar_total']
test['P_Teenagers']=test['Teenagers']/test['hogar_total']
test['RentHH']=test['v2a1']/test['hogar_total']

test['RentAdults']=test['v2a1']/test['Adults']
test['RentAdults'] = test['RentAdults'].fillna(test['v2a1']) # Replace NA value with the Rent value itself (Assume Adults = 0 as Adults = 1, there has to be 1 adult to pay the Rent amount.)

test['Tablet_PP']=test['v18q1']/test['hogar_total']
test['Phone_PP']=test['qmobilephone']/test['hogar_total']
test['Bedroom_PP']=test['bedrooms']/test['hogar_total']
test['Appliances']=test['refrig']+test['computer']+test['television']
test['Appliances_PP']=test['Appliances']/test['hogar_total']
test['HHS_Diff']=test['tamviv']-test['hhsize']

test["AdultsPerRoom"] = test['Adults']/test['rooms']
test["DependentsPerRoom"] = test['Dependents']/test['rooms']
test["TeenagersPerRoom"] = test['Teenagers']/test['rooms']

test["MalesPerRoom"] = test['r4h3']/test['rooms']
test["FemalesPerRoom"] = test['r4m3']/test['rooms']

test["BedroomPerRoom"] =  test['bedrooms']/test['rooms']
test["RentPerRoom"] =  test['v2a1']/test['rooms']

test["Schooling_Age"]  = test["escolari"]/test["age"]
test['SchoolingProp'] = test['rez_esc']/test['escolari']

#New Scoring For Education Level
test["EduLevel"] = 0 
test.loc[test["instlevel9"] == 1,"EduLevel"] = 6
test.loc[test["instlevel8"] == 1,"EduLevel"] = 5 #higher scoring for completing tertiary education
test.loc[test["instlevel7"] == 1,"EduLevel"] = 3
test.loc[test["instlevel5"] == 1,"EduLevel"] = 2
test.loc[(test[["instlevel4","instlevel3","instlevel6"]].sum(axis = 1) > 0),"EduLevel"] = 1

test.head()


# In[31]:


List_Mean = ['rez_esc', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5',
             'estadocivil6', 'estadocivil7', 'parentesco2','parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7',
             'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12','instlevel1', 'instlevel2', 'instlevel3',
             'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9','overcrowding']

List_Summary = ['age', 'escolari','dis','EduLevel', 'Schooling_Age']

trainGP = pd.DataFrame()
testGP = pd.DataFrame()

for item in List_Mean:
    group_train_mean = train[item].groupby(train['idhogar']).mean()
    group_test_mean = test[item].groupby(test['idhogar']).mean()
    new_col = item + '_mean'
    trainGP[new_col] = group_train_mean
    testGP[new_col] = group_test_mean

for item in List_Summary:
    for function in ['mean','std','min','max','sum']:
        group_train = train[item].groupby(train['idhogar']).agg(function)
        group_test = test[item].groupby(test['idhogar']).agg(function)
        new_col = item + '_' + function
        trainGP[new_col] = group_train
        testGP[new_col] = group_test
        
#adding one final feature
trainGP['age_range']=trainGP['age_max']-trainGP['age_min']
testGP['age_range']=testGP['age_max']-testGP['age_min']
trainGP['escolari_range']=trainGP['escolari_max']-trainGP['escolari_min']
testGP['escolari_range']=testGP['escolari_max']-testGP['escolari_min']

# Impute 0 to std columns (taking standard deviation on 1 value will yield NA)
trainGP["age_std"] = trainGP["age_std"].fillna(0)
trainGP["escolari_std"] = trainGP["escolari_std"].fillna(0)
trainGP["dis_std"] = trainGP["dis_std"].fillna(0)
trainGP["EduLevel_std"] = trainGP["EduLevel_std"].fillna(0)
trainGP["Schooling_Age_std"] = trainGP["Schooling_Age_std"].fillna(0)

testGP["age_std"] = testGP["age_std"].fillna(0)
testGP["escolari_std"] = testGP["escolari_std"].fillna(0)
testGP["dis_std"] = testGP["dis_std"].fillna(0)
testGP["EduLevel_std"] = testGP["EduLevel_std"].fillna(0)
testGP["Schooling_Age_std"] = testGP["Schooling_Age_std"].fillna(0)
    
trainGP.head()
testGP.head()


# In[32]:


#Sufficiency Features (self-defined conditions for a sufficient living conditions)

# =1 if predominant material on the outside wall is block or brick
# =1 if predominant material on the floor is mosaic, ceramic, terrazo
# =1 if the house has ceiling
# =1 if water provision inside the dwelling
# =1 electricity from CNFL,ICE, ESPH/JASEC
# =1 toilet connected to sewer or cesspool
# =1 main source of energy used for cooking electricity
# =1 if rubbish disposal mainly by tanker truck
# =1 if walls are good
# =1 if roof are good
# =1 if floor are good

train["GoodCondition"] = train["paredblolad"] +                                 train["pisomoscer"] +                                 train["cielorazo"] +                                 train["abastaguadentro"] +                                 train["public"] +                                 train["sanitario2"] +                                 train["energcocinar2"] +                                 train["elimbasu1"] +                                 train["epared3"] +                                 train["etecho3"] +                                 train["eviv3"] 
train["GoodCondition"] = train["GoodCondition"]/11   # Take the mean to get a GoodCondition score between 0 and 1     

# =1 if predominant material on the outside wall is socket (wood, zinc or abesto) OR prefabricated or cement
# =1 if predominant material on the roof is metal foil or zink
# =1 if water provision outside the dwelling
# =1 electricity from cooperative
# =1 toilet connected to  septic tank
# =1 main source of energy used for cooking gas
#  =1 if rubbish disposal mainly by botan hollow or buried
# =1 if walls are regular
#  =1 if roof are regular
#  =1 if floor are regular
train["AverageCondition"] = (train["paredzocalo"] + train["paredpreb"])/2 +                                 train["techozinc"] +                                 train["abastaguafuera"] +                                 train["coopele"] +                                 train["sanitario3"] +                                 train["energcocinar3"] +                                 train["elimbasu2"] +                                 train["epared2"] +                                 train["etecho2"] +                                 train["eviv2"] 

train["AverageCondition"] = train["AverageCondition"]/10   # Take the mean to get a GoodCondition score between 0 and 1     


# =1 if predominant material on the outside wall is waste material OR wood OR zinc
# =1 if predominant material on the floor is cement OR wood OR no floor
# =1 if no water provision
# =1 no electricity in the dwelling
# =1 no toilet in the dwelling OR toilet connected to black hole or letrine
# =1 no main source of energy used for cooking (no kitchen) OR  =1 main source of energy used for cooking wood charcoal
# =1 if rubbish disposal mainly by burning
# =1 if walls are bad
# =1 if roof are bad
# =1 if floor are bad
train["BadCondition"] = (train["pareddes"] + train["paredmad"] + train["paredzinc"])/3 +                         (train["pisocemento"] + train["pisonotiene"] + train["pisomadera"])/3 +                         train["abastaguano"] +                         train["noelec"] +                         (train["sanitario1"] + train["sanitario5"])/2 +                         (train["energcocinar1"] + train["energcocinar4"])/2 +                         train["elimbasu3"] +                         train["epared1"] +                         train["etecho1"] +                         train["eviv1"]

train["BadCondition"] = train["BadCondition"]/10   # Take the mean to get a GoodCondition score between 0 and 1     

# add the 3 features to the test set
test["GoodCondition"] = test["paredblolad"] +                                 test["pisomoscer"] +                                 test["cielorazo"] +                                 test["abastaguadentro"] +                                 test["public"] +                                 test["sanitario2"] +                                 test["energcocinar2"] +                                 test["elimbasu1"] +                                 test["epared3"] +                                 test["etecho3"] +                                 test["eviv3"] 
test["GoodCondition"] = test["GoodCondition"]/11   # Take the mean to get a GoodCondition score between 0 and 1     

test["AverageCondition"] = (test["paredzocalo"] + test["paredpreb"])/2 +                                 test["techozinc"] +                                 test["abastaguafuera"] +                                 test["coopele"] +                                 test["sanitario3"] +                                 test["energcocinar3"] +                                 test["elimbasu2"] +                                 test["epared2"] +                                 test["etecho2"] +                                 test["eviv2"] 

test["AverageCondition"] = test["AverageCondition"]/10   # Take the mean to get a GoodCondition score between 0 and 1     

test["BadCondition"] = (test["pareddes"] + test["paredmad"] + test["paredzinc"])/3 +                         (test["pisocemento"] + test["pisonotiene"] + test["pisomadera"])/3 +                         test["abastaguano"] +                         test["noelec"] +                         (test["sanitario1"] +test["sanitario5"])/2 +                         (test["energcocinar1"] + test["energcocinar4"])/2 +                         test["elimbasu3"] +                         test["epared1"] +                         test["etecho1"] +                         test["eviv1"]

test["BadCondition"] = test["BadCondition"]/10   # Take the mean to get a GoodCondition score between 0 and 1     


# In[33]:


trainGP = trainGP.reset_index()
testGP = testGP.reset_index()

trainML = pd.merge(train, trainGP, on='idhogar')
testML = pd.merge(test, testGP, on='idhogar')

trainML = trainML.query("parentesco1==1") #use only head of household for prediction

trainML.fillna(value=0, inplace=True)
testML.fillna(value=0, inplace=True)
trainML = trainML.replace([np.inf, -np.inf], 0)
testML = testML.replace([np.inf, -np.inf], 0)
          

train.shape, test.shape, trainGP.shape, testGP.shape,  trainML.shape, testML.shape 


# In[34]:


trainML.isna().any().any(),testML.isna().any().any()


# In[35]:


submission = testML.copy()[['Id']]


# In[36]:


trainML.drop(columns=['idhogar','Id','tamhog','r4t3','hhsize','hogar_adul','edjefe','edjefa'],inplace=True)
testML.drop(columns=['idhogar','Id','tamhog','r4t3','hhsize','hogar_adul','edjefe','edjefa'],inplace=True)


# In[37]:


correlation=trainML.corr()
correlation = correlation['Target'].sort_values(ascending=False)
print(f'The most 20 positively correlated feature: \n{correlation.head(20)}')
print('*'*50)
print(f'The most 20 negatively correlated feature: \n{correlation.tail(20)}')


# In[38]:


columns_remove = ['elimbasu5','estadocivil1','parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12']
trainML.drop(columns=columns_remove, inplace=True)
testML.drop(columns=columns_remove, inplace=True)


# In[39]:


X = trainML.drop(columns=['Target'])
y = trainML['Target']

Xtest = testML.copy()


# In[40]:


from sklearn.svm import SVC
svm_model = SVC(gamma='auto')
svm_model.fit(X, y) 


# In[41]:


test_prediction_svm = svm_model.predict(Xtest)


# In[42]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X,y)


# In[43]:


test_prediction_rf = rf_model.predict(Xtest)


# In[44]:


from sklearn.metrics import f1_score

def evaluate_macroF1_lgb(truth, predictions): 
    predictions = np.resize(predictions, new_shape=(4,len(truth)))
    pred_labels = predictions.argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    eval_name, val, is_higher_better = "macroF1", f1, True
    return (eval_name, val, is_higher_better) 

#parameter value is copied
# clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.01, objective='multiclass',
#                              random_state=123, silent=True, metric='None', 
#                              n_jobs=4, n_estimators=5000, class_weight='balanced',
#                              colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 15, subsample = 0.96)
clf = lgb.LGBMClassifier(max_depth=12, learning_rate=0.005, objective='multiclass',
                             random_state=69, silent=True, metric='None', 
                             n_jobs=4, n_estimators=2500, class_weight='balanced',
                             colsample_bytree =  0.89, min_child_samples = 80, num_leaves = 14, subsample = 0.96)

clf


# In[45]:


kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state = 123)


# In[46]:


X.shape, Xtest.shape


# In[47]:


val_predictions_logloss = []
test_predictions_logloss = []
test_predictions_logloss_proba = []
for train_index, val_index in kf.split(X, y):
    print("=======")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss",
            early_stopping_rounds=400, verbose=200)
    val_predictions_logloss.append(clf.predict(X_val)) 
    test_predictions_logloss.append(clf.predict(Xtest)) # store predictions on test set for Kaggle Submission
    test_predictions_logloss_proba.append(clf.predict_proba(Xtest)) # store predictions probability on test set for Kaggle Submission


# In[48]:


indices = np.argsort(clf.feature_importances_)[::-1]
top_100_indices = indices[:100]

# Visualise these with a barplot
plt.subplots(figsize=(20, 15))
g = sns.barplot(y=X.columns[top_100_indices], x = clf.feature_importances_[top_100_indices], orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("LightGBM feature importance")


# In[49]:


X2 = X.iloc[:,top_100_indices]
Xtest2 = testML.copy().iloc[:,top_100_indices]

# Re-training on top 100 features
test_predictions2 = []
test_predictions_proba2 = []
for train_index, val_index in kf.split(X2, y):
    print("=======")
    X_train, X_val = X2.iloc[train_index], X2.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss",
            early_stopping_rounds=400, verbose=200)
    test_predictions2.append(clf.predict(Xtest2)) # store predictions on test set for Kaggle Submission
    test_predictions_proba2.append(clf.predict_proba(Xtest2)) 


# In[50]:


# SVM Model Predictions
submission_svm = submission.copy()
submission_svm['Target'] = np.array(test_prediction_svm).astype(int)
submission_svm.to_csv("submission_01.csv", index=False)
print(submission_svm['Target'].value_counts()/submission_svm['Target'].value_counts().sum())
print("The Macro F1 Score (Test) is: 0.28218")


# In[51]:


# Random Forest Model Predictions
submission_rf = submission.copy()
submission_rf['Target'] = np.array(test_prediction_rf).astype(int)
submission_rf.to_csv("submission_02.csv", index=False)
print(submission_rf['Target'].value_counts()/submission_rf['Target'].value_counts().sum())
print("The Macro F1 Score (Test) is: 0.37434")


# In[52]:


# All Features Light GBM Model Predictions
submission_full_logloss = submission.copy()
voted_test_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=test_predictions_logloss) # Take the mode
submission_full_logloss['Target'] = np.array(voted_test_predictions).astype(int)
submission_full_logloss.to_csv("submission_03.csv", index=False)
print(submission_full_logloss['Target'].value_counts()/submission_full_logloss['Target'].value_counts().sum())
print("The Macro F1 Score (Test) is: 0.43970")


# In[53]:


# Top-100 Features Light GBM Model Predictions
submission_top100_logloss = submission.copy()
voted_test_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=test_predictions2) # Take the mode
submission_top100_logloss['Target'] = np.array(voted_test_predictions).astype(int)
submission_top100_logloss.to_csv("submission_04.csv", index=False)
print(submission_top100_logloss['Target'].value_counts()/submission_top100_logloss['Target'].value_counts().sum())
print("The Macro F1 Score (Test) is: 0.44019")

