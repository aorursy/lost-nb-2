#!/usr/bin/env python
# coding: utf-8

# In[17]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[26]:


df=pd.read_csv("../input/train.csv")


# In[27]:


df_test=pd.read_csv("../input/test.csv")


# In[28]:


###Replace -1 with NaN
df.replace(to_replace=-1,value=np.NaN,inplace=True)
df_test.replace(to_replace=-1,value=np.NaN,inplace=True)


# In[29]:


#####after the execution of the  below block all_vars variable will only have the numerical column 
all_vars=list(df.columns)
cat_var=[x for x in df.columns if "cat" in x]
bin_var=[x for x in df.columns if "bin" in x]
comb_vars=cat_var+bin_var
for x in comb_vars:
    all_vars.remove(x)

##removing the target and the id variable
#all_vars.remove("target")
all_vars.remove("id")


dep="target"


# In[30]:


#####treating missing values in Continous columns
df.loc[pd.isnull(df["ps_reg_03"]),"ps_reg_03"]=0.894047
df.loc[pd.isnull(df["ps_car_14"]),"ps_car_14"]=0.374691
df.loc[pd.isnull(df["ps_car_11"]),"ps_car_11"]=2.34
df.loc[pd.isnull(df["ps_car_12"]),"ps_car_12"]=0.379

df_test.loc[pd.isnull(df_test["ps_reg_03"]),"ps_reg_03"]=0.894047
df_test.loc[pd.isnull(df_test["ps_car_14"]),"ps_car_14"]=0.374691
df_test.loc[pd.isnull(df_test["ps_car_11"]),"ps_car_11"]=2.34
df_test.loc[pd.isnull(df_test["ps_car_12"]),"ps_car_12"]=0.379



#####treating high outliers in Continous columns
df.loc[df['ps_ind_14'] >= 0.395086,'ps_ind_14']=0.395086
df.loc[df['ps_reg_02'] >= 1.651976,'ps_reg_02']=1.651976
df.loc[df['ps_reg_03'] >= 1.930286,'ps_reg_03']=1.930286
df.loc[df['ps_car_12'] >= 0.554847,'ps_car_12']=0.554847
df.loc[df['ps_car_13'] >= 1.487029,'ps_car_13']=1.487029
df.loc[df['ps_car_14'] >= 0.511521,'ps_car_14']=0.511521
df.loc[df['ps_calc_05'] >= 5.290667,'ps_calc_05']=5.290667
df.loc[df['ps_calc_07'] >= 7.249515,'ps_calc_07']=7.249515
df.loc[df['ps_calc_09'] >= 6.079881,'ps_calc_09']=6.079881
df.loc[df['ps_calc_10'] >= 17.147381,'ps_calc_10']=17.147381
df.loc[df['ps_calc_11'] >= 12.439995,'ps_calc_11']=12.439995
df.loc[df['ps_calc_12'] >= 5.050807,'ps_calc_12']=5.050807
df.loc[df['ps_calc_13'] >= 7.956949,'ps_calc_13']=7.956949
df.loc[df['ps_calc_14'] >= 15.778982,'ps_calc_14']=15.778982

# #####treating low outliers in Continous columns
df.loc[df['ps_car_12'] <= 0.205047,'ps_car_12']=0.205047
df.loc[df['ps_car_14'] <= 0.237861,'ps_car_14']=0.237861
df.loc[df['ps_car_15'] <= 0.871801,'ps_car_15']=0.871801
df.loc[df['ps_calc_06'] <= 3.686509,'ps_calc_06']=3.686509
df.loc[df['ps_calc_08'] <= 4.846888,'ps_calc_08']=4.846888


# In[31]:


######Treating missing valeues in categorical variables

cat_var_havingmissing_vals=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat',
'ps_car_02_cat','ps_car_03_cat','ps_car_05_cat','ps_car_07_cat','ps_car_09_cat']

for var in cat_var_havingmissing_vals:
    category_with_max_counts=df[var].value_counts().sort_values(ascending=False).index[0]
    df.loc[pd.isnull(df[var]),var]=category_with_max_counts


# In[32]:


####converting the categorical to one hots

df_cat_one_hots=pd.get_dummies(df[cat_var],columns=cat_var)
df_test_cat_one_hots=pd.get_dummies(df_test[cat_var],columns=cat_var)


# In[36]:



#######3creating a new data frame having one hot encoded categorical variable and discarded the original categorical vars
df_cat_one_hots.reset_index(drop=True,inplace=True)

new_df=pd.concat([df[all_vars],df[bin_var],df_cat_one_hots],axis=1)


df_test_cat_one_hots.reset_index(drop=True,inplace=True)

new_df_test=pd.concat([df_test[['ps_ind_01',
 'ps_ind_03',
 'ps_ind_14',
 'ps_ind_15',
 'ps_reg_01',
 'ps_reg_02',
 'ps_reg_03',
 'ps_car_11',
 'ps_car_12',
 'ps_car_13',
 'ps_car_14',
 'ps_car_15',
 'ps_calc_01',
 'ps_calc_02',
 'ps_calc_03',
 'ps_calc_04',
 'ps_calc_05',
 'ps_calc_06',
 'ps_calc_07',
 'ps_calc_08',
 'ps_calc_09',
 'ps_calc_10',
 'ps_calc_11',
 'ps_calc_12',
 'ps_calc_13',
 'ps_calc_14']],df_test[bin_var],df_test_cat_one_hots],axis=1)


# In[9]:


####scaling the numerical column for PCA 
# from sklearn.preprocessing import StandardScaler
# s=StandardScaler()
# sf=s.fit(new_df[all_vars])
# scaled_num_df=pd.DataFrame(sf.transform(new_df[all_vars]),columns=new_df[all_vars].columns)


# In[10]:


# scaled_num_df.reset_index(drop=True,inplace=True)
# pca_dataset=pd.concat([new_df[bin_var],df_cat_one_hots,scaled_num_df ],axis=1)


# In[11]:


# from sklearn.decomposition import PCA,KernelPCA
# pca=PCA(n_components=52)
# reduced_df=pca.fit_transform(pca_dataset)

# pca=KernelPCA(n_components=30,kernel="poly",fit_inverse_transform=True)
# reduced_df=pca.fit_transform(pca_dataset)


# In[12]:


# np.cumsum(pca.explained_variance_ratio_)


# In[13]:


# ########3creating the pca reduced df with proper columns names
# cols=['PCA_1',
# 'PCA_2','PCA_3',
# 'PCA_4','PCA_5','PCA_6','PCA_7','PCA_8','PCA_9','PCA_10',
# 'PCA_11','PCA_12','PCA_13','PCA_14','PCA_15','PCA_16','PCA_17','PCA_18',
# 'PCA_19','PCA_20','PCA_21','PCA_22','PCA_23','PCA_24','PCA_25','PCA_26','PCA_27','PCA_28','PCA_29','PCA_30',
# 'PCA_31','PCA_32','PCA_33','PCA_34','PCA_35','PCA_36','PCA_37','PCA_38','PCA_39','PCA_40','PCA_41','PCA_42',
# 'PCA_43','PCA_44','PCA_45','PCA_46','PCA_47','PCA_48','PCA_49','PCA_50','PCA_51','PCA_52']

# pca_reduced_df=pd.DataFrame(reduced_df,columns=cols)


# In[14]:


###getting the dependednt column back in the pca reduced dataframe
# pca_reduced_df["target"]=df["target"].values


# In[15]:


# from sklearn.utils import resample
# pca_reduced_df_0=pca_reduced_df[pca_reduced_df["target"]==0]
# pca_reduced_df_1=pca_reduced_df[pca_reduced_df["target"]==1]
# pca_reduced_df_1_upsampled=resample(pca_reduced_df_1,replace=True,n_samples=400000,random_state=100)
# pca_reduced_df_sampled=pd.concat([pca_reduced_df_1_upsampled,pca_reduced_df],axis=0)

# # pca_reduced_df_0_downsampled=resample(pca_reduced_df_0,replace=False,n_samples=30000,random_state=100)
# # pca_reduced_df_sampled=pd.concat([pca_reduced_df_0_downsampled,pca_reduced_df_1],axis=0)
# # pca_reduced_df_sampled[dep].value_counts()


# In[16]:


# ######splittting the data in train and test and validation
# from sklearn.model_selection import train_test_split
# #train,test=train_test_split(pca_reduced_df,test_size=0.3,random_state=100)
# train,test=train_test_split(pca_reduced_df_sampled,test_size=0.3,random_state=100)


# In[17]:


# train[dep].value_counts()


# In[58]:


def randomforestclf(df,dep,indp):
    ###importing the essential components
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    ####sepearting the dependent 
    cols=indp
    
    m=RandomForestClassifier(random_state=10,n_jobs=-1,class_weight="balanced")
    ###defining the parameter list for grid search cv
    params={'n_estimators': [500,700,800] ,
           'max_depth':[3,4,5],
           'min_samples_split':[0.2,0.4]}
    ###giving the gridsearch parameters
    g=GridSearchCV(estimator=m,param_grid=params,scoring="f1")
    gfit=g.fit(df[cols],df[dep])
    return(gfit.best_estimator_)


# In[59]:


# rf=randomforestclf(train,dep,cols)


# In[43]:


# from sklearn.ensemble import RandomForestClassifier
# m=RandomForestClassifier(n_jobs=-1,n_estimators=250,max_depth=2,min_samples_split=50000,bootstrap=False)
# model=m.fit(train[cols],train[dep])


# In[60]:


# rf.get_params


# In[44]:


#  prediction=model.predict(train[cols])
#  from sklearn.metrics import classification_report
#  print(classification_report(train[dep],prediction))


# In[46]:


# prediction_test=model.predict(test[cols])
# print(classification_report(test[dep],prediction_test))


# In[32]:


# rf.get_params


# In[37]:


from sklearn.metrics import make_scorer,f1_score
# scorer=make_scorer(f1_score)


# In[81]:


# from sklearn.metrics import auc,roc_auc_score
# rf_predictions_train_proba=rf.predict_proba(train[cols])
# rf_predictions_train_proba=rf_predictions_train_proba[0:][0:,1]###prediction of only positive class
# roc_auc_score(train[dep],rf_predictions_train_proba)


# In[82]:


# rf_predictions_test_proba=rf.predict_proba(test[cols])
# rf_predictions_test_proba=rf_predictions_test_proba[0:][0:,1]###prediction of only positive class
# roc_auc_score(test[dep],rf_predictions_test_proba)


# In[80]:


# def xgbclf(df,dep,indp):
#     import xgboost as xgb
#     from sklearn.model_selection import GridSearchCV
#     ####sepearting the dependent 
#     cols=indp
#     xg=xgb.XGBClassifier(random_state=10,n_jobs=-1)
#     params={'n_estimators':range(800),
#             'max_depth':[3,4],
#             'learning_rate':[0.03]  } 
#     g=GridSearchCV(estimator=xg,param_grid=params,scoring="accuracy")
#     gfit=g.fit(df[indp],df[dep])
    
#     return(gfit.best_estimator_)


# In[81]:


# xg=xgbclf(train,dep,cols)


# In[78]:


# xg.get_params


# In[79]:


# xg_prediction=xg.predict(train[cols])
# from sklearn.metrics import classification_report
# print(classification_report(train[dep],xg_prediction))


# In[38]:


from sklearn.ensemble import RandomForestClassifier


   


# In[39]:


from sklearn.utils import resample
new_df_0=new_df[df["target"]==0]
new_df_1=new_df[df["target"]==1]
new_df_1_upsampled=resample(new_df_1,replace=True,n_samples=400000,random_state=100)
new_df_sampled=pd.concat([new_df_1_upsampled,new_df],axis=0)


# In[40]:


indp=['ps_ind_01',
 'ps_ind_03',
 'ps_ind_14',
 'ps_ind_15',
 'ps_reg_01',
 'ps_reg_02',
 'ps_reg_03',
 'ps_car_11',
 'ps_car_12',
 'ps_car_13',
 'ps_car_14',
 'ps_car_15',
 'ps_calc_01',
 'ps_calc_02',
 'ps_calc_03',
 'ps_calc_04',
 'ps_calc_05',
 'ps_calc_06',
 'ps_calc_07',
 'ps_calc_08',
 'ps_calc_09',
 'ps_calc_10',
 'ps_calc_11',
 'ps_calc_12',
 'ps_calc_13',
 'ps_calc_14',
 'ps_ind_06_bin',
 'ps_ind_07_bin',
 'ps_ind_08_bin',
 'ps_ind_09_bin',
 'ps_ind_10_bin',
 'ps_ind_11_bin',
 'ps_ind_12_bin',
 'ps_ind_13_bin',
 'ps_ind_16_bin',
 'ps_ind_17_bin',
 'ps_ind_18_bin',
 'ps_calc_15_bin',
 'ps_calc_16_bin',
 'ps_calc_17_bin',
 'ps_calc_18_bin',
 'ps_calc_19_bin',
 'ps_calc_20_bin',
 'ps_ind_02_cat_1.0',
 'ps_ind_02_cat_2.0',
 'ps_ind_02_cat_3.0',
 'ps_ind_02_cat_4.0',
 'ps_ind_04_cat_0.0',
 'ps_ind_04_cat_1.0',
 'ps_ind_05_cat_0.0',
 'ps_ind_05_cat_1.0',
 'ps_ind_05_cat_2.0',
 'ps_ind_05_cat_3.0',
 'ps_ind_05_cat_4.0',
 'ps_ind_05_cat_5.0',
 'ps_ind_05_cat_6.0',
 'ps_car_01_cat_0.0',
 'ps_car_01_cat_1.0',
 'ps_car_01_cat_2.0',
 'ps_car_01_cat_3.0',
 'ps_car_01_cat_4.0',
 'ps_car_01_cat_5.0',
 'ps_car_01_cat_6.0',
 'ps_car_01_cat_7.0',
 'ps_car_01_cat_8.0',
 'ps_car_01_cat_9.0',
 'ps_car_01_cat_10.0',
 'ps_car_01_cat_11.0',
 'ps_car_02_cat_0.0',
 'ps_car_02_cat_1.0',
 'ps_car_03_cat_0.0',
 'ps_car_03_cat_1.0',
 'ps_car_04_cat_0',
 'ps_car_04_cat_1',
 'ps_car_04_cat_2',
 'ps_car_04_cat_3',
 'ps_car_04_cat_4',
 'ps_car_04_cat_5',
 'ps_car_04_cat_6',
 'ps_car_04_cat_7',
 'ps_car_04_cat_8',
 'ps_car_04_cat_9',
 'ps_car_05_cat_0.0',
 'ps_car_05_cat_1.0',
 'ps_car_06_cat_0',
 'ps_car_06_cat_1',
 'ps_car_06_cat_2',
 'ps_car_06_cat_3',
 'ps_car_06_cat_4',
 'ps_car_06_cat_5',
 'ps_car_06_cat_6',
 'ps_car_06_cat_7',
 'ps_car_06_cat_8',
 'ps_car_06_cat_9',
 'ps_car_06_cat_10',
 'ps_car_06_cat_11',
 'ps_car_06_cat_12',
 'ps_car_06_cat_13',
 'ps_car_06_cat_14',
 'ps_car_06_cat_15',
 'ps_car_06_cat_16',
 'ps_car_06_cat_17',
 'ps_car_07_cat_0.0',
 'ps_car_07_cat_1.0',
 'ps_car_08_cat_0',
 'ps_car_08_cat_1',
 'ps_car_09_cat_0.0',
 'ps_car_09_cat_1.0',
 'ps_car_09_cat_2.0',
 'ps_car_09_cat_3.0',
 'ps_car_09_cat_4.0',
 'ps_car_10_cat_0',
 'ps_car_10_cat_1',
 'ps_car_10_cat_2',
 'ps_car_11_cat_1',
 'ps_car_11_cat_2',
 'ps_car_11_cat_3',
 'ps_car_11_cat_4',
 'ps_car_11_cat_5',
 'ps_car_11_cat_6',
 'ps_car_11_cat_7',
 'ps_car_11_cat_8',
 'ps_car_11_cat_9',
 'ps_car_11_cat_10',
 'ps_car_11_cat_11',
 'ps_car_11_cat_12',
 'ps_car_11_cat_13',
 'ps_car_11_cat_14',
 'ps_car_11_cat_15',
 'ps_car_11_cat_16',
 'ps_car_11_cat_17',
 'ps_car_11_cat_18',
 'ps_car_11_cat_19',
 'ps_car_11_cat_20',
 'ps_car_11_cat_21',
 'ps_car_11_cat_22',
 'ps_car_11_cat_23',
 'ps_car_11_cat_24',
 'ps_car_11_cat_25',
 'ps_car_11_cat_26',
 'ps_car_11_cat_27',
 'ps_car_11_cat_28',
 'ps_car_11_cat_29',
 'ps_car_11_cat_30',
 'ps_car_11_cat_31',
 'ps_car_11_cat_32',
 'ps_car_11_cat_33',
 'ps_car_11_cat_34',
 'ps_car_11_cat_35',
 'ps_car_11_cat_36',
 'ps_car_11_cat_37',
 'ps_car_11_cat_38',
 'ps_car_11_cat_39',
 'ps_car_11_cat_40',
 'ps_car_11_cat_41',
 'ps_car_11_cat_42',
 'ps_car_11_cat_43',
 'ps_car_11_cat_44',
 'ps_car_11_cat_45',
 'ps_car_11_cat_46',
 'ps_car_11_cat_47',
 'ps_car_11_cat_48',
 'ps_car_11_cat_49',
 'ps_car_11_cat_50',
 'ps_car_11_cat_51',
 'ps_car_11_cat_52',
 'ps_car_11_cat_53',
 'ps_car_11_cat_54',
 'ps_car_11_cat_55',
 'ps_car_11_cat_56',
 'ps_car_11_cat_57',
 'ps_car_11_cat_58',
 'ps_car_11_cat_59',
 'ps_car_11_cat_60',
 'ps_car_11_cat_61',
 'ps_car_11_cat_62',
 'ps_car_11_cat_63',
 'ps_car_11_cat_64',
 'ps_car_11_cat_65',
 'ps_car_11_cat_66',
 'ps_car_11_cat_67',
 'ps_car_11_cat_68',
 'ps_car_11_cat_69',
 'ps_car_11_cat_70',
 'ps_car_11_cat_71',
 'ps_car_11_cat_72',
 'ps_car_11_cat_73',
 'ps_car_11_cat_74',
 'ps_car_11_cat_75',
 'ps_car_11_cat_76',
 'ps_car_11_cat_77',
 'ps_car_11_cat_78',
 'ps_car_11_cat_79',
 'ps_car_11_cat_80',
 'ps_car_11_cat_81',
 'ps_car_11_cat_82',
 'ps_car_11_cat_83',
 'ps_car_11_cat_84',
 'ps_car_11_cat_85',
 'ps_car_11_cat_86',
 'ps_car_11_cat_87',
 'ps_car_11_cat_88',
 'ps_car_11_cat_89',
 'ps_car_11_cat_90',
 'ps_car_11_cat_91',
 'ps_car_11_cat_92',
 'ps_car_11_cat_93',
 'ps_car_11_cat_94',
 'ps_car_11_cat_95',
 'ps_car_11_cat_96',
 'ps_car_11_cat_97',
 'ps_car_11_cat_98',
 'ps_car_11_cat_99',
 'ps_car_11_cat_100',
 'ps_car_11_cat_101',
 'ps_car_11_cat_102',
 'ps_car_11_cat_103',
 'ps_car_11_cat_104']
dep="target"


# In[41]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(new_df_sampled,test_size=0.3,random_state=100)


# In[42]:


from sklearn.ensemble import RandomForestClassifier
m=RandomForestClassifier(max_depth=20,n_estimators=1500,n_jobs=-1)
m.fit(train[indp],train[dep])


# In[43]:


prediction=m.predict(train[indp])
from sklearn.metrics import classification_report
print(classification_report(train[dep],prediction))


# In[ ]:


prediction_test=m.predict(test[indp])
print(classification_report(test[dep],prediction_test))


# In[ ]:


predictions_test=m.predict_proba(new_df_test[indp])


# In[ ]:


predictions_test_new=predictions_test[0:,1]


# In[ ]:


submission_2=pd.DataFrame({'id':df_test["id"],
            'target':predictions_test_new})


# In[ ]:


submission_2.to_csv("./submission2.csv", index=False)

