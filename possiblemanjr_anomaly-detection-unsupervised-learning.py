#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score 
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve ,auc , log_loss ,  classification_report 
from sklearn.preprocessing import StandardScaler , Binarizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import time
import os, sys, gc, warnings, random, datetime
import math
import shap
import joblib
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold , cross_val_score
from sklearn.metrics import roc_auc_score


# In[2]:


df = pd.read_pickle('/kaggle/input/ieee-fe-with-some-eda/train_df.pkl')


# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

df = reduce_mem_usage(df)


# In[4]:


df.head()


# In[5]:



remove_features = pd.read_pickle('../input/ieee-fe-with-some-eda/remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)
print('Shape control:', df.shape)


# In[6]:


remove_features.remove('isFraud')


# In[7]:


remove_features += [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'is_december', 'is_holiday', 'card1_fq_enc', 'card2_fq_enc', 'card3_fq_enc', 'card5_fq_enc', 'uid_fq_enc', 'uid2_fq_enc', 'uid3_fq_enc', 'uid4_fq_enc', 'uid5_fq_enc', 'card3_DT_D_hour_dist', 'card3_DT_W_week_day_dist', 'card3_DT_M_month_day_dist', 'card3_DT_D_hour_dist_best', 'card3_DT_W_week_day_dist_best', 'card3_DT_M_month_day_dist_best', 'card5_DT_D_hour_dist', 'card5_DT_W_week_day_dist', 'card5_DT_M_month_day_dist', 'card5_DT_D_hour_dist_best', 'card5_DT_W_week_day_dist_best', 'card5_DT_M_month_day_dist_best', 'bank_type_DT_D_hour_dist', 'bank_type_DT_W_week_day_dist', 'bank_type_DT_M_month_day_dist', 'bank_type_DT_D_hour_dist_best', 'bank_type_DT_W_week_day_dist_best', 'bank_type_DT_M_month_day_dist_best', 'bank_type_DT_M', 'bank_type_DT_W', 'bank_type_DT_D', 'uid_D1_mean', 'uid_D1_std', 'uid2_D1_mean', 'uid2_D1_std', 'uid3_D1_mean', 'uid3_D1_std', 'uid4_D1_mean', 'uid4_D1_std', 'uid5_D1_mean', 'uid5_D1_std', 'bank_type_D1_mean', 'bank_type_D1_std', 'uid_D2_mean', 'uid_D2_std', 'uid2_D2_mean', 'uid2_D2_std', 'uid3_D2_mean', 'uid3_D2_std', 'uid4_D2_mean', 'uid4_D2_std', 'uid5_D2_mean', 'uid5_D2_std', 'bank_type_D2_mean', 'bank_type_D2_std', 'uid_D3_mean', 'uid_D3_std', 'uid2_D3_mean', 'uid2_D3_std', 'uid3_D3_mean', 'uid3_D3_std', 'uid4_D3_mean', 'uid4_D3_std', 'uid5_D3_mean', 'uid5_D3_std', 'bank_type_D3_mean', 'bank_type_D3_std', 'uid_D4_mean', 'uid_D4_std', 'uid2_D4_mean', 'uid2_D4_std', 'uid3_D4_mean', 'uid3_D4_std', 'uid4_D4_mean', 'uid4_D4_std', 'uid5_D4_mean', 'uid5_D4_std', 'bank_type_D4_mean', 'bank_type_D4_std', 'uid_D5_mean', 'uid_D5_std', 'uid2_D5_mean', 'uid2_D5_std', 'uid3_D5_mean', 'uid3_D5_std', 'uid4_D5_mean', 'uid4_D5_std', 'uid5_D5_mean', 'uid5_D5_std', 'bank_type_D5_mean', 'bank_type_D5_std', 'uid_D6_mean', 'uid_D6_std', 'uid2_D6_mean', 'uid2_D6_std', 'uid3_D6_mean', 'uid3_D6_std', 'uid4_D6_mean', 'uid4_D6_std', 'uid5_D6_mean', 'uid5_D6_std', 'bank_type_D6_mean', 'bank_type_D6_std', 'uid_D7_mean', 'uid_D7_std', 'uid2_D7_mean', 'uid2_D7_std', 'uid3_D7_mean', 'uid3_D7_std', 'uid4_D7_mean', 'uid4_D7_std', 'uid5_D7_mean', 'uid5_D7_std', 'bank_type_D7_mean', 'bank_type_D7_std', 'uid_D8_mean', 'uid_D8_std', 'uid2_D8_mean', 'uid2_D8_std', 'uid3_D8_mean', 'uid3_D8_std', 'uid4_D8_mean', 'uid4_D8_std', 'uid5_D8_mean', 'uid5_D8_std', 'bank_type_D8_mean', 'bank_type_D8_std', 'uid_D9_mean', 'uid_D9_std', 'uid2_D9_mean', 'uid2_D9_std', 'uid3_D9_mean', 'uid3_D9_std', 'uid4_D9_mean', 'uid4_D9_std', 'uid5_D9_mean', 'uid5_D9_std', 'bank_type_D9_mean', 'bank_type_D9_std', 'uid_D10_mean', 'uid_D10_std', 'uid2_D10_mean', 'uid2_D10_std', 'uid3_D10_mean', 'uid3_D10_std', 'uid4_D10_mean', 'uid4_D10_std', 'uid5_D10_mean', 'uid5_D10_std', 'bank_type_D10_mean', 'bank_type_D10_std', 'uid_D11_mean', 'uid_D11_std', 'uid2_D11_mean', 'uid2_D11_std', 'uid3_D11_mean', 'uid3_D11_std', 'uid4_D11_mean', 'uid4_D11_std', 'uid5_D11_mean', 'uid5_D11_std', 'bank_type_D11_mean', 'bank_type_D11_std', 'uid_D12_mean', 'uid_D12_std', 'uid2_D12_mean', 'uid2_D12_std', 'uid3_D12_mean', 'uid3_D12_std', 'uid4_D12_mean', 'uid4_D12_std', 'uid5_D12_mean', 'uid5_D12_std', 'bank_type_D12_mean', 'bank_type_D12_std', 'uid_D13_mean', 'uid_D13_std', 'uid2_D13_mean', 'uid2_D13_std', 'uid3_D13_mean', 'uid3_D13_std', 'uid4_D13_mean', 'uid4_D13_std', 'uid5_D13_mean', 'uid5_D13_std', 'bank_type_D13_mean', 'bank_type_D13_std', 'uid_D14_mean', 'uid_D14_std', 'uid2_D14_mean', 'uid2_D14_std', 'uid3_D14_mean', 'uid3_D14_std', 'uid4_D14_mean', 'uid4_D14_std', 'uid5_D14_mean', 'uid5_D14_std', 'bank_type_D14_mean', 'bank_type_D14_std', 'uid_D15_mean', 'uid_D15_std', 'uid2_D15_mean', 'uid2_D15_std', 'uid3_D15_mean', 'uid3_D15_std', 'uid4_D15_mean', 'uid4_D15_std', 'uid5_D15_mean', 'uid5_D15_std', 'bank_type_D15_mean', 'bank_type_D15_std', 'D9_not_na', 'D8_not_same_day', 'D8_D9_decimal_dist', 'D3_DT_D_min_max', 'D3_DT_D_std_score', 'D4_DT_D_min_max', 'D4_DT_D_std_score', 'D5_DT_D_min_max', 'D5_DT_D_std_score', 'D6_DT_D_min_max', 'D6_DT_D_std_score', 'D7_DT_D_min_max', 'D7_DT_D_std_score', 'D8_DT_D_min_max', 'D8_DT_D_std_score', 'D10_DT_D_min_max', 'D10_DT_D_std_score', 'D11_DT_D_min_max', 'D11_DT_D_std_score', 'D12_DT_D_min_max', 'D12_DT_D_std_score', 'D13_DT_D_min_max', 'D13_DT_D_std_score', 'D14_DT_D_min_max', 'D14_DT_D_std_score', 'D15_DT_D_min_max', 'D15_DT_D_std_score', 'D3_DT_W_min_max', 'D3_DT_W_std_score', 'D4_DT_W_min_max', 'D4_DT_W_std_score', 'D5_DT_W_min_max', 'D5_DT_W_std_score', 'D6_DT_W_min_max', 'D6_DT_W_std_score', 'D7_DT_W_min_max', 'D7_DT_W_std_score', 'D8_DT_W_min_max', 'D8_DT_W_std_score', 'D10_DT_W_min_max', 'D10_DT_W_std_score', 'D11_DT_W_min_max', 'D11_DT_W_std_score', 'D12_DT_W_min_max', 'D12_DT_W_std_score', 'D13_DT_W_min_max', 'D13_DT_W_std_score', 'D14_DT_W_min_max', 'D14_DT_W_std_score', 'D15_DT_W_min_max', 'D15_DT_W_std_score', 'D3_DT_M_min_max', 'D3_DT_M_std_score', 'D4_DT_M_min_max', 'D4_DT_M_std_score', 'D5_DT_M_min_max', 'D5_DT_M_std_score', 'D6_DT_M_min_max', 'D6_DT_M_std_score', 'D7_DT_M_min_max', 'D7_DT_M_std_score', 'D8_DT_M_min_max', 'D8_DT_M_std_score', 'D10_DT_M_min_max', 'D10_DT_M_std_score', 'D11_DT_M_min_max', 'D11_DT_M_std_score', 'D12_DT_M_min_max', 'D12_DT_M_std_score', 'D13_DT_M_min_max', 'D13_DT_M_std_score', 'D14_DT_M_min_max', 'D14_DT_M_std_score', 'D15_DT_M_min_max', 'D15_DT_M_std_score', 'D1_scaled', 'D2_scaled', 'TransactionAmt_check', 'card1_TransactionAmt_mean', 'card1_TransactionAmt_std', 'card2_TransactionAmt_mean', 'card2_TransactionAmt_std', 'card3_TransactionAmt_mean', 'card3_TransactionAmt_std', 'card5_TransactionAmt_mean', 'card5_TransactionAmt_std', 'uid_TransactionAmt_mean', 'uid_TransactionAmt_std', 'uid2_TransactionAmt_mean', 'uid2_TransactionAmt_std', 'uid3_TransactionAmt_mean', 'uid3_TransactionAmt_std', 'uid4_TransactionAmt_mean', 'uid4_TransactionAmt_std', 'uid5_TransactionAmt_mean', 'uid5_TransactionAmt_std', 'bank_type_TransactionAmt_mean', 'bank_type_TransactionAmt_std', 'TransactionAmt_DT_D_min_max', 'TransactionAmt_DT_D_std_score', 'TransactionAmt_DT_W_min_max', 'TransactionAmt_DT_W_std_score', 'TransactionAmt_DT_M_min_max', 'TransactionAmt_DT_M_std_score', 'product_type', 'product_type_DT_D', 'product_type_DT_W', 'product_type_DT_M', 'C1_fq_enc', 'C2_fq_enc', 'C3_fq_enc', 'C4_fq_enc', 'C5_fq_enc', 'C6_fq_enc', 'C7_fq_enc', 'C8_fq_enc', 'C9_fq_enc', 'C10_fq_enc', 'C11_fq_enc', 'C12_fq_enc', 'C13_fq_enc', 'C14_fq_enc', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'id_33_0', 'id_33_1', 'DeviceInfo_device', 'DeviceInfo_version', 'id_30_device', 'id_30_version', 'id_31_device']


# In[8]:


remove_features


# In[9]:


features_columns = [col for col in list(df) if col not in remove_features]


# In[10]:


features_columns


# In[11]:


df = df[features_columns]


# In[12]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['isFraud'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('isFraud')
ax[0].set_ylabel('')
sns.countplot('isFraud',data=df,ax=ax[1])
ax[1].set_title('isFraud')
plt.show()


# In[13]:


X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)


# In[14]:


get_ipython().system('pip install fastcluster')


# In[15]:


###Libraries


import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn import preprocessing as pp
#from sklearn import impute.SimpleImputer as pp
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import fastcluster
from scipy.cluster.hierarchy import dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
import os, sys, gc, warnings, random, datetime
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA


# In[16]:


def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


# In[17]:


def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds =         precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
    average_precision =         average_precision_score(preds['trueLabel'],preds['anomalyScore'])
    
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve: Average Precision =     {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'],                                      preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic:     Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.show()
    
    if returnPreds==True:
        return preds


# In[18]:


def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label",                data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using "+algoName)


# In[19]:


len(df.columns)


# In[20]:


df.head()


# In[21]:


df11 = df.fillna(pd.Series(-1, index=df.select_dtypes(exclude='category').columns))


# In[22]:


df11.head()


# In[23]:


df11.isnull().sum()


# In[24]:


len(df.columns)


# In[25]:


X = df11.drop('isFraud', axis=1)
from sklearn import preprocessing as pp
featuresToScale = X.columns
sX = pp.MinMaxScaler(copy=True)
X.loc[:,featuresToScale] = sX.fit_transform(X[featuresToScale])


y = df11['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)


# In[26]:


from sklearn.decomposition import PCA

n_components = 50
whiten = False
random_state = 2020

pca = PCA(n_components=n_components, whiten=whiten,           random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse,                                    index=X_train.index)

scatterPlot(X_train_PCA, y_train, "PCA")


# In[27]:


from sklearn.decomposition import PCA

n_components = range(30,50)

for i in n_components:
    print('Iteration ', i)
    whiten = False
    random_state = 2020

    pca = PCA(n_components=i, whiten=whiten,               random_state=random_state)

    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

    X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
    X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse,                                        index=X_train.index)
    
    anomalyScoresPCA = anomalyScores(X_train, X_train_PCA_inverse)
    preds = plotResults(y_train, anomalyScoresPCA, True)

    


# In[ ]:




