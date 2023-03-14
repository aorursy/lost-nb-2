#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcdefaults()
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[3]:


print('-' * 80)
print('df_transaction')
df_transaction = import_data('../input/ieee-fraud-detection/train_transaction.csv') # read the dataset train_transaction


# In[4]:


print('-' * 80)
print('df_identity')
df_identity = import_data('../input/ieee-fraud-detection/train_identity.csv')


# In[5]:


print('-' * 80)
print('test_transaction')
test_transaction = import_data('../input/ieee-fraud-detection/test_transaction.csv')


# In[6]:


print('-' * 80)
print('test_identity')
test_identity = import_data('../input/ieee-fraud-detection/test_identity.csv')


# In[7]:


print("The dataset has {} rows and {} columns.".format(*df_transaction.shape))
print("It contains {} duplicates.".format(df_transaction.duplicated().sum()))


# In[8]:


print("The dataset has {} rows and {} columns.".format(*df_identity.shape))
print("It contains {} duplicates.".format(df_identity.duplicated().sum()))


# In[9]:


print("The dataset has {} rows and {} columns.".format(*test_transaction.shape))
print("It contains {} duplicates.".format(test_transaction.duplicated().sum()))


# In[10]:


print("The dataset has {} rows and {} columns.".format(*test_identity.shape))
print("It contains {} duplicates.".format(test_identity.duplicated().sum()))


# In[11]:


df = pd.merge(df_transaction, df_identity,how="left",on="TransactionID")
print("The dataset has {} rows and {} columns.".format(*df.shape))
print("It contains {} duplicates.".format(df.duplicated().sum()))


# In[12]:


test = pd.merge(test_transaction, test_identity,how="left",on="TransactionID")
print("The dataset has {} rows and {} columns.".format(*test.shape))
print("It contains {} duplicates.".format(test.duplicated().sum()))


# In[13]:


df.head()


# In[14]:


df_raw=df


# In[15]:


df_raw.columns  


# In[16]:


df_grouped = pd.DataFrame(df.groupby(['card4'])['TransactionAmt'].agg(np.median)) #vizualisation of data
df_grouped.reset_index(inplace=True)
df_grouped.sort_values(by=['TransactionAmt'], ascending=True)          .plot(kind='barh', x='card4', y='TransactionAmt', 
                figsize=(9,5), legend=False, color='darkblue')# visualizatoin of transaction amount data
plt.xlabel('\nTypes of cards', fontsize=12)
plt.ylabel('Median Transaction Amount\n', fontsize=12)
plt.title('\nMedian Transaction Amount by Types of Cards\n', fontsize=14, fontweight='bold');


# In[17]:


df_grouped = pd.DataFrame(df.groupby(['card4'])['isFraud'].agg(np.sum))
df_grouped.reset_index(inplace=True)
df_grouped.sort_values(by=['isFraud'], ascending=True)          .plot(kind='barh', x='card4', y='isFraud', 
                figsize=(9,5), legend=False, color='darkblue')# visualization of fraud across different types of credit cards
plt.xlabel('\nTypes of Cards', fontsize=12)
plt.ylabel('Transaction Fraud\n', fontsize=12)
plt.title('\nTransaction Fraud by Types of Cards\n', fontsize=14, fontweight='bold');


# In[18]:


pd.set_option('display.max_rows', None)
df_raw.isna().sum() #checking whether there are missing values in the database


# In[19]:


pd.set_option('display.max_rows', None)
test.isna().sum()


# In[20]:


df_raw.info()


# In[21]:


test.info()


# In[22]:


#define a list to store the names of columns with fewer than 472432(80%) missing values
keep_col=[]
for col in df_raw.columns:
    if df_raw[col].isna().sum()<472432:
        keep_col.append(col)


# In[23]:


#The total number of variables to drop
len(keep_col)


# In[24]:


keep_col


# In[25]:


keep_col1=[]
for col in test.columns:
    if test[col].isna().sum()<405353:
        keep_col1.append(col)


# In[26]:


#The total number of variables to drop
len(keep_col1)


# In[27]:


keep_col1


# In[28]:


#Keep variables with fewer than 472432(80%) missing values
df_raw = df_raw[keep_col]


# In[29]:


test = test[keep_col1]


# In[30]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


# In[31]:


df_raw.dtypes


# In[32]:


test.dtypes


# In[33]:


# Get list of categorical variables
s = (df_raw.dtypes == 'category')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)


# In[34]:


# Get list of categorical variables
s1 = (test.dtypes == 'category')
object_cols1 = list(s1[s1].index)
print("Categorical variables:")
print(object_cols1)


# In[35]:


#Filling in missing values of categorical variables with most frequent values
df_raw = df_raw.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[36]:


test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[37]:


df_raw.isna().sum()


# In[38]:


test.isna().sum()


# In[39]:


df_raw['visa'] = df_raw['card4'].str.contains('visa')
df_raw['mastercard'] = df_raw['card4'].str.contains('mastercard')
df_raw['american express'] = df_raw['card4'].str.contains('american express')
df_raw['discover'] = df_raw['card4'].str.contains('discover')
df_raw['debit'] = df_raw['card6'].str.contains('debit')
df_raw['credit'] = df_raw['card6'].str.contains('credit')
df_raw['debit or credit'] = df_raw['card6'].str.contains('debit or credit')
df_raw['charge card'] = df_raw['card6'].str.contains('charge card')
df_raw['W'] = df_raw['ProductCD'].str.contains('W')
df_raw['C'] = df_raw['ProductCD'].str.contains('C')
df_raw['R'] = df_raw['ProductCD'].str.contains('R')
df_raw['H'] = df_raw['ProductCD'].str.contains('H')
df_raw['S'] = df_raw['ProductCD'].str.contains('S')
df_raw['desktop'] = df_raw['DeviceType'].str.contains('desktop')
df_raw['mobile'] = df_raw['DeviceType'].str.contains('mobile')


# In[40]:


test['visa'] = test['card4'].str.contains('visa')
test['mastercard'] = test['card4'].str.contains('mastercard')
test['american express'] = test['card4'].str.contains('american express')
test['discover'] = test['card4'].str.contains('discover')
test['debit'] = test['card6'].str.contains('debit')
test['credit'] = test['card6'].str.contains('credit')
test['debit or credit'] = test['card6'].str.contains('debit or credit')
test['charge card'] = test['card6'].str.contains('charge card')
test['W'] = test['ProductCD'].str.contains('W')
test['C'] = test['ProductCD'].str.contains('C')
test['R'] = test['ProductCD'].str.contains('R')
test['H'] = test['ProductCD'].str.contains('H')
test['S'] = test['ProductCD'].str.contains('S')
test['desktop'] = test['DeviceType'].str.contains('desktop')
test['mobile'] = test['DeviceType'].str.contains('mobile')


# In[41]:


df_raw['DeviceInfo'].unique()


# In[42]:


test['DeviceInfo'].unique()


# In[43]:


df_raw['P_emaildomain'].unique()


# In[44]:


test['P_emaildomain'].unique()


# In[45]:


df_raw['R_emaildomain'].unique()


# In[46]:


test['R_emaildomain'].unique()


# In[47]:


df_raw['M1'].unique()


# In[48]:


test['M1'].unique()


# In[49]:


df_raw['M2'].unique()


# In[50]:


test['M2'].unique()


# In[51]:


df_raw['M3'].unique()


# In[52]:


test['M3'].unique()


# In[53]:


df_raw['M4'].unique()


# In[54]:


test['M4'].unique()


# In[55]:


df_raw['M5'].unique()


# In[56]:


test['M5'].unique()


# In[57]:


df_raw['M6'].unique()


# In[58]:


test['M6'].unique()


# In[59]:


df_raw['M7'].unique()


# In[60]:


test['M7'].unique()


# In[61]:


df_raw['M8'].unique()


# In[62]:


test['M8'].unique()


# In[63]:


df_raw['M9'].unique()


# In[64]:


test['M9'].unique()


# In[65]:


df_raw['visa'] = pd.get_dummies(df_raw['visa'])
df_raw['mastercard'] = pd.get_dummies(df_raw['mastercard'])
df_raw['american express'] = pd.get_dummies(df_raw['american express'])
df_raw['discover'] = pd.get_dummies(df_raw['discover'])
df_raw['debit'] = pd.get_dummies(df_raw['debit'])
df_raw['credit'] = pd.get_dummies(df_raw['credit'])
df_raw['debit or credit'] = pd.get_dummies(df_raw['debit or credit'])
df_raw['charge card'] = pd.get_dummies(df_raw['charge card'])
df_raw['W'] = pd.get_dummies(df_raw['W'])
df_raw['C'] = pd.get_dummies(df_raw['C'])
df_raw['R'] = pd.get_dummies(df_raw['R'])
df_raw['H'] = pd.get_dummies(df_raw['H'])
df_raw['S'] = pd.get_dummies(df_raw['S'])
df_raw['desktop'] = pd.get_dummies(df_raw['desktop'])
df_raw['mobile'] = pd.get_dummies(df_raw['mobile'])


# In[66]:


test['visa'] = pd.get_dummies(test['visa'])
test['mastercard'] = pd.get_dummies(test['mastercard'])
test['american express'] = pd.get_dummies(test['american express'])
test['discover'] = pd.get_dummies(test['discover'])
test['debit'] = pd.get_dummies(test['debit'])
test['credit'] = pd.get_dummies(test['credit'])
test['debit or credit'] = pd.get_dummies(test['debit or credit'])
test['charge card'] = pd.get_dummies(test['charge card'])
test['W'] = pd.get_dummies(test['W'])
test['C'] = pd.get_dummies(test['C'])
test['R'] = pd.get_dummies(test['R'])
test['H'] = pd.get_dummies(test['H'])
test['S'] = pd.get_dummies(test['S'])
test['desktop'] = pd.get_dummies(test['desktop'])
test['mobile'] = pd.get_dummies(test['mobile'])


# In[67]:


df_raw['M1'] = pd.get_dummies(df_raw['M1'])
df_raw['M2'] = pd.get_dummies(df_raw['M2'])
df_raw['M3'] = pd.get_dummies(df_raw['M3'])
df_raw['M5'] = pd.get_dummies(df_raw['M5'])
df_raw['M6'] = pd.get_dummies(df_raw['M6'])
df_raw['M7'] = pd.get_dummies(df_raw['M7'])
df_raw['M8'] = pd.get_dummies(df_raw['M8'])
df_raw['M9'] = pd.get_dummies(df_raw['M9'])
df_raw['id_12'] = pd.get_dummies(df_raw['id_12'])
df_raw['id_15'] = pd.get_dummies(df_raw['id_15'])
df_raw['id_16'] = pd.get_dummies(df_raw['id_16'])
df_raw['id_28'] = pd.get_dummies(df_raw['id_28'])
df_raw['id_29'] = pd.get_dummies(df_raw['id_29'])
df_raw['id_31'] = pd.get_dummies(df_raw['id_31'])
df_raw['id_35'] = pd.get_dummies(df_raw['id_35'])
df_raw['id_36'] = pd.get_dummies(df_raw['id_36'])
df_raw['id_37'] = pd.get_dummies(df_raw['id_37'])


# In[68]:


test['M1'] = pd.get_dummies(test['M1'])
test['M2'] = pd.get_dummies(test['M2'])
test['M3'] = pd.get_dummies(test['M3'])
test['M5'] = pd.get_dummies(test['M5'])
test['M6'] = pd.get_dummies(test['M6'])
test['M7'] = pd.get_dummies(test['M7'])
test['M8'] = pd.get_dummies(test['M8'])
test['M9'] = pd.get_dummies(test['M9'])
test['id_12'] = pd.get_dummies(test['id_12'])
test['id_15'] = pd.get_dummies(test['id_15'])
test['id_16'] = pd.get_dummies(test['id_16'])
test['id_28'] = pd.get_dummies(test['id_28'])
test['id_29'] = pd.get_dummies(test['id_29'])
test['id_31'] = pd.get_dummies(test['id_31'])
test['id_35'] = pd.get_dummies(test['id_35'])
test['id_36'] = pd.get_dummies(test['id_36'])
test['id_37'] = pd.get_dummies(test['id_37'])


# In[69]:


s = (df_raw.dtypes == 'category')
object_cols = list(s[s].index)
object_cols.remove('ProductCD')
object_cols.remove('card4')
object_cols.remove('card6')
object_cols.remove('DeviceType')
print("Categorical variables:")
print(object_cols)


# In[70]:


#Label Encoding for variables with too many features('P_emaildomain', 'R_emaildomain' and 'DeviceInfo').
label_encoder = LabelEncoder()
for col in object_cols:
    df_raw[col] = label_encoder.fit_transform(df_raw[col])


# In[71]:


st = (test.dtypes == 'category')
object_colst = list(st[st].index)
object_colst.remove('ProductCD')
object_colst.remove('card4')
object_colst.remove('card6')
object_colst.remove('DeviceType')
print("Categorical variables:")
print(object_colst)


# In[72]:


label_encoder = LabelEncoder()
for col in object_colst:
    test[col] = label_encoder.fit_transform(test[col])


# In[73]:


i = (df_raw.dtypes == 'category')
category_cols = list(i[i].index)
print("Categorical variables:")
print(category_cols)


# In[74]:


n = (test.dtypes == 'category')
category_cols = list(n[n].index)
print("Categorical variables:")
print(category_cols)


# In[75]:


#drop categorical variables
df_raw=df_raw.select_dtypes(exclude=['category'])


# In[76]:


test=test.select_dtypes(exclude=['category'])


# In[77]:


pd.set_option('display.max_columns', None)
df_raw.head()


# In[78]:


pd.set_option('display.max_columns', None)
test.head()


# In[79]:


test.info()


# In[80]:


plotgroup1=['id_01','id_02','id_05','id_06','id_11','id_13','id_17','id_19','id_20']
plotgroup2=['id_29','id_31','id_35','id_36','id_37']
plotgroup3=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
plotgroup4=['D1','D2','D3','D4','D5','D10','D11','D15']
plotgroup5=['M1','M2','M3','M4','M5','M6','M7','M8','M9']
plotgroup6=['card1','card2','card3','card5']
plotgroup7=['P_emaildomain','R_emaildomain']


# In[81]:


#descriptive statistics 
for i in plotgroup1:
    print(df[i].describe())


# In[82]:


#Visualization of plotgroup1
from matplotlib.pyplot import subplot
red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_01'].plot(kind='box', xlim=((df_raw['id_01'].min()-100),df_raw['id_01'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[83]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_02'].plot(kind='box', xlim=((df_raw['id_02'].min()-100),df_raw['id_02'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[84]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_05'].plot(kind='box', xlim=((df_raw['id_05'].min()-100),df_raw['id_05'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[85]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_06'].plot(kind='box', xlim=((df_raw['id_06'].min()-100),df_raw['id_06'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[86]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_11'].plot(kind='box', xlim=((df_raw['id_11'].min()-10),df_raw['id_11'].max()+10), vert=False, flierprops=red_square, figsize=(16,2));


# In[87]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_13'].plot(kind='box', xlim=((df_raw['id_13'].min()-100),df_raw['id_13'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[88]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_17'].plot(kind='box', xlim=((df_raw['id_17'].min()-100),df_raw['id_17'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[89]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_19'].plot(kind='box', xlim=((df_raw['id_19'].min()-100),df_raw['id_19'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[90]:


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')
df_raw['id_20'].plot(kind='box', xlim=((df_raw['id_20'].min()-100),df_raw['id_20'].max()+100), vert=False, flierprops=red_square, figsize=(16,2));


# In[91]:


#Visualization of plotgroup2
# create plot on dummy variables
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[92]:


id29_count = df_raw['id_29'].value_counts()
sns.set(style="darkgrid")
sns.barplot(id29_count.index, id29_count.values, alpha=0.9)
plt.title('Frequency Distribution of id_29')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Dummy Variables', fontsize=12)
plt.show()


# In[93]:


id31_count = df_raw['id_31'].value_counts()
sns.set(style="darkgrid")
sns.barplot(id31_count.index, id31_count.values, alpha=0.9)
plt.title('Frequency Distribution of id_31')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Dummy Variables', fontsize=12)
plt.show()


# In[94]:


id35_count = df_raw['id_35'].value_counts()
sns.set(style="darkgrid")
sns.barplot(id35_count.index, id35_count.values, alpha=0.9)
plt.title('Frequency Distribution of id_35')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Dummy Variables', fontsize=12)
plt.show()


# In[95]:



id36_count = df_raw['id_36'].value_counts()
sns.set(style="darkgrid")
sns.barplot(id36_count.index, id36_count.values, alpha=0.9)
plt.title('Frequency Distribution of id_36')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Dummy Variables', fontsize=12)
plt.show()


# In[96]:



id37_count = df_raw['id_37'].value_counts()
sns.set(style="darkgrid")
sns.barplot(id37_count.index, id37_count.values, alpha=0.9)
plt.title('Frequency Distribution of id_37')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Dummy Variables', fontsize=12)
plt.show()


# In[97]:


for i in plotgroup3:
    print(df_raw[i].describe())


# In[98]:


for i in plotgroup4:
    print(df[i].describe())


# In[99]:


for i in plotgroup5:
    print(df_raw[i].describe())


# In[100]:


for i in plotgroup6:
    print(df_raw[i].describe())


# In[101]:


for i in plotgroup7:
    print(df_raw[i].describe())


# In[102]:


#The number of features are too many to include in the model. Therefore, we only select useful features for modeling.
useful_features = ['TransactionAmt','card3','addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C6', 'C7', 'C8',  'C10', 'C11', 'C12', 
                     'C14',  'D5','D11','M2', 'M3',
                   'M4',  'M6',  'M8', 'M9', 'V130', 'V131','V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174','V175', 'V176', 'V177',
                   'V178','V179', 'V180','V181','V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190','V191','V192','V193','V194','V195', 'V196','V197','V198','V199',
                   'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217','V218',  'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229','V230',  'V231','V232', 'V233', 'V234','V235', 'V236','V237','V238', 'V239', 'V240', 'V241', 'V248', 'V250',  'V252',  'V254',  'V255', 
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V260','V261','V269', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V282', 'V283', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V302','V303', 'V304', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317','id_01','id_02','id_05','id_06','id_11','id_12','id_13','id_17','id_19', 'id_20','id_31',
                   'id_35','id_36', 'id_38','visa','mastercard','american express','discover','debit','credit','debit or credit','charge card','W', 'C','R','H','S','desktop','mobile','DeviceInfo']


# In[103]:


df_rawmodel=df_raw[useful_features]


# In[104]:


#fit a logistic model to predict the probability
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()

X_train=df_rawmodel
Y_train = df_raw['isFraud']


# In[105]:


Y_train.isna().sum()


# In[106]:


X_train.isna().sum()


# In[107]:


model=logreg.fit(X_train, Y_train)


# In[108]:


#model validation for logistic model
from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(model,X_train, Y_train,cv=5,scoring='neg_mean_absolute_error')
print("Average MAE score (across experiments):")
print(scores.mean())


# In[109]:


X_test=test[useful_features]


# In[110]:


#predict by using logistic model
Y_test = logreg.predict_proba(X_test)


# In[111]:


Y_test = pd.DataFrame(Y_test)


# In[112]:


Y_test.info()


# In[113]:


X_test.head()


# In[114]:


new_test_data=pd.concat([test, Y_test], axis=1)


# In[115]:


new_test_data.head()


# In[116]:


new_test_data.info()


# In[117]:


sumbmit=pd.DataFrame(new_test_data,columns=['TransactionID', 1])


# In[118]:


sumbmit.head()


# In[119]:


sumbmit.columns = ['TransactionID','isFraud']


# In[120]:


sumbmit.head()


# In[121]:


sumbmit.to_csv('submission.csv')

