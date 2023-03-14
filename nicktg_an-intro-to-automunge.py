#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#First we'll grab the file paths
train_identify_filepath = "../input/ieee-fraud-detection/train_identity.csv"
train_transaction_filepath = "../input/ieee-fraud-detection/train_transaction.csv"
test_identity_filepath = "../input/ieee-fraud-detection/test_identity.csv"
test_transaction_filepath = "../input/ieee-fraud-detection/test_transaction.csv"
sample_submission_filepath = "../input/ieee-fraud-detection/sample_submission.csv"


# In[3]:


#Now let's import them as dataframes. Note both the identify and transaction sets include 
#a single common column, TransactionID, so we'll use that as an index column to merge

train_identify = pd.read_csv(train_identify_filepath, error_bad_lines=False, index_col="TransactionID")
train_transaction = pd.read_csv(train_transaction_filepath, error_bad_lines=False, index_col="TransactionID")
test_identity = pd.read_csv(test_identity_filepath, error_bad_lines=False, index_col="TransactionID")
test_transaction = pd.read_csv(test_transaction_filepath, error_bad_lines=False, index_col="TransactionID")
sample_submission = pd.read_csv(sample_submission_filepath, error_bad_lines=False)


# In[4]:


#Note that by inspection we end up with the identify columns on about a quarter 
#of the rows of the transaction data
print("   train_identify.shape = ", train_identify.shape)
print("train_transaction.shape = ", train_transaction.shape)


# In[5]:


#Here we'll concatinate the two sets based on the common index points
master_train = pd.concat([train_transaction, train_identify], axis=1, sort=False)
master_test = pd.concat([test_transaction, test_identity], axis=1, sort=False)

print("master_train.shape = ", master_train.shape)
print("master_test.shape = ", master_test.shape)


# In[6]:


#Because I'm going to be doing a whole bunch of demonstrations 
#in this notebook, I'm going to carve out a much smaller set 
#to speed up the writing.

columns_subset = list(master_train)[:15]
#(test columns won't have the label column)
test_columns_subset = list(master_test)[:14]

small_train = master_train[columns_subset]
small_test = master_test[test_columns_subset]

from sklearn.model_selection import train_test_split
big_train, tiny_train = train_test_split(small_train, test_size=0.002, random_state=42)
big_test, tiny_test = train_test_split(small_test, test_size=0.002, random_state=42)


print(list(tiny_train))
print("")
print("tiny_train.shape = ", tiny_train.shape)
print("big_train.shape = ", big_train.shape)
print("")
print("tiny_test.shape = ", tiny_test.shape)
print("big_test.shape = ", big_test.shape)


# In[7]:


#Ok here's where we import our tool with pip install. Note that this step requires  
#access to the internet. (Note this import procedure changed with version 2.58.)

get_ipython().system(' pip install Automunge')

# #or to upgrade (we currently roll out upgrades pretty frequently)
# ! pip install Automunge --upgrade


# In[8]:


#And then we initialize the class.

from Automunge import Automunger
am = Automunger.AutoMunge()


# In[9]:


#So first let's just try a generic application with our tiny_train set. Note tiny_train here
#represents our train set. If a labels column is available we should include and designate, 
#and any columns we want to exclude from processing we can designate as "ID columns" which
#will be carved out and consitnelty shuffled and partitioned.

#Note here we're only demonstrating on the set with the reduced number of features to save time.


train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', trainID_column = False,             testID_column = False, valpercent1=0.0, valpercent2 = 0.0,             shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False,             binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42,             numbercategoryheuristic = 15, pandasoutput = True, NArw_marker = False,             featureselection = False, featurepct = 1.0, featuremetric = .02,             featuremethod = 'pct', PCAn_components = None, PCAexcl = [],             ML_cmnd = {'MLinfill_type':'default',                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}},                        'PCA_type':'default',                        'PCA_cmnd':{}},             assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[],                          'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[],                          'bins':[], 'bint':[],                          'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[],                          'log0':[], 'log1':[], 'pwrs':[],                          'bnry':[], 'text':[], 'ordl':[], 'ord2':[],                          'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[],                          'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[]},             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[],                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[]},             transformdict = {}, processdict = {},             printstatus = True)


# In[10]:


#sffixes identifying steps of transformation
list(train)


# In[11]:


#And here's what the returned data looks like.
train.head()


# In[12]:


list(labels)


# In[13]:


#as you can see the returned values on the labels column are consistently encoded
#as were passed
labels['isFraud_bnry'].unique()


# In[14]:


#Note that if or original labels weren't yet binary encoded, we could inspect the 
#returned labelsencoding_dict object to determine the basis of encoding.

#Here we just see that the 1 value originated from values 1, and the 0 value
#originated from values 0 - a trivial example, but this could be helpful if
#we had passed a column containing values ['cat', 'dog'] for instance.

labelsencoding_dict


# In[15]:


test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=True)


# In[16]:


#And if we're doing our job right then this set should be formatted exaclty like that returned
#from automunge, let's take a look.

test.head()


# In[17]:


#Looks good! 

#So if we wanted to generate predictions from a machine learning model trained 
#on a train set processed with automunge, we now have a way to consistently 
#prepare data with postmunge.


# In[18]:


#great well let's try a few of these out. How about the ID columns, let's see what happens when we pass one.
#Let's just pick an arbitrary one, TransactionDT

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', trainID_column = 'TransactionDT',              valpercent1=0.20, shuffletrain = True, pandasoutput=True, printstatus = False)


# In[19]:


#Now we'll find that the TransactionDT column is missing from the train set, left 
#unaltered instead in the ID set, paired with the Transaction ID which was put
#in the ID set because it was a non-integer range index column (thus if we wanted
#to reassign the original index column we could simply copy the TransactionID column
#from the ID set back to the processed train set)

trainID.head()


# In[20]:


#note that since our automunge call included a validation ratio, we'll find 
#a portion of the sets partitioned in the validation sets, here for instance
#is the validaiton ID sets 

#(we'll also find returned sets in the validation1, and validationlabels1)

#note that since we activated the shuffletrain option these are randomly
#selected from the train set

validationID1.head()


# In[21]:


#Let's take a look at TrainLabelFreqLevel, which serves to copy rows such as to
#(approximately) levelize the frequency of labels found in the set.

#First let's look at the shape of a train set returtned from an automunge
#applicaiton without this option selected

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', TrainLabelFreqLevel=False,              pandasoutput=True, printstatus=False)

print("train.shape = ", train.shape)


# In[22]:


#OK now let's try again with the option selected. If there was a material discrepency in label frequency
#we should see more rows included in the returned set

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', TrainLabelFreqLevel=True,              pandasoutput=True, printstatus=False)

print("train.shape = ", train.shape)


# In[23]:


#binstransform just means that default numerical sets will include an additional set of bins identifying
#number of standard deviations from the mean. We have to be careful with this one if we don't have a lot
#of data as it adds a fair bit of dimensionality

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', binstransform=True,              pandasoutput=True, printstatus=False)

print("list(train):")
list(train)


# In[24]:


#so the interpretation should be for columns with suffix including "bint" that indicates 
#bins for number fo standard deviations from the mean. For example, nmbr_bint_t+01
#would indicated values between mean to +1 standard deviation.


# In[25]:


#So MLinfill changes the default infill method from standardinfill (which means mean for 
#numerical sets, most common for binary, and boolean marker for categorical), to a predictive
#method in which a machine learning model is trained for each column to predict infill based
#on properties of the rest of the set. This one's pretty neat, but caution that it performs 
#better with more data as you would expect.

#Let's demonstrate, first here's an applicaiton without MLinfill, we'll turn on the NArws option
#to output an identifier of rows subject to infill

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', MLinfill=False,              NArw_marker=True, pandasoutput=True, printstatus=False)

print("train.head()")
train.head()


# In[26]:


#So upon inspection it looks like we had a few infill points on
#columns originating from dist1 (as identified by the NArw columns)
#so let's focus on that

#As you can see the plug value here is just the mean which for a 
#z-score normalized set is 0

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# In[27]:


#Now let's try with MLinfill

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', MLinfill=True,              NArw_marker=True, pandasoutput=True, printstatus=False)

print("train[columns].head()")
train[columns].head()


# In[28]:


#As you can see the method predicted a unique infill value to each row subject to infill
#(as identified by the NArw column). We didn't include a lot of data with this small demonstration
#set, so I expect the accuracy of this method would improve with a bigger set


# In[29]:


# numbercategoryheuristic just changes the threshold for number of unique values in a categorical set
#between processing a categorical set via one-hot encoding or ordinal processing (sequential integer encoding)

#for example consiter the returned column for the email domain set in the data, if we look above we see the
#set was processed as ordinal, let's see why

print("number of unique values in P_emaildomain column pre-processing")
print(len(train['P_emaildomain_ordl']))


# In[30]:


#So yeah looks like that entry has a unique entry per row, so really not really a good candidate for inclusion at
#all, this might be better served carved out into the ID set until such time as we can extract some info from it
#prior to processing. But the poitn is if we had set numbercategoryheuristic to 1478 instead of 15 we would have 
#derived 1477 one-hot-encoded columns from this set which obviosuly would be an issue for this scale of data.


# In[31]:


#pandasoutput just tells whether to return pandas dataframe or numpy arrays (defaults to numpy which
#is a more universal elligible input to the different machine learning frameworks)

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',               pandasoutput=False, NArw_marker = False, printstatus=False)

print("type(train)")
print(type(train))


# In[32]:


#note that if we return numpy arrays and want to view the column headers 
#(which remember track the steps of transofmations in their suffix appenders)
#good news that's available in the returned finalcolumns_train
print("finalcolumns_train")
finalcolumns_train


# In[33]:


#or with pandasoutput = True

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',               pandasoutput=True, NArw_marker = True, printstatus=False)

print("type(train)")
print(type(train))


# In[34]:


#The NArw marker helpfully outputs from each column a marker indicating what rows were
#subject to infill. Let's quickly demonstrate. First here again are the returned columns
#without this feature activated.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=False, pandasoutput=True, printstatus=False)

print("list(train)")
list(train)


# In[35]:


#Now with NArw_marker turned on.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False)

print("list(train)")
list(train)


# In[36]:


#If we inspect one of these we'll see a marker for what rows were subject to infill
#(actually already did this a few cells ago but just to be complete)

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# In[37]:


#featureselection performs a feature importance evaluation with the permutaion method. 
#(basically trains a machine learning model, and then measures impact to accuaracy 
#after randomly shuffling each feature)

#Let's try it out. Note that this method requires the inclusion of a labels column.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              featureselection=True, pandasoutput=True, printstatus=False)


# In[38]:


#Now we can view the results like so.
#(a future iteration of tool will improve the reporting method, for now this works)
for keys,values in featureimportance.items():
    print(keys)
    print('shuffleaccuracy = ', values['shuffleaccuracy'])
    print('baseaccuracy = ', values['baseaccuracy'])
    print('metric = ', values['metric'])
    print('metric2 = ', values['metric2'])
    print()


# In[39]:


#I suspect the small size of this demonstration set impacted these results.

#Note that for interpretting these the "metric" represents the impact
#after shuffling the entire set originating from same feature and larger
#metric implies more importance
#and metric2 is derived after shuffling all but the current column originating from same
#feature and smaller metric2 implies greater relative importance in that set of
#derived features. In case you were wondering.


# In[40]:


#Now if we want to apply some kind of dimensionality reduction, we can conduct 
#via Principle Component Analysis (PCA), a type of unsupervised learning.

#a few defaults here is PCA is automatically performed if number of features > 50% number of rows
#(can be turned off via ML_cmnd)
#also the PCA type defaults to kernel PCA for all non-negative sets, sparse PCA otherwise, or regular
#PCA if PCAn_components pass as a percent. (All via scikit PCA methods)

#If there are any columns we want to exclude from PCA, we can specify in PCAexcl

#We can also pass parameters to the PCA call via the ML_cmnd

#Let's demosntrate, here we'll reduce to four PCA derived sets, arbitrarily excluding 
#from the transofrmation columns derived from dist1


train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              PCAn_components=4, PCAexcl=['dist1'],              pandasoutput=True, printstatus=False)

print("derived columns")
list(train)


# In[41]:


#Noting that any subsequently available data can easily be consistently prepared as follows
#with postmunge (by simply passing the postprocess_dict object returned from automunge, which
#you did remember to save, right? If not no worries it's also possible to consistnelty process
#by passing the test set with the exact saem original train set to automunge)

test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=False)

list(test)


# In[42]:


#Another useful method might be to exclude any boolean columns from the PCA
#dimensionality reduction. We can do that with ML_cmnd by passing following:

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              PCAn_components=4, PCAexcl=['dist1'],              pandasoutput=True, printstatus=False,              ML_cmnd = {'MLinfill_type':'default',                         'MLinfill_cmnd':{'RandomForestClassifier':{},                                          'RandomForestRegressor':{}},                         'PCA_type':'default',                         'PCA_cmnd':{'bool_PCA_excl':True}})

print("derived columns")
list(train)


# In[43]:


#A really important part is that we don't have to defer to the automated evaluation of
#column properties to determine processing methods, we can also assign distinct processing
#methods to specific columns.

#Now let's try assigning a few different methods to the numerical sets:

#remember we're assigninbg based on the original column names before the appended suffixes

#How about let's arbitrily select min-max scaling to these columns 
minmax_list = ['card1', 'card2', 'card3']

#And since we previously saw that Transaction_Amt might have some skewness based on our
#prior powertrasnform evaluation, let's set that to 'pwrs' which puts it into bins
#based on powers of 10
pwrs_list = ['TransactionAmt']

#Let's say we don't feel the P_emaildomain is very useful, we can just delete it with null
null_list = ['P_emaildomain']

#and if there's a column we want to exclude from processiong, we can exclude with excl
#note that any column we exclude from processing needs to be already numerically encoded
#if we want to use any of our predictive methods like MLinfill, feature improtance, PCA
#on other columns. (excl just passes data untouched, exc2 performs a modeinfill just in 
#case some missing points are found.)
exc2_list = ['card5']

#and we'll leave the rest to default methods

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              pandasoutput=True, printstatus=False,              assigncat = {'mnmx':minmax_list, 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[],                          'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[],                          'bins':[], 'bint':[],                          'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[],                          'log0':[], 'log1':[], 'pwrs':pwrs_list,                          'bnry':[], 'text':[], 'ordl':[], 'ord2':[],                          'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[],                          'excl':[], 'exc2':exc2_list, 'exc3':[], 'null':null_list, 'eval':[]})

print("derived columns")
list(train)


# In[44]:


#Here's what the resulting derivations look like
train.head()


# In[45]:


#We can also assign distinct infill methods to each column. Let's demonstrate. 
#I remember when we were looking at MLinfill that one of our columns had a few NArw
#(rows subject to infill), let's try a different infill method on those 

#how about we try adjinfill which carries the value from an adjacent row

#remember we're assigning columns based on their title prior to the suffix appendings

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False,              assigninfill = {'adjinfill':['dist1']})

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# In[46]:


#trasnformdict and processdict are for more advanced users. They allow the user to design
#custom compositions of transformations, or even incorporate their own custom defined
#trasnformation functions into use on the platform. I won't go into full detail on these methods
#here, I documented these a bunch in the essays which I'll link to below, but here's a taste.

#Say that we have a numerical set that we want to use to apply multiple trasnformations. Let's just
#make a few up, say that we have a set with fat tail characteristics, and we want to do multiple
#trasnformions including a bocx-cox trasnformation, a z-score trasnformation on that output, as
#well as a set of bins for powers of 10. Well our 'TransactionAmt' column might be a good candiate
#for that. Let's show how.

#Here we define our cusotm trasnform dict using our "family tree primitives"
#Note that we always need to uyse at least one replacement primitive, if a column is intended to be left
#intact we can include a excl trasnfo0rm as a replacement primitive.

#here are the primitive definitions
# 'parents' :           upstream / first generation / replaces column / with offspring
# 'siblings':           upstream / first generation / supplements column / with offspring
# 'auntsuncles' :       upstream / first generation / replaces column / no offspring
# 'cousins' :           upstream / first generation / supplements column / no offspring
# 'children' :          downstream parents / offspring generations / replaces column / with offspring
# 'niecesnephews' :     downstream siblings / offspring generations / supplements column / with offspring
# 'coworkers' :         downstream auntsuncles / offspring generations / replaces column / no offspring
# 'friends' :           downstream cousins / offspring generations / supplements column / no offspring

#So let's define our custom trasnformdict for a new root category we'll call 'cstm'
transformdict = {'cstm' : {'parents' : ['bxcx'],                            'siblings': [],                            'auntsuncles' : [],                            'cousins' : ['pwrs'],                            'children' : [],                            'niecesnephews' : [],                            'coworkers' : [],                            'friends' : []}}

#Note that since bxcx is a parent category, it will look for offspring in the primitives associated
#with bxcx root cateogry in the library, and find there a downstream nmbr category

#Note that since we are defining a new root category, we also have to define a few parameters for it
#demonstrate here. Further detail on thsi step available in documentation. If you're not sure you might
#want to try just copying an entry in the READ ME.

#Note that since cstm is only a root cateogry and not included in the family tree primitives we don't have to
#define a processing funciton (for the dualprocess/singleprocess/postprocess entries), we can just enter None

processdict = {'cstm' : {'dualprocess' : None,                          'singleprocess' : None,                          'postprocess' : None,                          'NArowtype' : 'numeric',                          'MLinfilltype' : 'numeric',                          'labelctgy' : 'nmbr'}}

#We can then pass this trasnformdict to the automunge call and assign the intended column in assigncat
train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False,              assigncat = {'cstm':['TransactionAmt']},              transformdict = transformdict, processdict = processdict)

print("list(train)")
list(train)


# In[47]:


#and then of course use also has the ability to define their own trasnformation functions to
#incorproate into the platform, I'll defer to the essays for that bit in the interest of brevity


# In[48]:


#And the final bit which I'll just reiterate here is that automunge facilitates the simplest means
#for consistent processing of subsequently available data with just a single function call
#all you need is the postprocess_dict object returned form the original automunge call

#This even works when we passed custom trasnformdict entries as was case with last postprocess_dict
#derived in last example, however if you're defining custom trasfnormation functions for now you
#need to save those custom function definitions are redefine in the new notewbook when applying postmunge

#Here again is a demosntration of postmunge. Since the last postprocess_dict we returned
#was with our custom transfomrations in preceding excample, the 'TransactionAmt' column will
#be processed consistently

test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=True)


# In[49]:


list(test)


# In[ ]:




