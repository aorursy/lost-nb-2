#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
import os.path as osp
import sys
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, clear_output


# In[3]:


get_ipython().run_cell_magic('time', '', "path = '/kaggle/input/data-science-bowl-2019/'\ntrain_df = pd.read_csv(osp.join(path, 'train.csv'))\ntest_df = pd.read_csv(osp.join(path, 'test.csv'))\ntrain_labels_df = pd.read_csv(osp.join(path, 'train_labels.csv'))\nspecs_df = pd.read_csv(osp.join(path, 'specs.csv'))\nsub_df = pd.read_csv(osp.join(path, 'sample_submission.csv'))")


# In[4]:


def show_df_info(df):
    display(df.head(2), df.columns, df.shape)


# In[5]:


show_df_info(train_df)


# In[6]:


show_df_info(train_labels_df)


# In[7]:


def get_shared_columns(df_1, df_2):
    return [x for x in df_1.columns if x in df_1.columns and x in df_2.columns]
    
shares_column_names = get_shared_columns(train_labels_df, train_df)
display(shares_column_names)


# In[8]:


show_df_info(test_df)


# In[9]:


get_shared_columns(train_labels_df, test_df)


# In[10]:


get_shared_columns(train_df, test_df)


# In[11]:


show_df_info(specs_df)


# In[12]:


display(get_shared_columns(specs_df, train_df),
        get_shared_columns(specs_df, train_labels_df),
        get_shared_columns(specs_df, test_df))


# In[13]:


show_df_info(sub_df)


# In[14]:


accuracy_group = np.array(train_labels_df['accuracy_group'])
display(set(accuracy_group))


# In[15]:


get_ipython().run_cell_magic('time', '', "train = pd.merge(train_df, train_labels_df, on = ['game_session', 'installation_id', 'title'])\nshow_df_info(train)")


# In[16]:


get_ipython().run_cell_magic('time', '', "train = pd.merge(train, specs_df, on = ['event_id'])\nshow_df_info(train)")


# In[17]:


get_ipython().run_cell_magic('time', '', "test = pd.merge(test_df, sub_df, on=['installation_id'])\nshow_df_info(test)")


# In[18]:


get_ipython().run_cell_magic('time', '', "test = pd.merge(test, specs_df, on=['event_id'])\nshow_df_info(test)")


# In[19]:


columns = get_shared_columns(train, test)
id_str = 'installation_id'
target_str = 'accuracy_group'
features = [column for column in columns if column not in [id_str, target_str]]

display(columns, len(columns), features, len(features))


# In[20]:


get_ipython().run_cell_magic('time', '', 'features_numbers = [len(set(train[feature])) for feature in features]\ndisplay(features, features_numbers)')


# In[21]:


train["title"].value_counts()


# In[22]:


train.info()


# In[23]:


train.describe()


# In[24]:



import matplotlib.pyplot as plt 
train.hist(bins=50, figsize=(20,15)) 
plt.show()


# In[25]:


show_df_info(train)


# In[26]:


corr_matrix = train.corr() 


# In[27]:


corr_matrix["accuracy_group"].sort_values(ascending=False) 


# In[28]:


train_try=train


# In[29]:


train_try["event_count*event_code"] = train["event_count"]*train["event_code"]


# In[30]:


corr_matrix = train_try.corr() 


# In[31]:


corr_matrix["accuracy_group"].sort_values(ascending=False) 


# In[32]:


train["event_count*event_code"]=train_try["event_count*event_code"]


# In[33]:


corr_matrix = train.corr() 
corr_matrix["accuracy_group"].sort_values(ascending=False) 


# In[34]:


train.head(2)


# In[35]:


cat =["args","info","accuracy_group","world","type","title","installation_id","event_data","timestamp","game_session","event_id"]


# In[36]:


train_cat=train[cat]


# In[37]:


train_args=train["args"]
train_info=train["info"]
train_accuracy_group=train["accuracy_group"]
train_world=train["world"]
train_title=train["title"]
train_installation_id=train["installation_id"]
train_event_data=train["event_data"]
train_timestamp=train["timestamp"]
train_game_session=train["game_session"]
train_event_id=train["event_id"]


# In[38]:


try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20


# In[39]:


cat_encoder = OneHotEncoder()
train_cat_1hot = cat_encoder.fit_transform(train_cat)
train_cat_1hot


# In[40]:



from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() 
world= encoder.fit_transform(train_world)
encoder.classes_
world = pd.DataFrame(world, columns = ['CRYSTALCAVES', 'MAGMAPEAK', 'TREETOPCITY']) 


# In[41]:


world


# In[42]:



from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() 
event_id= encoder.fit_transform(train_event_id)
event_id


# In[43]:


train_title=train["title"]
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() 
title= encoder.fit_transform(train_title)
title= pd.DataFrame(title, columns = (['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)','Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)','Mushroom Sorter (Assessment)']))
                     


# In[44]:


title


# In[45]:



from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() 
args= encoder.fit_transform(train_args)

encoder.classes_


# In[46]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer() 
info= encoder.fit_transform(train_info)
encoder.classes_




# In[47]:


encoder.classes_


# In[48]:


train_num=train_try[["event_count*event_code","accuracy","num_incorrect","num_correct"]]


# In[49]:


train_try.head(2)


# In[ ]:





# In[50]:


from sklearn.preprocessing import StandardScaler

train_num[["event_count*event_code","accuracy","num_incorrect","num_correct"]]=StandardScaler().fit_transform(train_num[["event_count*event_code","accuracy","num_incorrect","num_correct"]])


# In[51]:


train_num


# In[52]:


title


# In[53]:


Final_train=train_num


# In[54]:



Final_train=train_num
Final_train[['CRYSTALCAVES', 'MAGMAPEAK', 'TREETOPCITY']]=world[['CRYSTALCAVES', 'MAGMAPEAK', 'TREETOPCITY']]


# In[55]:


Final_train


# In[56]:


Final_train[['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)','Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)','Mushroom Sorter (Assessment)']]=title[['Bird Measurer (Assessment)', 'Cart Balancer (Assessment)','Cauldron Filler (Assessment)', 'Chest Sorter (Assessment)','Mushroom Sorter (Assessment)']]


# In[57]:


Final_training_set=Final_train
Final_train['accuracy_group']=train['accuracy_group']


# In[58]:


label=train['accuracy_group']


# In[59]:


corr_matrix = Final_train.corr() 


# In[60]:


corr_matrix["accuracy_group"].sort_values(ascending=False) 


# In[61]:


del Final_train["TREETOPCITY"]


# In[62]:


del Final_train["accuracy_group"]


# In[63]:


Final_training_set=Final_train


# In[64]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(Final_training_set, label)


# In[65]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# In[ ]:





# In[66]:


train          

 


# In[67]:


del train["event_count*event_code"]
del train["num_incorrect"]
del train["type"]
del train["event_code"]
del train["installation_id"]


# In[68]:


del train["installation_id"]


# In[69]:


train


# In[70]:




accuracy_group_list =train

accuracy_group_list


# In[71]:


label


# In[72]:



sub_df['accuracy_group'] = accuracy_group_list
sub_df.head()


# In[73]:


sub_df.to_csv('submission.csv', index=False)

