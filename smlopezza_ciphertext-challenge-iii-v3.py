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


# Importing packages
# For plots
from matplotlib import pyplot as plt
# To count letters and characters
from collections import Counter 


# In[3]:


test_path = '../input/test.csv'
train_path = '../input/train.csv'
sampleSubmission_path = '../input/sample_submission.csv'


# In[4]:


# Read and shows the head of the trainning data set
train_data = pd.read_csv(train_path)#, index_col = 0)
train_data.head()


# In[5]:


# Read and shows the head of the test data set
test_data = pd.read_csv(test_path)#, index_col = 0)
test_data.head()


# In[6]:


# Some Statistics from the train data set
train_data.describe()


# In[7]:


test_data.describe()


# In[8]:


# Columns name
print(train_data.columns)
print(test_data.columns)


# In[9]:


# first feature: create a 'length' column
train_data['length'] = train_data.text.apply(len)
train_data.head()


# In[10]:


test_data['length'] = test_data.ciphertext.apply(len)
test_data.head()


# In[11]:


# filter the test dataframes by cypher level
df_level_1 = test_data[test_data.difficulty==1].copy()
df_level_2 = test_data[test_data.difficulty==2].copy()
df_level_3 = test_data[test_data.difficulty==3].copy()
df_level_4 = test_data[test_data.difficulty==4].copy()

df_level_1.head(3)
#df_level_2.head(3)
#df_level_3.head(3)
#df_level_4.head(3)


# In[12]:


# Filter the train data set for difficult level
Level1_loc = df_level_1.loc[:].index.values
for i in range(len(Level1_loc)):
    train_data[train_data['index'] == Level1_loc[i]].text.values


# In[13]:


for i in range(len(Level1_loc)):
    train_level1 += train_data[train_data['index'] == Level1_loc[i]].text.values
    



#data_loc = train_data[train_data['index'] == Level1_loc[2]].index.values


# In[14]:


train_data[train_data['length']<=100]['length'].hist(bins=99)


# In[15]:


# using collections.Counter() to get count of each element in string  
plain_char_cntr = Counter(''.join(train_data['text'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)
print(plain_stats_train.head())


# In[16]:


# using collections.Counter() to get count of each element in string  
plain_char_test = Counter(''.join(df_level_1['ciphertext'].values))
plain_stats_test = pd.DataFrame([[x[0], x[1]] for x in plain_char_test.items()], columns=['Letter', 'Frequency'])
plain_stats_test = plain_stats_test.sort_values(by='Frequency', ascending=False)
print(plain_stats_test.head())


# In[17]:


# Plot the frequencies
f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Frequency'].values)
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values//4, alpha=.5,color='green')
plt.xticks(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Letter'].values)
plt.show()


# In[18]:


# Level of dificulty
A1 = df_level_1.length
print('Level 1 \n')
print(A1.describe())


# In[19]:


# then we look in the training data to find the passage with the corresponding length
matching_pieces = train_data[(train_data.length>=401) & (train_data.length<=500)]
matching_pieces
# only three unciphered texts length are in the interval: let's print them


# In[20]:


matching_pieces.text.values


# In[21]:


print('Unciphered text:\n', train_data.loc[13862].text, '\n\nCiphered text (level 1):\n', 
      df_level_1.loc[45272].ciphertext)


# In[22]:


# Function to decrypt the text
def decrypt_text(cipher_text):
    l = 'abcdefghijklmnopqrstuvwxy'
    u = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
    
    key =  [15, 24, 11, 4]
    key_index = 0
    plain = ''

    for character in cipher_text:
        test = l.find(character)
        if test != -1:
            p = (test - key[key_index]) % 25
            pc = l[p]
            key_index = (key_index + 1) % len(key)
        else:
            test2 = u.find(character)
            if test2 != -1:
                p = (test - key[key_index]) % 25
                pc = u[p]
                key_index = (key_index + 1) % len(key)
            else:
                pc = character
        
        plain += pc
        
    return plain


# In[23]:


# Function to encrypt the text
def encrypt_text(plain_text):
    l = 'abcdefghijklmnopqrstuvwxy'
    u = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
    
    key =  [15, 24, 11, 4]
    key_index = 0
    encrypted = ''

    for character in plain_text:
        test = l.find(character)
        if test != -1:
            p = (test + key[key_index]) % 25
            pc = l[p]
            key_index = (key_index + 1) % len(key)
        else:
            test2 = u.find(character)
            if test2 != -1:
                p = (test + key[key_index]) % 25
                pc = u[p]
                key_index = (key_index + 1) % len(key)
            else:
                pc = character
        
        encrypted += pc
        
    return encrypted


# In[24]:


plain_text = train_data.loc[13862].text
cipher_text = df_level_1.loc[45272].ciphertext

print('Plain text = \n', plain_text, '\n\n')
print('Decrypted text = \n', decrypt_text(cipher_text), '\n\n')
print('Encrypted text = \n', encrypt_text(plain_text), '\n\n')


# In[25]:


df_level_1.loc[Level1_loc[2]].ciphertext


# In[26]:


Level1_loc = df_level_1.loc[:].index.values
train_data[train_data['index'] == Level1_loc[2]]

data_loc = train_data[train_data['index'] == Level1_loc[2]].index.values
plain_text = train_data.loc[data_loc].text.values
cipher_text = df_level_1.loc[Level1_loc[2]].ciphertext


print('Plain text = \n', plain_text, '\n\n')
print('Decrypted text = \n', decrypt_text(cipher_text), '\n\n')


# In[27]:


KEYLEN = 4 # len('pyle')
def decrypt_level_1(ctext):
    
    key = [ord(c) - ord('a') for c in 'pyle']
    print('Key = ', key)
    
    key_index = 0
    plain = ''
    for c in ctext:
        cpos = 'abcdefghijklmnopqrstuvwxy'.find(c)
        print('c = ', c)
        print('cpos = ', cpos)
        if cpos != -1:
            p = (cpos - key[key_index]) % 25
            print('p = ', p)
            pc = 'abcdefghijklmnopqrstuvwxy'[p]
            key_index = (key_index + 1) % KEYLEN
        else:
            cpos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)
            if cpos != -1:
                p = (cpos - key[key_index]) % 25
                pc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]
                key_index = (key_index + 1) % KEYLEN
            else:
                pc = c
        plain += pc
                              
    return plain

def encrypt_level_1(ptext, key_index=0):
    key = [ord(c) - ord('a') for c in 'pyle']
    ctext = ''
    for c in ptext:
        pos = 'abcdefghijklmnopqrstuvwxy'.find(c)
        if pos != -1:
            p = (pos + key[key_index]) % 25
            cc = 'abcdefghijklmnopqrstuvwxy'[p]
            key_index = (key_index + 1) % KEYLEN
        else:
            pos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)
            if pos != -1:
                p = (pos + key[key_index]) % 25
                cc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]
                key_index = (key_index + 1) % KEYLEN
            else:
                cc = c
        ctext += cc
    return ctext

def test_decrypt_level_1(c_id):
    
    ciphertext = test_data[test_data['ciphertext_id'] == c_id].ciphertext.values
    print('Ciphertxt:', ciphertext)
    decrypted = decrypt_level_1(ciphertext)
    print('Decrypted:', decrypted)
    encrypted = encrypt_level_1(decrypted)
    print('Encrypted:', encrypted)
    print("Encrypted == Ciphertext:", encrypted == ciphertext)


c_id = 'ID_4a6fc1ea9'
test_decrypt_level_1(c_id) 


# In[28]:


test_data[test_data['ciphertext_id'] =='ID_4a6fc1ea9']


# In[ ]:





# In[29]:


A2 = df_level_2.length
print('\n Level 2 \n')
print(A2.describe())

