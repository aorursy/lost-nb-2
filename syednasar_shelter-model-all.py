#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# read the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams.update({'font.size': 12})


# In[3]:


from sklearn import preprocessing
import brewer2mpl


# In[4]:


in_file_train = '../input/train.csv'
in_file_test = '../input/test.csv'

print("Loading data...\n")
data = pd.read_csv(in_file_train)
pd_kaggle_test = pd.read_csv(in_file_test)


# In[5]:


data.head()


# In[6]:


The dataset used here is from a Kaggle competition - [Shelter Animal Outcome](https://www.kaggle.com/c/shelter-animal-outcomes/kernels)


# In[7]:


#TO DELETE

len(data['Color'].unique())

col_df = data['Color']


# In[8]:


#Do we have clean data?
data.count()


# In[9]:


def data_cleanup(data, train=True):
    

    #Convert to discrete numeric values of 1 for has a name, 0 for no name
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0,"HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    #Convert to discrete numeric values
    data['AnimalType'] = data['AnimalType'].map({'Cat':0,'Dog':1})
    if(train):
        data.drop(['AnimalID','OutcomeSubtype'],axis=1, inplace=True)
        #Assign numeric values to OutComeType
        data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})

    #Now lets fix the gender ('Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female','Unknown', nan)
    #sex = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5, np.nan:0}
    #data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)

    # Convert Breed to numeric classes
    data.SexuponOutcome.fillna('Unknown', inplace=True)
    sex_le = preprocessing.LabelEncoder()
    #to convert into numbers
    data.SexuponOutcome = sex_le.fit_transform(data.SexuponOutcome)


    #Discretizing the AgeUponOutcome to number of days
    def agetodays(x):
        try:
            y = x.split()
        except:
            return None 
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365/12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])

    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)
    data.loc[(data['AgeInDays'].isnull()),'AgeInDays'] = data['AgeInDays'].median()

    #Break date time components into Y,M,D,H,M components
    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)


    data['IsMix'] = data['Breed'].str.contains('mix',case=False).astype(int)

    # Convert Color to numeric classes
    color = preprocessing.LabelEncoder()
    #to convert into numbers
    data.Color = color.fit_transform(data.Color)

    # Convert Breed to numeric classes
    breed = preprocessing.LabelEncoder()
    #to convert into numbers
    data.Breed = breed.fit_transform(data.Breed)


    data['Name-n-Sex'] = data['HasName'] + data['SexuponOutcome']
    data['Type-n-Sex'] = data['AnimalType'] + data['SexuponOutcome']


    data.drop(['AgeuponOutcome','Name','DateTime'],axis=1, inplace=True)
    
    return data


# In[10]:


data = data_cleanup(data)

data.head()


# In[11]:


data.count()


# In[12]:





# In[12]:





# In[12]:


print (data.OutcomeType.unique())

#OutcomeTypes => 'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2

adoption = sum(data.loc[:, 'OutcomeType'] == 0)
died = sum(data.loc[:, 'OutcomeType'] == 2)
euth = sum(data.loc[:, 'OutcomeType'] == 3)
ret2own = sum(data.loc[:, 'OutcomeType'] == 4)

#functional , non_functional = sum(df2.loc[:, 'OutcomeType'] == 0), sum(df2.loc[:, 'status_group'] == 1)
print(adoption, died, euth, ret2own)


# In[13]:


set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors

font = {'family' : 'sans-serif',
        'color'  : 'teal',
        'weight' : 'bold',
        'size'   : 18,
        }
plt.rc('font',family='serif')
plt.rc('font', size=16)
plt.rc('font', weight='bold')
#plt.style.use('seaborn-poster')
#plt.style.use('bmh')
#plt.style.use('ggplot')
plt.style.use('seaborn-dark-palette')
#plt.style.use('presentation')
print (plt.style.available)

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 6 and height to 6
fig_size[0] = 6
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size


# In[14]:


from matplotlib import rcParams
rcParams['font.size'] = 12
#print (rcParams.keys())
rcParams['text.color'] = 'black'

piechart = plt.pie(
    (adoption, died, euth, ret2own),
    labels=('adopted', 'died','euthenized','returned'),
    shadow=False,
    colors=('teal', 'crimson', 'cyan', 'coral'),
    explode=(0.08,0.08,0.08,0.08), # space between slices 
    startangle=90,    # rotate conter-clockwise by 90 degrees
    autopct='%1.1f%%',# display fraction as percentages
)

plt.axis('equal')   
plt.title("Animal Shelter Outcome Train Data", y=1.08,fontdict=font)
plt.tight_layout()
plt.savefig('TWP-Status-Groups-train.png', bbox_inches='tight')


# In[15]:




