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


pip install pycountry_convert


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import pycountry_convert as pc
import pycountry
import functools


# In[4]:


train_df=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")


# In[5]:


train_df.head()


# In[6]:


train_df.info()


# In[7]:


test_df.info()


# In[8]:


print("Min train date: ",train_df["Date"].min())
print("Max train date: ",train_df["Date"].max())
print("Min test date: ",test_df["Date"].min())
print("Max test date: ",test_df["Date"].max())


# In[9]:


pop_info = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")
#Population Data


# In[10]:


pop_info.head()


# In[11]:


pop_info.rename(columns={'Density (P/Km²)': 'Density'}, inplace=True)


# In[12]:


pop_info.columns


# In[13]:


country_lookup=pop_info[["Country (or dependency)","Population (2020)","Density","Med. Age","Urban Pop %"]]


# In[14]:


country_lookup.head()


# In[15]:


pd.DataFrame.from_dict(country_lookup)
train_df_pop=pd.merge(train_df, country_lookup, how='left', left_on='Country_Region', right_on='Country (or dependency)')
#Train data joined with population data


# In[16]:


train_df_pop.info()


# In[17]:


#Some of the names don't match with the file, hence manually setting them
train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Population (2020)"]]=331002651 #United Sates
train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Density"]]=36
train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Med. Age"]]=38
train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Urban Pop %"]]="83%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Burma", ["Population (2020)"]]=54409800 #Myanmar
train_df_pop.loc[train_df_pop["Country_Region"]=="Burma", ["Density"]]=83
train_df_pop.loc[train_df_pop["Country_Region"]=="Burma", ["Med. Age"]]=29
train_df_pop.loc[train_df_pop["Country_Region"]=="Burma", ["Urban Pop %"]]="39%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Sao Tome and Principe", ["Population (2020)"]]=219159 #Sao Tome & Principe
train_df_pop.loc[train_df_pop["Country_Region"]=="Sao Tome and Principe", ["Density"]]=228
train_df_pop.loc[train_df_pop["Country_Region"]=="Sao Tome and Principe", ["Med. Age"]]=19
train_df_pop.loc[train_df_pop["Country_Region"]=="Sao Tome and Principe", ["Urban Pop %"]]="74%"
train_df_pop.loc[train_df_pop["Country_Region"]=="West Bank and Gaza", ["Population (2020)"]]=3340143 #Google Search
train_df_pop.loc[train_df_pop["Country_Region"]=="West Bank and Gaza", ["Density"]]=759
train_df_pop.loc[train_df_pop["Country_Region"]=="West Bank and Gaza", ["Med. Age"]]=17
train_df_pop.loc[train_df_pop["Country_Region"]=="West Bank and Gaza", ["Urban Pop %"]]="76%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Kosovo", ["Population (2020)"]]=1810463 #Taken from Wikipedia
train_df_pop.loc[train_df_pop["Country_Region"]=="Kosovo", ["Density"]]=159
train_df_pop.loc[train_df_pop["Country_Region"]=="Kosovo", ["Med. Age"]]=29
train_df_pop.loc[train_df_pop["Country_Region"]=="Kosovo", ["Urban Pop %"]]="55%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Population (2020)"]]=51269185 #South Korea
train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Density"]]=527
train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Med. Age"]]=44
train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Urban Pop %"]]="82%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Population (2020)"]]=10708981 #Czech Republic
train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Density"]]=139
train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Med. Age"]]=43
train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Urban Pop %"]]="74%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Population (2020)"]]=23816775 #Taiwan
train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Density"]]=673
train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Med. Age"]]=42
train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Urban Pop %"]]="79%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Population (2020)"]]=89561403 #DR Congo
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Density"]]=40
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Med. Age"]]=17
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Urban Pop %"]]="46%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Population (2020)"]]=5518087 #Congo
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Density"]]=16
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Med. Age"]]=19
train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Urban Pop %"]]="70%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Population (2020)"]]=26378274 #CÃ´te d'Ivoire
train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Density"]]=83
train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Med. Age"]]=19
train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Urban Pop %"]]="51%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Population (2020)"]]=53199 #Saint Kitts & Nevis
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Density"]]=205
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Med. Age"]]=36
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Urban Pop %"]]="33%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Population (2020)"]]=110940 #St. Vincent & Grenadines
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Density"]]=284
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Med. Age"]]=33
train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Urban Pop %"]]="53%"
train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Population (2020)"]]=3770 #Population and density are same since it is a cruise ship
train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Density"]]=3770
train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Med. Age"]]=62
train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Urban Pop %"]]="100%"
train_df_pop.loc[train_df_pop["Country_Region"]=="MS Zaandam", ["Population (2020)"]]=1432 #Population and density are same since it is a cruise ship
train_df_pop.loc[train_df_pop["Country_Region"]=="MS Zaandam", ["Density"]]=1432
train_df_pop.loc[train_df_pop["Country_Region"]=="MS Zaandam", ["Med. Age"]]=65
train_df_pop.loc[train_df_pop["Country_Region"]=="MS Zaandam", ["Urban Pop %"]]="100%"


# In[18]:


test_df_pop=pd.merge(test_df, country_lookup, how='left', left_on='Country_Region', right_on='Country (or dependency)')
#Test data joined with population data


# In[19]:


#Some of the names don't match with the file, hence manually setting them
test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Population (2020)"]]=331002651 #United Sates
test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Density"]]=36
test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Med. Age"]]=38
test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Urban Pop %"]]="83%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Burma", ["Population (2020)"]]=54409800 #Myanmar
test_df_pop.loc[test_df_pop["Country_Region"]=="Burma", ["Density"]]=83
test_df_pop.loc[test_df_pop["Country_Region"]=="Burma", ["Med. Age"]]=29
test_df_pop.loc[test_df_pop["Country_Region"]=="Burma", ["Urban Pop %"]]="39%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Sao Tome and Principe", ["Population (2020)"]]=219159 #Sao Tome & Principe
test_df_pop.loc[test_df_pop["Country_Region"]=="Sao Tome and Principe", ["Density"]]=228
test_df_pop.loc[test_df_pop["Country_Region"]=="Sao Tome and Principe", ["Med. Age"]]=19
test_df_pop.loc[test_df_pop["Country_Region"]=="Sao Tome and Principe", ["Urban Pop %"]]="74%"
test_df_pop.loc[test_df_pop["Country_Region"]=="West Bank and Gaza", ["Population (2020)"]]=3340143 #Google Search
test_df_pop.loc[test_df_pop["Country_Region"]=="West Bank and Gaza", ["Density"]]=759
test_df_pop.loc[test_df_pop["Country_Region"]=="West Bank and Gaza", ["Med. Age"]]=17
test_df_pop.loc[test_df_pop["Country_Region"]=="West Bank and Gaza", ["Urban Pop %"]]="76%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Kosovo", ["Population (2020)"]]=1810463 #Taken from Wikipedia
test_df_pop.loc[test_df_pop["Country_Region"]=="Kosovo", ["Density"]]=159
test_df_pop.loc[test_df_pop["Country_Region"]=="Kosovo", ["Med. Age"]]=29
test_df_pop.loc[test_df_pop["Country_Region"]=="Kosovo", ["Urban Pop %"]]="55%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Population (2020)"]]=51269185 #South Korea
test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Density"]]=527
test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Med. Age"]]=44
test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Urban Pop %"]]="82%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Population (2020)"]]=10708981 #Czech Republic
test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Density"]]=139
test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Med. Age"]]=43
test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Urban Pop %"]]="74%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Population (2020)"]]=23816775 #Taiwan
test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Density"]]=673
test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Med. Age"]]=42
test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Urban Pop %"]]="79%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Population (2020)"]]=89561403 #DR Congo
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Density"]]=40
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Med. Age"]]=17
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Urban Pop %"]]="46%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Population (2020)"]]=5518087 #Congo
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Density"]]=16
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Med. Age"]]=19
test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Urban Pop %"]]="70%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Population (2020)"]]=26378274 #CÃ´te d'Ivoire
test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Density"]]=83
test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Med. Age"]]=19
test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Urban Pop %"]]="51%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Population (2020)"]]=53199 #Saint Kitts & Nevis
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Density"]]=205
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Med. Age"]]=36
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Urban Pop %"]]="33%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Population (2020)"]]=110940 #St. Vincent & Grenadines
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Density"]]=284
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Med. Age"]]=33
test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Urban Pop %"]]="53%"
test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Population (2020)"]]=3770 #Population and density are same since it is a cruise ship
test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Density"]]=3770
test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Med. Age"]]=62
test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Urban Pop %"]]="100%"
test_df_pop.loc[test_df_pop["Country_Region"]=="MS Zaandam", ["Population (2020)"]]=1432 #Population and density are same since it is a cruise ship
test_df_pop.loc[test_df_pop["Country_Region"]=="MS Zaandam", ["Density"]]=1432
test_df_pop.loc[test_df_pop["Country_Region"]=="MS Zaandam", ["Med. Age"]]=65
test_df_pop.loc[test_df_pop["Country_Region"]=="MS Zaandam", ["Urban Pop %"]]="100%"


# In[20]:


train_df_pop.isnull().sum()


# In[21]:


test_df_pop.isnull().sum()


# In[22]:


train_df_pop.drop("Country (or dependency)", axis=1, inplace=True)
test_df_pop.drop("Country (or dependency)", axis=1, inplace=True)
#Irrelevant columns


# In[23]:


train_df_pop.rename(columns={'Country_Region':'Country'}, inplace=True)
test_df_pop.rename(columns={'Country_Region':'Country'}, inplace=True)

train_df_pop.rename(columns={'Province_State':'State'}, inplace=True)
test_df_pop.rename(columns={'Province_State':'State'}, inplace=True)


# In[24]:


#Creating a new column-"day_from_jan_first"
mo = train_df_pop['Date'].apply(lambda x: x[5:7])
da = train_df_pop['Date'].apply(lambda x: x[8:10])
mo_test = test_df_pop['Date'].apply(lambda x: x[5:7])
da_test = test_df_pop['Date'].apply(lambda x: x[8:10])
train_df_pop['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )
test_df_pop['day_from_jan_first'] = (da_test.apply(int)
                               + 31*(mo_test=='02') 
                               + 60*(mo_test=='03')
                               + 91*(mo_test=='04')  
                              )


# In[25]:


train_df_pop["Date"] = train_df_pop["Date"].apply(lambda x:x.replace("-",""))
train_df_pop["Date"] = train_df_pop["Date"].astype(int)


# In[26]:


test_df_pop["Date"] = test_df_pop["Date"].apply(lambda x:x.replace("-",""))
test_df_pop["Date"] = test_df_pop["Date"].astype(int)


# In[27]:


#Function to fill empty state values
EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


# In[28]:


train_copy = train_df_pop.copy()
#Copy of train set


# In[29]:


train_copy['State'].fillna(EMPTY_VAL, inplace=True)
train_copy['State'] = train_copy.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)


# In[30]:


train_copy.head()


# In[31]:


test_copy = test_df_pop.copy()


# In[32]:


test_copy['State'].fillna(EMPTY_VAL, inplace=True)
test_copy['State'] = test_copy.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)


# In[33]:


test_copy.head()


# In[34]:


#Function to check for invalid Country names
#https://medium.com/@richamonga86/do-you-want-to-check-if-country-name-coming-in-your-data-is-correct-or-not-5583cee1b960
def country_name_check():
    pycntrylst = list(pycountry.countries)
    alpha_2 = []
    alpha_3 = []
    name = []
    common_name = []
    official_name = []
    invalid_countrynames =[]
    tobe_deleted = ['IRAN','SOUTH KOREA','NORTH KOREA','SUDAN','MACAU','REPUBLIC OF IRELAND']
    for i in pycntrylst:
        alpha_2.append(i.alpha_2)
        alpha_3.append(i.alpha_3)
        name.append(i.name)
        if hasattr(i, "common_name"):
            common_name.append(i.common_name)
        else:
            common_name.append("")
        if hasattr(i, "official_name"):
            official_name.append(i.official_name)
        else:
            official_name.append("")
    for j in country_list:
        if j not in map(str.upper,alpha_2) and j not in map(str.upper,alpha_3) and j not in map(str.upper,name) and j not in map(str.upper,common_name) and j not in map(str.upper,official_name):
            invalid_countrynames.append(j)
    invalid_countrynames = list(set(invalid_countrynames))
    invalid_countrynames = [item for item in invalid_countrynames if item not in tobe_deleted]
    return print(invalid_countrynames)


# In[35]:


country_list=list(train_copy["Country"])


# In[36]:


country_list=[element.upper() for element in country_list]


# In[37]:


country_name_check()
#Invalid country names


# In[38]:


#Final function to extract continent name
@functools.lru_cache(maxsize=128)
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        if country in ['Congo (Brazzaville)', "Cote d'Ivoire", 'MS Zaandam', 'Diamond Princess', 'Holy See', 'Syria', 'West Bank and Gaza', 'Kosovo', 'Russia', 'Taiwan*', 'Korea, South', 'Burma', 'Congo (Kinshasa)', 'Laos', 'Brunei']:
            return np.nan
    else:
        return pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(result[0].alpha_2))


# In[39]:


train_copy.loc[train_copy["Country"]=="Holy See", ["Country"]]="Rome"
train_copy.loc[train_copy["Country"]=="Timor-Leste", ["Country"]]="Bali"
train_copy.loc[train_copy["Country"]=="Western Sahara", ["Country"]]="Rabat"
#This step is being done because otherwise an error is popping up, ie-
#KeyError: "Invalid Country Alpha-2 code: 'VA'" for the values-"Holy See", "Timor-Leste" and "Western Sahara"
#Replacing these values with certain city names to get the correct continent mapping
#After extracting the Continent name, the country name will be changed back to original


# In[40]:


test_copy.loc[test_copy["Country"]=="Holy See", ["Country"]]="Rome"
test_copy.loc[test_copy["Country"]=="Timor-Leste", ["Country"]]="Bali"
test_copy.loc[test_copy["Country"]=="Western Sahara", ["Country"]]="Rabat"
#This step is being done because otherwise an error is popping up, ie-
#KeyError: "Invalid Country Alpha-2 code: 'VA'" for the values-"Holy See", "Timor-Leste" and "Western Sahara"
#Replacing these values with certain city names to get the correct continent mapping
#After extracting the Continent name, the country name will be changed back to original


# In[41]:


train_copy["Continent"] = train_copy["Country"].apply(lambda country: do_fuzzy_search(country))


# In[42]:


test_copy["Continent"] = test_copy["Country"].apply(lambda country: do_fuzzy_search(country))


# In[43]:


train_copy["Continent"].value_counts()


# In[44]:


test_copy["Continent"].value_counts()


# In[45]:


#Filling in for missing values
train_copy.loc[train_copy["Country"]=="Burma", ["Continent"]]="Asia"
train_copy.loc[train_copy["Country"]=="MS Zaandam", ["Continent"]]="Others"
train_copy.loc[train_copy["Country"]=="Diamond Princess", ["Continent"]]="Others"
train_copy.loc[train_copy["Country"]=="Congo (Kinshasa)", ["Continent"]]="Africa"
train_copy.loc[train_copy["Country"]=="Congo (Brazzaville)", ["Continent"]]="Africa"
train_copy.loc[train_copy["Country"]=="West Bank and Gaza", ["Continent"]]="Asia"
train_copy.loc[train_copy["Country"]=="Taiwan*", ["Continent"]]="Asia"
train_copy.loc[train_copy["Country"]=="Korea, South", ["Continent"]]="Asia"
train_copy.loc[train_copy["Country"]=="Laos", ["Continent"]]="Asia"


# In[46]:


train_copy.isnull().sum()


# In[47]:


#Filling in for missing values
test_copy.loc[test_copy["Country"]=="Burma", ["Continent"]]="Asia"
test_copy.loc[test_copy["Country"]=="MS Zaandam", ["Continent"]]="Others"
test_copy.loc[test_copy["Country"]=="Diamond Princess", ["Continent"]]="Others"
test_copy.loc[test_copy["Country"]=="Congo (Kinshasa)", ["Continent"]]="Africa"
test_copy.loc[test_copy["Country"]=="Congo (Brazzaville)", ["Continent"]]="Africa"
test_copy.loc[test_copy["Country"]=="West Bank and Gaza", ["Continent"]]="Asia"
test_copy.loc[test_copy["Country"]=="Taiwan*", ["Continent"]]="Asia"
test_copy.loc[test_copy["Country"]=="Korea, South", ["Continent"]]="Asia"
test_copy.loc[test_copy["Country"]=="Laos", ["Continent"]]="Asia"


# In[48]:


test_copy.isnull().sum()


# In[49]:


train_copy.loc[train_copy["Country"]=="Rome", ["Country"]]="Holy See"
train_copy.loc[train_copy["Country"]=="Bali", ["Country"]]="Timor-Leste"
train_copy.loc[train_copy["Country"]=="Rabat", ["Country"]]="Western Sahara"
#Changing back to original values


# In[50]:


test_copy.loc[test_copy["Country"]=="Rome", ["Country"]]="Holy See"
test_copy.loc[test_copy["Country"]=="Bali", ["Country"]]="Timor-Leste"
test_copy.loc[test_copy["Country"]=="Rabat", ["Country"]]="Western Sahara"
#Changing back to original values


# In[51]:


train_copy[train_copy["Med. Age"]=="N.A."].groupby(["State","Country"]).sum()
#Checking for NA values for median age


# In[52]:


train_copy.loc[train_copy["Country"]=="Andorra", ["Med. Age"]]=44.9
train_copy.loc[train_copy["Country"]=="Dominica", ["Med. Age"]]=27
train_copy.loc[train_copy["Country"]=="Holy See", ["Med. Age"]]=25
train_copy.loc[train_copy["Country"]=="Liechtenstein", ["Med. Age"]]=41
train_copy.loc[train_copy["Country"]=="Monaco", ["Med. Age"]]=53
train_copy.loc[train_copy["Country"]=="San Marino", ["Med. Age"]]=45
#Filling in missing values for median age in training data 


# In[53]:


test_copy.loc[test_copy["Country"]=="Andorra", ["Med. Age"]]=44.9
test_copy.loc[test_copy["Country"]=="Dominica", ["Med. Age"]]=27
test_copy.loc[test_copy["Country"]=="Holy See", ["Med. Age"]]=25
test_copy.loc[test_copy["Country"]=="Liechtenstein", ["Med. Age"]]=41
test_copy.loc[test_copy["Country"]=="Monaco", ["Med. Age"]]=53
test_copy.loc[test_copy["Country"]=="San Marino", ["Med. Age"]]=45
#Filling in missing values for median age in test data 


# In[54]:


train_copy.info()


# In[55]:


train_copy["Med. Age"]=train_copy["Med. Age"].astype(int)
test_copy["Med. Age"]=test_copy["Med. Age"].astype(int)
#Converting median age to integer type


# In[56]:


train_copy[train_copy["Urban Pop %"]=="N.A."].groupby(["State","Country"]).sum()
#Checking for NA values for urban population percentage


# In[57]:


train_copy.loc[train_copy["Country"]=="Kuwait", ["Urban Pop %"]]="100%"
train_copy.loc[train_copy["Country"]=="Singapore", ["Urban Pop %"]]="100%"
train_copy.loc[train_copy["Country"]=="Holy See", ["Urban Pop %"]]="100%"
train_copy.loc[train_copy["Country"]=="Venezuela", ["Urban Pop %"]]="88%"
train_copy.loc[train_copy["Country"]=="Monaco", ["Urban Pop %"]]="100%"
#Filling in missing values for urban population percentage in training data 


# In[58]:


test_copy.loc[test_copy["Country"]=="Kuwait", ["Urban Pop %"]]="100%"
test_copy.loc[test_copy["Country"]=="Singapore", ["Urban Pop %"]]="100%"
test_copy.loc[test_copy["Country"]=="Holy See", ["Urban Pop %"]]="100%"
test_copy.loc[test_copy["Country"]=="Venezuela", ["Urban Pop %"]]="88%"
test_copy.loc[test_copy["Country"]=="Monaco", ["Urban Pop %"]]="100%"
#Filling in missing values for urban population percentage in test data 


# In[59]:


train_copy['Urban Pop %']=train_copy['Urban Pop %'].str.replace('%','').astype(float)/100
#Converting urban population percentage to float


# In[60]:


train_copy.head()


# In[61]:


test_copy['Urban Pop %']=test_copy['Urban Pop %'].str.replace('%','').astype(float)/100
#Converting urban population percentage to float


# In[62]:


test_copy.head()


# In[63]:


labelencoder = LabelEncoder()


# In[64]:


train_copy['Country'] = labelencoder.fit_transform(train_copy['Country'])
train_copy['State'] = labelencoder.fit_transform(train_copy['State'])
train_copy['Continent'] = labelencoder.fit_transform(train_copy['Continent'])
#Encoding country,state and continent values


# In[65]:


test_copy['Country'] = labelencoder.fit_transform(test_copy['Country'])
test_copy['State'] = labelencoder.fit_transform(test_copy['State'])
test_copy['Continent'] = labelencoder.fit_transform(test_copy['Continent'])
#Encoding country,state and continent values


# In[66]:


train_copy.info()


# In[67]:


train_copy.columns


# In[68]:


train_copy.head()


# In[69]:


corrMatrix = train_copy.corr()
plt.figure(figsize=(20,10))
sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[70]:


X=train_copy[['State', 'Country', 'Continent', 'day_from_jan_first', 'Population (2020)', 'Density', 'Med. Age', 'Urban Pop %']]


# In[71]:


y1=train_copy["ConfirmedCases"] #Confirmed Case
y2=train_copy["Fatalities"]     #Fatalities


# In[72]:


#Confirmed Cases
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(X, y1, test_size = .20, random_state = 42)


# In[73]:


#scaler = MinMaxScaler()
#X_train_confirmed = scaler.fit_transform(X_train_confirmed)
#X_test_confirmed = scaler.transform(X_test_confirmed)


# In[74]:


dt1=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)


# In[75]:


dt1.fit(X_train_confirmed, y_train_confirmed)


# In[76]:


y_pred_dt_confirmed=dt1.predict(X_test_confirmed)


# In[77]:


np.sqrt(mean_squared_log_error( y_test_confirmed, y_pred_dt_confirmed ))


# In[78]:


#Fatalities
X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X, y2, test_size = .20, random_state = 42)


# In[79]:


#scaler1 = MinMaxScaler()
#X_train_fatal = scaler1.fit_transform(X_train_fatal)
#X_test_fatal = scaler1.transform(X_test_fatal)


# In[80]:


dt2=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)


# In[81]:


dt2.fit(X_train_fatal, y_train_fatal)


# In[82]:


y_pred_dt_fatal=dt2.predict(X_test_fatal)


# In[83]:


np.sqrt(mean_squared_log_error( y_test_fatal, y_pred_dt_fatal ))


# In[84]:


test_copy.head()


# In[85]:


X_test=test_copy[['State', 'Country', 'Continent', 'day_from_jan_first','Population (2020)', 'Density', 'Med. Age', 'Urban Pop %']]


# In[86]:


#scaler2 = MinMaxScaler()
#X_test = scaler2.fit_transform(X_test)


# In[87]:


y_confirmed=dt1.predict(X_test)


# In[88]:


y_fatal=dt2.predict(X_test)


# In[89]:


submission=pd.DataFrame({'ForecastId': test_copy["ForecastId"], 'ConfirmedCases': y_confirmed, 'Fatalities': y_fatal})


# In[90]:


submission.head()


# In[91]:


submission.to_csv('submission.csv', index=False)

