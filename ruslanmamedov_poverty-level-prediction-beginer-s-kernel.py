#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sea #for visualization

# Set a few plotting defaults
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15
plt.rcParams['patch.edgecolor'] = 'k'


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.p


# In[ ]:


#let's look at all available files:
import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print ("Train Dataset: Rows, Columns: ", train_df.shape)
print ("Test Dataset: Rows, Columns: ", test_df.shape)


# In[ ]:


#the prediction will be based on the head of the household
submit = test_df[['Id','idhogar']]
#https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/


# In[ ]:


# a glimpse at train_df
train_df.head(5)


# In[ ]:


train_df.info()


# In[ ]:


#First, let's deal with non-numeric columns
train_df.select_dtypes(['object']).head(15)


# In[ ]:


#Id and idhogar won't be used for training so we'll take care of them later
#1. 'dependency'
train_df['dependency'].value_counts(ascending=False)


# In[ ]:


#Notice there is a column containing squared values for dependency, 'SQBdependency'. 
#see what are its analogs to 'yes' and 'no' of 'dependency':
print (train_df.loc[train_df['dependency']=='no',['SQBdependency']]['SQBdependency'].value_counts())
print (train_df.loc[train_df['dependency']=='yes',['SQBdependency']]['SQBdependency'].value_counts())


# In[ ]:


#Convert 'yes' to 1 and 'no' to 0
train_df['dependency'] = train_df['dependency'].replace(('yes', 'no'), (1, 0))
test_df['dependency'] = test_df['dependency'].replace(('yes', 'no'), (1, 0))
train_df['dependency']=train_df['dependency'].astype(float)
test_df['dependency']=test_df['dependency'].astype(float)


# In[ ]:


#2 and #3 'edjefe'/'edjefa'
train_df['edjefe'].value_counts()
#edjefe, years of education of male head of household, 
#based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0


# In[ ]:


train_df['edjefa'].value_counts()
#edjefa, years of education of female head of household, 
#based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0


# In[ ]:


#Again, correlate 'edjefe' with 'SQBedjefe'(squared value)
print (train_df.loc[train_df['edjefe']=='no',['SQBedjefe']]['SQBedjefe'].value_counts())
print (train_df.loc[train_df['edjefe']=='yes',['SQBedjefe']]['SQBedjefe'].value_counts())


# In[ ]:


#Based on 'SQBedjefe' column, convert 'no' to 0 and 'yes' to 1 to make the rows of 'edjefa'/'edjefe' numeric
train_df['edjefa'] = train_df['edjefa'].replace(('yes', 'no'), (1, 0))
train_df['edjefe'] = train_df['edjefe'].replace(('yes', 'no'), (1, 0))
test_df['edjefa'] = test_df['edjefa'].replace(('yes', 'no'), (1, 0))
test_df['edjefe'] = test_df['edjefe'].replace(('yes', 'no'), (1, 0))


# In[ ]:


#converting these object type columns to floats
train_df['edjefa']=train_df['edjefa'].astype(float)
train_df['edjefe']=train_df['edjefe'].astype(float)
test_df['edjefa']=test_df['edjefa'].astype(float)
test_df['edjefe']=test_df['edjefe'].astype(float)


# In[ ]:


#double checking that all columns are now numeric - except for Id and idhogar
print (train_df.select_dtypes(['object']).describe(), '\n')
print (test_df.select_dtypes(['object']).describe())


# In[ ]:


#Now let's take care of the missing columns
print ("Top Training Columns having missing values:")
missing_df = train_df.isnull().sum().to_frame()
missing_df = missing_df.sort_values(0, ascending = False)
missing_df.head()


# In[ ]:


print ("Top Testing Columns having missing values:")
missing_df = test_df.isnull().sum().to_frame()
missing_df = missing_df.sort_values(0, ascending = False)
missing_df.head()


# In[ ]:


#1 'v18q1' - number of tablets household owns
train_df.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[ ]:


#Every family that has nan for v18q1 does not own a tablet. 
#Therefore, we can fill in this missing value with zero.
train_df['v18q1'] = train_df['v18q1'].fillna(0)
test_df['v18q1'] = test_df['v18q1'].fillna(0)


# In[ ]:


#2 'rez_esc' - Years behind in school 
#let's see if high percentage of missing values in 'rez_esc' accounts for minors and people without education
print (train_df.loc[train_df['rez_esc'].isnull()]['age'].value_counts().head(6))
print (train_df.loc[train_df['rez_esc'].isnull()]['instlevel1'].value_counts())
print (train_df.loc[train_df['rez_esc'].isnull()]['instlevel2'].value_counts())


# In[ ]:


#another theory is that those 'na' are for individuals outside of school age
print (train_df.loc[train_df['rez_esc'].notnull()]['age'].describe())


# In[ ]:


#which is actually true: min age - 7, max age - 17. Assigning '0' to those people
train_df['rez_esc'] = train_df['rez_esc'].fillna(0)
test_df['rez_esc'] = test_df['rez_esc'].fillna(0)
train_df.loc[train_df['rez_esc'] > 5, 'rez_esc'] = 5
test_df.loc[test_df['rez_esc'] > 5, 'rez_esc'] = 5 #5 is a maximum value per competition's discussion, so here we're accounting for the outliers


# In[ ]:


#3 v2a1, Monthly rent payment
print(train_df['v2a1'].isnull().sum())


# In[ ]:


#Let's correlate it with tipovivi1, =1 own and fully paid house
print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi1'].value_counts())
print(train_df['tipovivi1'].value_counts())


# In[ ]:


#Replacing with '0' na for fully paid house 
train_df.loc[(train_df['v2a1'].isnull() & train_df['tipovivi1'] == 1), 'v2a1'] = 0
test_df.loc[(test_df['v2a1'].isnull() & test_df['tipovivi1'] == 1), 'v2a1'] = 0


# In[ ]:


print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi1'].value_counts())


# In[ ]:


#tipovivi2, "=1 own,  paying in installments"
#tipovivi3, =1 rented
#tipovivi4, =1 precarious
#tipovivi5, "=1 other(assigned,  borrowed)
print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi2'].value_counts())
print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi3'].value_counts())
print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi4'].value_counts())
print (train_df.loc[train_df['v2a1'].isnull()]['tipovivi5'].value_counts())


# In[ ]:


#Let's replace na for precarious with '0' as well
train_df.loc[(train_df['v2a1'].isnull() & train_df['tipovivi4'] == 1), 'v2a1'] = 0
test_df.loc[(test_df['v2a1'].isnull() & test_df['tipovivi4'] == 1), 'v2a1'] = 0


# In[ ]:


print (train_df.loc[train_df['v2a1'].isnull()]['Target'].value_counts())


# In[ ]:


#see if we can find a feature to correlate with those remaining missing values
v2a1_na_corr = train_df
v2a1_na_corr.v2a1.where(v2a1_na_corr.v2a1.isnull(), 1, inplace=True)
v2a1_na_corr['v2a1'].fillna(0, inplace = True)
print (v2a1_na_corr.corr()['v2a1'].sort_values())


# In[ ]:


#No luck. But since the property is 'assigned, borrowed', let's assume there's no monthly rent associated with it
train_df['v2a1'].fillna(train_df['v2a1'].mean(), inplace = True)
test_df['v2a1'].fillna(test_df['v2a1'].mean(), inplace = True)


# In[ ]:


print ("Top Training Columns having missing values:")
missing_df = train_df.isnull().sum().to_frame()
missing_df = missing_df.sort_values(0, ascending = False)
print (missing_df.head())
print ("Top Testing Columns having missing values:")
missing_df = test_df.isnull().sum().to_frame()
missing_df = missing_df.sort_values(0, ascending = False)
print (missing_df.head())


# In[ ]:


#the rest of the missing values can be replaced with mean as their percentage towards total number of entries is insignificant
train_df.fillna (train_df.mean(), inplace = True)
test_df.fillna(test_df.mean(), inplace = True)


# In[ ]:


print ('Columns having missing values:')
print (train_df.columns[train_df.isnull().any()])
print (test_df.columns[test_df.isnull().any()])


# In[ ]:


#top 30 features with best correlation to 'Target'
best_correlations = train_df.corr()['Target'].abs().sort_values().tail(30)
type(best_correlations)
best_correlations


# In[ ]:


best_correlation = best_correlations.index
best_correlation


# In[ ]:


d = {'dependency':'dependency, Dependency rate', 'v18q1':'v18q1, number of tablets household owns', 'epared1':'epared1, if walls are bad', 'qmobilephone':'qmobilephone, # of mobile phones', 
     'pisocemento':'pisocemento, =1 if predominant material on the floor is cement',
       'eviv1':'eviv1, =1 if floor are bad', 'instlevel8':'instlevel8, =1 undergraduate and higher education', 'rooms':'rooms,  number of all rooms in the house', 'r4h1':'r4h1, Males younger than 12 years of age', 
        'v18q': 'v18q, owns a tablet:', 'edjefe':'edjefe, years of education of male head of household', 'SQBedjefe':'SQBedjefe, years of education of male head of household squared',
       'etecho3':'etecho3, =1 if roof are good', 'r4m1':'r4m1, Females younger than 12 years of age', 'SQBovercrowding':'SQBovercrowding, overcrowding squared', 
       'paredblolad':'paredblolad, =1 if predominant material on the outside wall is block or brick', 'SQBmeaned':'SQBmeaned, square of the mean years of education of adults (>=18) in the household',
       'pisomoscer':'pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"', 'overcrowding':'overcrowding, # persons per room', 'epared3':'epared3, =1 if walls are good',
        'eviv3':'eviv3, =1 if floor are good', 'SQBescolari' :'SQBescolari, years of schooling squared',
       'escolari':'escolari, years of schooling', 'cielorazo':'cielorazo, =1 if the house has ceiling', 'SQBhogar_nin':'SQBhogar_nin, Number of children 0 to 19 in household, squared',
        'r4t1':'r4t1, persons younger than 12 years of age', 'hogar_nin':'hogar_nin, Number of children 0 to 19 in household',
       'meaneduc':'meaneduc,average years of education for adults (18+)', 'Target':'Target', 'elimbasu5':'elimbasu5, "=1 if rubbish disposal mainly by throwing in river,  creek or sea"'}


# In[ ]:


for i in best_correlation:
    if len(train_df[i].unique())>2:
        sea.boxplot(train_df[i])
        plt.xlabel(d.get(i))
        plt.show()
        sea.distplot(train_df[i])
        plt.xlabel(d.get(i))
        plt.show()


# In[ ]:


#we'll drop only the ones with less than 100 outliers
print(len(train_df.loc[(train_df['SQBmeaned']>900)]))
print(train_df['SQBmeaned'].value_counts(sort = True))


# In[ ]:


to_drop = train_df.loc[(train_df['rooms']>9)|(train_df['r4m1']>3)|
                       (train_df['r4t1']>5)|(train_df['hogar_nin']>6)|
                       (train_df['meaneduc']>25)|
                       (train_df['qmobilephone']>8)|(train_df['r4h1']>2)|
                       (train_df['SQBedjefe']>300)|(train_df['SQBescolari']>300)|
                       (train_df['SQBhogar_nin']>70)|(train_df['SQBovercrowding']>25)|
                       (train_df['SQBmeaned']>900)].index


# In[ ]:


len(to_drop)


# In[ ]:


train_df.drop(to_drop, inplace=True)


# In[ ]:


train_df.groupby('Target').mean()


# In[ ]:


train_df['Target'].hist()


# In[ ]:


#features with <5 possible values
for j in best_correlation:
    if len(train_df[j].unique())<5:
        sea.countplot(x=j, hue='Target', data=train_df)
        plt.xlabel(d.get(j))
        plt.ylabel("Count")
        #plt.title(str(j),' vs Target') 
        plt.figure()
        plt.show()


# In[ ]:


#if target distribution in each feature cathegory is similar to overall target distribution, 
#then the chances are that the feature will have a better correlation to Target
#I tried to combine a couple of features to produce better distributions/correlations
train_df['v18q+etecho3'] = train_df['v18q']+train_df['etecho3']
print (train_df.corr()['Target']['v18q'])
print (train_df.corr()['Target']['etecho3'])
print (train_df.corr()['Target']['v18q+etecho3'])
test_df['v18q+etecho3'] = test_df['v18q']+test_df['etecho3']


# In[ ]:


train_df['v18q+paredblolad'] = train_df['v18q']+train_df['paredblolad']
print (train_df.corr()['Target']['v18q'])
print (train_df.corr()['Target']['paredblolad'])
print (train_df.corr()['Target']['v18q+paredblolad'])
test_df['v18q+paredblolad'] = test_df['v18q']+test_df['paredblolad']


# In[ ]:


train_df['v18q+pisomoscer'] = train_df['v18q']+train_df['pisomoscer']
print (train_df.corr()['Target']['v18q'])
print (train_df.corr()['Target']['pisomoscer'])
print (train_df.corr()['Target']['v18q+pisomoscer'])
test_df['v18q+pisomoscer'] = test_df['v18q']+test_df['pisomoscer']


# In[ ]:


train_df['pisomoscer+instlevel8'] = train_df['pisomoscer']+train_df['instlevel8']
print (train_df.corr()['Target']['pisomoscer'])
print (train_df.corr()['Target']['instlevel8'])
print (train_df.corr()['Target']['pisomoscer+instlevel8'])
test_df['pisomoscer+instlevel8'] = test_df['pisomoscer']+test_df['instlevel8']


# In[ ]:


def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sea.FacetGrid(df, hue = target, size=4.0, aspect=1.3, sharex=False, sharey=False)
    facet.map(sea.kdeplot, var)
    facet.set(xlim = (0, df[var].max()))
    facet.add_legend()
    plt.xlabel(d.get(j))
    plt.show()


# In[ ]:


#features with >=5 possible values
for j in best_correlation:
    if len(train_df[j].unique())>5:
        plot_distribution(train_df, j, 'Target')

#In the first graph instead of 0's should be nulls(we changed these before). So there is no info about monthly rate payment for non vulnerable households 


# In[ ]:


#following the same logic, let's try to combine features to get a better distribution
train_df['edjefe+escolari'] = train_df['edjefe']+train_df['escolari']
print (train_df.corr()['Target']['edjefe'])
print (train_df.corr()['Target']['escolari'])
print (train_df.corr()['Target']['edjefe+escolari'])
test_df['edjefe+escolari'] = test_df['edjefe']+test_df['escolari']


# In[ ]:


#we can also do some pairwise feature comparison for various target cathegories with jointplots:
dict={4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"}
for i in range(1, 5):
    sea.set(font_scale=1, style="white")
    sea_jointplot = sea.jointplot('hogar_nin', 'age', data=train_df[train_df['Target'] == i], size=6,color = sea.color_palette("deep")[i], kind='kde', stat_func=None)
    plt.title(dict.get(i))
plt.show()


# In[ ]:


#finally, let's have some interactive plots as well - using plotly
# Standard plotly imports
pip install plotly chart-studio
import chart_studio.plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


train_df['Target1'] = train_df.Target
train_df.Target1 = train_df.Target1.apply(str)
train_df.iplot(
    x='hogar_nin',
    y='meaneduc',
    # Specify the category
    categories="Target1",
    xTitle = d.get('hogar_nin'),
    yTitle = d.get('meaneduc'),
    title="Number of children vs. Average education by Poverty level")
train_df = train_df.drop(['Target1'], axis =1)


# In[ ]:


train_df.pivot(columns='Target', values='meaneduc').iplot(
        kind='box',
        xTitle = d.get('meaneduc'),
        yTitle= 'Target',
        title='Education level per different poverty groups')


# In[ ]:


trace1 = go.Bar(
    x=train_df['Target'],
    y=train_df['meaneduc'],
    name=d.get('meaneduc')
)
trace2 = go.Bar(
    x=train_df['Target'],
    y=train_df['hogar_nin'],
    name=d.get('hogar_nin')
)
trace3 = go.Bar(
    x=train_df['Target'],
    y=train_df['rooms'],
    name=d.get('rooms')
)
data = [trace2, trace3,trace1]


layout = go.Layout(
    barmode="group",
    hovermode= 'closest',
    showlegend= True,
    xaxis ={"title":"Target"},
    yaxis ={"title":"Count"}
    
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# In[ ]:


best_correlation_df = train_df[['Target']]
for i in best_correlation:
    if len(train_df[i].unique())>2:
        best_correlation_df[i] = train_df[i]
#Correlation Heatmap
corrs = best_correlation_df.corr()
corrs.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


sea.clustermap(corrs)


# In[ ]:


c1 = corrs.abs().unstack().drop_duplicates()
c1.sort_values(ascending = True)


# In[ ]:


train_df_sample = train_df.sample(500) #sampling for better graph
sea.set(rc={'figure.figsize':(12,8)})
sea.swarmplot(x='rooms', y = 'hogar_nin', hue='Target', dodge = True, data=train_df_sample, size = 4)
#sea.violinplot(x='r4t1', y = 'hogar_nin', hue='Target', dodge = True, data=train_df_sample, size = 4)
plt.xlabel(d.get('rooms'))
plt.ylabel(d.get('hogar_nin'))
plt.figure()
plt.show()


# In[ ]:


#some of it has already been done in visualization part above
#poor materials used
train_df["Poor_materials"]=train_df['pareddes']+train_df['paredfibras']+train_df['pisonatur']+train_df['pisonotiene']+train_df['techocane']+train_df['epared1']+train_df['etecho1']+train_df['eviv1']
test_df["Poor_materials"]=test_df['pareddes']+test_df['paredfibras']+test_df['pisonatur']+test_df['pisonotiene']+test_df['techocane']+test_df['epared1']+test_df['etecho1']+test_df['eviv1']
print ('Pearson correlation coefficients:')
print ('Poor Materials (training set): ',train_df['Poor_materials'].corr( train_df['Target']))


# In[ ]:


#rich materials used
train_df["Rich_Materials"]=train_df['paredblolad']+train_df['pisomoscer']+train_df['techoentrepiso']+train_df['techootro']+train_df['cielorazo']+train_df['epared3']+train_df['etecho3']+train_df['eviv3']
test_df["Rich_Materials"]=test_df['paredblolad']+test_df['pisomoscer']+test_df['techoentrepiso']+test_df['techootro']+test_df['cielorazo']+test_df['epared3']+test_df['etecho3']+test_df['eviv3']
print ('Pearson correlation coefficients:')
print ('Materials (training set): ',train_df['Rich_Materials'].corr( train_df['Target']))


# In[ ]:


train_df["Poor_Infrastructure"]=train_df['abastaguano']+train_df['noelec']+train_df['epared1']+train_df['etecho1']+train_df['eviv1']+train_df['lugar3']+train_df['sanitario1']+train_df['energcocinar1']+train_df['elimbasu3']
test_df["Poor_Infrastructure"]=test_df['abastaguano']+test_df['noelec']+test_df['epared1']+test_df['etecho1']+test_df['eviv1']+test_df['lugar3']+test_df['sanitario1']+test_df['energcocinar1']+test_df['elimbasu3']
print ('Pearson correlation coefficients:')
print ('Materials (training set): ',train_df['Poor_Infrastructure'].corr( train_df['Target']))


# In[ ]:


train_df["Good_Infrastructure"]=train_df['sanitario2']+train_df['energcocinar2']+train_df['elimbasu1']+train_df['abastaguadentro']+train_df['planpri']+train_df['epared3']+train_df['etecho3']*(3)+train_df['eviv3']+train_df['lugar1']+train_df['lugar2']+train_df['lugar6']
test_df["Good_Infrastructure"]=test_df['sanitario2']+test_df['energcocinar2']+test_df['elimbasu1']+test_df['abastaguadentro']+test_df['planpri']+test_df['epared3']+test_df['etecho3']*(3)+test_df['eviv3']+test_df['lugar1']+test_df['lugar2']+test_df['lugar6']
print ('Pearson correlation coefficients:')
print ('Infrastructure (training set): ',train_df['Good_Infrastructure'].corr( train_df['Target']))


# In[ ]:


#overcrowding + total of persons younger than 12 years of age + no level of education + zona rural
train_df["overcrowding_total"] = train_df["hacdor"]+train_df["r4t1"] +train_df["instlevel1"] + train_df["area2"]
test_df["overcrowding_total"] = test_df["hacdor"]+ test_df["r4t1"] + test_df["instlevel1"] + test_df["area2"]
print ('overcrowding_total: ',train_df['overcrowding_total'].corr( train_df['Target']))


# In[ ]:


#years of schooling + overcdrowding
train_df["escolari+hacapo"] = train_df["escolari"]+train_df["hacapo"]
test_df["escolari+hacapo"] = test_df["escolari"]+test_df["hacapo"]
print (train_df['escolari+hacapo'].corr( train_df['Target']))


# In[ ]:


print(train_df.columns[-30:])


# In[ ]:


pip install --upgrade https://github.com/featuretools/featuretools/zipball/master


# In[ ]:


#credits to Will Koehrsen for his excellent kernel: https://www.kaggle.com/willkoehrsen/featuretools-for-good#Deep-Feature-Synthesis
#first we need to define variable types in the dataframe (boolean vs. ordered vs. continuous) 
#and group them into individual vs. household categories
ind_bool = list()
ind_ordered = list()
ind_cont = list()
hh_bool = list()
hh_ordered = list()
hh_cont = list()
print(train_df.drop(['age', 'SQBescolari','SQBage', 'agesq'], axis = 1).columns.get_loc('Target'))


# In[ ]:


train_list_hh = list(train_df.drop(['escolari', 'rez_esc'], axis = 1).loc[:,'v2a1':'eviv3'].columns)+list(train_df.loc[:,'idhogar':'meaneduc'].columns)+list(train_df.drop(['age', 'SQBescolari','SQBage', 'agesq','Target'], axis = 1).loc[:,'bedrooms':].columns)
train_list_ind = [column for column in list(train_df.columns) if column not in set (train_list_hh)]
print (len(train_list_hh)+len(train_list_ind)-len(list(train_df.columns)))
for i in train_list_hh:
    if len(train_df[i].unique())<=2:
        hh_bool.append(i)
    elif train_df[i].dtypes == 'int':
        hh_ordered.append(i)
    elif train_df[i].dtypes == 'float':
        hh_cont.append(i)

for i in train_list_ind:
    if len(train_df[i].unique())==2:
        ind_bool.append(i)
    elif train_df[i].dtypes == 'int':
        ind_ordered.append(i)
    elif train_df[i].dtypes == 'float':
        ind_cont.append(i) 


# In[ ]:


test_df['Target'] = np.nan

data = train_df.append(test_df, sort = True)
for variable in (hh_bool + ind_bool):
    data[variable] = data[variable].astype('bool')
for variable in (hh_cont + ind_cont):
    data[variable] = data[variable].astype(float)
for variable in (hh_ordered + ind_ordered):
    try:
        data[variable] = data[variable].astype(int)
    except Exception as e:
        print(f'Could not convert {variable} because of missing values.')


# In[ ]:


import featuretools as ft
es = ft.EntitySet(id = 'households') #creating the entity set
es.entity_from_dataframe(entity_id = 'data',    #adding first entity (table) to the entity set
                         dataframe = data, 
                         index = 'Id')

'''hh = hh_bool+hh_ordered+hh_cont+["Target"]+["idhogar"]
household = data.loc[data['parentesco1']==1, hh]
es.entity_from_dataframe(entity_id = 'household',    #adding second entity (table) to the entity set
                         dataframe = household, 
                         index = "idhogar")
household_rl = ft.Relationship(es["household"]["idhogar"],
                              es["data"]["idhogar"])
es = es.add_relationship(household_rl)'''
#we'll make a new entity by normalization of the original table 
es.normalize_entity(base_entity_id='data', 
                    new_entity_id='household', #adding household table to the entity set 
                    index = 'idhogar',
                   additional_variables = hh_bool + hh_ordered + hh_cont+["Target"])


# In[ ]:


feature_matrix, feature_names = ft.dfs(entityset=es, 
                                       target_entity = 'household', 
                                       max_depth = 2, 
                                       verbose = 1, 
                                       n_jobs = -1,
                                       chunk_size = 100)


# In[ ]:


all_features = [str(x.get_name()) for x in feature_names]
feature_matrix.head()


# In[ ]:


feature_matrix.shape


# In[ ]:


drop_cols = []
for col in feature_matrix:
    if col == 'Target':
        pass
    else:
        if 'Target' in col:
            drop_cols.append(col)
            
print(drop_cols)            
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]]         
feature_matrix.head()


# In[ ]:


train_df = feature_matrix[feature_matrix['Target'].notnull()].reset_index()
test_df = feature_matrix[feature_matrix['Target'].isnull()].reset_index()
test_df.head(5)


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


idhogar = test_df['idhogar']
train_df = train_df.select_dtypes(exclude=['object'])
test_df = test_df.select_dtypes(exclude=['object'])
train_df = train_df.dropna(axis='columns')
test_df = test_df.dropna(axis='columns')


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


#Removing columns with greater than 99% correlation as redundant
# Create correlation matrix
corr_matrix = train_df.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.99
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.99)]

print(f'There are {len(to_drop)} correlated columns to remove.')
print(to_drop)

train_df = train_df.drop(columns = to_drop)


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


#let's compare all the correlation coefficients now
print (train_df.corr()['Target'].abs().sort_values().tail(30))


# In[ ]:


#realligning two datasets based on the features selected in training
train_df_H20 = train_df # for use with autoML
y_df = train_df['Target']
train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)
print(f"Training set shape:{train_df.shape}, testing set shape:{test_df.shape}")


# In[ ]:


#converting to numpy array
X = train_df.values
y = y_df.values
y = y.reshape(-1, 1)
test_np = test_df.values
X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,test_size = 0.1, random_state = 123)
X_train.shape


# In[ ]:


#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X = sc.transform(X)
print (X)
test_np = sc.transform (test_np)
print (test_np)


# In[ ]:


'''#First, we'll try some autoML tool to generate a model
import h2o
from h2o.automl import H2OAutoML
h2o.init()

htrain = h2o.H2OFrame(train_df_H20)
htest = h2o.H2OFrame(test_df)
x = htrain.columns
y ="Target"
x.remove(y)
# This line is added in the case of classification
htrain[y] = htrain[y].asfactor()

aml = H2OAutoML(max_runtime_secs = 400)
aml.train(x=x, y =y, training_frame=htrain)
lb = aml.leaderboard
print (lb)'''


# In[ ]:


'''print("Generate predictions…")
test_y = aml.leader.predict(htest)
test_y = test_y.as_data_frame()'''


# In[ ]:


#AutoML can make a decent prediction, but not as good as the manually tuned model yet. For now we'll use LGBoost with early stopping for our final prediction
#credits to https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro for the parameters values
import lightgbm as lgb
classifier = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[ ]:


eval_set = [(X_train, y_train), (X_test, y_test)]
classifier.fit(X_train, y_train, eval_metric="multiclass", eval_set=eval_set, verbose=True, early_stopping_rounds=400) #LGBoost model model
y_pred = classifier.predict(X_test) 
y_pred = y_pred.reshape(-1, 1)


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
print (cm1)


# In[ ]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average ='macro')
print ('f1 score for LGBoost model:',f1)


# In[ ]:


y_pred = classifier.predict(test_np)
y_pred = y_pred.reshape(-1, 1)
y_pred = y_pred.astype(int)
print(plt.hist(y_pred))


# In[ ]:


# Visualise with a barplot
import seaborn as sns
indices = np.argsort(classifier.feature_importances_)[::-1]
indices = indices[:30]


plt.subplots(figsize=(40, 40))
g = sea.barplot(y=train_df.columns[indices], x = classifier.feature_importances_[indices], orient='h')
g.set_xlabel("Relative importance",fontsize=40)
g.set_ylabel("Features",fontsize=40)
g.tick_params(labelsize=40)
g.set_title("Feature importance", fontsize=40)


# In[ ]:


#Submitting the prediction
test_df['Target'] = y_pred.astype(int)
test_df['idhogar'] = idhogar
submit = submit.merge(test_df[['idhogar', 'Target']], on = 'idhogar', how = 'left').drop(columns = ['idhogar'])
#submit['TARGET'] = test_y['predict'].values - for autoML  


# In[ ]:


submit['Target'] = submit['Target'].fillna(4) #there is no head of the household, assigning '4' to those
submit['Target'] = submit['Target'].astype(int)


# In[ ]:


submit['Target'].hist()


# In[ ]:


submit


# In[ ]:


# Save the submission to a csv file
submit.to_csv('LGBClassification.csv', index = False)


# In[ ]:


print ('The prediction was based on LGBoost model with early stopping, trained on ', train_df.shape[1],' features. F1 score for the the training dataset was ',f1,'.')


# In[ ]:




