#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile,sqlite3,os; import pandas as pd
import numpy as np,sympy as sp,pylab as pl
from IPython.core.display import display, HTML
from IPython.core.magic import register_line_magic
pl.rcParams['xtick.major.pad']='1'
@register_line_magic
def get_query(q):
    sp.pprint(r'SQL Queries')
    tr=[]; cursor.execute(q)
    result=cursor.fetchall()
    for r in result: 
        tr+=[r]
    display(pd.DataFrame.from_records(tr)              .style.set_table_styles(style_dict))
def connect_to_db(dbf):
    sqlconn=None
    try:
        sqlconn=sqlite3.connect(dbf)
        return sqlconn
    except Error as err:
        print(err)
        if sqlconn is not None:
            sqlconn.close()
connection=connect_to_db('example.db')
if connection is not None:
    cursor=connection.cursor()
thp=[('font-size','15px'),('text-align','center'),
     ('font-weight','bold'),('padding','5px 5px'),
     ('color','white'),('background-color','slategray')]
tdp=[('font-size','14px'),('padding','5px 5px'),
     ('text-align','center'),('color','darkblue'),
     ('background-color','silver')]
style_dict=[dict(selector='th',props=thp),
            dict(selector='td',props=tdp)]
fp='../input/sberbank-russian-housing-market/'
fp2='../input/meals-programs-in-seattle/'
[os.listdir(),os.listdir('../input'),
os.listdir('../input/sberbank-russian-housing-market'),
os.listdir('../input/meals-programs-in-seattle')]


# In[2]:


fpath='https://data.cityofnewyork.us/resource/'
se=pd.read_json(fpath+'h7rb-945c.json')
fl=['dbn','total_students',
    'graduation_rate','attendance_rate',
    'latitude','longitude',
    'city','council_district']
se=se[fl].dropna().astype({'council_district':'int'})
se.to_sql('schooledu',con=connection,if_exists='replace')
se.head(int(10)).style  .set_table_styles(style_dict)


# In[3]:


# from csv
url='https://raw.githubusercontent.com/plotly/'+    'datasets/master/2016-weather-data-seattle.csv'
weather=pd.read_csv(url).dropna()
weather=weather.astype({'Mean_TemperatureC':'int64',
                        'Min_TemperatureC':'int64'})
weather.to_sql('weather',con=connection,
               if_exists='replace')
weather.describe().style       .set_table_styles(style_dict)


# In[4]:


# from csv
url='https://raw.githubusercontent.com/'+    'OlgaBelitskaya/machine_learning_engineer_nd009/'+    'master/Machine_Learning_Engineer_ND_P3/customers.csv'
customers=pd.read_csv(url).dropna()
customers.to_sql('customers',con=connection,
                 if_exists='replace')
customers.head(7).style         .set_table_styles(style_dict)


# In[5]:


# from csv
url=fp2+'meals-programs-in-seattle.csv'
meals=pd.read_csv(url).dropna()
meals.to_sql('meals',con=connection,
             if_exists='replace')
meals.tail(2).T.style     .set_table_styles(style_dict)


# In[6]:


# from csv
url=fp+'macro.csv.zip'
macro=pd.read_csv(url)
fl=['timestamp','oil_urals','brent','cpi','ppi',
    'usdrub','eurrub','salary','salary_growth',
    'unemployment','employment','average_life_exp',
    'pop_natural_increase','childbirth']
macro=macro[fl].dropna()
macro.to_sql('macro',con=connection,
             if_exists='replace')
macro.tail(int(7)).T.style     .set_table_styles(style_dict)


# In[7]:


# from csv
zipf=zipfile.ZipFile(fp+'train.csv.zip','r')
zipf.extractall(''); zipf.close()
housing=pd.read_csv('train.csv')
fnum=['timestamp','full_sq','floor','max_floor','num_room',
      'area_m','kremlin_km','big_road2_km','big_road1_km',
      'workplaces_km','stadium_km','swim_pool_km','fitness_km',
      'detention_facility_km','cemetery_km','radiation_km',
      'oil_chemistry_km','theater_km','exhibition_km','museum_km',
      'park_km','public_healthcare_km','metro_min_walk',
      'metro_km_avto', 'bus_terminal_avto_km',
      'public_transport_station_min_walk',
      'railroad_station_walk_min','railroad_station_avto_km',
      'kindergarten_km','school_km','preschool_km',
      'university_km','additional_education_km',
      'shopping_centers_km','big_market_km','ekder_all',
      'work_all','young_all']
fcat=['sub_area','ID_metro','office_raion',
      'raion_popul','healthcare_centers_raion',
      'school_education_centers_raion','sport_objects_raion',
      'preschool_education_centers_raion']
housing=housing[fnum+fcat].dropna()
housing.to_sql('housing',con=connection,
               if_exists='replace')
housing.tail(int(5)).T.style       .set_table_styles(style_dict)


# In[8]:


# from csv
url='https://raw.githubusercontent.com/noahgift/'+    'mma/master/data/ufc_fights_all.csv'
mma=pd.read_csv(url).dropna()
mma.to_sql('mma',con=connection,
           if_exists='replace')
mma.tail(int(2)).T.style   .set_table_styles(style_dict)


# In[9]:


get_ipython().run_line_magic('get_query', 'SELECT * FROM sqlite_master;')


# In[10]:


get_ipython().run_line_magic('get_query', 'PRAGMA table_info("macro")')


# In[11]:


get_ipython().run_line_magic('get_query', 'SELECT CURRENT_TIMESTAMP;')


# In[12]:


get_ipython().run_line_magic('get_query', "SELECT DISTINCT * FROM (VALUES ('x'),('y'),('z'),             ('X'),('Y'),('Z'),             ('x'),('y'),('z'))")


# In[13]:


get_ipython().run_line_magic('get_query', 'SELECT DISTINCT Name_of_Program As programs FROM meals WHERE People_Served="OPEN TO ALL";')


# In[14]:


pd.read_sql_query('''
SELECT "📑 "||dbn||" 🏙 "||city AS dbn_city,graduation_rate
FROM schooledu
WHERE city="Bronx" AND dbn LIKE "10%" AND
graduation_rate>.9 AND graduation_rate<>"N/A";
''',con=connection)\
.style.set_table_styles(style_dict)


# In[15]:


pd.read_sql_query('''
SELECT " 🏙 "||city||" 📑 "||dbn AS dbn_city,
       graduation_rate+attendance_rate 
       AS graduation_attendance_rate
FROM schooledu
WHERE graduation_attendance_rate<1.3 AND 
graduation_rate<>"N/A" AND attendance_rate<>"N/A";
''',con=connection)\
.style.set_table_styles(style_dict)


# In[16]:


q=pd.read_sql_query('''
SELECT salary,average_life_exp,
       employment,unemployment
FROM macro;
''',con=connection)
q.plot(kind='kde',figsize=(6,12),
       cmap=pl.cm.winter,
       subplots=True,sharex=False)
pl.show()


# In[17]:


q=pd.read_sql_query('''
SELECT usdrub,eurrub,oil_urals,brent
FROM macro;
''',con=connection)
fl=['usdrub','eurrub','oil_urals','brent']
qp=q.plot(kind='line',figsize=(10,6),
       secondary_y=fl[2:],cmap=pl.cm.winter)
lines=qp.get_lines()+qp.right_ax.get_lines()
qp.legend(lines,[l.get_label() for l in lines],
          bbox_to_anchor=(.2,.6))
pl.grid(); pl.show()


# In[18]:


display(weather        .groupby(weather['Date']                 .map(lambda x: x[-4:]))        .mean().tail().style        .set_table_styles(style_dict))
q=pd.read_sql_query('''
SELECT SUBSTR(Date,-4,4) AS Years,
       AVG(Max_TemperatureC),
       AVG(Mean_TemperatureC),
       AVG(Min_TemperatureC)
FROM weather
GROUP BY Years;
''',con=connection).set_index('Years')
q.plot(kind='area',figsize=(6,4),
       alpha=.7,cmap=pl.cm.tab10)
pl.show()
q.tail().style.set_table_styles(style_dict)


# In[19]:


fl=['Fresh','Milk','Grocery','Frozen',
    'Detergents_Paper','Delicatessen']
display(customers[fl].where(customers.Fresh>50000)        .dropna().sort_values('Fresh',ascending=False)        .astype('int64').style.set_table_styles(style_dict))
q=pd.read_sql_query('''
SELECT Fresh,Milk,Grocery,Frozen,
       Detergents_Paper,Delicatessen
FROM customers
WHERE Fresh>50000
ORDER BY Fresh DESC;
''',con=connection)
q.plot(kind='area',cmap=pl.cm.tab10,
       alpha=.7,figsize=(6,4))
pl.show()
q.style.set_table_styles(style_dict)


# In[20]:


fl=['workplaces_km','public_transport_station_min_walk',
    'school_km','public_healthcare_km']
display(housing[fl]        .groupby(housing['ID_metro'])        .mean().tail(3).T.style        .set_table_styles(style_dict))
q=pd.read_sql_query('''
SELECT ID_metro,
       AVG(workplaces_km),
       AVG(public_transport_station_min_walk),
       AVG(school_km),
       AVG(public_healthcare_km)
FROM housing
GROUP BY ID_metro;
''',con=connection).set_index('ID_metro')
q.iloc[:,int(1):].plot(kind='area',figsize=(10,5),
       alpha=.7,cmap=pl.cm.tab10)
pl.show()
q.tail(3).T.style.set_table_styles(style_dict)


# In[21]:


q=pd.read_sql_query('''
SELECT method,
       COUNT(*) AS number,
       AVG(round) AS avg_round
FROM mma
GROUP BY method;
''',con=connection).set_index('method')
q.plot(secondary_y='avg_round',
       figsize=(10,5),cmap=pl.cm.winter)
pl.show()
q.tail().style.set_table_styles(style_dict)


# In[22]:


q=pd.read_sql_query('''
SELECT method_d,
       COUNT(*) AS number,
       AVG(time) AS avg_time
FROM mma
GROUP BY method_d;
''',con=connection)
n,m=int(70),int(7)
pl.rcParams['xtick.major.pad']='-10'
pl.rcParams['xtick.bottom']=pl.rcParams['xtick.labelbottom']=False
pl.rcParams['xtick.top']=pl.rcParams['xtick.labeltop']=True
qp=q.tail(n).plot(secondary_y='avg_time',
                  figsize=(10,6),alpha=.5,
                  lw=3,cmap=pl.cm.winter)
qp.set_xticks(q.tail(n).index)
qp.set_xticklabels(q.method_d[:n],va='top',
                   fontsize=7,rotation=90)
lines=qp.get_lines()+qp.right_ax.get_lines()
qp.legend(lines,[l.get_label() for l in lines],
          bbox_to_anchor=(.9,.5))
pl.show()
q.tail(m).style.set_table_styles(style_dict)


# In[23]:


if connection is not None:
    connection.close()
if os.path.exists('example.db'):
    os.remove('example.db')
else:
    print('The file does not exist')
os.remove('train.csv')
os.listdir()

