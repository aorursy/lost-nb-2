#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\ndiv.h2 { background-color: #159957;\n         background-image: linear-gradient(120deg, #155799, #159957);\n         text-align: left;\n         color: white;              \n         padding:9px;\n         padding-right: 100px; \n         font-size: 20px; \n         max-width: 1500px; \n         margin: auto; \n         margin-top: 40px;}\n                                           \n                                           \nbody {font-size: 12px;} \n                                           \n                                           \ndiv.h3 { color: #159957; \n         font-size: 18px; \n         margin-top: 20px; \n         margin-bottom:4px;}\n   \n                                      \ndiv.h4 { color: #159957;\n         font-size: 15px; \n         margin-top: 20px; \n         margin-bottom: 8px;}\n   \n                                           \nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;}\n  \n                                      \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;}   \n    \n                                      \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n      <table align="left">\n    ...\n  </table>\n    background-color: white;}\n    \n                                      \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    text-align: center;} \n   \n                                                \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;}\n       \n        \n                                           \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;} \n   \n                                      \n                                      \ntable.rules tr.best\n{\n    color: green;}    \n    \n                                      \n.output { \n    align-items: left;}\n        \n                                      \n.output_png {\n    display: table-cell;\n    text-align: left;\n    margin:auto;}                                          \n                                                                    \n                                                             \n</style>  ')


# In[2]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#init_notebook_mode(connected=True)  # remove  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import sparklines
import colorcet as cc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from PIL import Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import scipy 
from scipy import constants
import math
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ styles ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###?sns.set_context('paper')  #Everything is smaller, use ? 
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##This helps set size of all fontssns.set(font_scale=1.5)
#~~~~~~~~~~~~~~~~~~~~~~~~~ B O K E H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.io import show
from bokeh.io import push_notebook
from bokeh.io import output_notebook
from bokeh.io import output_file
from bokeh.io import curdoc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.plotting import show                  
from bokeh.plotting import figure                  
from bokeh.plotting import output_notebook 
from bokeh.plotting import output_file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.models import ColumnDataSource
from bokeh.models import Circle
from bokeh.models import Grid 
from bokeh.models import LinearAxis
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import CategoricalColorMapper
from bokeh.models import FactorRange
from bokeh.models.tools import HoverTool
from bokeh.models import FixedTicker
from bokeh.models import PrintfTickFormatter
from bokeh.models.glyphs import HBar
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.core.properties import value
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.palettes import Blues4
from bokeh.palettes import Spectral5
from bokeh.palettes import Blues8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.layouts import gridplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.sampledata.perceptions import probly
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.transform import factor_cmap
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Altair ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Altair
import altair as alt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
import gc, pickle, tqdm, os, datetime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1. kaggle import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# My data sources: 
# ==================
# /kaggle/input/liverpool-ion-switching/train.csv
# /kaggle/input/liverpool-ion-switching/test.csv
# /kaggle/input/liverpool-ion-switching/sample_submission.csv


ion = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')



# ion.signal.describe()
# count    5.000000e+06
# mean     1.386246e+00
# std      3.336219e+00
# min     -5.796500e+00
# 25%     -1.594800e+00
# 50%      1.124000e+00
# 75%      3.690100e+00
# max      1.324400e+01

#  You will need to do this, get the library, and then comment out after:   
#  !pip install altair vega_datasets notebook vega



# the first time you need this (internet enabled), but then comment out afterwards... 
# !pip install altair vega_datasets notebook vega
# !pip install --upgrade altair vega_datasets notebook vega




# HTML(
#     "This block does the following:<br/><ul>"
#     "<li>Loads the column names and questions into <code>questions</code>.</li>"
#     "<li>Separates out groups of questions (e.g. 'AssessJob1-5') in <code>grouped_questions</code>.</li>"
#     "<li>Ingests the data into a data frame <code>df</code>.</li>"
#     "<li>Sets every column except <code>ConvertedSalary</code> to be a string.</li>"
#     "</ul>"
# )


#   !jupyter notebook --version   #  5.5.0




# source = pd.DataFrame({
#     'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
#     'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
# })

# alt.Chart(source).mark_bar().encode(
#     x='a',
#     y='b'
# )



# Define and register a kaggle renderer for Altair

import json
from IPython.display import HTML

KAGGLE_HTML_TEMPLATE = """
<style>
.vega-actions a {{
    margin-right: 12px;
    color: #757575;
    font-weight: normal;
    font-size: 13px;
}}
.error {{
    color: red;
}}
</style>
<div id="{output_div}"></div>
<script>
requirejs.config({{
    "paths": {{
        "vega": "{base_url}/vega@{vega_version}?noext",
        "vega-lib": "{base_url}/vega-lib?noext",
        "vega-lite": "{base_url}/vega-lite@{vegalite_version}?noext",
        "vega-embed": "{base_url}/vega-embed@{vegaembed_version}?noext",
    }}
}});
function showError(el, error){{
    el.innerHTML = ('<div class="error">'
                    + '<p>JavaScript Error: ' + error.message + '</p>'
                    + "<p>This usually means there's a typo in your chart specification. "
                    + "See the javascript console for the full traceback.</p>"
                    + '</div>');
    throw error;
}}
require(["vega-embed"], function(vegaEmbed) {{
    const spec = {spec};
    const embed_opt = {embed_opt};
    const el = document.getElementById('{output_div}');
    vegaEmbed("#{output_div}", spec, embed_opt)
      .catch(error => showError(el, error));
}});
</script>
"""


class KaggleHtml(object):
    def __init__(self, base_url='https://cdn.jsdelivr.net/npm'):
        self.chart_count = 0
        self.base_url = base_url
        
    @property
    def output_div(self):
        return "vega-chart-{}".format(self.chart_count)
        
    def __call__(self, spec, embed_options=None, json_kwds=None):
        # we need to increment the div, because all charts live in the same document
        self.chart_count += 1
        embed_options = embed_options or {}
        json_kwds = json_kwds or {}
        html = KAGGLE_HTML_TEMPLATE.format(
            spec=json.dumps(spec, **json_kwds),
            embed_opt=json.dumps(embed_options),
            output_div=self.output_div,
            base_url=self.base_url,
            vega_version=alt.VEGA_VERSION,
            vegalite_version=alt.VEGALITE_VERSION,
            vegaembed_version=alt.VEGAEMBED_VERSION
        )
        return {"text/html": html}
    
    

alt.data_transformers.disable_max_rows()
# IF YOU ARE EVER GOING TO PROCESS 

    
#alt.themes.enable('ggplot2')

    
# IMPORTANT: 
alt.renderers.register('kaggle', KaggleHtml())


# print("Define and register the kaggle renderer. Enable with\n\n"
#       "    alt.renderers.enable('kaggle')")

alt.renderers.enable('kaggle')

import warnings  
warnings.filterwarnings('ignore')


# In[3]:



correct = pd.DataFrame(ion.open_channels.value_counts())

#column_names = ["open_channels", "counts", "colors"]
correct.reset_index(inplace=True)
correct['color']='yellow'
correct.columns = ['ChannelNumber', 'ChannelCount', 'Color']


plt.style.use('dark_background')
plt.figure(figsize=(8,7))
plt.hlines(y=correct.ChannelNumber, xmin=0, xmax=correct.ChannelCount, color=correct.Color, alpha=0.6, linewidth=14)
plt.gca().set(ylabel='\n$Channel Number$\n', xlabel='\n$Instance Count$\n')
plt.yticks(fontsize=10)
plt.title('\nTraining Data:  Overall Open Channels Distribution\n(5M samples)\n\n', fontdict={'size':11})
plt.grid(linestyle='--', alpha=0.2)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))   
#plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(10))   
sns.despine(top=True, right=True, left=True, bottom=True)  
# plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# this was slightly wrong, fixed it up:
# temp = pd.DataFrame(columns = column_names)
# temp['counts'] = ion.open_channels.value_counts().values
# temp['open_channels'] = ion.open_channels.value_counts().index
# temp['colors'] = 'yellow'

# plt.style.use('dark_background')
# plt.figure(figsize=(8,7))
# plt.hlines(y=temp.index, xmin=0, xmax=temp.counts, color=temp.colors, alpha=0.6, linewidth=14)
# plt.gca().set(ylabel='\n$Channel Number$\n', xlabel='\n$Instance Count$\n')
# plt.yticks(fontsize=10)
# plt.title('\nTraining Data:  Overall Open Channels Distribution\n(5M samples)\n\n', fontdict={'size':11})
# plt.grid(linestyle='--', alpha=0.2)
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))   
# #plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(10))   
# sns.despine(top=True, right=True, left=True, bottom=True)  
# # plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show();

# column_names = ["open_channels", "counts", "colors"]
# temp = pd.DataFrame(columns = column_names)
# temp['counts'] = ion.open_channels.value_counts().values
# temp['open_channels'] = ion.open_channels.value_counts().index
# temp['colors'] = 'yellow'

# plt.style.use('dark_background')
# plt.figure(figsize=(8,7))
# plt.hlines(y=temp.index, xmin=0, xmax=temp.counts, color=temp.colors, alpha=0.6, linewidth=14)
# plt.gca().set(ylabel='\n$Channel Number$\n', xlabel='\n$Instance Count$\n')
# plt.yticks(fontsize=10)
# plt.title('\nTraining Data:  Overall Open Channels Distribution\n(5M samples)\n\n', fontdict={'size':11})
# plt.grid(linestyle='--', alpha=0.2)
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))   
# #plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(10))   
# sns.despine(top=True, right=True, left=True, bottom=True)  
# # plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show();



# In[4]:


temp2 = pd.DataFrame(ion.open_channels.value_counts())
temp2.index.name = 'Channel ID'
temp2.columns=['InstanceCount']
# cm = sns.light_palette("green", as_cmap=True)
temp2.style.format("{:,.0f}")
# temp2.style.set_caption('Instace Count of Channel Instances:').background_gradient(cmap=cm)


# In[5]:



bars = alt.Chart(correct).mark_bar(color='orange',size=20).encode(
    alt.X('ChannelNumber:Q', axis=alt.Axis(grid=False, tickCount=10, title='Channel Number'),   ),   
    alt.Y("ChannelCount:Q", axis=alt.Axis(grid=True, title='Channel InstanceCount'))
)


text = bars.mark_text(
    align='center',
    baseline='middle',
    dx=0, dy=-10,
    color="darkgrey"
).encode(
    text='ChannelCount:Q'
)

(bars + text).properties(width=600,height=500).interactive(bind_y=False).configure_view(strokeWidth=0).configure(background='white')


# correct['color']='yellow'
# correct.columns = ['ChannelNumber', 'ChannelCount', 'Color']

# plt.hlines(y=correct.ChannelNumber, xmin=0, xmax=correct.ChannelCount, color=correct.Color, alpha=0.6, linewidth=14)
# plt.gca().set(ylabel='\n$Channel Number$\n', xlabel='\n$Instance Count$\n')
# plt.yticks(fontsize=10)
# plt.title('\nTraining Data:  Overall Open Channels Distribution\n(5M samples)\n\n', fontdict={'size':11})
# plt.grid(linestyle='--', alpha=0.2)
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))   
# #plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(10))   
# sns.despine(top=True, right=True, left=True, bottom=True)  
# # plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show();


# In[6]:



# Hits pretty hard and slows down, pause for now...
# temp3 = ion[:50000]

# plt.style.use('dark_background')
# fig, ax = plt.subplots(figsize=(7,15))

# # Create a color if the group is "B"
# # my_color= np.where( (df04.group == 'NE') | (df04.group == 'NO') | (df04.group == 'LA') , 'orange', 'skyblue')
# my_color = 'orange'
# # my_size=np.where(df04['group']=='B', 70, 30)
 
# plt.hlines(y=temp3.index, xmin=0, xmax=temp3.signal, color=my_color, alpha=0.4, linewidth=1)
# #plt.scatter(df04.Yards, my_range, color=my_color, s=my_size, alpha=1)
 
    
# # Add title and exis names
# # plt.yticks(my_range, df04.group)
# plt.title("\nSignal (voltage) over the course of time (samples) \n\n", loc='center', fontsize=10)
# plt.xlabel('\n Signal (Volts)', fontsize=10)
# plt.ylabel('')
# ##############plt.ylabel('NFL\nTeam\n')

# ax.spines['top'].set_linewidth(.3)  
# ax.spines['left'].set_linewidth(.3)  
# ax.spines['right'].set_linewidth(.3)  
# ax.spines['bottom'].set_linewidth(.3)  


# # plt.text(0, 33.3, r'Top Three:  LA Rams, New England Patriots, and New Orleans Saints absolutely dominating the rushing game...', {'color': 'white', 'fontsize': 8.5})
# sns.despine(top=True, right=True, left=True, bottom=True)
# plt.gca().invert_yaxis()
# plt.grid(linestyle='--', alpha=0.15)

# plt.tight_layout()
# plt.show();





# data = pd.DataFrame({
#     'x': pd.date_range('2012-01-01', freq='D', periods=365),
#     'y1': rand.randn(365).cumsum(),
#     'y2': rand.randn(365).cumsum(),
#     'y3': rand.randn(365).cumsum()
# })

# data = data.melt('x')
# data.head()


# chart = alt.Chart(data).mark_line().encode(
#     x='x:T',
#     y='value:Q',
#     color='variable:N'
# ).interactive(bind_y=False)


# chart


# chart = alt.Chart(data).mark_circle().encode(
#     x='x:T',
#     y='value:Q',
#     color='variable:N'
# ).interactive(bind_y=False)


# chart





temp3 = ion[:50000]
# my_color = 'orange' 
# plt.hlines(y=temp3.index, xmin=0, xmax=temp3.signal, color=my_color, alpha=0.4, linewidth=1)
# plt.title("\nSignal (voltage) over the course of time (samples) \n\n", loc='center', fontsize=10)
# plt.xlabel('\n Signal (Volts)', fontsize=10)
# plt.ylabel('')


chart = alt.Chart(temp3).mark_circle(size=2,color='maroon').encode(
    
    alt.X('time:T',  axis=alt.Axis(title='TimeStamp') ),
    
    alt.Y('signal:Q', axis=alt.Axis(gridColor='grey', gridWidth=.2, title='SignalStrength')  )).properties(title='Batch-1: Signal Value (Interactive)',width=750, height=275).interactive(bind_y=False)
    
    #olor='variable:N'


chart.configure_view(strokeWidth=0)


# In[7]:


# dft = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
# dft.to_csv("submission.csv",index=False)


# In[8]:


# RiffRaff:




# rand = np.random.RandomState(578493)
# data = pd.DataFrame({
#     'x': pd.date_range('2012-01-01', freq='D', periods=365),
#     'y1': rand.randn(365).cumsum(),
#     'y2': rand.randn(365).cumsum(),
#     'y3': rand.randn(365).cumsum()
# })

# data = data.melt('x')
# data.head()


# chart = alt.Chart(data).mark_line().encode(
#     x='x:T',
#     y='value:Q',
#     color='variable:N'
# ).interactive(bind_y=False)


# chart


# chart.mark_circle()


