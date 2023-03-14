#!/usr/bin/env python
# coding: utf-8

# In[1]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR80B1-ealrPRrFhoJNri_GfULKGbbSIZHBkltJIqLE4afweR2X&usqp=CAU',width=400,height=400)


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


df = pd.read_csv('/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/metadata.csv')
df.head()


# In[4]:


title = df.copy()
title = title.dropna(subset=['title'])
title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)
title['title'] = title['title'].str.lower()


# In[5]:


title['keyword_biomarker'] = title['title'].str.find('biomarker')


# In[6]:


title.head()


# In[7]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRrwtPS-Yst8VfH_BazvJYgaRWfp_tctzl4pTCoMpPH6S-Dzj5s&usqp=CAU',width=400,height=400)


# In[8]:


included_biomarker = title.loc[title['keyword_biomarker'] != -1]
included_biomarker


# In[9]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs42256-020-0180-7/MediaObjects/42256_2020_180_Fig3_HTML.png?as=webp',width=400,height=400)


# In[10]:


import json
file_path = '/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/document_parses/pdf_json/b54932936d9dd6f8a399f23e19d0a1d0aeabd954.json'
with open(file_path) as json_file:
     json_file = json.load(json_file)
json_file


# In[11]:


import json
file_path = '/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/document_parses/pdf_json/0cea2b0d9b7187e2c5a596ed433be46208186d32.json'
with open(file_path) as json_file:
     json_file = json.load(json_file)
json_file


# In[12]:


biomarker = pd.read_csv('../input/cusersmarildownloadstcellcsv/TCell.csv', sep=';')
biomarker


# In[13]:


fig = go.Figure();
fig.add_trace(go.Scatter(x = biomarker['age_at_enrollment'].head(10),y = biomarker['study_id'],
                    mode='lines+markers',
                    name='study_id'));
fig.add_trace(go.Scatter(x = biomarker['age_at_enrollment'].head(10),y = biomarker['sex'],
                    mode='lines+markers',
                    name='sex'));
fig.add_trace(go.Scatter(x = biomarker['age_at_enrollment'].head(10),y = biomarker['dm'],
                    mode='lines+markers',
                    name='dm'));
fig.add_trace(go.Scatter(x = biomarker['age_at_enrollment'].head(10),y = biomarker['htn'],
                    mode='lines+markers',
                    name='htn'));
fig.add_trace(go.Scatter(x = biomarker['age_at_enrollment'].head(10),y = biomarker['anemia'],
                    mode='lines+markers',
                    name='anemia'));

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10);

fig.update_layout(autosize=False, width=1000,height=700, legend_orientation="h");

fig.show();


# In[14]:


get_ipython().system('pip install chart_studio')


# In[15]:


pip install bubbly


# In[16]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=biomarker, x_column='cd4', y_column='study_id', 
    bubble_column='age_at_enrollment',size_column='htn', color_column='age_at_enrollment', 
    x_title="CD4 Lymphocyte", y_title="Study ID", title='CD4 Lymphocyte Study ID',
     scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})


# In[17]:


ax = biomarker.plot(figsize=(15,8), title='CD4 Lymphocyte Study')
ax.set_xlabel('age_at_enrollment, sex, dm, htn')
ax.set_ylabel('study_id')


# In[18]:


biomarker.iloc[0]


# In[19]:


biomarker.plot.hist()


# In[20]:


biomarker.plot.scatter(x = 'study_id', y = 'cd4', c = 'htn', s = 190)


# In[21]:


#plt.style.use('dark_background')
from pandas.plotting import scatter_matrix
scatter_matrix(biomarker, figsize= (8,8), diagonal='kde', color = 'b')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# In[22]:


corr = biomarker.corr(method='pearson')
sns.heatmap(corr)


# In[23]:


biomarker_grp = biomarker.groupby(["study_id","age_at_enrollment"])[["dm","htn","pcp", "chronic_heart_disease", "ART_use", "tb", "cd4"]].sum().reset_index()
biomarker_grp.head()


# In[24]:


biomarker_grp.diff().hist(color = 'b', alpha = 0.1, figsize=(10,10))


# In[25]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'data:image/png;base64,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',width=400,height=400)

