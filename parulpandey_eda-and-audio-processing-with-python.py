#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install chart_studio')


# In[2]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
get_ipython().run_line_magic('matplotlib', 'inline')

# Preprocessing
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import datetime as dt
from datetime import datetime   

# Visualisation libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly.figure_factory as ff
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


# In[3]:


train = pd.read_csv('../input/birdsong-recognition/train.csv',)
train.head()


# In[4]:


train.info()


# In[5]:


len(train['ebird_code'].value_counts())


# In[6]:


x = train['ebird_code'].value_counts().index.to_list()
e_code_path = 'https://ebird.org/species/'
species = [e_code_path+p for p in x]


# In[7]:


from IPython.display import IFrame
IFrame(species[0], width=800, height=450)


# In[8]:


IFrame(species[100], width=800, height=450)


# In[9]:


IFrame(species[200], width=800, height=450)


# In[10]:


# Total number of people who provided the recordings
train['recordist'].nunique()


# In[11]:


# Top 10 recordists in terms of the number of recordings done
train['recordist'].value_counts()[:10].sort_values().iplot(kind='barh',color='#3780BF')


# In[12]:


train['playback_used'].fillna('Not Defined',inplace=True);
train['playback_used'].value_counts()


# In[13]:


train['playback_used'].value_counts()

labels = train['playback_used'].value_counts().index
values = train['playback_used'].value_counts().values
colors=['#3795bf','#bfbfbf']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',marker=dict(colors=colors))])
fig.show()


# In[14]:


train['rating'].value_counts().iplot(kind='bar',color='#3780BF')


# In[15]:


# Convert string to datetime64
train['date'] = train['date'].apply(pd.to_datetime,format='%Y-%m-%d', errors='coerce')
#train.set_index('date',inplace=True)
train['date'].value_counts().plot(figsize=(12,8))


# In[16]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1592397692077' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bi&#47;Birds_15923974075490&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Birds_15923974075490&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bi&#47;Birds_15923974075490&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1592397692077');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[17]:


# Total no of unique species in the dataset
print(len(train['species'].value_counts().index))


# In[18]:


train['species'].value_counts()


# In[19]:


train['species'].value_counts().iplot()


# In[20]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1592442148007' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ZN&#47;ZNDRZCHNN&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;ZNDRZCHNN' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ZN&#47;ZNDRZCHNN&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1592442148007');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[21]:


TRAIN_EXT_PATH = "../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv"
train_ext = pd.read_csv(TRAIN_EXT_PATH)
train_ext.head()


# In[22]:


len(train_ext['ebird_code'].value_counts())


# In[23]:


len(train_ext)


# In[24]:


df_original = train.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "original_recordings"})
df_extended = train_ext.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "extended_recordings"})

df = df_original.merge(df_extended, on = "species", how = "left").fillna(0)
df["total_recordings"] = df.original_recordings + df.extended_recordings
df = df.sort_values("total_recordings").reset_index().sort_values('total_recordings',ascending=False)
df.head()


# In[45]:


# Plot the total recordings
f, ax = plt.subplots(figsize=(10, 50))

sns.set_color_codes("pastel")
sns.barplot(x="total_recordings", y="species", data=df,
            label="total_recordings", color="r")

# Plot the original recordings
sns.set_color_codes("muted")
sns.barplot(x="original_recordings", y="species", data=df,
            label="original_recordings", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 2000), ylabel="",
       xlabel="Count")
sns.despine(left=True, bottom=True)


# In[26]:



audio_path = '../input/birdsong-recognition/train_audio/nutwoo/XC462016.mp3'
x , sr = librosa.load(audio_path)


# In[27]:


print(type(x), type(sr))


# In[28]:


print(x.shape, sr)


# In[29]:


librosa.load(audio_path, sr=44100)


# In[30]:


librosa.load(audio_path, sr=None)


# In[31]:


import IPython.display as ipd
ipd.Audio(audio_path)


# In[32]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[33]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# In[34]:


librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# In[35]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[36]:


# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()


# In[37]:


zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))


# In[38]:


spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape


# In[39]:


# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')


# In[40]:


spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')


# In[41]:


x, fs = librosa.load('../input/birdsong-recognition/train_audio/nutwoo/XC161356.mp3')
librosa.display.waveplot(x, sr=sr)


# In[42]:


mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)


# In[43]:


#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# In[44]:


mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

