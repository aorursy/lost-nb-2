#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import plotly.express as px
import plotly

from wordcloud import WordCloud
import datetime as dt
from sklearn import preprocessing
import librosa as lb
import librosa.display as lbd
import librosa.feature as lbf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline
sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (16,8)
import IPython.display as ipd
import ipywidgets as ipw
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings('ignore')


# In[3]:


link = 'https://ebird.org/species/'
audios = '../input/birdsong-recognition/train_audio/'


# In[4]:


train = pd.read_csv("/kaggle/input/birdsong-recognition/train.csv")
train.head()


# In[5]:


print(len(set(train["species"])))
print(len(set(train["ebird_code"])))


# In[6]:


train.describe()


# In[7]:


train.info()


# In[8]:


train['year'] = train['date'].apply(lambda x: x.split('-')[0])
train['month'] = train['date'].apply(lambda x: x.split('-')[1])
train['day_of_month'] = train['date'].apply(lambda x: x.split('-')[2])


# In[9]:


col = sorted(list(train['ebird_code'].unique()))

for temp in range(2,8):
    ## SCRAPING FOR BIRD DESCRIPTION and IMAGE URL
    URL = str(link+col[temp])
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    result = soup.find_all('p', class_='u-stack-sm')
    res = soup.find_all('figure', class_='MediaFeed-item'); 
    img = res[0].find_all('img')[0].get('src')
    description = result[0].text

    ## PUTTING EVERYTHING IN A VARIABLE
    ad = os.listdir(str(audios+col[temp]))[10]
    df = train[train['filename']==ad].reset_index()
    spec = str(df['species'][0]+' ('+df['sci_name'][0] + ')')
    loc = df['location'][0]
    time = str(df['date'][0]+' (yyyy-mm-dd) '+df['time'][0]+' hrs')
    recordist = df['recordist'][0]
    elev = df['elevation'][0]

    ## DISPLAYING IN THE NOTEBOOK
    ipd.display(ipd.HTML('<head> <body> <h1 style = "font-size:46px; font-family:sans; background-color:Lavender;"> {} </h1>                            <p style="text-align:left; color:black; font-family:verdana; font-size:19px;"> <br> <img src= {}                             style="float:right;width:50%;height:50%;"> <br> &emsp; {}<br><br><b> Located at:</b>{}<br><b>Date & Time                             of recording:</b> {}<br><b>Elevation: </b>{}<br><b>Recordist: </b>{}</br></p>                            <h3 style = "font-family:verdana; font-size:26px">Audio:</h3> </body></head>'.format(spec,img,description,loc,time,elev,recordist)))
    ipd.display(ipd.Audio(str(audios+col[temp]+'/'+ad), embed=True))


# In[10]:


temp = pd.DataFrame({'Number of Missing Values': pd.Series(train.isnull().sum().sort_values(ascending=False))[:5]})
temp['Feature'] = temp.index
temp = temp.reset_index(drop=True)
fig = px.bar(data_frame=temp,x="Feature",y="Number of Missing Values",color="Feature",orientation='v',title='Missing Values in the Train Data',hover_data=["Feature"])
fig.show()


# In[11]:


year = train["year"].value_counts()
year_df = pd.DataFrame({"year":year.index,"frequency":year.values})
year_df = year_df.sort_values(by="year",ascending=False)
fig = px.bar(data_frame=year_df[:30],x="year",y="frequency",color="year",title="On which Year Most Recordings Happen?")
fig.show()


# In[12]:


month = train["month"].value_counts()
month_df = pd.DataFrame({"month":month.index,"frequency":month.values})
month_df = month_df.sort_values(by="month",ascending=False)
fig = px.bar(data_frame=month_df[:30],x="month",y="frequency",color="month",title="On which Month Most Recordings Happen?")
fig.show()


# In[13]:


species = train["species"].value_counts()
species_df = pd.DataFrame({"species":species.index,"count":species.values})
fig = px.bar(data_frame=species_df,x="species",y="count",color="species",orientation='v',title='Count of Data Available for different bird species',hover_data=["species"])
fig.show()


# In[14]:


pitch = train["pitch"].value_counts()
pitch_df = pd.DataFrame({"pitch":pitch.index,"frequency":pitch.values})
pitch_df = pitch_df.sort_values(by="pitch",ascending=False)
fig = px.bar(data_frame=pitch_df,x="pitch",y="frequency",color="pitch",title="On which Pitch Most Bird sings?")
fig.show()


# In[15]:


seen = train["bird_seen"].value_counts()
seen_df = pd.DataFrame({"seen":seen.index,"frequency":seen.values})
seen_df = seen_df.sort_values(by="seen",ascending=False)
fig = px.bar(data_frame=seen_df[:30],x="seen",y="frequency",color="seen",title="Did the Author saw the bird?")
fig.show()


# In[16]:


group_seen = train.groupby(["bird_seen","species"]).size().reset_index()


# In[17]:


group_seen_no = group_seen[group_seen["bird_seen"]=="no"]
group_seen_yes = group_seen[group_seen["bird_seen"]=="yes"]


# In[18]:


def generate_word_cloud(text):
    wordcloud = WordCloud(
        width = 500,
        height = 1000,
        background_color = 'black').generate((" ").join(text))
    fig = plt.figure(
        figsize = (40, 30),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


# In[19]:


generate_word_cloud(list(group_seen_no.species.values))


# In[20]:


generate_word_cloud(list(group_seen_yes.species.values))


# In[21]:


adjusted_type = train['type'].apply(lambda x: x.split(',')).reset_index().explode("type")

# Strip of white spaces and convert to lower chars
adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()
adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')
top_15 = list(adjusted_type['type'].value_counts().reset_index()['index'])
data = adjusted_type[adjusted_type['type'].isin(top_15)]


# In[22]:


bird_call = data["type"].value_counts()
bird_call_df = pd.DataFrame({
    "type":bird_call.index,
    "frequency":bird_call.values
})
fig = px.bar(data_frame=bird_call_df[:15],x="type",y="frequency",color="type",hover_name="type",title="Which type of bird call did bird uses ?")
fig.show()


# In[23]:


elevation = train["elevation"].value_counts()
elevation_df = pd.DataFrame({"elevation":elevation.index,"frequency":elevation.values})
fig = px.bar(data_frame=elevation_df[:50],x="elevation",y="frequency",color="elevation",title="At Which Elevation did the bird found?")
fig.show()


# In[24]:


fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='world', title = 'Recordings from world'); 

plotly.offline.iplot(fig)


# In[25]:


fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='usa', title = 'Recordings from USA')

plotly.offline.iplot(fig)


# In[26]:


fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='europe', title = 'Recordings from EUROPE')

plotly.offline.iplot(fig)


# In[27]:


values_df= train.groupby(["species","author"]).size().reset_index()


# In[28]:


values_df = values_df.rename(columns={0: "Count"})
values_df = values_df.sort_values(by="Count",ascending=False)


# In[29]:


fig = px.bar(data_frame=values_df[:500],x="author",hover_name="species",y="Count",color="author")
fig.show()


# In[30]:


# Create Full Path so we can access data more easily
base_dir = '../input/birdsong-recognition/train_audio/'
train['full_path'] = base_dir + train['ebird_code'] + '/' + train['filename']

# Now let's sample a fiew audio files
amered = train[train['ebird_code'] == "amered"].sample(1, random_state = 33)['full_path'].values[0]


# In[31]:


ipd.Audio(amered)


# In[32]:


# Importing 1 file
y, sr = lb.load(amered)

print('y:', y, '\n')
print('y shape:', np.shape(y), '\n')
print('Sample Rate (KHz):', sr, '\n')

# Verify length of the audio
print('Check Len of Audio:', 661794/sr)


# In[33]:


audio_amered,sr_new = lb.effects.trim(y)


# In[34]:


lb.display.waveplot(y = audio_amered, sr = sr, color = "#A300F9")
plt.title("Sound Waves as 2D")
plt.ylabel("amered")


# In[35]:




# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

# Short-time Fourier transform (STFT)
D_amered = np.abs(lb.stft(audio_amered, n_fft = n_fft, hop_length = hop_length))


# In[36]:


print('Shape of D object:', np.shape(D_amered))


# In[37]:


DB_amered = lb.amplitude_to_db(D_amered, ref = np.max)


# In[38]:


lb.display.specshow(DB_amered, sr = sr, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool')


# In[39]:


# Create the Mel Spectrograms
S_amered = lb.feature.melspectrogram(y, sr=sr)
S_DB_amered = lb.amplitude_to_db(S_amered, ref=np.max)


# In[40]:


lb.display.specshow(S_DB_amered, sr = sr, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow')


# In[41]:


zero_amered = lb.zero_crossings(audio_amered, pad=False)
print("{}change rate is {}".format("armed",sum(zero_amered)))


# In[42]:




y_harm_haiwoo, y_perc_haiwoo = lb.effects.hpss(audio_amered)

plt.figure(figsize = (16, 6))
plt.plot(y_perc_haiwoo, color = '#FFB100')
plt.plot(y_harm_haiwoo, color = '#A300F9')
plt.legend(("Perceptrual", "Harmonics"))
plt.title("Harmonics and Perceptrual : Haiwoo Bird", fontsize=16);


# In[43]:


# Calculate the Spectral Centroids
spectral_centroids = lb.feature.spectral_centroid(audio_amered, sr=sr)[0]

# Shape is a vector
print('Centroids:', spectral_centroids, '\n')
print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = lb.frames_to_time(frames)

print('frames:', frames, '\n')
print('t:', t)

# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return preprocessing.minmax_scale(x, axis=axis)


# In[44]:


#Plotting the Spectral Centroid along the waveform
plt.figure(figsize = (16, 6))
lb.display.waveplot(audio_amered, sr=sr, alpha=0.4, color = '#A300F9', lw=3)
plt.plot(t, normalize(spectral_centroids), color='#FFB100', lw=2)
plt.legend(["Spectral Centroid", "Wave"])
plt.title("Spectral Centroid: Cangoo Bird", fontsize=16);


# In[45]:


# Increase or decrease hop_length to change how granular you want your data to be
hop_length = 5000

# Chromogram Vesspa
chromagram = lb.feature.chroma_stft(audio_amered, sr=sr, hop_length=hop_length)
print('Chromogram Vesspa shape:', chromagram.shape)

plt.figure(figsize=(16, 6))
lb.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.title("Chromogram Vesspa", fontsize=16);


# In[46]:


# Spectral RollOff Vector
spectral_rolloff = lb.feature.spectral_rolloff(audio_amered, sr=sr)[0]

# Computing the time variable for visualization
frames = range(len(spectral_rolloff))
# Converts frame counts to time (seconds)
t = lb.frames_to_time(frames)

# The plot
plt.figure(figsize = (16, 6))
lb.display.waveplot(audio_amered, sr=sr, alpha=0.4, color = '#A300F9', lw=3)
plt.plot(t, normalize(spectral_rolloff), color='#FFB100', lw=3)
plt.legend(["Spectral Rolloff", "Wave"])
plt.title("Spectral Rolloff: Amered Bird", fontsize=16)


# In[ ]:




