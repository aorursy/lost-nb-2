#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nlpaug')


# In[2]:


# tool box

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point, Polygon

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import plotly.express as px

import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import sklearn
import librosa.display as librosa_display
import nlpaug
import nlpaug.augmenter.audio as naa

import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
import keras
from keras import layers
import random
from keras.models import Sequential
from tqdm import tqdm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils, to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# General Settings

# display all the columns in the dataset
pd.pandas.set_option('display.max_columns', None)

# Setting color palette.
purple_black = [
"#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"
]

# Setting plot styling.
#plt.style.use('ggplot')
plt.style.use('fivethirtyeight')


# In[4]:


base_path='../input/birdsong-recognition/'
audio_path=base_path+'train_audio/'


# In[5]:


# training dataset
train = pd.read_csv("/kaggle/input/birdsong-recognition/train.csv")
train.shape


# In[6]:


# lets inspect first few rows of the dataset
train.head()


# In[7]:


# check null values
train.isnull().sum().sort_values(ascending = False)[train.isnull().sum()!=0]


# In[8]:


# visualize missing values:
plt.figure(constrained_layout=True, figsize=(12, 8))
percent = (train.isnull().sum().sort_values(ascending=False) / len(train) *
           100)[(train.isnull().sum().sort_values(ascending=False) / len(train) *
                 100) != 0]

missing = pd.DataFrame({"missing%":percent})

sns.barplot(x=missing.index,
            y='missing%',
            data=missing,
            palette=purple_black)
plt.title('Train Data Missing Values')


# In[9]:


test = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")
test.shape


# In[10]:


test


# In[11]:


# no of unique classes(birds) in the dataset
print("dataset has",train.species.nunique(),"unique bird's species")


# In[12]:


# count wise distribution of bird's species
count = train.species.value_counts().sort_values(ascending = False)
count


# In[13]:


# lets visualize class distribution in the dataset
fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)
fig.show()


# In[14]:


# country
print("training dataset has data from",train.country.nunique(),"unique countries")


# In[15]:


# lets visualize top 10 countries
plt.figure(constrained_layout=True, figsize=(16, 8))
sns.countplot(train.country,
              alpha=0.9,              
              palette=purple_black,
              order = train.country.value_counts().sort_values(ascending=False).iloc[:10].index,)
plt.xlabel("Country")
plt.ylabel("Count")
plt.title("Country wise Distribution")
plt.show()


# In[16]:


# world shape file
world_map = gpd.read_file("../input/worldshapefile/world_shapefile.shp")

# Coordinate reference system
crs = {"init" : "epsg:4326"}


# In[17]:


# let's filter out "not specified" values
df = train[train["latitude"] != "Not specified"]

# convert latitude and longitute to float variables
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)


# In[18]:


# create geometric list
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

# create geography dataframe
geo = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# Create ID for species
species = geo["species"].value_counts().reset_index()
species.insert(0, 'ID', range(0, 0 + len(species)))

species.columns = ["ID", "species", "count"]

# merge the dataframes
geo = pd.merge(geo, species, how="left", on="species")


# In[19]:


# visualize bird's on the world map!
fig, ax = plt.subplots(figsize = (20, 9))
world_map.plot(ax=ax, alpha=0.4, color="blue")

palette = iter(sns.hls_palette(len(species)))

for i in range(264):
    geo[geo["ID"] == i].plot(ax=ax, markersize=30, color=next(palette), marker="o");
    
plt.title("These colorful small circles are our birds :-)")


# In[20]:


# check the date format
train.date.head()


# In[21]:


# lets pull year from the given date

train['year'] = train['date'].apply(lambda x: x.split('-')[0])

# lets visualize year wise distribution

fig = plt.figure(constrained_layout=True, figsize=(20,8))

sns.countplot(train.year,             
              alpha=0.9,              
              palette=purple_black,           
              order = train.year.value_counts().sort_values(ascending=False).iloc[:15].index   
             )
plt.xlabel("Year")
plt.ylabel("Count")
plt.title('Year-Wise Distribution')

plt.show()


# In[22]:


# lets pull month from the date
train['month'] = train['date'].apply(lambda x: x.split('-')[1])

# lets visualize month wise distribution

fig = plt.figure(constrained_layout=True, figsize=(20,8))

sns.countplot(train.month,             
              alpha=0.9,              
              palette=purple_black,           
              order = train.month.value_counts().sort_values(ascending=False).index   
             )
plt.xlabel("Month")
plt.ylabel("Count")
plt.title('Month-Wise Distribution')
plt.show()


# In[23]:


print("There are",train.ebird_code.nunique(),"ebird codes in the dataset")
print("training dataset has",train.sci_name.nunique(),"unique sci_names")


# In[24]:


# lets visualize ebird code & sci_name

fig = plt.figure(constrained_layout=True, figsize=(20,8))

grid = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

ax1 = fig.add_subplot(grid[0, :2])
sns.countplot(train.ebird_code,             
              alpha=0.9,
              ax=ax1,
              palette=purple_black,           
              order = train.ebird_code.value_counts().sort_values(ascending = False).iloc[:15].index   
             )
plt.xlabel("Ebird Code")
plt.ylabel("Count")
plt.title('Ebird Code Distribution')
plt.xticks(rotation=30)

ax2 = fig.add_subplot(grid[0, 2:4])
sns.countplot(train.sci_name,             
              alpha=0.9,
              ax=ax2,
              palette=purple_black,           
              order = train.sci_name.value_counts().sort_values(ascending = False).iloc[:15].index   
             )
plt.xlabel("Scientific Name")
plt.ylabel("Count")
plt.title('Scientific Name Distribution')
plt.xticks(rotation=30)
plt.show()


# In[25]:


# lets check the no. of unique values for the field filename
print("training dataset has",train.filename.nunique(),"unique filenames")
print("training dataset has",train.title.nunique(),"unique titles")
print("training dataset has",train.description.nunique(),"unique descriptions")
print("training dataset has",train.xc_id.nunique(),"unique xc_id")
print("training dataset has",train.url.nunique(),"unique urls")


# In[26]:


# lets check top 3 descriptions and see how it looks like
fig = plt.figure(constrained_layout=True, figsize=(20, 12))
sns.countplot(train.description,
              alpha=0.9,              
              palette=purple_black,
              order= train.description.value_counts().sort_values(ascending = False).iloc[:3].index)

plt.xlabel("Description")
plt.ylabel("Count")
plt.title('Description Distribution')

fig.show()


# In[27]:


# lets visualize top Playback used, channel & Ratings fields

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid:
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# playback used
ax1 = fig.add_subplot(grid[0, :2])

sns.countplot(train.playback_used,
              alpha=0.9,
              ax=ax1,
              order= train.playback_used.value_counts().sort_values(ascending = False).index,
              palette=purple_black)

plt.xlabel("Playback_Used")
plt.ylabel("Count")
ax1.set_title('PlayBack Used Distribution')

# channels.
ax2 = fig.add_subplot(grid[0, 2:])

# Plot the countplot.
sns.countplot(train.channels,
              alpha=0.9,
              ax=ax2,
              order= train.channels.value_counts().sort_values(ascending = False).index,
              palette=purple_black)

plt.xlabel("Channels")
plt.ylabel("Count")
ax2.set_title('Channels Distribution')

# Ratings
ax3 = fig.add_subplot(grid[1, :])

sns.countplot(train.rating,
              alpha=0.9,
              ax = ax3,
              palette=purple_black,              
              order= train.rating.value_counts().sort_values(ascending = False).index)

plt.xlabel("Ratings")
plt.ylabel("Count")
ax3.set_title('Ratings Distribution')

plt.show()


# In[28]:


# lets visualize pitch, speed & no. of notes

fig = plt.figure(constrained_layout=True, figsize=(20, 9))
# Creating a grid:
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# pitch
ax1 = fig.add_subplot(grid[0, :2])

sns.countplot(train.pitch,
              alpha=0.9,
              ax=ax1,
              palette=purple_black,
              order= train.pitch.value_counts().sort_values(ascending = False).index)
plt.xlabel("Pitch")
plt.ylabel("Count")
ax1.set_title('Pitch Distribution')



# speed
ax2 = fig.add_subplot(grid[0, 2:])

# Plot the countplot.
sns.countplot(train.speed,
              alpha=0.9,
              ax=ax2,
              palette=purple_black,
              order= train.speed.value_counts().sort_values(ascending = False).index)

plt.xlabel("Speed")
plt.ylabel("Count")
ax2.set_title('Speed Distribution')

# number_of_notes
ax3 = fig.add_subplot(grid[1, :])

sns.countplot(train.number_of_notes,
              alpha=0.9,
              ax=ax3,
              palette=purple_black,
              order= train.number_of_notes.value_counts().sort_values(ascending = False).index)
plt.xlabel("Number Of Notes Distribution")
plt.ylabel("Count")
ax3.set_title('Number Of Notes')

plt.show()


# In[29]:


plt.figure(constrained_layout=True, figsize=(12, 8))
sns.distplot(train.duration,
            color='coral')

plt.xlabel("Duration")
plt.ylabel("Count")
plt.title('Duration Distribution')


# In[30]:


# lets check the no. of unique values for the field primary & secondary labels
print("training dataset has",train.primary_label.nunique(),"unique primary labels")

print("training dataset has",train.secondary_labels.nunique(),"unique secondary labels")


# In[31]:


# lets visualize primary labels
plt.figure(constrained_layout=True, figsize=(12, 8))

count = train.primary_label.value_counts().sort_values(ascending = False)[:50]

fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)

fig.show()


# In[32]:


# lets visualize seconary labels
plt.figure(constrained_layout=True, figsize=(12, 8))

count = train.secondary_labels.value_counts().sort_values(ascending = False)[:20]

fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)

fig.show()


# In[33]:


# lets visualize bird_seen, sampling rate and type fields

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid:
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# playback used
ax1 = fig.add_subplot(grid[0, :2])

sns.countplot(train.bird_seen,
              alpha=0.9,
              ax=ax1,
              palette=purple_black,              
              order = train.bird_seen.value_counts().sort_values(ascending = False).index)
plt.xlabel("Bird Seen Distribution")
plt.ylabel("Count")
ax1.set_title('Bird Seen')


# sampling_rate.
ax2 = fig.add_subplot(grid[0, 2:])

# Plot the countplot.
sns.countplot(train.sampling_rate,
              alpha=0.9,
              ax=ax2,
              palette=purple_black,
              order = train.sampling_rate.value_counts().sort_values(ascending = False).index)

plt.xlabel("Sampling Rate Distribution")
plt.ylabel("Count")
ax2.set_title('Sampling Rate')

# type              
ax3 = fig.add_subplot(grid[1, :])

sns.countplot(train.type              ,
              alpha=0.9,
              ax = ax3,
              palette=purple_black,           
              order = train.type.value_counts().sort_values(ascending = False).iloc[:10].index)
            
plt.xlabel("Type Distribution")
plt.ylabel("Count")
plt.xticks(rotation = 30)
ax3.set_title('Type')

plt.show()


# In[34]:


# lets visualize elevation,volume,length

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid:
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)


# sampling_rate.
ax1 = fig.add_subplot(grid[0, :2])

# Plot the countplot.
sns.countplot(train.length,
              alpha=0.9,
              ax=ax1,
              palette=purple_black,
              order = train.length.value_counts().sort_values(ascending = False).index)

plt.xlabel("Length Distribution")
plt.ylabel("Count")
ax1.set_title('Length')

# volume              
ax2 = fig.add_subplot(grid[0, 2:])

sns.countplot(train.volume,
              alpha=0.9,
              ax = ax2,
              palette=purple_black,           
              order = train.volume.value_counts().sort_values(ascending = False).index,   
             )
plt.xlabel("Volume Distribution")
plt.ylabel("Count")
ax2.set_title('Volume')

# elevation              
ax3 = fig.add_subplot(grid[1, :])
sns.countplot(train.elevation,
              alpha=0.9,
              ax = ax3,
              palette=purple_black,           
              order = train.elevation.value_counts().sort_values(ascending = False).iloc[:15].index,   
             )
plt.xlabel("Elevation Distribution")
plt.ylabel("Count")
ax3.set_title('Elevation')

plt.show()


# In[35]:


# lets visualize file type, license, bitrate_of_mp3

fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid:
grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

# playback used
ax1 = fig.add_subplot(grid[0, :2])

sns.countplot(train.file_type,
              alpha=0.9,
              ax=ax1,
              palette=purple_black,
              order = train.file_type.value_counts().sort_values(ascending = False).index)
plt.xlabel("File Type")
plt.ylabel("Count")
ax1.set_title('File Type Distribution')


# sampling_rate.
ax2 = fig.add_subplot(grid[0, 2:])

# using short forms for liecense values
train['license'] = train['license'].replace(["Creative Commons Attribution-NonCommercial-ShareAlike 4.0"],["CCA-NCSA4.0"])
train['license'] = train['license'].replace(["Creative Commons Attribution-NonCommercial-ShareAlike 3.0"],["CCA-NCSA3.0"])
train['license'] = train['license'].replace(["Creative Commons Attribution-ShareAlike 3.0"],["CCA-SA3.0"])
train['license'] = train['license'].replace(["Creative Commons Attribution-ShareAlike 4.0"],["CCA-SA4.0"])
                                          

# Plot the countplot.
sns.countplot(train.license,
              alpha=0.9,
              ax=ax2,
              palette=purple_black,
              order = train.license.value_counts().sort_values(ascending = False).index)

plt.xlabel("License")
plt.ylabel("Count")
ax2.set_title('License Distribution')

# type              
ax3 = fig.add_subplot(grid[1, :])

sns.countplot(train.bitrate_of_mp3,              
              alpha=0.9,
              ax = ax3,
              palette=purple_black,           
              order = train.bitrate_of_mp3.value_counts().sort_values(ascending = False).iloc[:10].index)
plt.xlabel("Bitrate Of Mp3")
plt.ylabel("Count")
ax3.set_title('Bitrate Of Mp3  Distribution')

plt.show()


# In[36]:


# lets visualize background
plt.figure(constrained_layout=True, figsize=(12, 8))

count = train.background.value_counts().sort_values(ascending = False)[:20]

fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)

fig.show()


# In[37]:


# lets visualize author
plt.figure(constrained_layout=True, figsize=(12, 8))

count = train.author.value_counts().sort_values(ascending = False)[:20]

fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)

fig.show()


# In[38]:


# lets visualize recordist
plt.figure(constrained_layout=True, figsize=(12, 8))

count = train.recordist.value_counts().sort_values(ascending = False)[:20]

fig = px.pie(count,
             values=count.values,
             names=count.index,
             color_discrete_sequence=purple_black,
             hole=.4)
fig.update_traces(textinfo='percent', pull=0.05)

fig.show()


# In[39]:


print('Minimum samples per category = ', min(train.ebird_code.value_counts()))
print('Maximum samples per category = ', max(train.ebird_code.value_counts()))


# In[40]:


perfal = '/kaggle/input/birdsong-recognition/train_audio/perfal/XC463087.mp3'   # Hi-hat
ipd.Audio(perfal)


# In[41]:


lotduc = '/kaggle/input/birdsong-recognition/train_audio/lotduc/XC121426.mp3'   # Hi-hat
ipd.Audio(lotduc)


# In[42]:


rewbla = '/kaggle/input/birdsong-recognition/train_audio/rewbla/XC135672.mp3'   # Hi-hat
ipd.Audio(rewbla)


# In[43]:


warvir = '/kaggle/input/birdsong-recognition/train_audio/warvir/XC192521.mp3'   # Hi-hat
ipd.Audio(warvir)


# In[44]:


lecthr = '/kaggle/input/birdsong-recognition/train_audio/lecthr/XC141435.mp3'   # Hi-hat
ipd.Audio(lecthr)


# In[45]:


def audioinfo(filename, species):   
    # The load functions loads the audio file and converts it into an array of values which represent the amplitude if a sample at a 
    # given point of time.

    data,sample_rate1 = librosa.load(filename, res_type='kaiser_best')

    print("data:",data,"\n")
    print("Sample Rate (KHz):",sample_rate1)

    # lenth of the audio
    print('Audio Length:', np.shape(data)[0]/sample_rate1)
    
    # ----------------------------------------------------------WAVE PLOT-----------------------------------------------------------
    plt.figure(figsize=(30,20))
    plt.subplot(3,1,1)
    
    # Amplitude and frequency are important parameters of the sound and are unique for each audio. 

    # librosa.display.waveplot is used to plot waveform of amplitude vs time where the first axis is an amplitude and second axis is time
   
    librosa.display.waveplot(data,sr=sample_rate1,color = 'darkblue')
    plt.xlabel("Time (seconds) -->")
    plt.ylabel("Amplitude")
    plt.title("Waveplot for - " + species)
    
    # --------------------------------------------------------SPECTOGRAM------------------------------------------------------------
    plt.subplot(3,1,2)
     # .stft converts data into short term Fourier transform. STFT converts signal such that we can know the amplitude of given 
     # frequency at a given time. Using STFT we can determine the amplitude of various frequencies playing at a given time of an audio
     # signal. 
    X = librosa.stft(data)

    Xdb = librosa.amplitude_to_db(abs(X))

    #.specshow is used to display spectogram.
    librosa.display.specshow(Xdb, sr=sample_rate1, x_axis='time', y_axis='hz',cmap = 'winter') 

    plt.colorbar()
    plt.xlabel("Time (seconds) -->")
    plt.ylabel("Amplitude")
    plt.title("Spectogram for - " + species)
    
    # ----------------------------------------------------MEL SPECTOGRAM----------------------------------------------------------
    plt.subplot(3,1,3)
    librosa.feature.melspectrogram(y=data, sr=sample_rate1)

    D = np.abs(librosa.stft(data))**2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate1)

    librosa.display.specshow(librosa.power_to_db(S,ref=np.max),x_axis='time',cmap = 'rainbow')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel spectrogram for species - " + species)
    plt.xlabel("Time (seconds) -->")
    plt.ylabel("Amplitude")   
     
    plt.show()


# In[46]:


audioinfo('/kaggle/input/birdsong-recognition/train_audio/perfal/XC463087.mp3',"perfal")


# In[47]:


audioinfo('/kaggle/input/birdsong-recognition/train_audio/lotduc/XC121426.mp3',"lotduc")


# In[48]:


audioinfo("/kaggle/input/birdsong-recognition/train_audio/rewbla/XC135672.mp3","rewbla")


# In[49]:


audioinfo( '/kaggle/input/birdsong-recognition/train_audio/warvir/XC192521.mp3',"warvir")


# In[50]:


audioinfo("/kaggle/input/birdsong-recognition/train_audio/lecthr/XC141435.mp3","lecthr")


# In[51]:


def zero_cross(filename):
    data,sample_rate1 = librosa.load(filename)
    # Zooming in
    n0 = 9000
    n1 = 9100
    plt.figure(figsize=(20, 5))
    plt.plot(data[n0:n1],color = "gold")
    plt.grid()
    
    zero_crossings = librosa.zero_crossings(data, pad=False)
    print("Zero Crossing Shape:",zero_crossings.shape)
    
    print("Total Zero Crossings:",sum(zero_crossings))


# In[52]:


zero_cross(perfal)


# In[53]:


zero_cross(lotduc)


# In[54]:


zero_cross(rewbla)


# In[55]:


zero_cross(warvir)


# In[56]:


zero_cross(lecthr)


# In[57]:


def spectral_centroid(filename):
    data,sample_rate1 = librosa.load(filename)
    
    spectral_centroids = librosa.feature.spectral_centroid(data, sr=sample_rate1)[0]
    spectral_centroids.shape

    # Computing the time variable for visualization
    plt.figure(figsize=(20,5))
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    # Normalising the spectral centroid for visualisation
    def normalize(data, axis=0):
        return sklearn.preprocessing.minmax_scale(data, axis=axis)

    #Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(data, sr=sample_rate1, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')


# In[58]:


spectral_centroid(perfal)


# In[59]:


spectral_centroid(lotduc)


# In[60]:


spectral_centroid(rewbla)


# In[61]:


spectral_centroid(warvir)


# In[62]:


spectral_centroid(lecthr)


# In[63]:


def rolloff(filename):
    data,sample_rate1 = librosa.load(filename)
    
    spectral_centroids = librosa.feature.spectral_centroid(data, sr=sample_rate1)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    
    def normalize(data, axis=0):
        return sklearn.preprocessing.minmax_scale(data, axis=axis)

    plt.figure(figsize=(20,5))
    spectral_rolloff = librosa.feature.spectral_rolloff(data+0.01, sr=sample_rate1)[0]
    librosa.display.waveplot(data, sr=sample_rate1, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color='g')
    plt.grid()


# In[64]:


rolloff(perfal)


# In[65]:


rolloff(lotduc)


# In[66]:


rolloff(rewbla)


# In[67]:


rolloff(warvir)


# In[68]:


rolloff(lecthr)


# In[69]:


# MFCC
def mfcc(filename):
    data,sample_rate1 = librosa.load(filename)
    plt.figure(figsize=(20,5))
    mfccs = librosa.feature.mfcc(data, sr=sample_rate1)
    print(mfccs.shape)

    librosa.display.specshow(mfccs, sr=sample_rate1, x_axis='time')


# In[70]:


mfcc(perfal)


# In[71]:


mfcc(lotduc)


# In[72]:


mfcc(rewbla)


# In[73]:


mfcc(warvir)


# In[74]:


mfcc(lecthr)


# In[75]:


def chrom_freq(filename):
    data,sample_rate1 = librosa.load(filename)
    
    hop_length = 512
    chromagram = librosa.feature.chroma_cqt(data, sr=sample_rate1, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length)


# In[76]:


chrom_freq(perfal)


# In[77]:


chrom_freq(lotduc)


# In[78]:


chrom_freq(rewbla)


# In[79]:


chrom_freq(warvir)


# In[80]:


chrom_freq(lecthr)


# In[81]:


def fundamental_frequency(filename):
    y, sr = librosa.load(filename)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')


# In[82]:


fundamental_frequency(perfal)


# In[83]:


fundamental_frequency(warvir)


# In[84]:


fundamental_frequency(lotduc)


# In[85]:


fundamental_frequency(lecthr)


# In[86]:


fundamental_frequency(rewbla)


# In[87]:


def compute_tempogram(filename):
    # computing local onset autocorrelation
    y,sr = librosa.load(filename)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
    
    # Computing global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    
    # Estimating global tempo
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                           hop_length=hop_length)[0]
    
    # plotting
    
    fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    ax[0].plot(times, oenv, label='Onset strength')
    ax[0].label_outer()
    ax[0].legend(frameon=True)
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[1])
    ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                label='Estimated tempo={:g}'.format(tempo))
    ax[1].legend(loc='upper right')
    ax[1].set(title='Tempogram')
    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                    num=tempogram.shape[0])
    ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    ax[2].set(xlabel='Lag (seconds)')
    ax[2].legend(frameon=True)
    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                 label='Mean local autocorrelation', basex=2)
    ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                 label='Global autocorrelation', basex=2)
    ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
                label='Estimated tempo={:g}'.format(tempo))
    ax[3].legend(frameon=True)
    ax[3].set(xlabel='BPM')
    ax[3].grid(True)


# In[88]:


compute_tempogram(lecthr)


# In[89]:


compute_tempogram(warvir)


# In[90]:


compute_tempogram(perfal)


# In[91]:


compute_tempogram(lotduc)


# In[92]:


def decompose_audio(filename):
    y,sr = librosa.load(filename)
    D = librosa.stft(y)
    #y_harmonic, y_percussive = librosa.effects.hpss(D, margin=(1.0,5.0)) # we will get more isolated percussive component by increasing margin 
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    # Pre-compute a global reference power from the input spectrum
    rp = np.max(np.abs(D))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
    plt.colorbar()
    plt.title('Full spectrogram')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
    plt.colorbar()
    plt.title('Harmonic spectrogram')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Percussive spectrogram')
    plt.tight_layout()


# In[93]:


decompose_audio(perfal)


# In[94]:


decompose_audio(lotduc)


# In[95]:


decompose_audio(warvir)


# In[96]:


decompose_audio(lecthr)


# In[97]:


def pitch_speed(filename):
    data, sr = librosa.load(filename)
    pitch_speed = data.copy()
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.0  / length_change
    print("resample length_change = ",length_change)
    tmp = np.interp(np.arange(0,len(pitch_speed),speed_fac),np.arange(0,len(pitch_speed)),pitch_speed)
    minlen = min(pitch_speed.shape[0], tmp.shape[0])
    pitch_speed *= 0
    pitch_speed[0:minlen] = tmp[0:minlen]
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(pitch_speed, sr=sr, color='r', alpha=0.25)
    plt.title('augmented pitch and speed')
    return ipd.Audio(data, rate=sr)


# In[98]:


pitch_speed(perfal)


# In[99]:


pitch_speed(lotduc)


# In[100]:


pitch_speed(rewbla)


# In[101]:


pitch_speed(warvir)


# In[102]:


pitch_speed(lecthr)


# In[103]:


def pitch(filename):
    data, sr = librosa.load(filename)
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    print("pitch_change = ",pitch_change)
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), 
                                          sr, n_steps=pitch_change, 
                                          bins_per_octave=bins_per_octave)
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(y_pitch, sr=sr, color='r', alpha=0.25)
    plt.title('augmented pitch only')
    plt.tight_layout()
    plt.show()
    return ipd.Audio(data, rate=sr)


# In[104]:


pitch(perfal)


# In[105]:


pitch(lotduc)


# In[106]:


pitch(rewbla)


# In[107]:


pitch(warvir)


# In[108]:


pitch(lecthr)


# In[109]:


def speed(filename):
    data, sr = librosa.load(filename)
    aug = naa.SpeedAug()
    augmented_data = aug.augment(data)

    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)
    plt.title('augmented speed only')
    plt.tight_layout()
    plt.show()
    return ipd.Audio(augmented_data, rate=sr)


# In[110]:


speed(perfal)


# In[111]:


speed(lotduc)


# In[112]:


speed(rewbla)


# In[113]:


speed(warvir)


# In[114]:


speed(lecthr)


# In[115]:


def augmentation(filename):
    data, sr = librosa.load(filename)
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5,high=3)
    print("dyn_change = ",dyn_change)
    y_aug = y_aug * dyn_change
    print(y_aug[:50])
    print(data[:50])
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(y_aug, sr=sr, color='r', alpha=0.25)
    plt.title('amplify value')
    return ipd.Audio(y_aug, rate=sr)


# In[116]:


augmentation(perfal)


# In[117]:


augmentation(lotduc)


# In[118]:


augmentation(rewbla)


# In[119]:


augmentation(warvir)


# In[120]:


augmentation(lecthr)


# In[121]:


def add_noise(filename):
    data, sr = librosa.load(filename)
    y_noise = data.copy()
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(y_noise, sr=sr, color='r', alpha=0.25)
    return ipd.Audio(y_noise, rate=sr)


# In[122]:


add_noise(perfal)


# In[123]:


add_noise(lotduc)


# In[124]:


add_noise(rewbla)


# In[125]:


add_noise(warvir)


# In[126]:


add_noise(lecthr)


# In[127]:


def random_shift(filename):
    data, sr = librosa.load(filename)
    y_shift = data.copy()
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    print("timeshift_fac = ",timeshift_fac)
    start = int(y_shift.shape[0] * timeshift_fac)
    print(start)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(y_shift, sr=sr, color='r', alpha=0.25)
    return ipd.Audio(y_shift, rate=sr)


# In[128]:


random_shift(perfal)


# In[129]:


random_shift(lotduc)


# In[130]:


random_shift(rewbla)


# In[131]:


random_shift(warvir)


# In[132]:


random_shift(lecthr)


# In[133]:


def hpss(filename):
    data, sr = librosa.load(filename)
    y_hpss = librosa.effects.hpss(data.astype('float64'))
    print(y_hpss[1][:10])
    print(data[:10])
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(y_hpss[1], sr=sr, color='r', alpha=0.25)
    plt.title('apply hpss')
    return ipd.Audio(y_hpss[1], rate=sr)


# In[134]:


hpss(perfal)


# In[135]:


hpss(lotduc)


# In[136]:


hpss(rewbla)


# In[137]:


hpss(warvir)


# In[138]:


hpss(lecthr)


# In[139]:


def streching(filename):
    data, sr = librosa.load(filename)
    input_length = len(data)
    streching = data.copy()
    streching = librosa.effects.time_stretch(streching.astype('float'), 1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(streching, sr=sr, color='r', alpha=0.25)
    
    plt.title('stretching')
    return ipd.Audio(streching, rate=sr)


# In[140]:


streching(perfal)


# In[141]:


streching(lotduc)


# In[142]:


streching(rewbla)


# In[143]:


streching(warvir)


# In[144]:


streching(lecthr)


# In[145]:


def crop(filename):
    data, sr = librosa.load(filename)
    aug = naa.CropAug(sampling_rate=sr)
    augmented_data = aug.augment(data)

    librosa_display.waveplot(augmented_data, sr=sr, alpha=0.5)
    librosa_display.waveplot(data, sr=sr, color='r', alpha=0.25)

    plt.tight_layout()
    plt.show()

    return ipd.Audio(augmented_data, rate=sr)


# In[146]:


crop(perfal) 


# In[147]:


crop(lotduc)


# In[148]:


crop(rewbla)


# In[149]:


crop(warvir)


# In[150]:


crop(lecthr)


# In[151]:


def loudnessaug(filename):
    data, sr = librosa.load(filename)
    aug = naa.LoudnessAug(loudness_factor=(2, 5))
    augmented_data = aug.augment(data)

    librosa_display.waveplot(augmented_data, sr=sr, alpha=0.25)
    librosa_display.waveplot(data, sr=sr, color='r', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return ipd.Audio(augmented_data,rate=sr)


# In[152]:


loudnessaug(perfal) 


# In[153]:


loudnessaug(lotduc)


# In[154]:


loudnessaug(rewbla)


# In[155]:


loudnessaug(warvir)


# In[156]:


loudnessaug(lecthr)


# In[157]:


def mask(filename):
    data, sr = librosa.load(filename)
    aug = naa.MaskAug(sampling_rate=sr, mask_with_noise=False)
    augmented_data = aug.augment(data)

    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

    plt.tight_layout()
    plt.show()
    
    return ipd.Audio(augmented_data, rate=sr)


# In[158]:


mask(perfal)


# In[159]:


mask(lotduc)


# In[160]:


mask(rewbla)


# In[161]:


mask(warvir)


# In[162]:


mask(lecthr)


# In[163]:


def shift(filename):
    data, sr = librosa.load(filename)
    aug = naa.ShiftAug(sampling_rate=sr)
    augmented_data = aug.augment(data)

    librosa_display.waveplot(data, sr=sr, alpha=0.5)
    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

    plt.tight_layout()
    plt.show()
    
    return ipd.Audio(augmented_data, rate=sr)


# In[164]:


shift(perfal)


# In[165]:


shift(lotduc)


# In[166]:


shift(rewbla)


# In[167]:


shift(warvir)


# In[168]:


shift(lecthr)


# In[169]:


train_set= train.copy()
birds_key=train["ebird_code"].unique()
birds_key


# In[170]:


random.shuffle(birds_key)
train_set = train_set.query("ebird_code in @birds_key")

idBirdDict = {}
ebirdDict = {}
ebirdDict["nocall"] = 0
idBirdDict[0] = "nocall"
for idx, unique_ebird_code in enumerate(train_set.ebird_code.unique()):
    ebirdDict[unique_ebird_code] = str(idx+1)
    idBirdDict[idx+1] = str(unique_ebird_code)


# In[171]:


ebirdDict


# In[172]:


idBirdDict


# In[173]:


#Let create a Sample Set as Whote data set will run for long hours
sample_set=pd.DataFrame(columns=['ebird_code','audio_File_path',"song_sample","bird"])


# In[174]:


#Using Francois's code to extract the data/ run model

def get_sample(filename, bird, sample_set):
    min_max_Scaler=MinMaxScaler()
    wave_data, wave_rate = librosa.load(filename)
    data_point_per_second = 10
    
    #Take 10 data points every second
    prepared_sample = wave_data[0::int(wave_rate/data_point_per_second)]
    #We normalize each sample before extracting 5s samples from it
    normalized_sample = min_max_Scaler.fit_transform(prepared_sample.reshape(-1, 1))
    normalized_sample = normalized_sample.flatten()
    
    #only take 5s samples and add them to the dataframe
    song_sample = []
    sample_length = 5*data_point_per_second
    for idx in range(0,len(normalized_sample),sample_length): 
        song_sample = normalized_sample[idx:idx+sample_length]
        if len(song_sample)>=sample_length:
            sample_set = sample_set.append({"song_sample":np.asarray(song_sample).astype(np.float32),
                                            "bird":ebirdDict[bird],
                                           "audio_File_path":filename,
                                           "ebird_code":bird}, 
                                           ignore_index=True)
                     
    return sample_set


# In[175]:


# we will run for 5000 records for total Trains set to prepare for Model 
with tqdm(total=5000) as pbar:
    for idx, row in train_set[:5000].iterrows():
        pbar.update(1)
        #print(idx)
        sample_set = get_sample(row.audio_File_path, row.ebird_code, sample_set)


# In[176]:


#Now out of the complete sequence length we will choose with the fixed 50 sequence length for the above input array on Sample Set
# also divide the sample set into train and val set on the basis of 80:20
sequence_length = 50
split_per = 0.80
train_item_count = int(len(sample_set)*split_per)
val_item_count = len(sample_set)-int(len(sample_set)*split_per)
training_set = sample_set[:train_item_count]
validation_set = sample_set[train_item_count:]


# In[177]:


# we will have Sequential LSTM with dropout and 3 layer as SOftMax and Optimizer is ADAM
model = Sequential()
model.add(LSTM(32, return_sequences=True, recurrent_dropout=0.2,input_shape=(None, sequence_length)))
model.add(LSTM(32,recurrent_dropout=0.2))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(len(ebirdDict.keys()), activation="softmax"))

model.summary()

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss="categorical_crossentropy", optimizer='adam')


# In[178]:


# Take the Xtrain and Y train from train Set from Sample Set data frame to be feed into LSTM Model
X_train = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in training_set["song_sample"]]),(train_item_count,1,sequence_length))).astype(np.float32)
train_gd = np.asarray([np.asarray(x) for x in training_set["bird"]]).astype(np.float32)
Y_train = to_categorical(
                train_gd, num_classes=len(ebirdDict.keys()), dtype='float32'
            )


X_val = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in validation_set["song_sample"]]),(val_item_count,1,sequence_length))).astype(np.float32)
val_gd = np.asarray([np.asarray(x) for x in validation_set["bird"]]).astype(np.float32)
Y_val = to_categorical(
                val_gd, num_classes=len(ebirdDict.keys()), dtype='float32'
            )


# In[179]:


# Fit the LSTM model and plot the Train and validation Loss for 100 Epochs and batch Size of 32
model_his1 = model.fit(X_train, Y_train, 
          epochs = 100, 
          batch_size = 32, 
          validation_data=(X_val, Y_val), 
          callbacks=callbacks)

plt.plot(model_his1.history['loss'])
plt.plot(model_his1.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# In[180]:


# make the predictions function to predict on Unsenn data from the Model trained
model.load_weights("best_model.h5")

def make_prediction(df, audio_file_path):
        
    loaded_audio_sample = []
    previous_filename = ""
    data_point_per_second = 10
    sample_length = 5*data_point_per_second
    wave_data = []
    wave_rate = None
    
    for idx,row in df.iterrows():
        if previous_filename == "" or previous_filename!=row.filename:
            filename = '{}/{}.mp3'.format(audio_file_path, row.filename)
            wave_data, wave_rate = librosa.load(filename)
            sample = wave_data[0::int(wave_rate/data_point_per_second)]
        previous_filename = row.filename
        
        #basically allows to check if we are running the examples or the test set.
        if "site" in df.columns:
            if row.site=="site_1" or row.site=="site_2":
                song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])
            elif row.site=="site_3":
                #for now, I only take the first 5s of the samples from site_3 as they are groundtruthed at file level
                song_sample = np.array(sample[0:sample_length])
        else:
            #same as the first condition but I isolated it for later and it is for the example file
            song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])

        input_data = np.reshape(np.asarray([song_sample]),(1,sequence_length)).astype(np.float32)
        prediction = model.predict(np.array([input_data]))
        predicted_bird = idBirdDict[np.argmax(prediction)]

        df.at[idx,"birds"] = predicted_bird
    return df


# In[181]:


#Let see how our model performs on example set given
example_set = pd.read_csv(base_path+"example_test_audio_summary.csv")
example_set["filename"] = [ "BLKFR-10-CPL_20190611_093000.pt540" if filename=="BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for filename in example_set["filename"]]
example_set


# In[182]:


example_audio_file_path = base_path +"example_test_audio"
if os.path.exists(example_audio_file_path):
    example_set = make_prediction(example_set, example_audio_file_path)
example_set


# In[183]:


# Now lets predict on the test Set and prepare the Submission File
test_audio_file_path = base_path+"test_audio/"
submission_set = pd.read_csv(base_path+"sample_submission.csv")
submission_set.head()


# In[184]:


if os.path.exists(test_audio_file_path):
    submission_set = make_prediction(test, test_audio_file_path)


# In[185]:


submission_set[:20]


# In[186]:


submission_set.to_csv("submission.csv", index=False)


# In[187]:


submission_set

