#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np 
import pandas as pd
import datetime as dt
from sklearn import preprocessing as prep
import librosa as lb
import librosa.display as lbd
import librosa.feature as lbf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline
sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (16,8)
import IPython.display as ipd
import ipywidgets as ipw
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings('ignore')

link = 'https://ebird.org/species/'
audios = '../input/birdsong-recognition/train_audio/'


# In[2]:


train = pd.read_csv('../input/birdsong-recognition/train.csv')


# In[3]:


train


# In[4]:


train.info()


# In[5]:


temp = pd.DataFrame({'Number of Missing Values': pd.Series(train.isnull().sum().sort_values(ascending=False)[:5])})
temp['Feature'] = temp.index; temp = temp.reset_index(drop=True)


plotly.offline.iplot(px.bar(temp, x = 'Number of Missing Values', y = 'Feature', orientation='h', title = 'Missing Values in training data', 
                            color = 'Feature', height=400, text = 'Number of Missing Values'))


# In[6]:


temp = train.groupby(['ebird_code', 'species']).count().reset_index().sort_values(by='filename',ascending=True)

plotly.offline.iplot(px.bar(temp, x='filename', y='species', orientation='h', labels={'filename': 'Count', 'species':'Bird Name'}, hover_data=['species'],
       height=1500, width=800, color = 'filename', title='Count of Data Available for different bird species'))


# In[7]:


col = sorted(list(train['ebird_code'].unique()))

for temp in range(2,8):
    ## SCRAPING FOR BIRD DESCRIPTION and IMAGE URL
    URL = str(link+col[temp]); page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser'); result = soup.find_all('p', class_='u-stack-sm')
    res = soup.find_all('figure', class_='MediaFeed-item'); 
    img = res[0].find_all('img')[0].get('src'); description = result[0].text

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


# In[8]:


## WILL GIVE UP A WARNING TO USE 'audioread' INSTEAD. SUPPRESS IT BY 'warnings.filterwarnings'.
## GIVES WARNING AS LIBROSA IS NOT CAPABLE OF READING '.mp3' FILES AND FALLS BACK TO 'audioread' 
## WHEN GIVEN ONE. TAKES A BIT TIME TO LOAD.
ad = 'XC233159.mp3'
x, sr = lb.load(str(audios+col[temp]+'/'+ad))


# In[9]:


lbd.waveplot(x,sr=sr); 
plt.title('Waveform of the audio recording:', size=20); plt.ylabel('Amplitude'); plt.show()


# In[10]:


X = lb.stft(x)

lbd.specshow(lb.amplitude_to_db(abs(X)),sr=sr,x_axis='time', y_axis='hz')
plt.xlabel('Time (in min:sec)'); plt.title('Short-time Fourier transform (Spectrogram):', size=20); plt.show()


# In[11]:


sc = lbf.spectral_centroid(x, sr=sr)[0]; frames = range(len(sc)); t = lb.frames_to_time(frames)

lbd.waveplot(x, sr=sr)
plt.title('Spectral Centroid visualized', size=25)
## NORMALIZING THE VALUES OF SPECTRAL CENTROID BEFORE PLOTTING TO COMPARE WITH WAVEFORM
plt.plot(t, prep.minmax_scale(sc), linewidth=0.8, color='r'); plt.show()


# In[12]:


X = lb.stft(x)

plt.figure(figsize=(20,8)); lbd.specshow(lb.amplitude_to_db(abs(X)),sr=sr,x_axis='time', y_axis='log')
plt.xlabel('Time (in min:sec)'); plt.title('Spectrogram with spectral centroids:', size=25); plt.plot(t, sc, linewidth=0.8, color='black'); plt.colorbar(); plt.show()


# In[13]:


spectral_rolloff = lbf.spectral_rolloff(x+0.01, sr=sr)[0]

lbd.waveplot(x, sr=sr)
plt.plot(t, prep.minmax_scale(spectral_rolloff), color='r', label='Spectral Roll-off'); plt.title('Spectral Roll-off visualized', size=25); plt.legend(); plt.show()


# In[14]:


spectral_bandwidth = lbf.spectral_bandwidth(x+0.01, sr=sr, p=2)[0]

lbd.waveplot(x,sr)
plt.plot(t, prep.minmax_scale(spectral_bandwidth), color='r', label='Spectral Bandwidth'); plt.title('Spectral Bandwidth visualized', size=25); plt.legend(); plt.show()


# In[15]:


spectral_flux = lb.onset.onset_strength(x, sr); onset_default = lb.onset.onset_detect(x, sr, units='time')

plt.plot(t, spectral_flux, label='Spectral flux'); plt.vlines(onset_default, 0, spectral_flux.max(), color='r', label='Onsets');plt.title('Spectral Flux Visualized', size=25)
plt.legend(); plt.show()


# In[16]:


spectral_contrast = lbf.spectral_contrast(x, sr)

plt.figure(figsize=(20,8)); lbd.specshow(spectral_contrast, x_axis='time'); plt.ylabel('Frequency bands'); plt.colorbar(); plt.title('Spectral contrast Visualized', size=25); plt.show()


# In[17]:


flatness = lbf.spectral_flatness(x); 

plt.figure(figsize=(20,8)); lbd.specshow(flatness, x_axis='time'); plt.colorbar(); plt.title('Spectral flatness Visualized', size=25); plt.show()


# In[18]:


p, q = 1650, 1700
plt.figure(figsize=(16,8)); plt.plot(x[p:q]); plt.title('Zero crossing rate between {} and {}:'.format(p,q)); plt.show()


# In[19]:


zc = lb.zero_crossings(x[p:q])
zc.shape


# In[20]:


print('Zero crossings: ', sum(zc))


# In[21]:


zcr = lbf.zero_crossing_rate(x)

plt.figure(figsize=(16,8)); plt.plot(zcr[0]); plt.title('Zero crossing rate over time:', size=25); plt.show()


# In[22]:


ar = lb.autocorrelate(x, max_size=10000)

plt.figure(figsize=(16,8)); plt.plot(ar); plt.xlim(0,10000); plt.xlabel('Lag'); plt.title('Autocorrelation over lag:', size=25); plt.show()


# In[23]:


## SETTING THE SEARCH RANGE OF THE PITCH
f_hi = lb.midi_to_hz(120)
f_lo = lb.midi_to_hz(12)

## SETTING THE INVALID PITCH RANGE TO ZERO
t_lo = sr/f_hi
t_hi = sr/f_lo
ar[:int(t_lo)] = 0
ar[int(t_hi):] = 0


# In[24]:


plt.figure(figsize=(16,8)); plt.plot(ar[:1400]); plt.title('Autocorrelation within a range (0 to 1400)', size=25); plt.show()


# In[25]:


float(sr)/ar.argmax()


# In[26]:


hop_length = 200 ## Samples to be taken per frame
onset = lb.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)

## PLOTTING THE ONSET ENVELOPE
frames = range(len(onset)); t = lb.frames_to_time(frames,sr,hop_length=hop_length)

plt.figure(figsize=(16,8)); plt.plot(t,onset); plt.xlabel('Time (in sec)'); plt.title('Novelty Function', size=25); plt.show()


# In[27]:


S = lb.stft(onset, hop_length=1, n_fft=512)
ft = np.absolute(S)

plt.figure(figsize=(16,8)); lbd.specshow(ft,sr=sr,hop_length=hop_length, x_axis='time'); plt.title('Fourier Tempogram', size=25); plt.xlabel('Time (in min:sec)'); plt.show()


# In[28]:


tempogram = lbf.tempogram(onset_envelope=onset, sr=sr, hop_length=hop_length, win_length=400)

plt.figure(figsize=(16,8)); lbd.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo');
plt.title('Tempogram Visualized', size=25); plt.xlabel('Time (in min:sec)'); plt.show()


# In[29]:


tempo = lb.beat.tempo(x, sr=sr)
print(tempo[0])


# In[30]:


t = len(x)/float(sr)
seconds_per_beat = 60/tempo[0]
beat_times = np.arange(0, t, seconds_per_beat)

plt.figure(figsize=(16,8)); lbd.waveplot(x); plt.vlines(beat_times, -1, 1, color='r'); plt.title('Tempo Visualized', size=25); plt.show();


# In[31]:


mfccs = lbf.mfcc(x, sr)
mfccs.shape


# In[32]:


plt.figure(figsize=(20,8)); lbd.specshow(prep.minmax_scale(mfccs, axis=1), sr=sr, x_axis='time')
plt.colorbar(); plt.title('MFCCs Visualized', size=25); plt.show()


# In[33]:


chrom = lbf.chroma_stft(x)

plt.figure(figsize=(20,8)); lbd.specshow(chrom, sr=sr, y_axis='chroma', x_axis='time'); plt.colorbar(); 
plt.title('Chromagram Visualized', size=25); plt.show()


# In[34]:


countries = train['country'].unique()
colors = ['blue', 'red', 'green', 'black', 'lavender']

fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='world', title = 'Recordings from world'); 

plotly.offline.iplot(fig)


# In[35]:


fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='usa', title = 'Recordings from USA'); 

plotly.offline.iplot(fig)


# In[36]:


fig = go.Figure(data=go.Scattergeo(lon=train['longitude'], lat = train['latitude'], mode='markers', text = train['location'], marker=dict(size=4,
                                                                                                               opacity=0.6,
                                                                                                               symbol='square',
                                                                                                               line=dict(width=1,
                                                                                                                        color='white'),
                                                                                                               colorscale='Blues',
                                                                                                               color='blue')))
fig.update_layout(geo_scope='europe', title = 'Recordings from Europe'); 

plotly.offline.iplot(fig)


# In[37]:


fig = make_subplots(2,2,
                    specs = [[{},{}],[{'colspan':2},None]],
                    subplot_titles=('Where are the recordists from?', 'How many recordings each recordist has contributed?', 'What are the average ratings of recordists?'))

temp = train.groupby(['country']).count().reset_index().sort_values(by='recordist'); 
fig.add_trace(go.Bar(x= temp['recordist'], y = temp['country'], orientation='h', name='Ans 1', marker=dict(color='violet',line=dict(color='black',width=0.3))),1,1)

temp =train.groupby(['recordist']).count().reset_index().sort_values(by='filename'); 
fig.add_trace(go.Bar(x=temp['filename'], y=temp['recordist'],orientation='h',name='Ans 2',marker=dict(color='indianred',line=dict(color='black',width=0.3))), 1,2)

temp = pd.DataFrame(train.groupby(['recordist'])['rating'].sum()).reset_index(); temp1 = pd.DataFrame(train.groupby(['recordist'])['filename'].count()).reset_index()
temp['rating'] = round(temp['rating']/temp1['filename'],1); 
fig.add_trace(go.Bar(x=temp.sort_values('rating')['rating'], y = temp.sort_values('rating')['recordist'],orientation='h',name='Ans 3', marker=dict(color='green',line=dict(color='black',width=0.1))),2,1)

fig.update_layout(height=1500, title_text='Exploring recordists'); 
plotly.offline.iplot(fig)

