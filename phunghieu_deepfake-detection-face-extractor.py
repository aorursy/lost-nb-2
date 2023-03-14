#!/usr/bin/env python
# coding: utf-8

# In[1]:


TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
TMP_DIR = '/kaggle/tmp/'
ZIP_NAME = 'dfdc_train_faces_sample.zip'
METADATA_PATH = TRAIN_DIR + 'metadata.json'

SCALE = 0.25
N_FRAMES = None


# In[2]:


get_ipython().system('pip install facenet-pytorch > /dev/null 2>&1')
get_ipython().system('apt install zip > /dev/null 2>&1')


# In[3]:


import os
import glob
import json
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


# In[4]:


class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()


# In[5]:


with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)


# In[6]:


train_df = pd.DataFrame(
    [
        (video_file, metadata[video_file]['label'], metadata[video_file]['split'], metadata[video_file]['original'] if 'original' in metadata[video_file].keys() else '')
        for video_file in metadata.keys()
    ],
    columns=['filename', 'label', 'split', 'original']
)

train_df.head()


# In[7]:


# Load face detector
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()


# In[8]:


# Define face extractor
face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)


# In[9]:


# Get the paths of all train videos
all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))


# In[10]:


get_ipython().system('mkdir -p $TMP_DIR')


# In[11]:


with torch.no_grad():
    for path in tqdm(all_train_videos):
        file_name = path.split('/')[-1]

        save_dir = os.path.join(TMP_DIR, file_name.split(".")[0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Detect all faces appear in the video and save them.
        face_extractor(path, save_dir)


# In[12]:


cd $TMP_DIR


# In[13]:


train_df.to_csv('metadata.csv', index=False)


# In[14]:


get_ipython().system('zip -r -m -q /kaggle/working/$ZIP_NAME *')

