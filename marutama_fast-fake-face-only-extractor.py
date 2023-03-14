#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install facenet-pytorch > /dev/null 2>&1')
get_ipython().system('apt install zip > /dev/null 2>&1')


# In[2]:


ls ../input/dfdc-full-data-csv


# In[3]:


mkdir /kaggle/working/tmp


# In[4]:


##################################
TMP_DIR = '/kaggle/working/tmp'
FULL_DATA_CSV = '../input/dfdc-full-data-csv/full_data.csv'
DFDC_FULL_PATH ='../input/dfdc-train-part-04/'
CANDIDATE_ALL = 'candidate_all'

##################################
# for MTCNN and CV2
SCALE = 0.25
N_FRAMES = None

##################################
# parameters
##################################
# for MTCNN
MARGIN=0
SIZE=160
FACTOR = 0.9  # 0.5: 27s, 0.7:30s, 0.9:70s, 1.0:no end
#MAX_FRAMES=27   # for DEBUG
MAX_FRAMES=300

##################################
# for FACE cluster matching
# matching range in M_FRAME frames
M_FRAME = 50
# video frame size is 1920x1080 or 1080x1920
NEAR_X = 100 # about 5% of 1920 
NEAR_Y = 50  # about 5% of 1080 
FACE_RATE = 1.1  # for face size matching
##################################
# For CopyCandidate
# number os train candidate faces
CANDIDATE = 10
# threshold of std
STD_TH = 4.5
##################################

def auto_contrast(img):
    # BGR->HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # V histogram equalize
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    # HSV->BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    
    return bgr

###########################################################
import PIL
PIL.PILLOW_VERSION = PIL.__version__

import os
import sys
import shutil
import re
import glob
import json
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm
from facenet_pytorch import MTCNN

from functools import wraps
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

###########################################################
DEBUG = False
def isSameCenter(sxc, syc, xc, yc):
    ret = True
    
    if (sxc-NEAR_X > xc) | (sxc+NEAR_X < xc):
        if DEBUG :
            print(f'False1:{sxc},{xc}')
        return False
    elif (syc-NEAR_Y > yc) | (syc+NEAR_Y < yc):
        if DEBUG :
            print(f'False2:{syc},{yc}')
        return False
    # currently not using FACE size, if it used then clusters could be more divided
    
    return ret

###########################################################
def InitDrawImage(draw_flag, ax, frame):
    if draw_flag is not True:
        return

    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    return

def DrawLandmarks(draw_flag, ax, box, landmark):
    if draw_flag is not True:
        return

    xl, yb, xr, yt = box[0], box[1], box[2], box[3]
    width, height=xr-xl, yt-yb
    
    #draw corner
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    #draw landmark
    ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
    #draw rectangle
    r = patches.Rectangle(xy=(xl, yb), width=width, height=height, ec='green', fill=False)
    ax.add_patch(r)

    return

def SaveLandmarkPlot(draw_flag, save_dir, file_base, f_num):
    if draw_flag is not True:
        plt.close('all')
        return
    
    #save plot
    save_path = os.path.join(save_dir, file_base+f'{f_num:04}.png')
    plt.savefig(save_path) 
    plt.close('all')
    
    return

def SaveCharts(df, save_dir, file_base):
    plt.figure()
    df.plot(kind='scatter', x='xc', y='yc', c='c_id', cmap='rainbow', xlim=(0,1920), ylim=(1080,0), title=f'{file_base}:xc-yc graph')
    plt.savefig(os.path.join(save_dir, 'graph-xc-yc'), dpi=300)
    plt.close('all')
            
    plt.figure()
    df.plot(kind='scatter', x='frame', y='yc', c='c_id', cmap='rainbow', ylim=(1080,0), title=f'{file_base}:frame-yc graph')
    plt.savefig(os.path.join(save_dir, 'graph-frame-yc'), dpi=300)
    plt.close('all')
    
    plt.figure()            
    ax = df.plot(kind='scatter', x='frame', y='xc', c='c_id', cmap='rainbow', ylim=(1920,0), title=f'{file_base}:frame-xc graph')
    plt.savefig(os.path.join(save_dir, 'graph-frame-xc'), dpi=300)
    plt.close('all')

    return

def SaveStdCharts(df, save_dir, file_base):
    plt.figure()
    df.plot(kind='scatter', x='frame', y='std', c='c_id', cmap='rainbow', title=f'{file_base}:frame-std graph')
    plt.savefig(os.path.join(save_dir, 'graph-frame-std'), dpi=300)
    plt.close('all')
    
    return

###########################################################
def MoveClusters(df, save_dir):
    # get TOP 2 clusters
    Clusters = df['cluster'].value_counts().index[:2]
    #if 2nd cluster length is < 5, only use TOP 1 cluster
    if len(Clusters) > 1:
        if len(df[df['cluster']==Clusters[1]]) < 5:
            # use only 1st cluster
            Clusters = df['cluster'].value_counts().index[:1]
    
    # move to cluster folders
    for c in Clusters:
        path_c = os.path.join(save_dir, c)
        if not os.path.exists(path_c):
            os.makedirs(path_c)

        #for i, row in df[df['cluster']==c].iterrows():
        #    png_file = os.path.join(save_dir, row['png'])
        #    if os.path.exists(png_file):
        #        shutil.move(png_file, path_c)
        for png in df[df['cluster']==c]['png']:
            png_file = os.path.join(save_dir, png)
            if os.path.exists(png_file):
                shutil.move(png_file, path_c)
    
    # remaining png -> etc
    path_etc = os.path.join(save_dir, 'etc')
    if not os.path.exists(path_etc):
        os.makedirs(path_etc)
    remaining_png = glob.glob(os.path.join(save_dir, '*_????_????.png'))
    for png_file in remaining_png:
        if os.path.exists(png_file):
            shutil.move(png_file, path_etc)
    
    return

###########################################################
class FaceCExtractor:
    def __init__(self, detector, n_frames=None, resize=None, max_frames=None, margin=0, size=160):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
            max_frames [int] -- Maximum number of frames to process.
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
        self.max_frames = max_frames
        self.margin = margin
        self.size = size
    
    def __call__(self, filename, tmp_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            tmp_dir  [str] -- The tmp directory for save
        """
        
        file_base = os.path.splitext(os.path.basename(filename))[0]
        save_dir = os.path.join(tmp_dir, file_base)
        if os.path.exists(save_dir):
            #shutil.rmtree(save_dir)
            # continue with using save_dir
            return
        else:
            os.makedirs(save_dir)

        face_df = pd.DataFrame()
        idx = 0
        
        # high speed DataFrame generate : list -> pd 
        df = pd.DataFrame()
        video_list = [] 
        png_list = [] 
        frame_list = [] 
        xc_list = [] 
        yc_list = [] 
        l_list = []
        w_list = [] 
        h_list = []  
        xl_list = []
        xr_list = []
        yb_list = []
        yt_list = []
        cluster_list = []

        
        save_dir = os.path.join(tmp_dir, file_base)

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # for temporary
        loop = v_len if self.max_frames is None else min(v_len, self.max_frames)
        # Loop through frames
        for j in tqdm(range(loop), desc=f'MTCNN f{FACTOR}:{file_base}'):
        #for j in range(loop):
            success = v_cap.grab()

            # Load frame
            success, frame_cv2 = v_cap.retrieve()
            if not success:
                continue
                    
            frame_cv2 = auto_contrast(frame_cv2)

            # Resize frame to desired size
            if self.resize is not None:
                frame_cv2_re = cv2.resize(frame_cv2, dsize=None, fx=self.resize, fy=self.resize)
            else:
                frame_cv2_re = frame_cv2
            
            frame_h, frame_w, frame_ch = frame_cv2_re.shape[:3]
                
            # Detect face
            boxes, probs, landmarks = self.detector.detect(frame_cv2_re, landmarks=True)
            
            # Visualize for DEBUG
            draw_flag = True if j==0 else False
            #draw_flag = True # all frame draw for DEBUG
            
            ### Draw
            #plt.figure()
            fig, ax = plt.subplots(figsize=(16/2, 12/2))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            InitDrawImage(draw_flag, ax, frame_cv2_re)

            if boxes is None:
                ### just Image draw and closing and skip frame loop
                SaveLandmarkPlot(draw_flag, save_dir, file_base, j)
                continue
            
            for box, landmark in zip(boxes, landmarks):
                xl, yb, xr, yt = box[0], box[1], box[2], box[3]
                if xl<0 or xl>frame_w or xr<0 or xr>frame_w or yb<0 or yb>frame_h or yt<0 or yt>frame_h:
                    # print('SKIP!!')
                    # invalid float number: skip this face
                    continue
                width, height=xr-xl, yt-yb
                ### Draw
                DrawLandmarks(draw_flag, ax, box, landmark)
                    
                # cut & save face rectangle
                length = width if width > height else height
                #print(f'frame {j}:length={length}')
                        
                xc, yc = (xl+xr)/2, (yb+yt)/2
                    
                # Square BOX corners with MARGIN
                # clip in resized small framesize for MTCNN
                bxl = min(frame_w, max(0, xc - (length+self.margin)/2))
                bxr = min(frame_w, max(0, xc + (length+self.margin)/2))
                byb = min(frame_h, max(0, yc - (length+self.margin)/2))
                byt = min(frame_h, max(0, yc + (length+self.margin)/2))
                if (bxr-bxl)<10 or (byt-byb)<10:
                    # print('SKIP!!')
                    # too small or out of screen : skip this face
                    continue

                # SCALED corners
                sxl = bxl/self.resize
                sxr = bxr/self.resize
                syb = byb/self.resize
                syt = byt/self.resize
                sxc = xc/self.resize
                syc = yc/self.resize
                sl  = length/self.resize
                sw  = width/self.resize
                sh  = height/self.resize
                
                face_cv2 = frame_cv2[int(syb):int(syt), int(sxl):int(sxr)]
                #print(f'frame {j}: {syb}, {syt}, {sxl}, {sxr}')  # for DEBUG
                face_cv2 = cv2.resize(face_cv2, dsize=(self.size, self.size))
                png_file = f'{file_base}_r_{j:04}_{int(sxc):04}_{int(syc):04}.png'
                save_face_path = os.path.join(save_dir, png_file)
                cv2.imwrite(save_face_path, face_cv2)
                
                #idx = f'{file_base}_{j:04}_{int(sxc):04}_{int(syc):04}'
                # high speed DataFrame generate : list -> pd 
                video_list.append(file_base) 
                png_list.append(png_file) 
                frame_list.append(j) 
                xc_list.append(sxc) 
                yc_list.append(syc) 
                l_list.append(sl)
                w_list.append(sw) 
                h_list.append(sh) 
                xl_list.append(sxl) 
                xr_list.append(sxr) 
                yb_list.append(syb) 
                yt_list.append(syt) 

                
                c = f'{int(sxc):04}_{int(syc):04}'
                if j >= 1:
                    #for index, row in face_df[(face_df['video']==file_base) & (face_df['frame']==j-1)].iterrows():
                    # matching range in M_FRAME frames
                    stop = len(xc_list)-1 # until previous frame
                    start = max(0, stop-M_FRAME)  # matching range
                    for i in range(start, len(xc_list)-1):
                        if isSameCenter(sxc, syc, xc_list[i], yc_list[i]):
                            c = cluster_list[i]
                    
                #face_df.loc[idx, 'cluster'] = c
                cluster_list.append(c)

            ### Draw
            SaveLandmarkPlot(draw_flag, save_dir, file_base, j)
                
        v_cap.release()
        
        # high speed DataFrame generate : list -> pd 
        df = pd.DataFrame(
            data={'video': video_list, 'png': png_list, 'frame': frame_list, 
                  'xc': xc_list, 'yc': yc_list, 
                  'length': l_list, 'width': w_list, 'height' : h_list, 
                  'xl': xl_list, 'xr' : xr_list, 'yb' : yb_list, 'yt' : yt_list, 
                  'cluster' : cluster_list},
            columns=['video', 'png', 'frame', 'xc', 'yc', 'length', 'width', 'height', 'xl', 'xr', 'yb', 'yt', 'cluster']
        )
        
        
        # create cluster_id
        df['c_id'] = pd.factorize(df['cluster'])[0]

        # Draw & Save Charts
        SaveCharts(df, save_dir, file_base)
        
        # Cluster: move to sub-folders
        MoveClusters(df, save_dir)
        
        # save csv for face information
        df.to_csv(os.path.join(save_dir, file_base+'.csv'))

        return
    
###########################################################
def CopyCandidate(fake_df, fake_tmp_dir, fake_base, real_tmp_dir, real_base):
    if not os.path.exists(fake_tmp_dir):
        print('ERROR! not exists:', fake_tmp_dir)
        return
    if not os.path.exists(real_tmp_dir):
        print('ERROR! not exists:', real_tmp_dir)
        return
    
    # reduce fake_df
    c_df = fake_df.loc[:, ['png', 'frame', 'cluster', 'std']]
    #c_df = fake_df.copy()
    # sort by 'std'
    c_df = c_df.sort_values(by=['std'], ascending=False)
    # get clusters form FAKE foder
    Clusters = glob.glob(os.path.join(fake_tmp_dir, fake_base, '????_????'))

    candidate_dir = os.path.join(fake_tmp_dir, fake_base, 'candidate')
    os.makedirs(candidate_dir)
    
    candidate_all_dir = os.path.join(real_tmp_dir, '..', CANDIDATE_ALL)
    if not os.path.exists(candidate_all_dir):
        os.makedirs(candidate_all_dir)
    candidate_all_dir = os.path.join(candidate_all_dir, real_base)
    if not os.path.exists(candidate_all_dir):
        os.makedirs(candidate_all_dir)
    
    for path_c in Clusters :
        c = os.path.basename(path_c)
        #print(c)
        df = c_df[c_df['cluster']==c]
        df = df.reset_index()
        for i, row in df.iterrows():
            #print(i, row['png'], row['cluster'], row['std'])
            fake_png = row['png']
            base, ext = os.path.splitext(fake_png)
            std = row['std']
            fake_new_png = real_base + '-' + base + f'_{std:05.1f}' + ext
            
            cluster = row['cluster']
            real_png = fake_png.replace(fake_base, real_base).replace('_f_', '_r_')
            if row['std'] > STD_TH:
                shutil.copy2(os.path.join(fake_tmp_dir, fake_base, cluster, fake_png), os.path.join(candidate_dir, fake_new_png))
                shutil.copy2(os.path.join(real_tmp_dir, real_base, cluster, real_png), os.path.join(candidate_dir, real_png))

                shutil.copy2(os.path.join(fake_tmp_dir, fake_base, cluster, fake_png), os.path.join(candidate_all_dir, fake_new_png))
                shutil.copy2(os.path.join(real_tmp_dir, real_base, cluster, real_png), os.path.join(candidate_all_dir, real_png))

            # copy only CANDIDATE files
            if i >= CANDIDATE-1:
                break
    
    return

###########################################################
def fake_diff_extractor(filename, fake_tmp_dir, real_tmp_dir, real_base):
    file_base = os.path.splitext(os.path.basename(filename))[0]
    fake_base = file_base
    save_dir = os.path.join(fake_tmp_dir, file_base)
    if os.path.exists(save_dir):
        #shutil.rmtree(save_dir)
        # continue with using save_dir
        return
    else:
        os.makedirs(save_dir)

    real_dir = os.path.join(real_tmp_dir, real_base)
    real_csv = os.path.join(real_dir, real_base+'.csv')
    real_df = pd.read_csv(real_csv, index_col=0)
    fake_df = real_df
    fake_df['video'] = fake_base
    
    save_dir = os.path.join(fake_tmp_dir, file_base)
    save_csv = os.path.join(save_dir, real_base+'.csv')

    # copy: csv, init_image, charts
    #shutil.copy2(real_csv, save_csv)
    shutil.copy2(os.path.join(real_dir, real_base+'0000.png'), os.path.join(save_dir, real+'0000.png'))
    shutil.copy2(os.path.join(real_dir, 'graph-frame-xc.png'), os.path.join(save_dir, 'graph-frame-xc.png'))
    shutil.copy2(os.path.join(real_dir, 'graph-frame-yc.png'), os.path.join(save_dir, 'graph-frame-yc.png'))
    shutil.copy2(os.path.join(real_dir, 'graph-xc-yc.png'), os.path.join(save_dir, 'graph-xc-yc.png'))
    
    # Create video reader and find length
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fake_df['std'] = fake_df['png']
    loop = min(v_len, MAX_FRAMES)
    # Loop through frames
    for j in range(loop):
        success = v_cap.grab()

        # Load frame
        success, frame_cv2 = v_cap.retrieve()
        if not success:
            continue
                    
        frame_cv2 = auto_contrast(frame_cv2)

        # Detect face
        #boxes, probs, landmarks = self.detector.detect(frame_cv2_re, landmarks=True)
        frame_df = real_df[real_df['frame']==j]
        if len(frame_df)==0:   # case : no face
            continue
        
        for real_png, sxc, syc, sxl, sxr, syb, syt, c in zip(frame_df['png'], frame_df['xc'], frame_df['yc'], 
                                                             frame_df['xl'], frame_df['xr'], frame_df['yb'], frame_df['yt'],
                                                            frame_df['cluster']):

            fake_face_cv2 = frame_cv2[int(syb):int(syt), int(sxl):int(sxr)]
            fake_face_cv2 = cv2.resize(fake_face_cv2, dsize=(SIZE, SIZE))
            fake_png = f'{fake_base}_f_{j:04}_{int(sxc):04}_{int(syc):04}.png'
            fake_face_path = os.path.join(save_dir, fake_png)
            cv2.imwrite(fake_face_path, fake_face_cv2)
            
            fake_df['png'] = fake_df['png'].str.replace(real_png, fake_png)
            
            real_face_path = os.path.join(real_dir, c, real_png)
            if os.path.exists(real_face_path):
                real_face_cv2 = cv2.imread(real_face_path)
                
                img_diff=abs(fake_face_cv2.astype("int") - real_face_cv2.astype("int")).astype("uint8")
                #img_diff=abs(auto_contrast(fake_face_cv2).astype("int") - auto_contrast(real_face_cv2).astype("int")).astype("uint8")
                
                #diff_mean = np.mean(img_diff)
                diff_std  = np.std(img_diff)
                fake_df['std'] = fake_df['std'].replace(real_png, diff_std)
                

    v_cap.release()
    
    # out of clusters
    fake_df['std'] = fake_df['std'].replace(real_base, 0, regex=True)
    
    # Cluster: move to sub-folders
    MoveClusters(fake_df, save_dir)

    # save csv for face information
    fake_df.to_csv(os.path.join(save_dir, fake_base+'.csv'))
    
    # save std charts
    SaveStdCharts(fake_df, os.path.join(fake_tmp_dir, fake_base), fake_base)
    
    # Copy the face-png file of the high std score along with the corresponding real-png to the 'candidate' folder
    CopyCandidate(fake_df, fake_tmp_dir, fake_base, real_tmp_dir, real_base)
    
    return

###########################################################
### log
import datetime

#path_log = "../tmp-c/tmp/log.txt"

def log_file_base(label_str, file_base, file_part, tmp_dir):
    m=re.match('^part_\d+$', file_part)
    if not m:
        file_part = "no-part"

    path_log = os.path.join(tmp_dir, 'log.txt')
    with open(path_log, mode='a') as f:
        dt_now = datetime.datetime.now()
        if label_str is None:
            head = ''
        elif label_str == 'FAKE':
            head = '  FAKE '
        else:
            head ='REAL '
        
        f.write(dt_now.strftime(head+'[%Y%m%d %H:%M:%S] '))
        f.write(file_base + ':' + file_part +  " ... ")  
    return

def log_skip(label_str, file_base, file_part, tmp_dir):
    path_log = os.path.join(tmp_dir, 'log.txt')
    with open(path_log, mode='a') as f:
        dt_now = datetime.datetime.now()
        if label_str == 'REAL':
            head = '[SKIP REAL]'
        else:
            head = '  [SKIP FAKE]'
        f.write(dt_now.strftime(head+'[%Y%m%d %H:%M:%S] '))

    return
    
def log_done(tmp_dir):
    path_log = os.path.join(tmp_dir, 'log.txt')
    with open(path_log, mode='a') as f:
        f.write("done\n")
    return

def log_fixdone(tmp_dir):
    path_log = os.path.join(tmp_dir, 'log.txt')
    with open(path_log, mode='a') as f:
        f.write("fix done\n")
    return

def log_finish(tmp_dir):
    path_log = os.path.join(tmp_dir, 'log.txt')
    with open(path_log, mode='a') as f:
        f.write("finish!\n")
    return

###########################################################
def load_full_real():
    ########################
    # load full_df
    ########################
    '''
    [NOTICE] These videos are missing.
    Part18：
      pvohowzowy.mp4 : Fake
      wipjitfmta.mp4 : Fake
      wpuxmawbkj.mp4 : Fake
    Part35
      cfxiikrhep.mp4 : Fake
      dzjjtfwiqc.mp4 : Fake
      glleqxulnn.mp4 : Fake
      innmztffzd.mp4 : Fake
      zzfhqvpsyp.mp4 : Fake
    
    '''
    full_df = pd.read_csv(FULL_DATA_CSV, index_col=0)
    # delete missing video rows
    #     not is in [list]
    full_df = full_df[~full_df['filename'].isin([
        'pvohowzowy.mp4', 
        'wipjitfmta.mp4', 
        'wpuxmawbkj.mp4',
        'cfxiikrhep.mp4', 
        'dzjjtfwiqc.mp4', 
        'glleqxulnn.mp4', 
        'innmztffzd.mp4', 
        'zzfhqvpsyp.mp4'])]

    ########################
    # make real_df
    ########################
    #### create real_df
    real_df = full_df[['orig_part', 'original', 'orig_label']]
    # counts duplicated number
    vc = real_df['original'].value_counts()
    # drop duplicates
    real_df = real_df.drop_duplicates()
    # set index
    real_df.set_index('original', inplace=True)
    # add duplicated number
    for real in vc.index:
        real_df.loc[real, 'fake_num'] = vc[real]
    # sort by reverse order, top is big number
    real_df=real_df.sort_values(by=['fake_num'], ascending=False)
    
    return full_df, real_df

###########################################################
###########################################################
### Load face detector
#face_detector = MTCNN(image_size=SIZE, margin=14, keep_all=True, factor=0.5, device=device).eval()
face_detector = MTCNN(image_size=SIZE, margin=MARGIN, keep_all=True, factor=FACTOR, device=device).eval()

## Define face extractor
face_extractor = FaceCExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE, 
                                max_frames=MAX_FRAMES, margin=MARGIN, size=SIZE)
###########################################################
# about 5 sec
full_df, real_df = load_full_real()

###########################################################
# Real->Fake loop
######################
# select input data
######################

# get command line args
args = sys.argv
print(args)
# check invalid args and jupyter notebook 
if (len(args) == 2) and (not args[1]=='-f()') : # check args and no jupyter notebook
    # set run mode by command line args
    m = re.match(r'(part_\d\d*)', args[1])
    if m :
        SELECT_PART = m.group()
        real_df=real_df[real_df['orig_part']==SELECT_PART]
    else :
        print('ERROR : invalid args')
        exit(1)
else : 
    # set run mode by manualy here
    SELECT_PART=''    # MUST need this line
    ### select part
    #SELECT_PART='part_4'

    # remove comment out, if you want to use SELECT_PART
    #real_df=real_df[real_df['orig_part']==SELECT_PART]

    ##### SELECT_PART or SELECT_REAL #####

    ### select real
    #SELECT_REAL=['bxgkydnxzv.mp4', 'vudstovrck.mp4']
    SELECT_REAL=['bgpoldvzrh.mp4']

    # remove comment out, if you want to use SELECT_REAL
    real_df=real_df.reset_index()
    real_df=real_df[real_df['original'].isin(SELECT_REAL)]
    real_df=real_df.set_index('original')


###########################################################
###########################
### main Real-to-Fake loop
###########################
if SELECT_PART == '':
    print('selecting real:', SELECT_REAL)
    TMP_DIR_PART=TMP_DIR
else:
    print('selecting part:', SELECT_PART)
    TMP_DIR_PART=TMP_DIR+'_'+ SELECT_PART

##################################################
# Check TMP_DIR_PART with log.txt and try continue
##################################################
if os.path.exists(TMP_DIR_PART) and SELECT_PART != "":
    print(f'Checking [{TMP_DIR_PART}] for continue')
    
    log_real = None
    log_fake = None
    continue_flag = False
    if os.path.exists(os.path.join(TMP_DIR_PART, 'log.txt')):
        with open(os.path.join(TMP_DIR_PART, 'log.txt'),'r') as f:
            for line in f:
                # for DEBUG
                #print(line.strip())
                log_label = None
                log_base = None
                m = re.search(r'^finish!$', line)
                if m :
                    print("previous log.txt is finishd!")
                    exit(0)
                m = re.search(r'^\s*(REAL|FAKE).* (..........):part.*', line)
                if m :
                    log_label = m.group(1)
                    log_base = m.group(2)
                    if log_label == 'REAL':
                        log_real = log_base
                    else:
                        log_fake = log_base
                    #print(f'label:{log_label}, base:{log_base}, log_real:{log_real}')
                else :
                    print('log.txt format error:', line)
                    exit(1)

                m = re.search(r'(done)$', line)
                if not m :
                    if log_label == "REAL":
                        path_log_base = os.path.join(TMP_DIR_PART, log_real)
                        if os.path.exists(path_log_base):
                                shutil.rmtree(path_log_base)
                    else:
                        path_log_base = os.path.join(TMP_DIR_PART, log_real, 'FAKES', log_fake)
                        if os.path.exists(path_log_base):
                                shutil.rmtree(path_log_base)

                    print('Continue OK:', line.strip())
                    continue_flag = True

    if continue_flag:
        log_fixdone(TMP_DIR_PART)
    else:
        print(f'Fail continue')
        print(f'Removing [{TMP_DIR_PART}] ... ', end='')
        shutil.rmtree(TMP_DIR_PART)
        print('done')
        os.makedirs(TMP_DIR_PART)
else:
    if os.path.exists(TMP_DIR_PART):
        shutil.rmtree(TMP_DIR_PART)
        os.makedirs(TMP_DIR_PART)

##################
#### real loop
##################
with tqdm(real_df.index) as preal:
    for i, real in enumerate(preal):
        #print(i, real, real_df.loc[real, 'fake_num'])
        real_base = os.path.splitext(os.path.basename(real))[0]
        preal.set_description(f'[REAL]{real_base}')
        
        fake_df = full_df[full_df['original']==real]
        fake_df.set_index('filename', inplace=True)
        
        real_part = real_df.loc[real]['orig_part']
        path_movie = os.path.join(DFDC_FULL_PATH, 'dfdc_train_'+real_part, real)

        if not os.path.exists(path_movie):
            log_skip('REAL', real_base, real_part, TMP_DIR_PART)
            continue
        
        #### extract real

        #### for DEBUG
        #print('REAL:', real, real_part)
        
        log_file_base('REAL', real_base, real_part, TMP_DIR_PART)
        #start = time.time()
        face_extractor(path_movie, TMP_DIR_PART)
        #elapsed_time = time.time() - start
        #preal.set_postfix(time=elapsed_time)
        log_done(TMP_DIR_PART)
        
        ##################
        #### fake loop
        ##################
        real_tmp_dir = os.path.join(TMP_DIR_PART, real_base)
        fake_df.to_csv(os.path.join(real_tmp_dir, real_base+'-'+real_part+'-fakes.csv'))
                       
        fake_tmp_dir = os.path.join(real_tmp_dir, 'FAKES')
        if not os.path.exists(fake_tmp_dir):
            os.makedirs(fake_tmp_dir)

        with tqdm(fake_df.index, desc='FAKE') as pfake:
            for j, fake in enumerate(pfake):
                fake_base = os.path.splitext(os.path.basename(fake))[0]
                pfake.set_description(f' [FAKE]{fake_base}')

                fake_part = fake_df.loc[fake, 'part']
                real = fake_df.loc[fake, 'original']
            
                real_base = os.path.splitext(os.path.basename(real))[0]

                #### for DEBUG
                #print("FAKE:", fake, fake_part, real, real_part)
            
                fake_part=fake_df.loc[fake, 'part']
                path_movie = os.path.join(DFDC_FULL_PATH, 'dfdc_train_'+fake_part, fake)

                if not os.path.exists(path_movie):
                    log_skip('SKIP', fake_base, fake_part, TMP_DIR_PART)
                    continue
                
                #### extract fake
                log_file_base('FAKE', fake_base, fake_part, TMP_DIR_PART)
                fake_diff_extractor(path_movie, fake_tmp_dir, TMP_DIR_PART, real_base)
                log_done(TMP_DIR_PART)
            
                ### for DEBUG
                ### break FAKE loop
                #if j>=1:
                #    break
        
        ### for DEBUG
        ### break REAL loop
        #if i=>0:
        #    break

log_finish(TMP_DIR_PART)

###########################################################


# In[5]:


ls /kaggle/working/candidate_all/bgpoldvzrh


# In[6]:


ls /kaggle/working/tmp/bgpoldvzrh/


# In[7]:


def draw_png(png):
    ffig, ax = plt.subplots(figsize=(16/2, 12/2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    f = cv2.imread(png)
    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.show()
    return

def draw_png1_4(png1, png2, png3, png4):
    f1 = cv2.imread(png1)
    f2 = cv2.imread(png2)
    f3 = cv2.imread(png3)
    f4 = cv2.imread(png4)

    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')

    ax1.imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB))
    ax4.imshow(cv2.cvtColor(f4, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.close('all')

    return

def draw_png2_2(png1, png2, png3, png4):
    f1 = cv2.imread(png1)
    f2 = cv2.imread(png2)
    f3 = cv2.imread(png3)
    f4 = cv2.imread(png4)

    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')

    ax1.imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(f3, cv2.COLOR_BGR2RGB))
    ax4.imshow(cv2.cvtColor(f4, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.close('all')

    return


# In[8]:


draw_png2_2('/kaggle/working/tmp/bgpoldvzrh/bgpoldvzrh0000.png',
          '/kaggle/working/tmp/bgpoldvzrh/graph-xc-yc.png',
          '/kaggle/working/tmp/bgpoldvzrh/graph-frame-xc.png',
          '/kaggle/working/tmp/bgpoldvzrh/graph-frame-yc.png')


# In[9]:


ls /kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa


# In[10]:


draw_png('/kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/graph-frame-std.png')


# In[11]:


ls /kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/candidate


# In[12]:


draw_png1_4(
    '/kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/candidate/bgpoldvzrh-txdcmspaaa_f_0092_1431_0419_006.7.png',
    '/kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/candidate/bgpoldvzrh-txdcmspaaa_f_0095_1427_0421_006.7.png',
    '/kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/candidate/bgpoldvzrh-txdcmspaaa_f_0103_0540_0508_007.7.png',
    '/kaggle/working/tmp/bgpoldvzrh/FAKES/txdcmspaaa/candidate/bgpoldvzrh-txdcmspaaa_f_0113_0562_0515_006.0.png')


# In[ ]:





# In[ ]:




