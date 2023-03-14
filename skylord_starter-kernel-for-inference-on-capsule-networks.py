#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install facenet-pytorch
get_ipython().system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl')

from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
torch_home = get_torch_home()

# Copy model checkpoints to torch cache so they are loaded automatically by the package
get_ipython().system('mkdir -p $torch_home/checkpoints/')
get_ipython().system('cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt')
get_ipython().system('cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt')
get_ipython().system('cp /kaggle/input/superresolution-pets/vgg19-dcbb9e9d.pth $torch_home/checkpoints/vgg19-dcbb9e9d.pth')


# In[2]:


# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face


# In[3]:


import os
import sys
import pathlib
from pathlib import Path


# In[4]:


sys.path.append('/kaggle/input/superresolution-pets')
sys.path.append('/kaggle/input/facent-pytorch-vggface2')
sys.path.append('/kaggle/input/resnet')


# In[5]:


import model_big
import utilDeepFake


# In[6]:


import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# In[7]:


# setup a dict with the required command line default parameters 

opt = dict()

opt['dataset'] =''
opt['imageSize'] = 160
opt['gpu_id'] = 0 if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {opt["gpu_id"]}')
opt['outf'] = '/kaggle/input/superresolution-pets/checkpoints/binary_faceforensicspp'
opt['id'] = 21
opt['random'] = False


# In[8]:


transform_fwd = transforms.Compose([
        transforms.Resize((opt['imageSize'], opt['imageSize'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


# In[9]:


#vgg_ext = torch.load('/kaggle/input/superresolution-pets/vgg19-dcbb9e9d.pth')
vgg_ext = model_big.VggExtractor()


# In[10]:


capnet = model_big.CapsuleNet(2, opt['gpu_id'])


# In[11]:


capnet.load_state_dict(torch.load(os.path.join(opt['outf'],'capsule_' + str(opt['id']) + '.pt')))


# In[12]:


capnet.eval();


# In[13]:


if opt['gpu_id'] >= 0:
    vgg_ext.cuda(opt['gpu_id'])
    capnet.cuda(opt['gpu_id'])


# In[14]:


tol_label = np.array([], dtype=np.float)
tol_pred = np.array([], dtype=np.float)
tol_pred_prob = np.array([], dtype=np.float)

count = 0
loss_test = 0


# In[15]:


testPath = Path('/kaggle/input/deepfake-detection-challenge/test_videos')
test_videos = [f for f in testPath.glob("**/*") if f.is_file() and '.mp4' in str(f)]
len(test_videos)


# In[16]:


import time
from fastai.vision import *
from PIL import Image
import torchvision.transforms.functional as TF


# In[17]:


def predict_face(fileList):
    frameSample = len(fileList)
    predsF = list()
    
    for faceFile in fileList:
        img = Image.open(faceFile)
        x = TF.to_tensor(img).cuda(opt['gpu_id'])
        x.unsqueeze_(0)
        input_v = Variable(x)
        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt['random'])
        outputs = class_.data.cpu()
        predsF.append(float(outputs[0][0]))
        #print(outputs)
        os.remove(faceFile)
    print(sum(predsF)/frameSample) 
    return sum(predsF)/frameSample


# In[18]:


vName = str(test_videos[0]).split('/')[-1]
frames = utilDeepFake.extractFrames(str(test_videos[0]), frameSample = 10)
faceCrops = utilDeepFake.detect_facenet_pytorch(frames, 16)
fileList = utilDeepFake.saveFaces(faceCrops, vName, '2', faceDir = '/kaggle/working/')


# In[19]:


faceFile = fileList[0]

img = Image.open(faceFile)
x = TF.to_tensor(img).cuda(opt['gpu_id'])
x.unsqueeze_(0)
input_v = Variable(x)
x = vgg_ext(input_v)
classes, class_ = capnet(x, random=opt['random'])


# In[20]:


class_


# In[21]:


outputs = class_.data.cpu()
predsF.append(float(outputs[0][0]))
#print(outputs)
os.remove(faceFile)


# In[22]:


start = time.time()
# Sample run 
vName = str(test_videos[0]).split('/')[-1]
frames = utilDeepFake.extractFrames(str(test_videos[0]), frameSample = 10)
faceCrops = utilDeepFake.detect_facenet_pytorch(frames, 16)
fileList = utilDeepFake.saveFaces(faceCrops, vName, '2', faceDir = '/kaggle/working/')
preds = predict_face(fileList)
end = time.time()
print("Prediction: ", preds)
print("Time for processing single video ", (end - start))
print("Is it good to go? ", (end - start)< 8)


# In[23]:


import time 

classP =list()
prediction = list()
filename = list()

start = time.time()

for video in test_videos:
    vName = str(video).split('/')[-1]
    
    try:

        frames = utilDeepFake.extractFrames(str(video), frameSample = 10)
        faceCrops = utilDeepFake.detect_facenet_pytorch(frames, 16)
        fileList = utilDeepFake.saveFaces(faceCrops, vName, '2', faceDir = '/kaggle/working/')
        preds = predict_face(fileList)
        #classP.append(pred_class)
        prediction.append(preds)
        filename.append(vName)
        
    except:
        print("Error in file: {}, appending 0.5 prob".format(vName) )
        #classP.append('NA')
        prediction.append(0.5)
        filename.append(vName)
    #break
    
end = time.time()


# In[24]:


print("Total time elapsed: ", (end-start) )
print("Average time on each video: ", (end-start)/len(test_videos))
print("Cool, so we have a GO!")


# In[25]:


submission_df = pd.DataFrame({"filename": filename, "label": prediction})
submission_df.to_csv("submission.csv", index=False)


# In[26]:


print(submission_df.shape)
submission_df.head()


# In[27]:


submission_df.label.hist()


# In[ ]:




