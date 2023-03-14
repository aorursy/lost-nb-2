#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import seaborn as sns
from sklearn.metrics import log_loss
XCEPTION_MODEL = '../input/deepfakemodelspackages/xception-b5690688.pth'


# In[2]:


get_ipython().run_cell_magic('time', '', '# Install packages\n!pip install ../input/deepfakemodelspackages/Pillow-6.2.1-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/munch-2.5.0-py2.py3-none-any.whl -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/numpy-1.17.4-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/six-1.13.0-py2.py3-none-any.whl -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/torchvision-0.4.2-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index\n!pip install ../input/deepfakemodelspackages/tqdm-4.40.2-py2.py3-none-any.whl -f ./ --no-index')


# In[3]:


get_ipython().run_cell_magic('time', '', '!pip install ../input/deepfakemodelspackages/dlib-19.19.0/dlib-19.19.0/ -f ./ --no-index')


# In[4]:


## xception.py
"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'],             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model


# In[5]:


## models.py
"""
Author: Andreas Rössler
"""
import os
import argparse


import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
# from network.xception import xception
import math
import torchvision


def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            #'/home/ondyari/.torch/models/xception-b5690688.pth')
            XCEPTION_MODEL)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)

# if __name__ == '__main__':
#     model, image_size, *_ = model_selection('resnet18', num_out_classes=2)
#     print(model)
#     model = model.cuda()
#     from torchsummary import summary
#     input_s = (3, image_size, image_size)
#     print(summary(model, input_s))


# In[6]:


## transform.py
"""
Author: Andreas Rössler
"""
from torchvision import transforms

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


# In[7]:


## detect_from_video.py
"""
Evaluates a folder of video files or a single file with a xception binary
classification network.
Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>
Author: Andreas Rössler
"""
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm.notebook import tqdm

# from network.models import model_selection
# from dataset.transform import xception_default_data_transforms

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required
    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def test_full_image_network(video_path, model, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    
    # Modified to take in the model file instead of model
    """
    #print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
#     model, *_ = model_selection(modelname='xception', num_out_classes=2)
#     if model_path is not None:
#         model = torch.load(model_path)
#         print('Model found in {}'.format(model_path))
#     else:
#         print('No model found, initializing random model.')
#     if cuda:
#         model = model.cuda()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
#         print('getting image size')
        height, width = image.shape[:2]

        # Init output writer
#         print('init output writer')
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

        # 2. Detect with dlib
#         print('detect with dlib')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------

            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        # Show
#         print('show result')
        # cv2.imshow('test', image)
#         cv2.waitKey(33)     # About 30 fps
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        #print('Finished! Output saved under {}'.format(output_path))
    else:
        pass
        #print('Input video file was empty')
    return


# In[8]:


get_ipython().system('mkdir network')


# In[9]:


get_ipython().run_cell_magic('writefile', 'network/__init__.py', '# init')


# In[10]:


get_ipython().run_cell_magic('writefile', 'network/xception.py', '"""\nPorted to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)\n@author: tstandley\nAdapted by cadene\nCreates an Xception Model as defined in:\nFrancois Chollet\nXception: Deep Learning with Depthwise Separable Convolutions\nhttps://arxiv.org/pdf/1610.02357.pdf\nThis weights ported from the Keras implementation. Achieves the following performance on the validation set:\nLoss:0.9173 Prec@1:78.892 Prec@5:94.292\nREMEMBER to set your image size to 3x299x299 for both test and validation\nnormalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],\n                                  std=[0.5, 0.5, 0.5])\nThe resize parameter of the validation transform should be 333, and make sure to center crop at 299x299\n"""\nimport math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.utils.model_zoo as model_zoo\nfrom torch.nn import init\n\npretrained_settings = {\n    \'xception\': {\n        \'imagenet\': {\n            \'url\': \'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth\',\n            \'input_space\': \'RGB\',\n            \'input_size\': [3, 299, 299],\n            \'input_range\': [0, 1],\n            \'mean\': [0.5, 0.5, 0.5],\n            \'std\': [0.5, 0.5, 0.5],\n            \'num_classes\': 1000,\n            \'scale\': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299\n        }\n    }\n}\n\n\nclass SeparableConv2d(nn.Module):\n    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):\n        super(SeparableConv2d,self).__init__()\n\n        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)\n        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)\n\n    def forward(self,x):\n        x = self.conv1(x)\n        x = self.pointwise(x)\n        return x\n\n\nclass Block(nn.Module):\n    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):\n        super(Block, self).__init__()\n\n        if out_filters != in_filters or strides!=1:\n            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)\n            self.skipbn = nn.BatchNorm2d(out_filters)\n        else:\n            self.skip=None\n\n        self.relu = nn.ReLU(inplace=True)\n        rep=[]\n\n        filters=in_filters\n        if grow_first:\n            rep.append(self.relu)\n            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))\n            rep.append(nn.BatchNorm2d(out_filters))\n            filters = out_filters\n\n        for i in range(reps-1):\n            rep.append(self.relu)\n            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))\n            rep.append(nn.BatchNorm2d(filters))\n\n        if not grow_first:\n            rep.append(self.relu)\n            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))\n            rep.append(nn.BatchNorm2d(out_filters))\n\n        if not start_with_relu:\n            rep = rep[1:]\n        else:\n            rep[0] = nn.ReLU(inplace=False)\n\n        if strides != 1:\n            rep.append(nn.MaxPool2d(3,strides,1))\n        self.rep = nn.Sequential(*rep)\n\n    def forward(self,inp):\n        x = self.rep(inp)\n\n        if self.skip is not None:\n            skip = self.skip(inp)\n            skip = self.skipbn(skip)\n        else:\n            skip = inp\n\n        x+=skip\n        return x\n\n\nclass Xception(nn.Module):\n    """\n    Xception optimized for the ImageNet dataset, as specified in\n    https://arxiv.org/pdf/1610.02357.pdf\n    """\n    def __init__(self, num_classes=1000):\n        """ Constructor\n        Args:\n            num_classes: number of classes\n        """\n        super(Xception, self).__init__()\n        self.num_classes = num_classes\n\n        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)\n        self.bn1 = nn.BatchNorm2d(32)\n        self.relu = nn.ReLU(inplace=True)\n\n        self.conv2 = nn.Conv2d(32,64,3,bias=False)\n        self.bn2 = nn.BatchNorm2d(64)\n        #do relu here\n\n        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)\n        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)\n        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)\n\n        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n\n        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)\n\n        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)\n\n        self.conv3 = SeparableConv2d(1024,1536,3,1,1)\n        self.bn3 = nn.BatchNorm2d(1536)\n\n        #do relu here\n        self.conv4 = SeparableConv2d(1536,2048,3,1,1)\n        self.bn4 = nn.BatchNorm2d(2048)\n\n        self.fc = nn.Linear(2048, num_classes)\n\n        # #------- init weights --------\n        # for m in self.modules():\n        #     if isinstance(m, nn.Conv2d):\n        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels\n        #         m.weight.data.normal_(0, math.sqrt(2. / n))\n        #     elif isinstance(m, nn.BatchNorm2d):\n        #         m.weight.data.fill_(1)\n        #         m.bias.data.zero_()\n        # #-----------------------------\n\n    def features(self, input):\n        x = self.conv1(input)\n        x = self.bn1(x)\n        x = self.relu(x)\n\n        x = self.conv2(x)\n        x = self.bn2(x)\n        x = self.relu(x)\n\n        x = self.block1(x)\n        x = self.block2(x)\n        x = self.block3(x)\n        x = self.block4(x)\n        x = self.block5(x)\n        x = self.block6(x)\n        x = self.block7(x)\n        x = self.block8(x)\n        x = self.block9(x)\n        x = self.block10(x)\n        x = self.block11(x)\n        x = self.block12(x)\n\n        x = self.conv3(x)\n        x = self.bn3(x)\n        x = self.relu(x)\n\n        x = self.conv4(x)\n        x = self.bn4(x)\n        return x\n\n    def logits(self, features):\n        x = self.relu(features)\n\n        x = F.adaptive_avg_pool2d(x, (1, 1))\n        x = x.view(x.size(0), -1)\n        x = self.last_linear(x)\n        return x\n\n    def forward(self, input):\n        x = self.features(input)\n        x = self.logits(x)\n        return x\n\n\ndef xception(num_classes=1000, pretrained=\'imagenet\'):\n    model = Xception(num_classes=num_classes)\n    if pretrained:\n        settings = pretrained_settings[\'xception\'][pretrained]\n        assert num_classes == settings[\'num_classes\'], \\\n            "num_classes should be {}, but is {}".format(settings[\'num_classes\'], num_classes)\n\n        model = Xception(num_classes=num_classes)\n        model.load_state_dict(model_zoo.load_url(settings[\'url\']))\n\n        model.input_space = settings[\'input_space\']\n        model.input_size = settings[\'input_size\']\n        model.input_range = settings[\'input_range\']\n        model.mean = settings[\'mean\']\n        model.std = settings[\'std\']\n\n    # TODO: ugly\n    model.last_linear = model.fc\n    del model.fc\n    return model')


# In[11]:


get_ipython().run_cell_magic('writefile', 'network/models.py', '"""\nAuthor: Andreas Rössler\n"""\nimport os\nimport argparse\n\n\nimport torch\n#import pretrainedmodels\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom network.xception import xception\nimport math\nimport torchvision\n\n\ndef return_pytorch04_xception(pretrained=True):\n    # Raises warning "src not broadcastable to dst" but thats fine\n    model = xception(pretrained=False)\n    if pretrained:\n        # Load model in torch 0.4+\n        model.fc = model.last_linear\n        del model.last_linear\n        state_dict = torch.load(\n            \'/home/ondyari/.torch/models/xception-b5690688.pth\')\n        for name, weights in state_dict.items():\n            if \'pointwise\' in name:\n                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)\n        model.load_state_dict(state_dict)\n        model.last_linear = model.fc\n        del model.fc\n    return model\n\n\nclass TransferModel(nn.Module):\n    """\n    Simple transfer learning model that takes an imagenet pretrained model with\n    a fc layer as base model and retrains a new fc layer for num_out_classes\n    """\n    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):\n        super(TransferModel, self).__init__()\n        self.modelchoice = modelchoice\n        if modelchoice == \'xception\':\n            self.model = return_pytorch04_xception()\n            # Replace fc\n            num_ftrs = self.model.last_linear.in_features\n            if not dropout:\n                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)\n            else:\n                print(\'Using dropout\', dropout)\n                self.model.last_linear = nn.Sequential(\n                    nn.Dropout(p=dropout),\n                    nn.Linear(num_ftrs, num_out_classes)\n                )\n        elif modelchoice == \'resnet50\' or modelchoice == \'resnet18\':\n            if modelchoice == \'resnet50\':\n                self.model = torchvision.models.resnet50(pretrained=True)\n            if modelchoice == \'resnet18\':\n                self.model = torchvision.models.resnet18(pretrained=True)\n            # Replace fc\n            num_ftrs = self.model.fc.in_features\n            if not dropout:\n                self.model.fc = nn.Linear(num_ftrs, num_out_classes)\n            else:\n                self.model.fc = nn.Sequential(\n                    nn.Dropout(p=dropout),\n                    nn.Linear(num_ftrs, num_out_classes)\n                )\n        else:\n            raise Exception(\'Choose valid model, e.g. resnet50\')\n\n    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):\n        """\n        Freezes all layers below a specific layer and sets the following layers\n        to true if boolean else only the fully connected final layer\n        :param boolean:\n        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3\n        :return:\n        """\n        # Stage-1: freeze all the layers\n        if layername is None:\n            for i, param in self.model.named_parameters():\n                param.requires_grad = True\n                return\n        else:\n            for i, param in self.model.named_parameters():\n                param.requires_grad = False\n        if boolean:\n            # Make all layers following the layername layer trainable\n            ct = []\n            found = False\n            for name, child in self.model.named_children():\n                if layername in ct:\n                    found = True\n                    for params in child.parameters():\n                        params.requires_grad = True\n                ct.append(name)\n            if not found:\n                raise Exception(\'Layer not found, cant finetune!\'.format(\n                    layername))\n        else:\n            if self.modelchoice == \'xception\':\n                # Make fc trainable\n                for param in self.model.last_linear.parameters():\n                    param.requires_grad = True\n\n            else:\n                # Make fc trainable\n                for param in self.model.fc.parameters():\n                    param.requires_grad = True\n\n    def forward(self, x):\n        x = self.model(x)\n        return x\n\n\ndef model_selection(modelname, num_out_classes,\n                    dropout=None):\n    """\n    :param modelname:\n    :return: model, image size, pretraining<yes/no>, input_list\n    """\n    if modelname == \'xception\':\n        return TransferModel(modelchoice=\'xception\',\n                             num_out_classes=num_out_classes), 299, \\\n               True, [\'image\'], None\n    elif modelname == \'resnet18\':\n        return TransferModel(modelchoice=\'resnet18\', dropout=dropout,\n                             num_out_classes=num_out_classes), \\\n               224, True, [\'image\'], None\n    else:\n        raise NotImplementedError(modelname)\n\n\nif __name__ == \'__main__\':\n    model, image_size, *_ = model_selection(\'resnet18\', num_out_classes=2)\n    print(model)\n    model = model.cuda()\n    from torchsummary import summary\n    input_s = (3, image_size, image_size)\n    print(summary(model, input_s))')


# In[12]:


metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

def predict_model(video_fn, model,
                  start_frame=0, end_frame=30,
                  plot_every_x_frames = 5):
    """
    Given a video and model, starting frame and end frame.
    Predict on all frames.
    
    """
    fn = video_fn.split('.')[0]
    label = metadata.loc[video_fn]['label']
    original = metadata.loc[video_fn]['original']
    video_path = f'../input/deepfake-detection-challenge/train_sample_videos/{video_fn}'
    output_path = './'
    test_full_image_network(video_path, model, output_path, start_frame=0, end_frame=30, cuda=False)
    # Read output
    vidcap = cv2.VideoCapture(f'{fn}.avi')
    success,image = vidcap.read()
    count = 0
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes = axes.flatten()
    i = 0
    while success:
        # Show every xth frame
        if count % plot_every_x_frames == 0:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[i].imshow(image)
            axes[i].set_title(f'{fn} - frame {count} - true label: {label}')
            axes[i].xaxis.set_visible(False)
            axes[i].yaxis.set_visible(False)
            i += 1
        success,image = vidcap.read()
        count += 1
    plt.tight_layout()
    plt.show()


# In[13]:


model_path = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/full/xception/full_raw.p'
model = torch.load(model_path, map_location=torch.device('cpu'))


# In[14]:


predict_model('bbhtdfuqxq.mp4', model)
predict_model('crezycjqyk.mp4', model)
predict_model('ebchwmwayp.mp4', model)


# In[15]:


model_path_full_c40 = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/full/xception/full_c40.p'
model_full_c40 = torch.load(model_path_full_c40, map_location=torch.device('cpu'))


# In[16]:


predict_model('bbhtdfuqxq.mp4', model_full_c40)
predict_model('crezycjqyk.mp4', model_full_c40)
predict_model('ebchwmwayp.mp4', model_full_c40)


# In[17]:


model_path_full23 = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/full/xception/full_c23.p'
model_full_c23 = torch.load(model_path_full23, map_location=torch.device('cpu'))


# In[18]:


predict_model('bbhtdfuqxq.mp4', model_full_c23)
predict_model('crezycjqyk.mp4', model_full_c23)
predict_model('ebchwmwayp.mp4', model_full_c23)


# In[19]:


model_path_face_allraw = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/face_detection/xception/all_raw.p'
model_face_allraw = torch.load(model_path_face_allraw, map_location=torch.device('cpu'))


# In[20]:


predict_model('bbhtdfuqxq.mp4', model_face_allraw)
predict_model('crezycjqyk.mp4', model_face_allraw)
predict_model('ebchwmwayp.mp4', model_face_allraw)


# In[21]:


model_path_face_all_c40 = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/face_detection/xception/all_c40.p'
model_face_all_c40 = torch.load(model_path_face_all_c40, map_location=torch.device('cpu'))


# In[22]:


predict_model('bbhtdfuqxq.mp4', model_face_all_c40)
predict_model('crezycjqyk.mp4', model_face_all_c40)
predict_model('ebchwmwayp.mp4', model_face_all_c40)


# In[23]:


model, *_ = model_selection(modelname='xception', num_out_classes=2)
model_path_face_allc23 = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/face_detection/xception/all_c23.p'
model_face_all_c23 = torch.load(model_path_face_allc23, map_location=torch.device('cpu'))


# In[24]:


predict_model('bbhtdfuqxq.mp4', model_face_all_c23)
predict_model('crezycjqyk.mp4', model_face_all_c23)
predict_model('ebchwmwayp.mp4', model_face_all_c23)


# In[25]:


def video_file_frame_pred(video_path, model,
                          start_frame=0, end_frame=300,
                          cuda=True, n_frames=5):
    """
    Predict and give result as numpy array
    """
    pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]
    predictions = []
    outputs = []
    # print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1
        if frame_num in pred_frames:
            height, width = image.shape[:2]
            # 2. Detect with dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0]
                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]

                # Actual prediction using our model
                prediction, output = predict_with_model(cropped_face, model,
                                                        cuda=cuda)
                predictions.append(prediction)
                outputs.append(output)
                # ------------------------------------------------------------------
        if frame_num >= end_frame:
            break
    # Figure out how to do this with torch
    preds_np = [x.detach().cpu().numpy()[0][1] for x in outputs]
    if len(preds_np) == 0:
        return predictions, outputs, 0.5, 0.5, 0.5
    try:
        mean_pred = np.mean(preds_np)
    except:
        # couldnt find faces
        mean_pred = 0.5
    min_pred = np.min(preds_np)
    max_pred = np.max(preds_np)
    return predictions, outputs, mean_pred, min_pred, max_pred


# In[26]:


torch.nn.Module.dump_patches = True
model_path_23 = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/face_detection/xception/all_c23.p'
model_23 = torch.load(model_path_23, map_location=torch.device('cpu'))
model_path_raw = '../input/deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/face_detection/xception/all_raw.p'
model_raw = torch.load(model_path_raw, map_location=torch.device('cpu'))


# In[27]:


# Read metadata
metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T

# Predict Fake
for video_fn in tqdm(metadata.query('label == "FAKE"').sample(77).index):
    video_path = f'../input/deepfake-detection-challenge/train_sample_videos/{video_fn}'
    predictions, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(video_path, model_23, n_frames=4, cuda=False)
    metadata.loc[video_fn, 'avg_pred_c23'] = mean_pred
    metadata.loc[video_fn, 'min_pred_c23'] = min_pred
    metadata.loc[video_fn, 'max_pred_c23'] = max_pred
    predictions, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(video_path, model_raw, n_frames=4, cuda=False)
    metadata.loc[video_fn, 'avg_pred_raw'] = mean_pred
    metadata.loc[video_fn, 'min_pred_raw'] = min_pred
    metadata.loc[video_fn, 'max_pred_raw'] = max_pred
    
# Predict Real
for video_fn in tqdm(metadata.query('label == "REAL"').sample(77).index):
    video_path = f'../input/deepfake-detection-challenge/train_sample_videos/{video_fn}'
    predictions, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(video_path, model_23, n_frames=4, cuda=False)
    metadata.loc[video_fn, 'avg_pred_c23'] = mean_pred
    metadata.loc[video_fn, 'min_pred_c23'] = min_pred
    metadata.loc[video_fn, 'max_pred_c23'] = max_pred
    predictions, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(video_path, model_raw, n_frames=4, cuda=False)
    metadata.loc[video_fn, 'avg_pred_raw'] = mean_pred
    metadata.loc[video_fn, 'min_pred_raw'] = min_pred
    metadata.loc[video_fn, 'max_pred_raw'] = max_pred


# In[28]:


preds_df = metadata.dropna(subset=['avg_pred_raw']).copy()
preds_df['label_binary'] = 0
preds_df.loc[preds_df['label'] == "FAKE", 'label_binary'] = 1
preds_df[['min_pred_c23','max_pred_c23',
          'min_pred_raw','max_pred_raw']] = preds_df[['min_pred_c23','max_pred_c23',
                                                      'min_pred_raw','max_pred_raw']].fillna(0.5)
preds_df['naive_pred'] = 0.5
score_avg23 = log_loss(preds_df['label_binary'], preds_df['avg_pred_c23'])
score_min23 = log_loss(preds_df['label_binary'], preds_df['min_pred_c23'])
score_max23 = log_loss(preds_df['label_binary'], preds_df['max_pred_c23'])
score_avgraw = log_loss(preds_df['label_binary'], preds_df['avg_pred_raw'])
score_minraw = log_loss(preds_df['label_binary'], preds_df['min_pred_raw'])
score_maxraw = log_loss(preds_df['label_binary'], preds_df['max_pred_raw'])
score_naive = log_loss(preds_df['label_binary'], preds_df['naive_pred'])
preds_df['max_pred_clipped'] = preds_df['max_pred_c23'].clip(0.4, 1)
score_max_clipped = log_loss(preds_df['label_binary'], preds_df['max_pred_clipped'])
preds_df['max_pred_clipped_raw'] = preds_df['max_pred_raw'].clip(0.4, 1)
score_max_clipped_raw = log_loss(preds_df['label_binary'], preds_df['max_pred_clipped_raw'])
print('Score using average prediction of all frames all_c23.p: {:0.4f}'.format(score_avg23))
print('Score using minimum prediction of all frames all_c23.p: {:0.4f}'.format(score_min23))
print('Score using maximum prediction of all frames all_c23.p: {:0.4f}'.format(score_max23))
print('Score using 0.5 prediction of all frames: {:0.4f}'.format(score_naive))
print('Score using maximum clipped prediction of all frames: {:0.4f}'.format(score_max_clipped))
print('Score using average prediction of all frames all_raw.p: {:0.4f}'.format(score_avgraw))
print('Score using minimum prediction of all frames all_raw.p: {:0.4f}'.format(score_minraw))
print('Score using maximum prediction of all frames all_raw.p: {:0.4f}'.format(score_maxraw))
print('Score using maximum clipped prediction of all frames all_raw.p: {:0.4f}'.format(score_max_clipped_raw))


# In[29]:


fig, ax = plt.subplots(1,1, figsize=(10, 10))
sns.scatterplot(x='avg_pred_c23', y='max_pred_c23', data=metadata.dropna(subset=['avg_pred_c23']), hue='label')
plt.show()


# In[30]:


fig, ax = plt.subplots(1,1, figsize=(10, 10))
sns.scatterplot(x='avg_pred_raw', y='max_pred_raw', data=metadata.dropna(subset=['avg_pred_raw']), hue='label')
plt.show()


# In[31]:


for i, d in metadata.groupby('label'):
    d['avg_pred_c23'].plot(kind='hist', figsize=(15, 5), bins=20, alpha=0.8, title='Average Prediction distribution c23')
    plt.legend(['FAKE','REAL'])
plt.show()
for i, d in metadata.groupby('label'):
    d['max_pred_c23'].plot(kind='hist', figsize=(15, 5), bins=20, title='Max Prediction distribution c23', alpha=0.8)
    plt.legend(['FAKE','REAL'])
plt.show()


# In[32]:


for i, d in metadata.groupby('label'):
    d['avg_pred_raw'].plot(kind='hist',
                           figsize=(15, 5),
                           bins=20,
                           alpha=0.8,
                           title='Average Prediction distribution raw')
    plt.legend(['FAKE','REAL'])
plt.show()
for i, d in metadata.groupby('label'):
    d['max_pred_raw'].plot(kind='hist',
                           figsize=(15, 5),
                           bins=20,
                           title='Max Prediction distribution raw',
                           alpha=0.8)
    plt.legend(['FAKE','REAL'])
plt.show()


# In[33]:


metadata['max_pred_c23'] = metadata['max_pred_c23'].round(6)
metadata.dropna(subset=['max_pred_c23']).sort_values('label')


# In[34]:


metadata['label_binary'] = 0
metadata.loc[metadata['label'] == "FAKE", 'label_binary'] = 1


# In[35]:


import pandas as pd
ss = pd.read_csv('../input/deepfake-detection-challenge/sample_submission.csv')


# In[36]:


for video_fn in tqdm(ss['filename'].unique()):
    video_path = f'../input/deepfake-detection-challenge/test_videos/{video_fn}'
    predictions, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(video_path, model, n_frames=4, cuda=False)
    ss.loc[ss['filename'] == video_fn, 'avg_pred'] = mean_pred
    ss.loc[ss['filename'] == video_fn, 'min_pred'] = min_pred
    ss.loc[ss['filename'] == video_fn, 'max_pred'] = max_pred


# In[37]:


# Use the Maximum frame predicted as "Fake" to be the final prediction
ss['label'] = ss['max_pred'].fillna(0.5).clip(0.4, 0.8)


# In[38]:


ss['label'].plot(kind='hist', figsize=(15, 5), bins=50)
plt.show()


# In[39]:


ss[['filename','label']].to_csv('submission.csv', index=False)
ss.to_csv('submission_min_max.csv', index=False)


# In[40]:


ss.head(20)

