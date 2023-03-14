#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, random, zipfile
from importlib import reload  # Python 3
import numpy as np
import shutil
from glob import glob


# In[2]:


#NOTE - notebook assumes data has been unzipped
LABEL_LIST = "breedList.txt" #made a textfile from the list of breeds provided
ID_MAP = "labels.csv"

STORAGE_DIR = "{input dir}"
TRAIN_DIR = STORAGE_DIR + "train/"
VAL_DIR = STORAGE_DIR + "valid/"

SAMPLE_DIR = STORAGE_DIR + "sample/"
SAMPLE_TRAIN = SAMPLE_DIR + "train/"
SAMPLE_VAL = SAMPLE_DIR + "valid/"

RESULTS_DIR = STORAGE_DIR + "results/"


# In[3]:


get_ipython().run_line_magic('cd', '$STORAGE_DIR')

if not os.path.isdir(TRAIN_DIR):
    get_ipython().run_line_magic('mkdir', '$TRAIN_DIR')

if not os.path.isdir(VAL_DIR):
    get_ipython().run_line_magic('mkdir', '$VAL_DIR')
    
if not os.path.isdir(SAMPLE_DIR):
    get_ipython().run_line_magic('mkdir', '$SAMPLE_DIR')
    
if not os.path.isdir(SAMPLE_TRAIN):
    get_ipython().run_line_magic('mkdir', '$SAMPLE_TRAIN')

if not os.path.isdir(SAMPLE_VAL):
    get_ipython().run_line_magic('mkdir', '$SAMPLE_VAL')
    
if not os.path.isdir(RESULTS_DIR):
    get_ipython().run_line_magic('mkdir', '$RESULTS_DIR')


# In[4]:


def processLabels(labels, path):
    get_ipython().run_line_magic('cd', '$path')
    for label in labels:
        if not os.path.isdir(label):
            get_ipython().run_line_magic('mkdir', '$label')


# In[5]:


def selectRandomFiles(num, path):
    if len(os.listdir(path)) < num:
        num = len(os.listdir(path))
        
    selected = []
    while num >= len(selected):
            choice = random.choice(os.listdir(path))
            print("ch: " + choice)
            if choice not in selected:
                selected.append(choice)
    return selected


# In[6]:


def createSampleSet(labels, samplePath, sourcePath, num):
    processLabels(labels, samplePath)
    for label in labels:
        existing = len(os.listdir(samplePath + label + "/"))
        if existing < num:
            files = selectRandomFiles(num - existing, sourcePath + label + "/")
            for file in files:
                src = sourcePath + label + "/" + file
                dest = samplePath + label + "/"
                shutil.copy2(src, dest)


# In[7]:


def removeUnsortedFiles(path):
    files = glob(path + "*.jpg")
    for file in files:
        os.remove(file)


# In[8]:


def createTrainingSet(labels, items):
    processLabels(labels, TRAIN_DIR)
    for item in items:
        file = TRAIN_DIR + item[0] + ".jpg"
        dest = TRAIN_DIR + item[1] + "/" + item[0] + ".jpg"
        if os.path.isfile(file):
            shutil.copy2(file, dest)
    createSampleSet(labels, SAMPLE_TRAIN, TRAIN_DIR, 16)
    removeUnsortedFiles(TRAIN_DIR)


# In[9]:


def createValidationSet(lables):
    processLabels(labels, VAL_DIR)
    for label in labels:
        existing = len(os.listdir(VAL_DIR + label))
        if(existing > 0):
             continue
        files = os.listdir(TRAIN_DIR + label)
        numForValid = int(np.ceil(len(files) * .2))
        for file in files[-numForValid:]:
            src = TRAIN_DIR + label + "/" + file
            dest = VAL_DIR + label + "/" + file
            shutil.copy2(src , dest)
            os.remove(src)
    createSampleSet(labels, SAMPLE_VAL, VAL_DIR, 4)


# In[10]:


def sortData(labels):
    items = np.loadtxt(STORAGE_DIR + ID_MAP, delimiter=',', dtype=str, skiprows=1)
    createTrainingSet(labels, items)
    createValidationSet(labels)


# In[11]:


def getLabels():
    if os.path.isfile(STORAGE_DIR + LABEL_LIST):
        try: 
            f = open(STORAGE_DIR + LABEL_LIST, 'r')
            fileString = f.read()
            return fileString.splitlines()
        except IOError:
            print("Could not read file: ", LABEL_LIST)
    return []


# In[12]:


labels = getLabels()
sortData(labels)

