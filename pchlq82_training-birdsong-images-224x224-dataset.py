#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from typing import List
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import pylab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


@dataclass
class Project:

    base_dir: Path = Path(".").absolute().parent
    birdsong_imgs_dir = base_dir / "input/birdsong-log-mel-spectrograms"
    fold_0 = birdsong_imgs_dir / "fold_0/fold_0"
    get_n_images = 6
    
proj = Project()


# In[3]:


# Number of images in each fold
for i in range(5):
    print("fold: ", i, "-->", len(list(proj.birdsong_imgs_dir.glob(f"fold_{i}/fold_{i}/*/*.png"))))


# In[4]:


labels = np.unique([i.name for i in list(proj.fold_0.glob("*"))])
random_lables = random.choices(labels, k=proj.get_n_images)

# selecting 6 different labels
label_paths = [random.choice( list(proj.fold_0.glob(f"{label}/*.png")) ) for label in random_lables]
label_paths


# In[5]:


def plot_imgs(imgs_lst: List[Path]=label_paths) -> None:
    plt.figure(figsize=[20,14])
    ncols = 3
    nrows = np.ceil( len(imgs_lst)/ncols ).astype(int)
    for i, im_path in enumerate(imgs_lst):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(im_path.parent.name, fontsize=12)
        ax.imshow(Image.open(im_path))
        ax.set_xlabel("time")
        ax.set_ylabel("Hz")
    
    plt.suptitle('Mel Spectrograms [224x224 pixels]', y=1.05,  fontsize=16)   
    plt.tight_layout()


# In[6]:


plot_imgs()


# In[7]:


get_ipython().run_cell_magic('html', '', "<marquee style='width: 50%; color: blue;'><b>THANKS FOR YOUR ATTENTION!</b></marquee>")

