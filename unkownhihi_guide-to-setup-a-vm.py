#!/usr/bin/env python
# coding: utf-8

# In[1]:


sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-418


# In[2]:


sudo reboot


# In[3]:


nvidia-smi


# In[4]:


sudo apt-get install --no-install-recommends     cuda-10-0     libcudnn7=7.6.4.38-1+cuda10.1      libcudnn7-dev=7.6.4.38-1+cuda10.1
sudo apt-get install -y --no-install-recommends libnvinfer5=6.0.1-1+cuda10.1     libnvinfer-dev=6.0.1-1+cuda10.1


# In[5]:


sudo apt-get install python3-pip
pip3 install tensorflow
pip3 install keras
pip3 install opencv

