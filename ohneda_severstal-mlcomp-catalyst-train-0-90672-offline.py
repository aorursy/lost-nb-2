#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install mlcomp')


# In[2]:


get_ipython().run_cell_magic('script', 'bash --bg --out script_out', '\n! mlcomp-server start')


# In[3]:


ls ../input/severstal/severstal/configs_kaggle


# In[4]:


cat ../input/severstal/severstal/configs_kaggle/catalyst_kaggle.yml


# In[5]:


cat ../input/severstal/severstal/configs_kaggle/kaggle.yml


# In[6]:


ls ../input/severstal/severstal


# In[7]:


cat ../input/severstal/severstal/executors/preprocess.py


# In[8]:


cat ../input/severstal/severstal/executors/masks.py


# In[9]:


get_ipython().system(' mkdir -p ~/mlcomp/data/severstal')

get_ipython().system(' ln -s /kaggle/input/severstal-steel-defect-detection/ ~/mlcomp/data/severstal/input')
get_ipython().system(' ln -s /kaggle/working/ ~/mlcomp/db')


# In[10]:


get_ipython().system(' sleep 5')
get_ipython().system(' mlcomp dag ../input/severstal/severstal/configs_kaggle/kaggle.yml --params=executors/train/params/data_params/max_count:50 --params=executors/train/params/num_epochs:3')


# In[11]:


from mlcomp.utils.describe import describe, describe_task_names
describe_task_names(dag=1)


# In[12]:


describe(dag=1, metrics=['loss', 'dice'], wait=True, task_with_metric_count=3, fig_size=(10 ,15))


# In[13]:


get_ipython().system(' cp ~/mlcomp/tasks/3/trace.pth unet_resnet34.pth')
get_ipython().system(' cp ~/mlcomp/tasks/4/trace.pth unet_se_resnext50_32x4d.pth')
get_ipython().system(' cp ~/mlcomp/tasks/5/trace.pth unet_mobilenet2.pth')

