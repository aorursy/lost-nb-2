#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from IPython.display import display

data_path = '../input'


# In[2]:


get_ipython().system('ls -lSh $data_path/*.csv')


# In[3]:


files_names = get_ipython().getoutput('ls $data_path/*.csv')


# In[4]:


data_dict = {}

for name in files_names:
    data_dict[name.split('/')[-1][:-4]] = pd.read_csv(name)


# In[5]:


for k in data_dict.keys():
    display(k)
    display(data_dict[k].head())


# In[6]:


get_ipython().run_cell_magic('time', '', "df_complete = data_dict['train'].copy()\ndf_complete = df_complete.join(data_dict['potential_energy'].set_index('molecule_name'), on='molecule_name')\ndf_complete = df_complete.join(data_dict['dipole_moments'].set_index('molecule_name'), on='molecule_name', lsuffix='dipole_moments_')\ndf_complete = df_complete.join(data_dict['magnetic_shielding_tensors'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0')\ndf_complete = df_complete.join(data_dict['magnetic_shielding_tensors'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1')\ndf_complete = df_complete.join(data_dict['mulliken_charges'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0')\ndf_complete = df_complete.join(data_dict['mulliken_charges'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1')\ndf_complete = df_complete.join(data_dict['scalar_coupling_contributions'].set_index(['molecule_name', 'atom_index_0', 'atom_index_1']), on=['molecule_name', 'atom_index_0', 'atom_index_1'], rsuffix='_scc')\ndf_complete = df_complete.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_complete = df_complete.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# In[7]:


get_ipython().run_cell_magic('time', '', "df_train = data_dict['train'].copy()\ndf_train = df_train.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_train = df_train.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# In[8]:


get_ipython().run_cell_magic('time', '', "df_test = data_dict['test'].copy()\ndf_test = df_test.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_test = df_test.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# In[9]:


get_ipython().run_cell_magic('time', '', "for df in [df_complete, df_train, df_test]:    \n    distance_foo = np.linalg.norm(df[['x_atom1_structure', 'y_atom1_structure', 'z_atom1_structure']].values - df[['x', 'y', 'z']].values, axis=1)\n    df['distance'] = distance_foo")


# In[10]:


df_complete.to_msgpack('./complete.msg')
df_train.to_msgpack('./train.msg')
df_test.to_msgpack('./test.msg')

