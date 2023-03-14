#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Not necessary but you can get some logs.
get_ipython().system('wandb login 22f6b417e27a2c1fd23e5d45d687926b3f9e3b85')


# In[2]:


import pandas as pd


train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df = pd.concat([test_df, train_df])
df.to_csv("train_test.csv", index=False)


# In[3]:


# If you want TPU.
"""
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!export XLA_USE_BF16=1
"""


# In[4]:


get_ipython().run_cell_magic('time', '', '!python ../input/pytorchtransformers/examples/language-modeling/run_language_modeling.py \\\n--output_dir=fine_tuned_roberta_update                                                     \\\n--model_type=roberta                                                                 \\\n--model_name_or_path=roberta-base                                                     \\\n--do_train                                                                             \\\n--train_data_file=train_test.csv                                                             \\\n--mlm \\\n--num_train_epochs 5')

