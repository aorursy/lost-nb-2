#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def get_product_agg(cols):
    def_train = pd.read_csv('../input/train.csv', usecols=['Semana','Produco_ID']+cols, 
                            dtype={'Semana':'int32',
                                   'Producto':'int32',
                                   'Venta_hoy':'float32',
                                   'Venta_uni_hoy':'int32',
                                   'Dev_uni_proxima':'int32',
                                   'Dev_proxima':'float32',
                                   'Demanda_uni_equil':'int32'})
        agg = df_train.groupby(['Semana','Producto_ID'], as_index=False).agg(['count','sum','min',
                                                                              'median','mean'])
        agg.columns =['_'.join(col).strip() for col in agg.columns.values]
        del(df_train)
        return agg
    


# In[3]:




