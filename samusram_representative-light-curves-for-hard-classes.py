#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
Image("../input/plasstic-intro/first_iter_plasstic.png")


# In[2]:


Image("../input/plasstic-intro/agg_1.png")


# In[3]:


Image("../input/plasstic-intro/later_iter_plasstic.png")


# In[4]:


Image("../input/plasstic-intro/agg_later.png")


# In[5]:


Image("../input/plasstic-intro/partial_match.png")


# In[6]:


Image("../input/plasstic-intro/comparison.png")


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0, 'axes.titlesize':22, 'axes.labelsize': 22, 'lines.linewidth' : 2, 'lines.markersize' : 7})
import os
import pickle
from scipy.stats import sigmaclip
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool, cpu_count
from functools import partial

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
class_to_process = 62
df_meta = df_meta[df_meta['target']==class_to_process]

# dealing with two measurements during the same night based on CPMP's kernel
df['mjd'] += 0.3

# time scale correction based on https://physics.stackexchange.com/questions/172474/redshift-of-supernova-light-curve, https://physics.stackexchange.com/questions/156618/tired-light-red-shift-hypothesis
df = df.merge(df_meta[['object_id', 'hostgal_specz']], on='object_id')
df['mjd'] /= df['hostgal_specz'] + 1

# aggregating measurements from the same night
df['mjd_int'] = df['mjd'].map(lambda x: np.modf(x)[1])
df.sort_values(by=['object_id', 'passband', 'mjd_int'], inplace=True)
df[['object_id_lag', 'passband_lag', 'mjd_int_lag', 'flux_lag', 'flux_err_lag']] = df[
    ['object_id', 'passband', 'mjd_int', 'flux', 'flux_err']].shift(1)
df[['object_id_lead', 'passband_lead', 'mjd_int_lead', 'flux_lead', 'flux_err_lead']] = df[
    ['object_id', 'passband', 'mjd_int', 'flux', 'flux_err']].shift(-1)
previous_flux_the_same_night = (df['object_id'] == df['object_id_lag']) & (df['passband'] == df['passband_lag']) & (
        df['mjd_int'] == df['mjd_int_lag'])
next_flux_the_same_night = (df['object_id'] == df['object_id_lead']) & (df['passband'] == df['passband_lead']) & (
        df['mjd_int'] == df['mjd_int_lead'])
df.loc[previous_flux_the_same_night, 'flux'] = (df.loc[previous_flux_the_same_night, 'flux'] + df.loc[
    previous_flux_the_same_night, 'flux_lag']) / 2
df.loc[previous_flux_the_same_night, 'flux_err'] = (df.loc[previous_flux_the_same_night, 'flux_err'] ** 2 + df.loc[
    previous_flux_the_same_night, 'flux_err_lag'] ** 2).map(np.sqrt)
df = df[~next_flux_the_same_night]
df.drop(
    ['object_id_lag', 'passband_lag', 'mjd_int_lag', 'flux_lag', 'flux_err_lag', 'object_id_lead', 'passband_lead',
     'mjd_int_lead', 'flux_lead', 'flux_err_lead'], axis=1, inplace=True)

first_timestamp_per_object = df.groupby('object_id')['mjd'].agg(lambda x: np.modf(np.min(x))[1])
first_timestamp_per_objects = df['object_id'].map(lambda x: first_timestamp_per_object[x])

# timeorder stand for order of the observation night
df['timeorder'] = df['mjd_int'] - first_timestamp_per_objects

df = df[['object_id', 'passband', 'timeorder', 'flux', 'flux_err']]


# In[9]:


def plot_object(object_id, class_id='', with_errors=True):
    object_df = df[df['object_id'] == object_id]
    min_timeorder = df.loc[(df['object_id'] == object_id) & ~np.isnan(df['flux']), 'timeorder'].min()
    max_timeorder = df.loc[(df['object_id'] == object_id) & ~np.isnan(df['flux']), 'timeorder'].max()
    for passband in range(6):
        plt.figure(figsize=(30,5))
        object_df_passband = object_df[object_df['passband'] == passband]
        if with_errors:
            plt.errorbar(object_df_passband['timeorder'], object_df_passband['flux'], yerr=object_df_passband['flux_err'])
        else:            
            plt.plot(object_df_passband['timeorder'], object_df_passband['flux'])
        plt.scatter(object_df_passband['timeorder'], object_df_passband['flux'])
        plt.xlim(min_timeorder, max_timeorder)
        plt.title(f'object id: {object_id}, passband: {passband}')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')

id_to_check = df['object_id'].unique()[33]        
plot_object(id_to_check)


# In[10]:


def moving_mean_smoothed_aggregations(df, column_names=['flux', 'flux_err'], agg_window_width=10,
                                      exp_decaying_tail_width=7, exp_decaying_const=0.3):
    # imputing anchor timestamps for future aggregations
    existing_ts_entries = set(map(tuple,
                                  df.loc[df['timeorder'] % agg_window_width == 0, ['object_id', 'passband',
                                                                                   'timeorder']].astype(
                                      np.int).drop_duplicates().values.tolist()))

    max_timeorder = df['timeorder'].max()

    triples_to_add = np.array([[object_id, passband, timeorder, np.nan, np.nan]
                               for object_id, passband in df[['object_id', 'passband']].drop_duplicates().values
                               for timeorder in np.arange(0, int(max_timeorder) + 1, agg_window_width)
                               if (object_id, passband, timeorder) not in existing_ts_entries])
    if len(triples_to_add) > 0:
        df_to_append = pd.DataFrame(triples_to_add, columns=df.columns)
        df = df.append(df_to_append, ignore_index=True)
        df.sort_values(by=['object_id', 'passband', 'timeorder'], inplace=True)

    # lag/lead pandas-efficiency-friendly computation
    list_of_shifts = []
    values_counts = [(~df[column_name].isnull()).map(int) for column_name in column_names]
    for shift_step in range(1, agg_window_width + exp_decaying_tail_width + 1):
        shifted_values = df.shift(-1 * shift_step)
        shifted_values.loc[(shifted_values['object_id'] != df['object_id']) | (
                shifted_values['passband'] != df['passband']) |
                           (shifted_values['timeorder'] - df[
                               'timeorder'] >= agg_window_width + exp_decaying_tail_width), column_names] = np.nan
        # if exp_decaying_const != 0:
        decaying_weights = np.exp(-exp_decaying_const *
                                  (shifted_values['timeorder'] - df['timeorder']).map(
                                      lambda x: 0 if x < agg_window_width
                                      else x - agg_window_width + 1))
        for column_name in column_names:
            shifted_values[column_name] *= decaying_weights
        list_of_shifts.append(shifted_values[column_names])

        for i, column_name in enumerate(column_names):
            values_counts[i] += (~shifted_values[column_name].isnull()).map(int) * decaying_weights.fillna(1)

    for shift_step in range(-exp_decaying_tail_width, 0):
        shifted_values = df.shift(-shift_step)
        shifted_values.loc[(shifted_values['object_id'] != df['object_id']) | (
                shifted_values['passband'] != df['passband']) | (
                                   shifted_values['timeorder'] - df[
                               'timeorder'] < -exp_decaying_tail_width), column_names] = np.nan
        decaying_weights = np.exp(exp_decaying_const * (shifted_values['timeorder'] - df['timeorder']))
        for column_name in column_names:
            shifted_values[column_name] *= decaying_weights
        list_of_shifts.append(shifted_values[column_names])
        for i, column_name in enumerate(column_names):
            values_counts[i] += (~shifted_values[column_name].isnull()).map(int) * decaying_weights.fillna(1)

    df[column_names] = df[column_names].fillna(0)
    for shifted_series in list_of_shifts:
        for column_name in column_names:
            df[column_name] += shifted_series[column_name].fillna(0)
    for i, column_name in enumerate(column_names):
        df.loc[df[column_name] == 0, column_name] = np.nan
        df[column_name] /= values_counts[i]

    df = df[df['timeorder'] % agg_window_width == 0]
    df['timeorder'] /= agg_window_width
    return df

df = moving_mean_smoothed_aggregations(df, agg_window_width=7, exp_decaying_tail_width=3)
plot_object(id_to_check)


# In[11]:


flux_err_sum_per_passband = df.groupby(['object_id', 'passband'])['flux_err'].agg(np.nansum)
df['flux_abs'] = df['flux'].map(abs)
flux_sum_per_passband = df.groupby(['object_id', 'passband'])['flux_abs'].agg(np.nansum)

flux_err_sum_per_passband = flux_err_sum_per_passband.reset_index()
flux_sum_per_passband = flux_sum_per_passband.reset_index()
flux_sum_per_passband['flux_err_rel_sum'] = flux_err_sum_per_passband['flux_err']/flux_sum_per_passband['flux_abs']

max_per_object_passband = df.groupby(['object_id', 'passband'])['flux_abs'].max().reset_index()
max_per_object_passband.columns = ['object_id', 'passband', 'flux_abs_max']
max_per_object_passband['flux_object_abs_max'] = max_per_object_passband['object_id'].map(
    max_per_object_passband.groupby('object_id')['flux_abs_max'].max())
max_per_object_passband['passband_to_total_max_ration'] = max_per_object_passband['flux_abs_max']/max_per_object_passband['flux_object_abs_max']
max_per_object_passband['object_id_passband'] = max_per_object_passband['object_id'].map(str) + '_' + max_per_object_passband['passband'].map(str)
max_per_object_passband_map = max_per_object_passband.set_index('object_id_passband')['passband_to_total_max_ration']
flux_sum_per_passband['object_id_passband'] = flux_sum_per_passband['object_id'].map(str) + '_' + flux_sum_per_passband['passband'].map(str)
flux_sum_per_passband_map = flux_sum_per_passband.set_index('object_id_passband')['flux_err_rel_sum']

accepted_flux_err_rel_sum_per_object = flux_sum_per_passband.groupby(['object_id'])['flux_err_rel_sum'].agg(lambda x: sorted(x)[2])
accepted_passband_to_total_max_ration_per_object = max_per_object_passband.groupby(['object_id'])['passband_to_total_max_ration'].agg(lambda x: sorted(x,key=lambda i: -i)[3])

flux_sum_per_passband = flux_sum_per_passband[
    (flux_sum_per_passband['flux_err_rel_sum'] <= flux_sum_per_passband['object_id'].map(accepted_flux_err_rel_sum_per_object)) & (
    (flux_sum_per_passband['object_id'].map(str) + '_' + flux_sum_per_passband['passband'].map(str)).map(max_per_object_passband_map) >= flux_sum_per_passband['object_id'].map(accepted_passband_to_total_max_ration_per_object))]

reliable_passbands_per_object = flux_sum_per_passband.groupby('object_id')['passband'].agg(set)
object_passband_max_flux = max_per_object_passband[['flux_abs_max', 'object_id', 'passband']]
object_passband_max_flux['object_id_passband'] = object_passband_max_flux['object_id'].map(str) + '_' + object_passband_max_flux['passband'].map(str)
object_passband_max_flux['flux_abs_max_sqrt'] = object_passband_max_flux['flux_abs_max'].map(np.sqrt)
object_passband_max_flux = object_passband_max_flux.set_index('object_id_passband')['flux_abs_max_sqrt']

reliable_passbands = df['object_id'].map(reliable_passbands_per_object)
reliable_passbands_without_current_passband = reliable_passbands - df['passband'].map(lambda x: {x})
reliable_passbands_entries = reliable_passbands_without_current_passband.map(len) != reliable_passbands.map(len)


# In[12]:


max_per_object_passband = df.groupby(['object_id', 'passband'])['flux_abs'].max().reset_index()
max_per_object_passband.columns = ['object_id', 'passband', 'flux_abs_max']
max_per_object_passband['object_id_passband'] = max_per_object_passband['object_id'].map(str) + '_' + max_per_object_passband['passband'].map(str)
max_per_object_passband_map = max_per_object_passband[['object_id_passband', 'flux_abs_max']].set_index('object_id_passband')['flux_abs_max']
df_object_id_passband = df['object_id'].map(str) + '_' + df['passband'].map(str)
max_flux_per_object_passband = df_object_id_passband.map(max_per_object_passband_map)
df['flux'] /= max_flux_per_object_passband


# In[13]:


object_passband = df['object_id'].map(str) + '_' + df['passband'].map(str)
df['passband_max_flux_sqrt'] = object_passband.map(max_per_object_passband_map)/(1 + object_passband.map(flux_sum_per_passband_map))
df['flux_weight'] = list(zip(df['flux'], df['passband_max_flux_sqrt']))

def weighted_mean(timestamp_values):
    non_nans = list(filter(lambda x: not np.isnan(x[0]), timestamp_values))
    if len(non_nans) == 0:
        return np.nan
    values = list(map(lambda x: x[0], non_nans))
    if len(values) > 2:
        mean = np.mean(values)
        std = np.std(values)
        lb = mean - std
        ub = mean + std
        non_nans = list(filter(lambda x: x[0] >= lb and x[0] <= ub, non_nans))
    elif len(values) == 2:
        val_range = max(values) - min(values)
        if val_range > 0.5:
            return np.nan
    elif len(values) == 1:
        return values[0]
    
    if len(non_nans) == 0:
        return np.nan
    
    values = list(map(lambda x: x[0], non_nans))
    weights = list(map(lambda x: x[1], non_nans))
    result = np.average(values, weights=weights)
    return result
    
df = df.groupby(['object_id', 'timeorder'], as_index=False)['flux_weight'].agg(weighted_mean)
df.columns = ['object_id', 'timeorder', 'flux']

# omitting tailing missing values
max_timeorder_per_object = df[~df['flux'].isnull()].groupby(['object_id'])['timeorder'].max()
min_timeorder_per_object = df[~df['flux'].isnull()].groupby(['object_id'])['timeorder'].min()
df = df[(df['timeorder'] <= df['object_id'].map(max_timeorder_per_object)) & 
        (df['timeorder'] >= df['object_id'].map(min_timeorder_per_object))]


# In[14]:


def plot_agg_light_curve(object_id=None, object_df=None, class_id='', save=False):
    assert object_id is not None or object_df is not None
    if object_df is None:
        object_df = df[df['object_id']==object_id]
    else:
        object_id = object_df['object_id'].iloc[0]
    plt.figure(figsize=(30,5))
    plt.plot(object_df['timeorder'], object_df['flux'])
    plt.scatter(object_df['timeorder'], object_df['flux'])
    plt.title(f'object id: {object_id}', fontsize=30)
    if save:
        if not os.path.exists(f'../input/class_{class_to_proceed}'):
            os.makedirs(f'../input/class_{class_to_proceed}')
        plt.savefig(f'../input/class_{class_to_proceed}/{object_id}.png')
    
plot_agg_light_curve(id_to_check)


# In[15]:


# object_ids_with_full_pick = [int(name.split('.')[0]) for name in os.listdir(f'../input/class_{class_to_process}_init_base')]
with open('../input/plasticcwip/object_ids_with_full_pick_class_62.pkl', 'rb') as f:
    object_ids_with_full_pick = pickle.load(f)
object_ids_class_train = set(df_meta.loc[(df_meta['target'] == class_to_process), 'object_id'].values)
object_ids_with_full_pick_train = [object_id for object_id in object_ids_with_full_pick if object_id in object_ids_class_train]


# In[16]:


def rob_mean(timestamp_values):
        values = list(filter(lambda x: not np.isnan(x), timestamp_values))
        if len(values) == 1:
            return values[0]
        if len(values) == 0:
            return np.nan
        if len(values) > 2:
            mean = np.mean(values)
            std = np.std(values)
            lb = mean - std
            ub = mean + std
            values = list(filter(lambda x: x >= lb and x <= ub, values))
        elif len(values) == 2:
            val_range = max(values) - min(values)
            if val_range > 0.5:
                return np.nan
        if len(values) == 0:
            return np.nan
        result = np.mean(values)
        return result
    
def shift_array_and_vector(ts_1_input, ts_collection, shift_0, original_ts_collection_region=False):
    # shift = 'ts_collection' - 'ts_1'
    if not original_ts_collection_region:
        if shift_0 < 0:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((ts_1_input,
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
                ts_collection = np.concatenate((np.full((-shift_0, ts_collection.shape[1]), np.nan),
                                                ts_collection,
                                                np.full((max(0, len(ts_1_input) - ts_collection.shape[0] + shift_0),
                                                         ts_collection.shape[1]), np.nan)), axis=0)
            else:
                ts_1_shifted_0 = np.concatenate((ts_1_input, np.full(-shift_0 - len(ts_1_input) + ts_collection.shape[0], np.nan)))
                ts_collection = np.concatenate((np.full((-shift_0, ts_collection.shape[1]), np.nan), ts_collection), axis=0)
        else:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan), ts_1_input))
                ts_collection = np.concatenate((ts_collection,
                                                np.full((shift_0 + len(ts_1_input) - ts_collection.shape[0], ts_collection.shape[1]), np.nan)), axis=0)
            else:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input,
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
                ts_collection = np.concatenate((ts_collection, np.full((max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0]),
                                                                        ts_collection.shape[1]), np.nan)), axis=0)
    else:
        if shift_0 < 0:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((ts_1_input[-shift_0: len(ts_1_input) - max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0])],
                                                np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
            else:
                ts_1_shifted_0 = np.concatenate((ts_1_input[-shift_0:],
                                                 np.full(-shift_0 - len(ts_1_input) + ts_collection.shape[0], np.nan)))
        else:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input[:-shift_0 - len(ts_1_input) + ts_collection.shape[0]]))
            else:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input[:len(ts_1_input) - max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0])],
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))

    return ts_1_shifted_0, ts_collection

def match_full_picks(ts_1_input, ts_collection, debug=False, iteration=''):
    
    ts_2_input = np.apply_along_axis(rob_mean, 1, ts_collection)

    ts_1 = np.nan_to_num(ts_1_input)
    ts_2 = np.nan_to_num(ts_2_input)
    idx_max_ts_1 = np.argmax(ts_1)
    idx_max_ts_2 = np.argmax(ts_2)
        
    shift_0 = idx_max_ts_2 - idx_max_ts_1
    if shift_0 < 0:
        if len(ts_1) > len(ts_2):
            ts_1_shifted_0 = np.concatenate((ts_1, np.zeros(max(0, -shift_0 - (len(ts_1) - len(ts_2))))))
            ts_2_shifted_0 = np.concatenate((np.zeros(-shift_0), ts_2, np.zeros(max(0, len(ts_1) - len(ts_2) + shift_0))))
        else:            
            ts_1_shifted_0 = np.concatenate((ts_1, np.zeros(-shift_0 - len(ts_1) + len(ts_2))))
            ts_2_shifted_0 = np.concatenate((np.zeros(-shift_0), ts_2))
    else:
        if len(ts_1) > len(ts_2):
            ts_1_shifted_0 = np.concatenate((np.zeros(shift_0), ts_1))
            ts_2_shifted_0 = np.concatenate((ts_2, np.zeros(shift_0 + len(ts_1) - len(ts_2))))
        else:            
            ts_1_shifted_0 = np.concatenate((np.zeros(shift_0), ts_1, np.zeros(max(0, -shift_0 - len(ts_1) + len(ts_2)))))
            ts_2_shifted_0 = np.concatenate((ts_2, np.zeros(max(0, shift_0 + len(ts_1) - len(ts_2)))))
        
    if debug:
        plot_args = list(range(len(ts_1_shifted_0)))
        plt.figure(figsize=(20,4))
        plt.plot(plot_args, ts_2_shifted_0)
        plt.plot(plot_args, ts_1_shifted_0)
        plt.scatter(plot_args, ts_2_shifted_0)
        plt.scatter(plot_args, ts_1_shifted_0)
        plt.title(f'Class representative so far + next curve')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.legend(['class representative', 'light curve'])
        
    min_msa = mean_squared_error(ts_1_shifted_0, ts_2_shifted_0)
    min_shift_added = 0
    ts_1_shifted, ts_2_shifted = ts_1_shifted_0, ts_2_shifted_0
    for shift_added in range(1, 6):
        ts_1_shifted = np.concatenate((np.zeros(shift_added), ts_1_shifted))
        ts_2_shifted = np.concatenate((ts_2_shifted, np.zeros(shift_added)))
        msa = mean_squared_error(ts_1_shifted, ts_2_shifted)
        if msa < min_msa:
            min_msa = msa
            min_shift_added = shift_added
            
    ts_1_shifted, ts_2_shifted = ts_1_shifted_0, ts_2_shifted_0  
    for shift_added in range(-5, 0):
        ts_1_shifted = np.concatenate((ts_1_shifted, np.zeros(-shift_added)))
        ts_2_shifted = np.concatenate((np.zeros(-shift_added), ts_2_shifted))
        msa = mean_squared_error(ts_1_shifted, ts_2_shifted)
        if msa < min_msa:
            min_msa = msa
            min_shift_added = shift_added
            
    shift_0 += min_shift_added  
    
    ts_1_shifted_0, ts_collection = shift_array_and_vector(ts_1_input, ts_collection, shift_0, original_ts_collection_region=True)
    
    ts_collection = np.concatenate((ts_collection, ts_1_shifted_0.reshape(-1, 1)), axis=1)
    
    if debug:
        ts_final = np.apply_along_axis(rob_mean, 1, ts_collection)

        plt.figure(figsize=(20,4))
        plt.plot(range(len(ts_final)), ts_final)
        plt.scatter(range(len(ts_final)), ts_final)        
        plt.title(f'Class {class_to_process} representative after {iteration} iterations')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.figure(figsize=(20,4))
        plt.plot(range(len(ts_final)), np.ones(len(ts_final)))
    return ts_1_shifted_0, ts_collection


# In[17]:


def rob_mean(timestamp_values):
        values = list(filter(lambda x: not np.isnan(x), timestamp_values))
        if len(values) == 1:
            return values[0]
        if len(values) == 0:
            return np.nan
        if len(values) > 2:
            mean = np.mean(values)
            std = np.std(values)
            lb = mean - std
            ub = mean + std
            values = list(filter(lambda x: x >= lb and x <= ub, values))
        elif len(values) == 2:
            val_range = max(values) - min(values)
            if val_range > 0.5:
                return np.nan
        if len(values) == 0:
            return np.nan
        result = np.mean(values)
        return result
    
def shift_array_and_vector(ts_1_input, ts_collection, shift_0, original_ts_collection_region=False):
    # shift = 'ts_collection' - 'ts_1'
    if not original_ts_collection_region:
        if shift_0 < 0:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((ts_1_input,
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
                ts_collection = np.concatenate((np.full((-shift_0, ts_collection.shape[1]), np.nan),
                                                ts_collection,
                                                np.full((max(0, len(ts_1_input) - ts_collection.shape[0] + shift_0),
                                                         ts_collection.shape[1]), np.nan)), axis=0)
            else:
                ts_1_shifted_0 = np.concatenate((ts_1_input, np.full(-shift_0 - len(ts_1_input) + ts_collection.shape[0], np.nan)))
                ts_collection = np.concatenate((np.full((-shift_0, ts_collection.shape[1]), np.nan), ts_collection), axis=0)
        else:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan), ts_1_input))
                ts_collection = np.concatenate((ts_collection,
                                                np.full((shift_0 + len(ts_1_input) - ts_collection.shape[0], ts_collection.shape[1]), np.nan)), axis=0)
            else:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input,
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
                ts_collection = np.concatenate((ts_collection, np.full((max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0]),
                                                                        ts_collection.shape[1]), np.nan)), axis=0)
    else:
        if shift_0 < 0:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((ts_1_input[-shift_0: len(ts_1_input) - max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0])],
                                                np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))
            else:
                ts_1_shifted_0 = np.concatenate((ts_1_input[-shift_0:],
                                                 np.full(-shift_0 - len(ts_1_input) + ts_collection.shape[0], np.nan)))
        else:
            if len(ts_1_input) > ts_collection.shape[0]:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input[:-shift_0 - len(ts_1_input) + ts_collection.shape[0]]))
            else:
                ts_1_shifted_0 = np.concatenate((np.full(shift_0, np.nan),
                                                 ts_1_input[:len(ts_1_input) - max(0, shift_0 + len(ts_1_input) - ts_collection.shape[0])],
                                                 np.full(max(0, -shift_0 - len(ts_1_input) + ts_collection.shape[0]), np.nan)))

    return ts_1_shifted_0, ts_collection

def match_full_picks(ts_1_input, ts_collection, debug=False, iteration=''):
    
    ts_2_input = np.apply_along_axis(rob_mean, 1, ts_collection)

    ts_1 = np.nan_to_num(ts_1_input)
    ts_2 = np.nan_to_num(ts_2_input)
    idx_max_ts_1 = np.argmax(ts_1)
    idx_max_ts_2 = np.argmax(ts_2)
        
    shift_0 = idx_max_ts_2 - idx_max_ts_1
    if shift_0 < 0:
        if len(ts_1) > len(ts_2):
            ts_1_shifted_0 = np.concatenate((ts_1, np.zeros(max(0, -shift_0 - (len(ts_1) - len(ts_2))))))
            ts_2_shifted_0 = np.concatenate((np.zeros(-shift_0), ts_2, np.zeros(max(0, len(ts_1) - len(ts_2) + shift_0))))
        else:            
            ts_1_shifted_0 = np.concatenate((ts_1, np.zeros(-shift_0 - len(ts_1) + len(ts_2))))
            ts_2_shifted_0 = np.concatenate((np.zeros(-shift_0), ts_2))
    else:
        if len(ts_1) > len(ts_2):
            ts_1_shifted_0 = np.concatenate((np.zeros(shift_0), ts_1))
            ts_2_shifted_0 = np.concatenate((ts_2, np.zeros(shift_0 + len(ts_1) - len(ts_2))))
        else:            
            ts_1_shifted_0 = np.concatenate((np.zeros(shift_0), ts_1, np.zeros(max(0, -shift_0 - len(ts_1) + len(ts_2)))))
            ts_2_shifted_0 = np.concatenate((ts_2, np.zeros(max(0, shift_0 + len(ts_1) - len(ts_2)))))
        
    if debug:
        plot_args = list(range(len(ts_1_shifted_0)))
        plt.figure(figsize=(20,4))
        plt.plot(plot_args, ts_2_shifted_0)
        plt.plot(plot_args, ts_1_shifted_0)
        plt.scatter(plot_args, ts_2_shifted_0)
        plt.scatter(plot_args, ts_1_shifted_0)
        plt.title(f'Class representative so far + next curve')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.legend(['class representative', 'light curve'])
        
    min_msa = mean_squared_error(ts_1_shifted_0, ts_2_shifted_0)
    min_shift_added = 0
    ts_1_shifted, ts_2_shifted = ts_1_shifted_0, ts_2_shifted_0
    for shift_added in range(1, 6):
        ts_1_shifted = np.concatenate((np.zeros(shift_added), ts_1_shifted))
        ts_2_shifted = np.concatenate((ts_2_shifted, np.zeros(shift_added)))
        msa = mean_squared_error(ts_1_shifted, ts_2_shifted)
        if msa < min_msa:
            min_msa = msa
            min_shift_added = shift_added
            
    ts_1_shifted, ts_2_shifted = ts_1_shifted_0, ts_2_shifted_0  
    for shift_added in range(-5, 0):
        ts_1_shifted = np.concatenate((ts_1_shifted, np.zeros(-shift_added)))
        ts_2_shifted = np.concatenate((np.zeros(-shift_added), ts_2_shifted))
        msa = mean_squared_error(ts_1_shifted, ts_2_shifted)
        if msa < min_msa:
            min_msa = msa
            min_shift_added = shift_added
            
    shift_0 += min_shift_added  
    
    ts_1_shifted_0, ts_collection = shift_array_and_vector(ts_1_input, ts_collection, shift_0, original_ts_collection_region=True)
    
    ts_collection = np.concatenate((ts_collection, ts_1_shifted_0.reshape(-1, 1)), axis=1)
    
    if debug:
        ts_final = np.apply_along_axis(rob_mean, 1, ts_collection)

        plt.figure(figsize=(20,4))
        plt.plot(range(len(ts_final)), ts_final)
        plt.scatter(range(len(ts_final)), ts_final)        
        plt.title(f'Class {class_to_process} representative after {iteration} iterations')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.figure(figsize=(20,4))
        plt.plot(range(len(ts_final)), np.ones(len(ts_final)))
    return ts_1_shifted_0, ts_collection

# the matching
base_prototype = df.loc[df['object_id']==object_ids_with_full_pick_train[0], 'flux'].values.reshape(-1, 1)

for i, object_id in enumerate(object_ids_with_full_pick_train[1:]):
    _, base_prototype = match_full_picks(df.loc[df['object_id']==object_id, 'flux'].values, base_prototype, debug=i%40==0, iteration=i+1)


# In[18]:


base_prototype = base_prototype[37:67,:]
final_base_prototype_ts = np.apply_along_axis(rob_mean, 1, base_prototype)
plt.figure(figsize=(20,4))
plt.plot(range(len(final_base_prototype_ts)), final_base_prototype_ts)
plt.scatter(range(len(final_base_prototype_ts)), final_base_prototype_ts)
plt.xlabel('timestamp order')
plt.ylabel('flux')
plt.title(f'Class {class_to_process} representative light curve')


# In[19]:


object_ids_with_full_pick_train = set(object_ids_with_full_pick_train)
object_ids_without_full_pick_train = [object_id for object_id in object_ids_class_train if object_id not in object_ids_with_full_pick_train]

def squared_error_sum(x, y):
    return np.sum([(x[i] - y[i])**2 for i in range(len(x))])

def match_partials(ts_1_input, ts_collection, debug=False, iteration=''):
    
    ts_2_input = np.apply_along_axis(rob_mean, 1, ts_collection)    
    ts_2 = np.nan_to_num(ts_2_input)
    
    squared_error_min = float('inf')
    scale_factor_opt = -1
    ts_1_pos_under_ts_2_start_opt = -1
    
    for scale_factor in np.arange(0.1,1.01,0.1):
        ts_1 = scale_factor*ts_1_input
            
        sum_squares_current = np.nansum(ts_1[:-4]**2)
        sum_squares_total = np.nansum(ts_1**2)
        
        # shifting a ts_1_input, abstracting ts_2 as a solid, not moving part  
        for ts_1_pos_under_ts_2_start in range(len(ts_1) - 5, -(len(ts_2) - 5), -1):
            ts_1_intersection_part = ts_1[max(0, ts_1_pos_under_ts_2_start):
                                          min(len(ts_1), 
                                              len(ts_2) + ts_1_pos_under_ts_2_start)]
            ts_2_intersection_part = ts_2[max(0, -ts_1_pos_under_ts_2_start): 
                                          min(len(ts_2), 
                                             len(ts_1) - ts_1_pos_under_ts_2_start)]         
                            
            mask = ~np.isnan(ts_1_intersection_part)
            
            squared_error_intersection = squared_error_sum(ts_1_intersection_part[mask], ts_2_intersection_part[mask])
            
            # managing non-intersection part of ts_1
            if ts_1_pos_under_ts_2_start > 0:
                val = ts_1[ts_1_pos_under_ts_2_start]**2 if not np.isnan(ts_1[ts_1_pos_under_ts_2_start]) else 0
                sum_squares_current -= val
                
            if len(ts_1) - ts_1_pos_under_ts_2_start > len(ts_2):
                val = ts_1[len(ts_2) + ts_1_pos_under_ts_2_start]**2 if not np.isnan(ts_1[len(ts_2) + ts_1_pos_under_ts_2_start]) else 0
                sum_squares_current += val
                
            squared_error_total = (squared_error_intersection + sum_squares_current)/sum_squares_total
            if squared_error_total < squared_error_min:
                squared_error_min = squared_error_total
                scale_factor_opt = scale_factor
                ts_1_pos_under_ts_2_start_opt = ts_1_pos_under_ts_2_start
                
    ts_1_input *= scale_factor_opt
    
    ts_1_shifted_0, ts_collection = shift_array_and_vector(ts_1_input, ts_collection, -ts_1_pos_under_ts_2_start_opt, 
                                                           original_ts_collection_region=True)
    t2_shifted = np.apply_along_axis(rob_mean, 1, ts_collection)
                
    if debug:
        plt.figure(figsize=(20,4))
        plt.plot(range(len(t2_shifted)), t2_shifted)
        plt.plot(range(len(ts_1_shifted_0)), ts_1_shifted_0)
        plt.scatter(range(len(t2_shifted)), t2_shifted)
        plt.scatter(range(len(ts_1_shifted_0)), ts_1_shifted_0)
        plt.title(f'Class representative so far + next curve')
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.legend(['class representative', 'light curve'])
    
    ts_collection = np.concatenate((ts_collection, ts_1_shifted_0.reshape(-1, 1)), axis=1)
    
    if debug:
        ts_final = np.apply_along_axis(rob_mean, 1, ts_collection)
        plt.figure(figsize=(20,4))
        plt.plot(range(len(ts_final)), ts_final)
        plt.scatter(range(len(ts_final)), ts_final)  
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.title(f'Class {class_to_process} representative after {iteration} iterations')
    return ts_collection

for i, object_id in enumerate(object_ids_without_full_pick_train):
    base_prototype = match_partials(df.loc[df['object_id']==object_id, 'flux'].values, base_prototype, debug=i%70==0, iteration=i)


# In[20]:


final_prototype_ts = np.apply_along_axis(rob_mean, 1, base_prototype)
plt.figure(figsize=(20,4))
plt.plot(range(len(final_prototype_ts)), final_prototype_ts)
plt.scatter(range(len(final_prototype_ts)), final_prototype_ts)
plt.xlabel('timestamp order')
plt.ylabel('flux')
plt.title(f'Final class {class_to_process} representative light curve')


# In[21]:


fold_idx = 0
classes = [42, 52, 62, 67, 90, 6, 15, 95, 64]
class_prototypes = dict()
for class_name in classes:
    with open(f'../input/plasticcwip/final_prototype_ts_class_{class_name}_fold_{fold_idx}.pkl', 'rb') as f:
        plt.figure(figsize=(20,4))
        final_prototype_ts = pickle.load(f)
        final_prototype_ts, base_prototype = match_full_picks(final_prototype_ts, base_prototype)
        class_prototypes[class_name] = np.nan_to_num(final_prototype_ts)
        plt.plot(range(len(final_prototype_ts)), final_prototype_ts)
        plt.scatter(range(len(final_prototype_ts)), final_prototype_ts)
        plt.xlabel('timestamp order')
        plt.ylabel('flux')
        plt.title(f'Final {class_name} class representative light curve')


# In[22]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[23]:


get_ipython().run_cell_magic('cython', '', '\ncimport numpy as cnp\nfrom numpy cimport ndarray\nctypedef unsigned char uint8\nimport numpy as np\nimport matplotlib.pyplot as plt\n\ncnp.import_array()\ncdef double[:,:] return_empty_2d(int dim1, int dim2):\n    cdef cnp.npy_intp* dims = [dim1, dim2]\n    return cnp.PyArray_SimpleNew(2, dims, cnp.NPY_DOUBLE)\n\ncdef double[:] return_empty_1d_double(int dim):\n    cdef cnp.npy_intp dims = dim\n    return cnp.PyArray_SimpleNew(1, &dims, cnp.NPY_DOUBLE)\n\ncdef int[:] return_empty_1d_int(int dim):\n    cdef cnp.npy_intp dims = dim\n    return cnp.PyArray_SimpleNew(1, &dims, cnp.NPY_INT)\n\ncdef extern from "math.h":\n    double hypot(double x, double y)\n\ncdef extern from "math.h":\n    double INFINITY\n\ncdef extern from "math.h":\n    bint isnan(double x)\n\ncdef extern from "math.h":\n    double fabs(double x)\n\ncdef int INT_NAN = -999\ncdef int BIG_M_SEQ_LEN = 10000 # no sequence can be longer\ncdef double NOISE_UB = 0.04\ncdef double NOISE_LB = -0.01\ncdef int INITIAL_INTERSECTION_WIDTH = 5\n\n#cdef double weight_direction_error = 70\n\n\ncdef double[:] squared_error_sum(double[:] pred, double[:] true):\n    cdef int N = pred.shape[0]\n    cdef double sum_err = 0\n    cdef double sum_sq_true = 0\n    cdef int i\n    for i in range(N):\n        if not isnan(pred[i]): # prototypes have no missing values, therefore there is no need to check y\n            sum_err += (pred[i] - true[i])**2\n            sum_sq_true += true[i]**2\n    cdef double[:] result = return_empty_1d_double(2)\n    result[0] = sum_err\n    result[1] = sum_sq_true\n    return result\n\ncdef double[:,:] get_segments(double[:,:] ts_points, NUMBER_OF_POINTS_TS_1=-1):\n    cdef int N\n    if NUMBER_OF_POINTS_TS_1 == -1:\n        N = ts_points.shape[0] - 1\n    else:\n        N = NUMBER_OF_POINTS_TS_1 - 1\n\n    if N < 0:\n        return return_empty_2d(0, 2)\n    cdef double[:,:] segments = return_empty_2d(N, 2)\n    cdef int i\n    for i in range(N):\n        segments[i, 0] = ts_points[i + 1, 0] - ts_points[i, 0]\n        segments[i, 1] = ts_points[i + 1, 1] - ts_points[i, 1]\n    return segments\n\n\ncdef double cos_dist(double x1, double y1, double x2, double y2):\n    cdef double vector_product = x1 * x2 + y1 * y2\n    cdef double cos = vector_product / (hypot(x1, y1) * hypot(x2, y2))\n    return 1 - cos\n\ncdef double cos_dist_ts_to_be_masked(double[:,:] segments_ts_1_intersection, double[:] ts_2_intersection_part, double[:] ts_1_intersection_part):\n    if len(segments_ts_1_intersection) == 0:\n        return 0\n    cdef int number_of_points = len(segments_ts_1_intersection) + 1\n    cdef double[:,:] ts_2_points = return_empty_2d(number_of_points, 2)\n    cdef int current_point_idx = 0\n    cdef int i\n    cdef int N = ts_1_intersection_part.shape[0]\n    for i in range(N):\n        if not isnan(ts_1_intersection_part[i]):\n            ts_2_points[current_point_idx,0] = i\n            ts_2_points[current_point_idx,1] = ts_2_intersection_part[i]\n            current_point_idx += 1\n    cdef double[:,:] ts_2_segments = get_segments(ts_2_points)\n    cdef double cos_dist_sum = 0\n    for i in range(len(segments_ts_1_intersection)):\n        cos_dist_sum += cos_dist(segments_ts_1_intersection[i,0], segments_ts_1_intersection[i,1], ts_2_segments[i,0], ts_2_segments[i,1])\n    return cos_dist_sum\n\n\ndef estimate_scale(double[:] ts_1_input, double[:] ts_2, int debug=1, class_name=\'\', time_scale=\'\'):\n\n    cdef double error_sq_min = INFINITY\n    cdef double error_cos_min = INFINITY\n    cdef double scale_factor_opt_sq = -1.0\n    cdef double scale_factor_opt_cos = -1.0\n    cdef double[:] scale_factor_array = np.arange(0.1, 1.001, 0.1)\n    cdef int LEN_TS_1 = len(ts_1_input)\n    cdef int LEN_TS_2 = len(ts_2)\n    cdef double[:] ts_1 = return_empty_1d_double(LEN_TS_1)\n    cdef double[:] ts_1_points_orig = return_empty_1d_double(LEN_TS_1)\n    cdef double[:,:] ts_1_points = return_empty_2d(LEN_TS_1, 2)\n    cdef double[:,:] ts_1_segments, ts_1_segment_values, segments_ts_1_before_intersection, segments_ts_1_after_intersection, segments_ts_1_non_intersection\n    cdef double[:] segment_base_cos_dists = return_empty_1d_double(LEN_TS_1)\n    cdef double[:] sq_errors, segments_cos_dists_before_intersection, segments_cos_dists_after_intersection, segments_cos_dists_non_intersection, cos_dist_before_intersection, cos_dist_after_intersection\n    cdef double segment, sum_squares_outside_intersection_current, sum_squares_total, sum_segment_cos_dists_total, sum_segment_cos_dists_outside_intersection_current, scale_factor, val, cos_dist_intersection, squared_error_intersection\n    cdef int[:] timestamp_segment_start_2_segment_idx = np.ones(LEN_TS_1, dtype=np.int32) * INT_NAN\n    cdef int[:] timestamp_segment_end_2_segment_idx = np.ones(LEN_TS_1, dtype=np.int32) * INT_NAN\n    cdef int[:] timestamp_2_start_segment_idx_higher = return_empty_1d_int(LEN_TS_1)\n    cdef int[:] timestamp_2_end_segment_idx_lower = return_empty_1d_int(LEN_TS_1)\n    cdef int[:] timestamp_2_point_idx_higher = return_empty_1d_int(LEN_TS_1)\n    cdef int i, segment_idx, start_point_idx, scale_factor_idx, ts_1_pos_under_ts_2_start, current_min, current_max, number_of_segments_outside_intersection_current, initial_intersection_point_idx, ts_1_start_idx, ts_1_end_idx, NUMBER_OF_POINTS_TS_1\n\n    # init\n    NUMBER_OF_POINTS_TS_1 = 0\n    for i in range(LEN_TS_1):\n        if not isnan(ts_1_input[i]):\n            ts_1_points[NUMBER_OF_POINTS_TS_1, 0] = i\n            ts_1_points_orig[NUMBER_OF_POINTS_TS_1] = ts_1_input[i]\n            NUMBER_OF_POINTS_TS_1 += 1\n\n    for segment_idx in range(NUMBER_OF_POINTS_TS_1 - 1):\n        start_point_idx = <int>ts_1_points[segment_idx, 0]\n        timestamp_segment_start_2_segment_idx[start_point_idx] = segment_idx\n    for i in range(1, NUMBER_OF_POINTS_TS_1):\n        end_point_idx = <int>ts_1_points[i, 0]\n        timestamp_segment_end_2_segment_idx[end_point_idx] = i - 1\n\n    current_min = BIG_M_SEQ_LEN\n    timestamp_2_start_segment_idx_higher[-1] = current_min\n    # the wrongly filled "tail" of timestamp_2_start_segment_idx_higher/ "head" of timestamp_2_end_segment_idx_lower don\'t bother us, it would result into an empty slice selection\n    for i in range(LEN_TS_1 - 2, -1, -1):\n        if timestamp_segment_start_2_segment_idx[i] != INT_NAN:\n            current_min = timestamp_segment_start_2_segment_idx[i]\n        timestamp_2_start_segment_idx_higher[i] = current_min\n    current_max = -1\n    timestamp_2_end_segment_idx_lower[0] = current_max\n    for i in range(LEN_TS_1):\n        if timestamp_segment_end_2_segment_idx[i] != INT_NAN:\n            current_max = timestamp_segment_end_2_segment_idx[i]\n        timestamp_2_end_segment_idx_lower[i] = current_max\n    current_min = NUMBER_OF_POINTS_TS_1\n    timestamp_2_point_idx_higher[-1] = current_min\n    for i in range(LEN_TS_1 - 2, -1, -1):\n        if not isnan(ts_1_input[i]):\n            current_min -= 1\n        timestamp_2_point_idx_higher[i] = current_min\n\n    # main loop checking different scales\n    for scale_factor_idx in range(10):\n        scale_factor = scale_factor_array[scale_factor_idx]\n        for i in range(LEN_TS_1):\n            ts_1[i] = scale_factor * ts_1_input[i]\n            if ts_1[i] < NOISE_UB and ts_1[i] > NOISE_LB:\n                ts_1[i] = 0\n\n        for i in range(NUMBER_OF_POINTS_TS_1):\n            ts_1_points[i, 1] = scale_factor * ts_1_points_orig[i]\n\n        ts_1_segments = get_segments(ts_1_points, NUMBER_OF_POINTS_TS_1)\n\n        # a helping variable to keep track of values of interest outside of intersection\n        sum_segment_cos_dists_total = 0.0\n        number_of_segments_outside_intersection_current = ts_1_segments.shape[0]\n        for i in range(ts_1_segments.shape[0]):\n            segment_base_cos_dists[i] = cos_dist(ts_1_segments[i, 0], ts_1_segments[i, 1], ts_1_segments[i, 0], 0.0)\n            sum_segment_cos_dists_total += segment_base_cos_dists[i]\n\n        sum_squares_total = 0.0\n        for i in range(NUMBER_OF_POINTS_TS_1):\n            sum_squares_total += ts_1_points[i, 1]**2\n        if sum_squares_total == 0:\n            continue\n\n        initial_intersection_point_idx = NUMBER_OF_POINTS_TS_1 - 1\n\n        sum_squares_outside_intersection_current = sum_squares_total\n        while initial_intersection_point_idx >= 0 and ts_1_points[initial_intersection_point_idx, 0] >= LEN_TS_1 - (INITIAL_INTERSECTION_WIDTH - 1):\n            sum_squares_outside_intersection_current -= ts_1_points[initial_intersection_point_idx, 1]**2\n            initial_intersection_point_idx -= 1\n\n        sum_segment_cos_dists_outside_intersection_current = sum_segment_cos_dists_total\n        for i in range(LEN_TS_1 - (INITIAL_INTERSECTION_WIDTH - 1), LEN_TS_1):\n            if timestamp_segment_start_2_segment_idx[i] != INT_NAN:\n                segment_idx = timestamp_segment_start_2_segment_idx[i]\n                sum_segment_cos_dists_outside_intersection_current -= segment_base_cos_dists[segment_idx]\n                number_of_segments_outside_intersection_current -= 1\n\n        # shifting ts_1_input against ts_2\n        for ts_1_pos_under_ts_2_start in range(LEN_TS_1 - INITIAL_INTERSECTION_WIDTH, -(LEN_TS_2 - INITIAL_INTERSECTION_WIDTH), -1):\n            ts_1_start_idx, ts_1_end_idx = max(0, ts_1_pos_under_ts_2_start), min(LEN_TS_1,\n                                                                                  LEN_TS_2 + ts_1_pos_under_ts_2_start)\n            ts_1_intersection_part = ts_1[ts_1_start_idx: ts_1_end_idx]\n            ts_2_intersection_part = ts_2[max(0, -ts_1_pos_under_ts_2_start):\n                                          min(LEN_TS_2,\n                                              LEN_TS_1 - ts_1_pos_under_ts_2_start)]\n\n            sq_errors = squared_error_sum(ts_1_intersection_part, ts_2_intersection_part)\n            squared_error_intersection = sq_errors[0]\n            squared_baseline_intersection = sq_errors[1]\n            segments_ts_1_intersection = ts_1_segments[timestamp_2_start_segment_idx_higher[ts_1_start_idx]:\n                                                       timestamp_2_end_segment_idx_lower[ts_1_end_idx - 1] + 1]\n\n            segment_cos_dists_intersection = cos_dist_ts_to_be_masked(segments_ts_1_intersection, ts_2_intersection_part, ts_1_intersection_part)\n            # managing non-intersection part of ts_1\n            if ts_1_pos_under_ts_2_start > 0:\n                if not isnan(ts_1[ts_1_pos_under_ts_2_start]):\n                    val = ts_1[ts_1_pos_under_ts_2_start] ** 2\n                    sum_squares_outside_intersection_current -= val\n                    if timestamp_segment_start_2_segment_idx[ts_1_pos_under_ts_2_start] != INT_NAN:\n                        segment_idx = timestamp_segment_start_2_segment_idx[ts_1_pos_under_ts_2_start]\n                        sum_segment_cos_dists_outside_intersection_current -= segment_base_cos_dists[segment_idx]\n                        number_of_segments_outside_intersection_current -= 1\n            if LEN_TS_1 - ts_1_pos_under_ts_2_start > LEN_TS_2:\n                if not isnan(ts_1[LEN_TS_2 + ts_1_pos_under_ts_2_start]):\n                    val = ts_1[LEN_TS_2 + ts_1_pos_under_ts_2_start] ** 2\n                    sum_squares_outside_intersection_current += val\n                    if timestamp_segment_end_2_segment_idx[LEN_TS_2 + ts_1_pos_under_ts_2_start] != INT_NAN:\n                        segment_idx = timestamp_segment_end_2_segment_idx[LEN_TS_2 + ts_1_pos_under_ts_2_start]\n                        sum_segment_cos_dists_outside_intersection_current += segment_base_cos_dists[segment_idx]\n                        number_of_segments_outside_intersection_current += 1\n\n            squared_error_total = squared_error_intersection + sum_squares_outside_intersection_current\n            squared_error_unreduced_portion = squared_error_total / sum_squares_total\n\n            segment_cos_dists_unreduced_portion = (segment_cos_dists_intersection + sum_segment_cos_dists_outside_intersection_current) / sum_segment_cos_dists_total\n\n            if squared_error_unreduced_portion < error_sq_min:\n                error_sq_min = squared_error_unreduced_portion\n                scale_factor_opt_sq = scale_factor\n                squared_error_unreduced_portion_opt_sq, segment_cos_dists_unreduced_portion_opt_sq = squared_error_unreduced_portion, segment_cos_dists_unreduced_portion\n\n            if segment_cos_dists_unreduced_portion < error_cos_min:\n                error_cos_min = segment_cos_dists_unreduced_portion\n                scale_factor_opt_cos = scale_factor\n                squared_error_unreduced_portion_opt_cos, segment_cos_dists_unreduced_portion_opt_cos = squared_error_unreduced_portion, segment_cos_dists_unreduced_portion\n\n    return scale_factor_opt_cos, error_cos_min, scale_factor_opt_sq, error_sq_min, segment_cos_dists_unreduced_portion_opt_sq, squared_error_unreduced_portion_opt_cos')


# In[24]:


def compute_scale_based_on_class_prototype(class_name, df_grouped, time_scale, class_prototypes, object_ids):
    class_ts_2 = class_prototypes[class_name]
    mapping = df_grouped.agg(lambda x: estimate_scale(x.values, class_ts_2))
    added_columns = [f'x_sc_{class_name}_t_sc_{time_scale:.2f}_cos', f'err_{class_name}_t_sc_{time_scale:.2f}_cos',
                     f'x_sc_{class_name}_t_sc_{time_scale:.2f}_sq',
                     f'err_{class_name}_t_sc_{time_scale:.2f}_sq',
                     f'err_{class_name}_t_sc_{time_scale:.2f}_cos_from_sq',
                     f'err_{class_name}_t_sc_{time_scale:.2f}_sq_from_cos']
    df_added_columns = pd.DataFrame(np.column_stack(object_ids.map(lambda x: mapping[x])).T,
                                    columns=added_columns, index=object_ids.index)
    return df_added_columns

df_grouped = df.groupby('object_id')['flux']

time_scales = [1.0, 0.99] # for illustration
for time_scale in time_scales:
    compute_scale_based_on_class_prototype_current = partial(compute_scale_based_on_class_prototype,
                                                             time_scale=time_scale,
                                                             df_grouped=df_grouped,
                                                             class_prototypes=class_prototypes,
                                                             object_ids=df_meta['object_id'])

    with Pool(cpu_count() - 1) as p:
        results_list = p.map(compute_scale_based_on_class_prototype_current, classes)
    all_new_columns = pd.concat(results_list, axis=1)
    df_meta = pd.concat((df_meta, all_new_columns), axis=1)


# In[25]:


################################################
# Counting matches of cos/sq best similarities
################################################
for class_name in classes:
    df_meta[f'cos_sq_matches_{class_name}'] = (df_meta[f'err_{class_name}_t_sc_{time_scales[0]:.2f}_cos']==df_meta[f'err_{class_name}_t_sc_{time_scales[0]:.2f}_cos_from_sq']).map(int)
    for time_scale in time_scales[1:]:
        df_meta[f'cos_sq_matches_{class_name}'] += (df_meta[f'err_{class_name}_t_sc_{time_scale:.2f}_cos']==df_meta[f'err_{class_name}_t_sc_{time_scale:.2f}_cos_from_sq']).map(int)

################################################
# Time scale estimation
################################################
def gather_time_scale_info(suffix):
    gathered_time_scale_values = dict()
    for time_scale in time_scales:
        classes_iterator = iter(classes)
        class_name = next(classes_iterator)
        gathered_time_scale_values[time_scale] = np.expand_dims(np.row_stack(df_meta[[f'err_{class_name}_t_sc_{time_scale:.2f}_{suffix}',
                                                                                      f'x_sc_{class_name}_t_sc_{time_scale:.2f}_{suffix}']].values), axis=1)
        for class_name in classes_iterator:
            helping_array = np.expand_dims(np.row_stack(df_meta[[f'err_{class_name}_t_sc_{time_scale:.2f}_{suffix}',
                                                                 f'x_sc_{class_name}_t_sc_{time_scale:.2f}_{suffix}']].values),
                                           axis=1)
            gathered_time_scale_values[time_scale] = np.concatenate(
                (gathered_time_scale_values[time_scale], helping_array), axis=1)
    return gathered_time_scale_values, np.concatenate(
        list(map(lambda x: np.expand_dims(x, 0), gathered_time_scale_values.values())))

# gathered_time_scale_values_cos, axes: time_scale, object, class, [err, x_scale]
gathered_time_scale_values_cos_dict, gathered_time_scale_values_cos = gather_time_scale_info('cos')
gathered_time_scale_values_sq_dict, gathered_time_scale_values_sq = gather_time_scale_info('sq')

def get_trimmed_means_per_time_scale(gathered_time_scale_values, number_of_best_scored_classes_to_take=4):
    gathered_time_scale_values_sorted_per_classes = np.sort(gathered_time_scale_values, axis=2)
    trimmed_means_per_time_scale = np.apply_along_axis(np.mean, axis=2,
                                                       arr=gathered_time_scale_values_sorted_per_classes[:, :, :number_of_best_scored_classes_to_take, 0])
    return trimmed_means_per_time_scale

def normalize_scores(object_trimmed_means_per_time_scale):
    max_score = np.max(object_trimmed_means_per_time_scale)
    object_trimmed_means_per_time_scale = object_trimmed_means_per_time_scale / max_score
    return object_trimmed_means_per_time_scale

trimmed_means_per_time_scale_sq = get_trimmed_means_per_time_scale(gathered_time_scale_values_sq)
trimmed_means_per_time_scale_sq_normed = np.apply_along_axis(normalize_scores, axis=0,
                                                             arr=trimmed_means_per_time_scale_sq)

def neighb_smooth(trimmed_means_per_time_scale_normed):
    max_idx = trimmed_means_per_time_scale_normed.shape[0] - 1

    def neighb_mean(i):
        if i > 0 and i < max_idx:
            return (0.3 * trimmed_means_per_time_scale_normed[i - 1]
                    + trimmed_means_per_time_scale_normed[i]
                    + 0.3 * trimmed_means_per_time_scale_normed[i + 1]) / 1.6
        elif i == 0:
            return (trimmed_means_per_time_scale_normed[0] + 0.3 * trimmed_means_per_time_scale_normed[1]) / 1.3
        else:
            return (trimmed_means_per_time_scale_normed[-1] + 0.3 * trimmed_means_per_time_scale_normed[-2]) / 1.3

    trimmed_means_per_time_scale_normed_smoothed = np.array(
        list(map(neighb_mean, np.arange(trimmed_means_per_time_scale_normed.shape[0]))))
    return trimmed_means_per_time_scale_normed_smoothed

trimmed_means_per_time_scale_sq_normed_smoothed = np.apply_along_axis(neighb_smooth, axis=0,
                                                                      arr=trimmed_means_per_time_scale_sq_normed)
trimmed_means_per_time_scale_sq_smoothed = np.apply_along_axis(neighb_smooth, axis=0, arr=trimmed_means_per_time_scale_sq)
df_meta['estimated_t_sc_idx_orig'] = np.argmin(trimmed_means_per_time_scale_sq_smoothed, axis=0)
df_meta['estimated_t_sc'] = df_meta['estimated_t_sc_idx_orig'].map(lambda i: time_scales[i])
################################################
# red shift estimation
################################################
df_meta['z_approx'] = df_meta['estimated_t_sc']*(df_meta['hostgal_photoz'] + 1) - 1

################################################
# Amplitude scale estimation
################################################
# time_scale, object, class, [err, x_scale]
def get_best10matches_amplitude_scales(gathered_time_scale_values):
    gathered_err = np.swapaxes(gathered_time_scale_values[:, :, :, 0], 1, 2).reshape(-1,gathered_time_scale_values.shape[1])
    gathered_scale = np.swapaxes(gathered_time_scale_values[:, :, :, 1], 1, 2).reshape(-1,gathered_time_scale_values.shape[1])
    gathered_val_pos = np.argsort(gathered_err, axis=0)
    # https://stackoverflow.com/questions/6155649/sort-a-numpy-array-by-another-array-along-a-particular-axis
    second_scale_indices = list(range(gathered_scale.shape[1]))
    gathered_scale_sorted_by_err = gathered_scale[gathered_val_pos, second_scale_indices]
    gathered_scale_best10 = gathered_scale_sorted_by_err[:10, :]
    return gathered_scale_best10

gathered_scale_best10_cos = get_best10matches_amplitude_scales(gathered_time_scale_values_cos)
gathered_scale_best10_sq = get_best10matches_amplitude_scales(gathered_time_scale_values_sq)
gathered_scale_best20 = np.concatenate((gathered_scale_best10_sq, gathered_scale_best10_cos))
df_meta['estimated_x_sc'] = np.median(gathered_scale_best20, axis=0)

################################################
# Tournament scores
################################################
def err_to_rank_points(err_array):
    results = np.zeros_like(err_array)
    if np.all(np.isnan(err_array)):
        return results
    pos_sorted = np.argsort(err_array)
    for i, points in enumerate([4, 3, 2, 1]):
        results[pos_sorted[i]] += points
    return results

def compute_tournament_points(gathered_time_scale_values, suffix):
    global df_meta
    gathered_time_scale_err = gathered_time_scale_values[:, :, :, 0] # axes: time_scale, object, class, [err, x_scale]
    gathered_time_scale_tournament_per_scale_points = np.apply_along_axis(err_to_rank_points, axis=2, arr=gathered_time_scale_err)
    gathered_time_scale_tournament_total_points = np.sum(gathered_time_scale_tournament_per_scale_points, axis=0)

    columns_to_add = [f'tournament_points_{class_name}_{suffix}' for class_name in classes]
    df_tournament_points = pd.DataFrame(gathered_time_scale_tournament_total_points,
                                       columns=columns_to_add, index=df_meta.index)
    df_meta = pd.concat((df_meta, df_tournament_points), axis=1)

compute_tournament_points(gathered_time_scale_values_cos, 'cos')
compute_tournament_points(gathered_time_scale_values_sq, 'sq')

################################################
# std of amplitude matching scores
################################################
def std_matching_scores(gathered_time_scale_values, suffix):
    global df_meta
    gathered_time_scale_err = gathered_time_scale_values[:, :, :, 1]  # axes: time_scale, object, class
    gathered_time_scale_err_sorted_per_time_scale = np.sort(gathered_time_scale_err, axis=0)
    gathered_time_scale_err_2best = gathered_time_scale_err_sorted_per_time_scale[:5, :, :]
    gathered_time_scale_err_2best_mean = np.std(gathered_time_scale_err_2best, axis=0)
    # gathered_time_scale_err_2best_sum_min = np.min(gathered_time_scale_err_2best_sum, axis=1)
    gathered_time_scale_err_2best_sum_normed = gathered_time_scale_err_2best_mean  # gathered_time_scale_err_2best_sum/ (np.ones_like(gathered_time_scale_err_2best_sum) *
    #    gathered_time_scale_err_2best_sum_min.reshape(-1, 1))

    columns_to_add = [f'best2amp_std_{class_name}_{suffix}' for class_name in classes]
    df_gathered_time_scale_err_2best_sum = pd.DataFrame(gathered_time_scale_err_2best_sum_normed,
                                                        columns=columns_to_add, index=df_meta.index)
    df_meta = pd.concat((df_meta, df_gathered_time_scale_err_2best_sum), axis=1)

std_matching_scores(gathered_time_scale_values_cos, 'cos')
std_matching_scores(gathered_time_scale_values_sq, 'sq')

