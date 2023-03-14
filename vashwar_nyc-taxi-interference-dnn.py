#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


os.listdir("../input/dnn-model")


# In[3]:


df_test=pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


# In[4]:


df_test.head()


# In[5]:


def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    #print(targ_pre)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','hour',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
        
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
###Harv Distance        
def distance( data):

    radius = 6371 # km
    lon1=data[:,0]
    lat1=data[:,1]
    lon2=data[:,2]
    lat2=data[:,3]
    #print(lat2-lat1)
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1))             * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d


# In[6]:


class DNN_Model:
    def __init__(self,feature,predict,hidden_layers,neurons,layer_dropout,iterations,path_read,path_write,save_model,load_model):
        self.x_f=feature
        self.y_pr=predict
        self.n_hidden_layers=hidden_layers
        self.n_neurons=neurons
        self.dropout=layer_dropout
        self.n_iterations=iterations
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.path_r=path_read
        self.path_w=path_write
        self.save_model=save_model
        self.load_model=load_model

    def dnn(self,inputs,training):

        with tf.variable_scope("dnn"):
            for layer in range(self.n_hidden_layers):


                inputs = tf.layers.dropout(inputs, self.dropout[layer], training=self.training)
                inputs = tf.layers.dense(inputs, self.n_neurons[layer],
                                         kernel_initializer=self.initializer,
                                         name="hidden%d" % (layer + 1))

                #inputs=tf.layers.batch_normalization(inputs, momentum=0.9,training=training)
                inputs=tf.nn.relu(inputs, name="hidden%d_out" % (layer + 1))

            return inputs
    def build_graph(self):
        n_inputs=14
        n_outputs=1
        self.X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
        self.y=tf.placeholder(tf.float32,shape=(None,1),name='y')
        self.training=tf.placeholder_with_default(False,shape=(),name='training')
        with tf.name_scope("dnn"):
            self.y_pred=tf.layers.dense(self.dnn(self.X,self.training),n_outputs,name="Outputs",kernel_initializer=self.initializer,activation=tf.nn.relu)
        with tf.name_scope("loss"):

            error=self.y_pred-self.y
            self.mse=tf.reduce_mean(tf.square(error),name='MSE')
        l_rate=0.0001
#l_rate=0.0005
        with tf.name_scope("train"):

            optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
            self.t_op=optimizer.minimize(self.mse)
        


    def train_data(self):
                
        np.random.seed(4)
        index_shuffled=np.random.permutation(x_sc.shape[0])
        ind_train=index_shuffled[0:54000000]
        ind_valid=index_shuffled[54000000:]
        er_=np.zeros((1,2))
        batch_size=128
        tf.reset_default_graph()
        self.build_graph()
        #init=tf.global_variables_initializer()
        
        saver=tf.train.Saver()
        #saver.restore(sess,)
        sess=tf.InteractiveSession()
        if load_model==1:
            saver.restore(sess,self.path_r+'/model_.ckpt')
        else:
            sess.run(init)

        
        for i in range(0,0+self.n_iterations):

            ind_use=np.random.randint(0,54000000,batch_size)
            rand_test=np.random.randint(0,ind_valid.shape[0],1000)
            X_batch=self.x_f[ind_use,:]
            y_batch=self.y_pr[ind_use,:]
            sess.run(self.t_op,feed_dict={self.X:X_batch,self.y:y_batch,self.training:True})
            temp_er=np.vstack((self.mse.eval(feed_dict={self.X:X_batch,self.y:y_batch}),
            self.mse.eval(feed_dict={self.X:x_sc[ind_valid[rand_test]],self.y:y_sc[ind_valid[rand_test]]}))).T
            er_=np.concatenate((er_,temp_er))
        if self.save_model==1:
            saver.save(sess,path_r+'/model_.ckpt')
        sess.close()
        return er_
    def model_interference(self):
        tf.reset_default_graph()
        self.build_graph()
        sess=tf.InteractiveSession()
        saver=tf.train.Saver()
        saver.restore(sess,self.path_r+'/model_.ckpt')
        y_prediction=self.y_pred.eval(feed_dict={self.X:self.x_f}) 
        sess.close()
        return y_prediction


# In[7]:


df_test['Herv_Dist'] =distance(np.float64(df_test.values[:,2:6]))


# In[8]:


add_datepart(df_test, 'pickup_datetime', drop=True)


# In[9]:


feature_cols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count', 'pickup_datetimeYear',
       'pickup_datetimeMonth', 'pickup_datetimeWeek', 'pickup_datetimeDay',
       'pickup_datetimeDayofweek', 'pickup_datetimeDayofyear',
       'pickup_datetimehour', 'pickup_datetimeElapsed', 'Herv_Dist']


# In[10]:


df_test[feature_cols].head()


# In[11]:


##Mean and sigma for scaling
mu=np.array([ -7.39752352e+01,   4.07510864e+01,  -7.39743620e+01,
         4.07514412e+01,   1.69111912e+00,   2.01173779e+03,
         6.26937910e+00,   2.54649417e+01,   1.57119467e+01,
         3.04109087e+00,   1.75307310e+02,   1.35101716e+01,
         1.33224990e+09,   3.34143936e+00])
sigma=np.array([  4.26467712e-02,   3.18110081e-02,   4.13962939e-02,
         3.48417371e-02,   1.30694141e+00,   1.86550121e+00,
         3.43641982e+00,   1.49473195e+01,   8.68516050e+00,
         1.94912410e+00,   1.04798866e+02,   6.51677611e+00,
         5.84916113e+07,   4.08371701e+00])


# In[12]:


x_test_unscl=df_test[feature_cols].values


# In[13]:


x_test=(x_test_unscl-mu)/sigma


# In[14]:





# In[14]:


path_model='../input/dnn-model'
n_neurons_=[2000,1000,500,250,125,50,25,10]
dropout_=[0.25,0,0,0,0,0,0,0]
model=DNN_Model(x_test,[],8,n_neurons_,dropout_,1,path_model,[],0,1)
y_test=model.model_interference()


# In[15]:


my_submission =pd.DataFrame(np.concatenate((df_test['key'].values.reshape(-1,1),y_test),axis=1),columns=['key','fare_amount'])
my_submission.to_csv('submission.csv', index=False)


# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:





# In[16]:




