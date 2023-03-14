#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings; warnings.filterwarnings('ignore')
import pandas as pd,numpy as np,sqlite3,os
import seaborn as sn,pylab as pl
import keras as ks,tensorflow as tf
from IPython import display
from sklearn.model_selection import train_test_split
from sklearn import datasets,linear_model,svm
from sklearn.metrics import mean_squared_error,median_absolute_error,mean_absolute_error,r2_score,explained_variance_score
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,RationalQuadratic,RBF
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.isotonic import IsotonicRegression
from keras.datasets import boston_housing
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,GlobalAveragePooling1D
from keras.layers import Activation,Flatten,Dropout,BatchNormalization
from keras.layers import Conv1D,MaxPooling1D,GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU,LeakyReLU
fw='weights.boston.hdf5'


# In[2]:


def connect_to_db(dbf):
    sqlconn=None
    try:
        sqlconn=sqlite3.connect(dbf)
        return sqlconn
    except Error as err:
        print(err)
        if sqlconn is not None:
            sqlconn.close()
def history_plot(fit_history,n):
    pl.figure(figsize=(12,10))    
    pl.subplot(211)
    pl.plot(fit_history.history['loss'][n:],
            color='slategray',label='train')
    pl.plot(fit_history.history['val_loss'][n:],
            color='#348ABD',label='valid')
    pl.xlabel('Epochs'); pl.ylabel('Loss')
    pl.legend(); pl.title('Loss Function')      
    pl.subplot(212)
    pl.plot(fit_history.history['mae'][n:],
            color='slategray',label='train')
    pl.plot(fit_history.history['val_mae'][n:],
            color='#348ABD',label='valid')
    pl.xlabel('Epochs'); pl.ylabel('MAE')    
    pl.legend(); pl.title('Mean Absolute Error')
    pl.show()
def nnpredict(y1,y2,y3,ti):
    pl.figure(figsize=(12,6))
    pl.scatter(range(n),y_test[:n],marker='*',s=100,
               color='black',label='Real data')
    pl.plot(y1[:n],label='MLP')
    pl.plot(y2[:n],label='CNN')
    pl.plot(y3[:n],label='RNN')
    pl.xlabel("Data Points")
    pl.ylabel("Predicted and Real Target Values")
    pl.legend(); pl.title(ti); pl.show()


# In[3]:


connection=connect_to_db('boston.db')
if connection is not None:
    cursor=connection.cursor()
boston_data=datasets.load_boston()
columns=boston_data.feature_names
boston_df=pd.DataFrame(boston_data.data,columns=columns)
boston_df['MEDV']=boston_data.target
boston_df.to_sql('main',con=connection,if_exists='replace')
boston_df.head()


# In[4]:


pearson=boston_df.corr(method='pearson')
corr_with_prices=pearson.iloc[-int(1)][:-int(1)]
pd.DataFrame(corr_with_prices[abs(corr_with_prices)             .argsort()[::-int(1)]])


# In[5]:


pd.read_sql_query('''
SELECT ZN,
       AVG(LSTAT),
       AVG(RM),
       AVG(PTRATIO),
       AVG(INDUS),
       AVG(TAX)
FROM main
GROUP BY ZN;
''',con=connection)\
.set_index('ZN').head(int(7))


# In[6]:


if connection is not None:
    connection.close()
if os.path.exists('boston.db'):
    os.remove('boston.db')
else:
    print('The file does not exist')
os.listdir()


# In[7]:


n=int(51)
(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
x_valid,y_valid=x_test[:n],y_test[:n]
x_test,y_test=x_test[n:],y_test[n:]
t=[["Training feature's shape:",x_train.shape],
   ["Training target's shape",y_train.shape],
   ["Validating feature's shape:",x_valid.shape],
   ["Validating target's shape",y_valid.shape],
   ["Testing feature's shape:",x_test.shape],
   ["Testing target's shape",y_test.shape]]
pd.DataFrame(t)


# In[8]:


pl.style.use('seaborn-whitegrid')
pl.figure(1,figsize=(10,4))
pl.subplot(121)
sn.distplot(y_train,color='#348ABD',bins=30)
pl.ylabel("Distribution"); pl.xlabel("Prices")
pl.subplot(122)
sn.distplot(np.log(y_train),color='#348ABD',bins=30)
pl.ylabel("Distribution"); pl.xlabel("Logarithmic Prices")
pl.suptitle('Boston Housing Data',fontsize=15)
pl.show()


# In[9]:


def mlp_model():
    model=Sequential() 
    model.add(Dense(832,input_dim=13))
    model.add(LeakyReLU(alpha=.025))   
    model.add(Dense(104))     
    model.add(LeakyReLU(alpha=.025))   
    model.add(Dense(1,kernel_initializer='normal'))    
    model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])
    return model
mlp_model=mlp_model()
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=0,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=mlp_model.fit(x_train,y_train,batch_size=24, 
                      validation_data=(x_valid,y_valid),
                      epochs=1000,verbose=2, 
                      callbacks=[checkpointer,lr_reduction,estopping])


# In[10]:


history_plot(history,2)
mlp_model.load_weights(fw)
y_train_mlp=mlp_model.predict(x_train)
y_valid_mlp=mlp_model.predict(x_valid)
y_test_mlp=mlp_model.predict(x_test)
score_train_mlp=r2_score(y_train,y_train_mlp)
score_valid_mlp=r2_score(y_valid,y_valid_mlp)
score_test_mlp=r2_score(y_test,y_test_mlp)
pd.DataFrame([['Train R2 score:',score_train_mlp],
              ['Valid R2 score:',score_valid_mlp],
              ['Test R2 score:',score_test_mlp]])


# In[11]:


def cnn_model():
    model=Sequential()       
    model.add(Conv1D(13,5,padding='valid',
                     input_shape=(13,1)))
    model.add(LeakyReLU(alpha=.025))
    model.add(MaxPooling1D(pool_size=2))   
    model.add(Conv1D(128,3,padding='valid'))
    model.add(LeakyReLU(alpha=.025))
    model.add(MaxPooling1D(pool_size=2))   
    model.add(Flatten())      
    model.add(Dense(26,activation='relu',
                    kernel_initializer='normal'))
    model.add(Dropout(.1))  
    model.add(Dense(1,kernel_initializer='normal'))  
    model.compile(loss='mse',optimizer='nadam',metrics=['mae'])
    return model
cnn_model=cnn_model()
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=0,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=cnn_model.fit(x_train.reshape(-1,13,1),y_train, 
                          validation_data=(x_valid.reshape(-1,13,1),y_valid),
                          epochs=1000,batch_size=14,verbose=2, 
                          callbacks=[checkpointer,lr_reduction,estopping])


# In[12]:


history_plot(history,2)
cnn_model.load_weights(fw)
y_train_cnn=cnn_model.predict(x_train.reshape(-1,13,1))
y_valid_cnn=cnn_model.predict(x_valid.reshape(-1,13,1))
y_test_cnn=cnn_model.predict(x_test.reshape(-1,13,1))
score_train_cnn=r2_score(y_train,y_train_cnn)
score_valid_cnn=r2_score(y_valid,y_valid_cnn)
score_test_cnn=r2_score(y_test,y_test_cnn)
pd.DataFrame([['Train R2 score:',score_train_cnn],
              ['Valid R2 score:',score_valid_cnn],
              ['Test R2 score:',score_test_cnn]])


# In[13]:


def rnn_model():
    model=Sequential()   
    model.add(LSTM(104,return_sequences=True,
                   input_shape=(1,13)))
    model.add(LSTM(104,return_sequences=True))
    model.add(LSTM(104,return_sequences=False))   
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])       
    return model
rnn_model=rnn_model()
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                               verbose=0,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=rnn_model.fit(x_train.reshape(-1,1,13),y_train, 
                          validation_data=(x_valid.reshape(-1,1,13),y_valid),
                          epochs=1000,batch_size=16,verbose=2, 
                          callbacks=[checkpointer,lr_reduction,estopping])


# In[14]:


history_plot(history,2)
rnn_model.load_weights(fw)
y_train_rnn=rnn_model.predict(x_train.reshape(-1,1,13))
y_valid_rnn=rnn_model.predict(x_valid.reshape(-1,1,13))
y_test_rnn=rnn_model.predict(x_test.reshape(-1,1,13))
score_train_rnn=r2_score(y_train,y_train_rnn)
score_valid_rnn=r2_score(y_valid,y_valid_rnn)
score_test_rnn=r2_score(y_test,y_test_rnn)
pd.DataFrame([['Train R2 score:',score_train_rnn],
              ['Valid R2 score:',score_valid_rnn],
              ['Test R2 score:',score_test_rnn]])


# In[15]:


ti="Train Set; Neural Network Predictions vs Real Data"
y1,y2,y3=y_train_mlp,y_train_cnn,y_train_rnn
nnpredict(y1,y2,y3,ti)
ti="Validation Set; Neural Network Predictions vs Real Data"
y1,y2,y3=y_valid_mlp,y_valid_cnn,y_valid_rnn
nnpredict(y1,y2,y3,ti)
ti="Test Set; Neural Network Predictions vs Real Data"
y1,y2,y3=y_test_mlp,y_test_cnn,y_test_rnn
nnpredict(y1,y2,y3,ti)


# In[16]:


def regressor_fit_score(regressor,regressor_name,dataset,
                        x_train,x_test,y_train,y_test,n=6):
    regressor_list.append(str(regressor))
    regressor_names.append(regressor_name)
    reg_datasets.append(dataset)    
    regressor.fit(x_train,y_train)
    y_reg_train=regressor.predict(x_train)
    y_reg_test=regressor.predict(x_test)    
    r2_reg_train=round(r2_score(y_train,y_reg_train),n)
    r2_train.append(r2_reg_train)
    r2_reg_test=round(r2_score(y_test,y_reg_test),n)
    r2_test.append(r2_reg_test)    
    ev_reg_train=round(explained_variance_score(y_train,y_reg_train),n)
    ev_train.append(ev_reg_train)
    ev_reg_test=round(explained_variance_score(y_test, y_reg_test),n)
    ev_test.append(ev_reg_test)    
    mse_reg_train=round(mean_squared_error(y_train,y_reg_train),n)
    mse_train.append(mse_reg_train)
    mse_reg_test=round(mean_squared_error(y_test,y_reg_test),n)
    mse_test.append(mse_reg_test)
    mae_reg_train=round(mean_absolute_error(y_train,y_reg_train),n)
    mae_train.append(mae_reg_train)
    mae_reg_test=round(mean_absolute_error(y_test,y_reg_test),n)
    mae_test.append(mae_reg_test)
    mdae_reg_train=round(median_absolute_error(y_train,y_reg_train),n)
    mdae_train.append(mdae_reg_train)
    mdae_reg_test=round(median_absolute_error(y_test,y_reg_test),n)
    mdae_test.append(mdae_reg_test)    
    return [y_reg_train,y_reg_test,r2_reg_train,r2_reg_test,
            ev_reg_train,ev_reg_test,
            mse_reg_train,mse_reg_test,mae_reg_train,mae_reg_test,
            mdae_reg_train,mdae_reg_test]
def get_regressor_results():
    return pd.DataFrame({'regressor':regressor_list,
                         'regressor_name':regressor_names,
                         'dataset':reg_datasets,
                         'r2_train':r2_train,'r2_test':r2_test,
                         'ev_train':ev_train,'ev_test':ev_test,
                         'mse_train':mse_train,'mse_test':mse_test,
                         'mae_train':mae_train,'mae_test':mae_test,
                         'mdae_train':mdae_train,'mdae_test':mdae_test})


# In[17]:


(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
regressor_list,regressor_names,reg_datasets=[],[],[]
r2_train,r2_test,ev_train, ev_test,mse_train,mse_test,mae_train,mae_test,mdae_train,mdae_test=[],[],[],[],[],[],[],[],[],[]
df_list=['regressor_name','r2_train','r2_test','ev_train','ev_test',
         'mse_train','mse_test','mae_train','mae_test',
         'mdae_train','mdae_test']
reg=[linear_model.LinearRegression(),
     linear_model.Ridge(),linear_model.RidgeCV(),
     linear_model.Lasso(),linear_model.LassoLarsCV(),
     linear_model.RANSACRegressor(),
     linear_model.BayesianRidge(),linear_model.ARDRegression(),
     linear_model.HuberRegressor(),linear_model.TheilSenRegressor(),
     PLSRegression(),DecisionTreeRegressor(),ExtraTreeRegressor(),
     BaggingRegressor(),AdaBoostRegressor(),
     GradientBoostingRegressor(),RandomForestRegressor(),
     linear_model.PassiveAggressiveRegressor(max_iter=1000,tol=.001),
     linear_model.ElasticNet(),
     linear_model.SGDRegressor(max_iter=1000,tol=.001),
     svm.SVR(),KNeighborsRegressor(),
     RadiusNeighborsRegressor(radius=1.5),GaussianProcessRegressor()]
listreg=['LinearRegression','Ridge','RidgeCV',
         'Lasso','LassoLarsCV','RANSACRegressor',
         'BayesianRidge','ARDRegression','HuberRegressor',
         'TheilSenRegressor','PLSRegression','DecisionTreeRegressor',
         'ExtraTreeRegressor','BaggingRegressor','AdaBoostRegressor',
         'GradientBoostingRegressor','RandomForestRegressor']


# In[18]:


yreg=[]
for i in range(len(listreg)):
    yreg.append(regressor_fit_score(reg[i],listreg[i],'Boston',
                                    x_train,x_test,
                                    y_train,y_test)[:2])
[[y_train101,y_test101],[y_train102,y_test102],[y_train103,y_test103],
 [y_train104,y_test104],[y_train105,y_test105],[y_train106,y_test106],
 [y_train107,y_test107],[y_train108,y_test108],[y_train109,y_test109],
 [y_train110,y_test110],[y_train111,y_test111],[y_train112,y_test112],
 [y_train113,y_test113],[y_train114,y_test114],[y_train115,y_test115],
 [y_train116,y_test116],[y_train117,y_test117]]=yreg


# In[19]:


df_regressor_results=get_regressor_results()
df_regressor_results.to_csv('regressor_results.csv')
df_regressor_results[df_list].sort_values('r2_test',ascending=False)


# In[20]:


pl.figure(figsize=(12,6)); n=30; x=range(n)
pl.scatter(x,y_test[:n],marker='*',s=100,
           color='black',label='Real data')
pl.plot(x,y_test116[:n],lw=2,label='Gradient Boosting')
pl.plot(x,y_test117[:n],lw=2,label='Random Forest')
pl.plot(x,y_test114[:n],lw=2,label='Bagging')
pl.plot(x,y_test115[:n],lw=2,label='Ada Boost')
pl.plot(x,y_test113[:n],lw=2,label='ExtraTree')
pl.xlabel('Observations'); pl.ylabel('Targets')
pl.title('Regressors. Test Results. Boston')
pl.legend(loc=2,fontsize=10); pl.show()

