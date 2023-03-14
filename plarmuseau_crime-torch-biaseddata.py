#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import numpy as np

import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1777)
torch.manual_seed(1777)
if device == 'cuda' :
    torch.cuda.manual_seed_all(777)

# 학습 파라미터 설정
learning_rate = 0.01
training_epochs = 2020
batch_size = 15

# Data load
train_data = pd.read_csv('../input/crime-types/train_data.csv', header=None, skiprows=1, usecols=range(0, 13))
test_data = pd.read_csv('../input/crime-types/test_data.csv', header=None, skiprows=1, usecols=range(0, 12))

# Data 파싱
x_train_data = train_data.loc[:, 1:13]
y_train_data = train_data.loc[:, 0]

# 파싱한 Data를 numpy의 array로 변환
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)

test_data = np.array(test_data)

# 변환한 numpy의 array를 Tensor로 변환
x_train_data = torch.FloatTensor(x_train_data)
y_train_data = torch.LongTensor(y_train_data)

test_data = torch.FloatTensor(test_data)

# data_loader에 이용할 하나의 train Dataset으로 변환
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

# data_loader 설정
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# 모델 설계
linear1 = torch.nn.Linear(12, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 256, bias=True)
linear4 = torch.nn.Linear(256, 512, bias=True)
linear5 = torch.nn.Linear(512, 1024, bias=True)
linear6 = torch.nn.Linear(1024, 1024, bias=True)
linear7 = torch.nn.Linear(1024, 512, bias=True)
linear8 = torch.nn.Linear(512, 512, bias=True)
linear9 = torch.nn.Linear(512, 256, bias=True)
linear10 = torch.nn.Linear(256, 256, bias=True)
linear11 = torch.nn.Linear(256, 128, bias=True)
linear12 = torch.nn.Linear(128, 128, bias=True)
linear13 = torch.nn.Linear(128, 10, bias=True)
elu = torch.nn.ELU()

torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_normal_(linear6.weight)
torch.nn.init.xavier_normal_(linear7.weight)
torch.nn.init.xavier_normal_(linear8.weight)
torch.nn.init.xavier_uniform_(linear9.weight)
torch.nn.init.xavier_normal_(linear10.weight)
torch.nn.init.xavier_uniform_(linear11.weight)
torch.nn.init.xavier_normal_(linear12.weight)
torch.nn.init.xavier_normal_(linear13.weight)

model = torch.nn.Sequential(linear1, elu,
                            linear2, elu,
                            linear3, elu,
                            linear4, elu,
                            linear5, elu,
                            linear6, elu,
                            linear7, elu,
                            linear8, elu,
                            linear9, elu,
                            linear10, elu,
                            linear11, elu,
                            linear12, elu,
                            linear13).to(device)

loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=1e-20, eps=1e-20)

# 모델 학습
total_batch = len(data_loader)

for epoch in range(training_epochs) :
    avg_cost = 0

    for X, Y in data_loader :

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = loss(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch : {:4d}'.format(epoch+1), 'Cost : {:.9f}'.format(avg_cost))

print('Learning Finishied')

# 모델 평가
with torch.no_grad() :
    test_data = test_data.to(device)

    prediction = model(test_data)
    prediction = torch.argmax(prediction, 1)
    prediction = prediction.cpu().numpy().reshape(-1, 1)


# In[3]:



submit = pd.read_csv('../input/crime-types/submission_format.csv')

for i in range(len(prediction)) :
    submit['Lable'][i] = prediction[i].item()

submit.to_csv('result.csv', index=False, header=True)


# In[4]:


train_=train_data.loc[:, 1:13]
train_=torch.FloatTensor(train_.values)
train_=train_.to(device)
predtr = model( train_)
predtr = torch.argmax(predtr, 1)
predtr =predtr.detach().numpy().reshape(-1, 1)
predtr


# In[5]:


prediction.shape


# In[6]:


def verslag(titel,label2,yval,ypred,ypred2,mytrain):
        from sklearn.metrics import classification_report    
        yval=pd.Series(yval)
        #ypred=pd.Series(ypred)
        #ypred2=pd.Series(ypred2)
        print('shape yval/dropna ypred/dropna',yval.dropna().shape,yval.shape,ypred.shape,np.array(ypred2).shape)
        ypred=ypred
        ypred2=ypred2
        print(titel+'\n', classification_report(yval,ypred )  )
        vsubmit = pd.DataFrame({        label2[0]: mytrain[len(yval):].reset_index().index,        label2[1]: np.array(ypred2)    })
        #print(label2,label2[0],label2[1 ],vsubmit.shape,vsubmit.head(3))
        vsubmit[label2[1]]=vsubmit[label2[1]].astype('int')#-1
        print('submission header',vsubmit.head())
        vsubmit[label2].to_csv(titel+'submission.csv',index=False)
        print(titel,vsubmit[label2].groupby(label2[1]).count() )
        return  
    
verslag('test',['Index','Lable'],train_data[0],predtr,[x for x in prediction],submit)


# In[7]:


len(submit),prediction.shape

