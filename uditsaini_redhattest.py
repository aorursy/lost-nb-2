#!/usr/bin/env python
# coding: utf-8

# In[2]:


library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)


train=fread('../input/act_train.csv') %>% as.data.frame()
test=fread('../input/act_test.csv') %>% as.data.frame()



# In[3]:


#people data frame
people=fread('../input/people.csv') %>% as.data.frame()
people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

#reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'


# In[ ]:




