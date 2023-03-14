#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


# In[2]:


df_cities = pd.read_csv('../input/cities.csv')
#10% of dataframe
df_cities.sample(frac=0.1, replace = True)


# In[3]:


# source = https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths
# To improve the performance, instead of checking whether each member is a prime, 
# we first a generate a list where each element tells whether the number indicated 
# by the position is a prime or not. 

# using sieve of eratosthenes
start = time.time()
def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)
prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))
end = time.time()


# In[4]:


#Algorithm run time
print(end - start)


# In[5]:


#https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths
start = time.time()
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

dumbest_path = list(df_cities.CityId[:].append(pd.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(df_cities,dumbest_path)))
end = time.time()


# In[6]:


#Algorithm run time
print(end - start)

