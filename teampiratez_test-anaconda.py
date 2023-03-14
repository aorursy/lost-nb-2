#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import tree
import csv
train_data = []
test_data = []
test_target = []
train_target = []
n = []
m = []
l = 0

with open('../input/train.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
        l = l + 1
		n = [row['Semana'],row['Agencia_ID'], row['Canal_ID'],row['Ruta_SAK'],row['Cliente_ID'],row['Producto_ID']]
		train_data.append(n)
		m = [row['Demanda_uni_equil']]
		train_target.append(m)
        if(l > 1000):
            break
 print(train_data)
		

