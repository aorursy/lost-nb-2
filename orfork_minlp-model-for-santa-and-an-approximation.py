#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyomo')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pyomo.environ import * 
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.core import * 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data = pd.read_csv('/kaggle/input//santa-workshop-tour-2019/family_data.csv', index_col='family_id')

# Any results you write to the current directory are saved as output.


# In[3]:


#Functions for preparation of cost function. 

def _build_choice_array(data, n_days):
    choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
    choice_array_num = np.full((data.shape[0], n_days + 1), -1)

    for i, choice in enumerate(choice_matrix):
        for d, day in enumerate(choice):
            choice_array_num[i, day] = d
    
    return choice_array_num

def _precompute_penalties(choice_array_num, family_size):
    penalties_array = np.array([
        [
            0,
            50,
            50 + 9 * n,
            100 + 9 * n,
            200 + 9 * n,
            200 + 18 * n,
            300 + 18 * n,
            300 + 36 * n,
            400 + 36 * n,
            500 + 36 * n + 199 * n,
            500 + 36 * n + 398 * n
        ]
        for n in range(family_size.max() + 1)
    ])
    
    penalty_matrix = np.zeros(choice_array_num.shape)
    N = family_size.shape[0]
    for i in range(N):
        choice = choice_array_num[i]
        n = family_size[i]
        
        for j in range(penalty_matrix.shape[1]):
            penalty_matrix[i, j] = penalties_array[n, choice[j]]
    
    return penalty_matrix

def _precompute_accounting(max_day_count, max_diff):
    accounting_matrix = np.zeros((max_day_count+1, max_diff+1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(1, max_day_count+1):
        for diff in range(max_diff+1):
            accounting_cost = (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0)
            accounting_matrix[today_count, diff] = max(0, accounting_cost)
    
    return accounting_matrix


# In[4]:


#Model Section

N_DAYS = 100
N_OPT = 10
N_FAMILY = 5000
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125


choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
family_size = data.n_people.values
choice_array_num = _build_choice_array(data, N_DAYS)

model = ConcreteModel("Santa")

days = range(N_DAYS)
assigned_to = range(N_OPT) 
family = range(N_FAMILY)
occupancy_days = range(MIN_OCCUPANCY,MAX_OCCUPANCY+1)

model.fx = Var(family, assigned_to, within=Binary) 
model.day_size = Var(days, within=NonNegativeReals, bounds=(125,300))
model.day_size_var = Var(days, occupancy_days, within=Binary)

def assign_con_rule(model, i):
    return quicksum(model.fx[i,s] for s in assigned_to) == 1.0
model.assign_con = Constraint(family, rule=assign_con_rule)

def daily_loads_rule(model, j):
    return quicksum(family_size[i]*model.fx[i,s] for i in family for s in assigned_to if j == (choice_matrix[i,s]-1))             == model.day_size[j]
model.daily_loads = Constraint(days, rule=daily_loads_rule)

def load_determine_rule(model, j):
    return quicksum(k*model.day_size_var[j,k] for k in occupancy_days) == model.day_size[j]
model.load_determine = Constraint(days, rule=load_determine_rule)

penalty_matrix = _precompute_penalties(choice_array_num, family_size)
accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY, max_diff=175)

def objective_rule(model):
    penalty_cost = quicksum(penalty_matrix[i,choice_matrix[i][s]]*model.fx[i,s] for i in family for s in assigned_to)
    accounting_cost = quicksum(accounting_matrix[k_1,abs(k_2-k_1)]*model.day_size_var[i,k_1]*model.day_size_var[i+1,k_2]                              for i in range(N_DAYS-1) for k_1 in occupancy_days for k_2 in occupancy_days)
    last_day_acc_cost = quicksum(accounting_matrix[k_1,0]*model.day_size_var[99,k_1] for k_1 in occupancy_days)
    return penalty_cost + accounting_cost + last_day_acc_cost
#
model.object = Objective(rule=objective_rule, sense=minimize)


# In[5]:


#For chekinn model, it is written to lp file.
#model.write(filename="santa_model.lp",io_options={"symbolic_solver_labels":True})

#This section will be open based on selected solver.
#opt = SolverFactory('Solver_Name', executable="Executable_Path")
#opt_success = opt.solve(model, tee=True)
#print("time", opt_success.Solver[0])
#fx_value = {}
#for ib in family:
#    for j in assigned_to:
#        fx_value[(i,j)] = model.fx[i,j].value


# In[6]:


model = ConcreteModel("Santa")

days = range(N_DAYS)
assigned_to = range(N_OPT) 
family = range(N_FAMILY)
occupancy_days = range(MIN_OCCUPANCY,MAX_OCCUPANCY+1)

model.fx = Var(family, assigned_to, within=Binary) 
model.day_size = Var(days, within=NonNegativeReals, bounds=(125,300))
model.day_size_diff = Var(days, within=NonNegativeReals)
model.day_size_devp = Var(days, within=NonNegativeReals)
model.day_size_devn = Var(days, within=NonNegativeReals)

def assign_con_rule(model, i):
    return quicksum(model.fx[i,s] for s in assigned_to) == 1.0
model.assign_con = Constraint(family, rule=assign_con_rule)

def daily_loads_rule(model, j):
    return quicksum(family_size[i]*model.fx[i,s] for i in family for s in assigned_to if j == (choice_matrix[i,s]-1))             == model.day_size[j]
model.daily_loads = Constraint(days, rule=daily_loads_rule)

#The first approach constraints
def load_determine_1_rule(model, j):
    return model.day_size[j] - model.day_size[j+1] <= model.day_size_diff[j]
model.load_determine_1 = Constraint(range(N_DAYS-1), rule=load_determine_1_rule)

def load_determine_2_rule(model, j):
    return model.day_size[j+1] - model.day_size[j] <= model.day_size_diff[j]
model.load_determine_2 = Constraint(range(N_DAYS-1), rule=load_determine_2_rule)

#The second approach constraints
def load_determine_3_rule(model, j):
    return model.day_size[j] - model.day_size_devn[j] + model.day_size_devp[j] == 210
model.load_determine_3 = Constraint(days, rule=load_determine_3_rule)


penalty_matrix = _precompute_penalties(choice_array_num, family_size)
accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY, max_diff=175)

def objective_rule(model):
    penalty_cost = quicksum(penalty_matrix[i,choice_matrix[i][s]]*model.fx[i,s] for i in family for s in assigned_to)
    day_diff_ = quicksum(100*model.day_size_diff[j] for j in range(N_DAYS-1)) #Objective function part for the first approach
    #day_diff_ = quicksum(100*model.day_size_devn[j] for j in days) + quicksum(100*model.day_size_devp[j] for j in days) #Objective function part for the second approach
    return penalty_cost + day_diff_
#
model.object = Objective(rule=objective_rule, sense=minimize)

