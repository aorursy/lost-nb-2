#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
print(df.shape)
df.head()


# In[3]:


family_size_dict = df[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]
choice_dict = df[cols].to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))

def cost_function(prediction):

    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):

        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty

submission = pd.read_csv('../input/santa-workshop-tour-2019/sample_submission.csv')
cost_function(submission['assigned_day'])


# In[4]:


ary_n_people = df.n_people.values
ary_n_people


# In[5]:


ary_choices = df.iloc[:, 1:-1].values-1
ary_choices


# In[6]:


N_FAMILY = len(df)
N_DAYS = 100
data = np.zeros((N_FAMILY, N_DAYS))
for i, d in enumerate(submission['assigned_day']):
    data[i, d-1]=1
# for i, d in enumerate(ary_choices[:,0]):
#     data[i, d-1]=1
data


# In[7]:


N_STATE = 11 * 2**10
N_STATE


# In[8]:


def get_state(data, family_id, day):
    choice_order = np.where(ary_choices[family_id] == day)[0]
    if choice_order.shape[0] > 0:
        choice_order = choice_order[0]
    else:
        choice_order = 10
    occupancy_flag = np.zeros(10)
    for i, choiced_day in enumerate(ary_choices[family_id]):
        if (data[:,choiced_day] * ary_n_people).sum() > 300:
            occupancy_flag[i] = 1
    return choice_order, occupancy_flag

get_state(data, 0, np.argmax(data[0]))


# In[9]:


def get_state_row(data, family_id):
    day = np.argmax(data[family_id])
    choice_order, occupancy_flag = get_state(data, family_id, day)
    res = choice_order + 11 * np.sum(occupancy_flag * np.array([512,256,128,64,32,16,8,4,2,1]))
    return res.astype(int)

get_state_row(data, 0)


# In[10]:


N_ACTION = 12


# In[11]:


def get_action(next_state_row, episode, q_table):
    epsilon = 0.5 * (0.99** episode)
    if epsilon <= np.random.uniform(0, 1): 
        next_action = np.argmax(q_table[next_state_row])
    else:
        next_action = np.random.choice(range(N_ACTION))
    return next_action


# In[12]:


def get_score(data):
    pred = 1+np.argmax(data, axis=1)
    return round(cost_function(pred))

get_score(data)


# In[13]:


def update_data(data, family_id, action):
    if action<10:
        # move to choice_0 - choice_9
        new_day = ary_choices[family_id, action]
    elif action == 10:
        # no action
        return data
    elif action == 11:
        # move most less people day
        new_day = np.argmin([np.sum(data[:,i]*ary_n_people) for i in range(100)])
    new_data = data.copy()
    new_row = np.zeros(N_DAYS)
    new_row[new_day] = 1
    new_data[family_id] = new_row
    return new_data

get_score(update_data(data, 0, 11))


# In[14]:


def make_qtable(*size):
    return np.random.uniform(low = -1, high = 1, size = size)

make_qtable(N_STATE, N_ACTION).shape


# In[15]:


def update_qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.50
    next_max = max(q_table[next_state])
    q_table[state, action] = (1-alpha)*q_table[state, action] +    alpha * (reward + gamma * next_max)
    return q_table


# In[16]:


def learn(data, q_table=None, n_loop=10_000, step=10):
    if not hasattr(q_table, 'shape'):
        q_table = make_qtable(N_STATE, N_ACTION)
    history = []
    best_data = data.copy()
    best_score = get_score(data)
    score = best_score
    reward = 0   
    with tqdm(total=n_loop) as pbar:
        for episode in range(n_loop):
            new_data = best_data.copy()
            state = get_state_row(new_data, 0)
            action = np.argmax(q_table[state])

            for family_id in np.random.choice(range(N_FAMILY), step):
                next_state = get_state_row(new_data, family_id)
                q_table = update_qtable(q_table, state, action, reward, next_state)
                action = get_action(next_state, episode, q_table)
                state = next_state
                new_data = update_data(new_data, family_id, action)

            new_score = get_score(new_data)
            reward = np.clip(score - new_score, -1, 1)
            score = new_score

            if best_score > new_score:
                best_data = new_data.copy()
                best_score = new_score
            history.append(best_score)
            pbar.set_description(f'best_score={best_score:,}')
            pbar.update()
    plt.plot(history)
    plt.show()
    return best_data, q_table


# In[17]:


best_data, q_table = learn(data, n_loop=30_000, step=5)


# In[18]:


for i in [3,5,10,15,20,30,50,100]:
    print(f'step={i}')
    _, _ = learn(best_data, n_loop=100, step=i)


# In[19]:


for i in range(100):
    best_data, _ = learn(best_data, step=3)


# In[20]:


pred = np.argmax(best_data, axis=1)+1
pred


# In[21]:


cost_function(pred)


# In[22]:


submission['assigned_day'] = pred
submission.to_csv('submission.csv', index=False)
submission.head()

