#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import json


# In[2]:


pd_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


# In[3]:


train = np.array(pd_train)
test = np.array(pd_test)


# In[4]:


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


# In[5]:


get_ipython().system('mkdir data')


# In[6]:


# Convert training data

output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in train:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    answers = []
    answer = line[2]
    if type(answer) != str or type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answer_starts = find_all(context, answer)
    for answer_start in answer_starts:
        answers.append({'answer_start': answer_start, 'text': answer})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/train.json', 'w') as outfile:
    json.dump(output, outfile)


# In[7]:


# Convert test data

output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in test:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    if type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answers = []
    answers.append({'answer_start': 1000000, 'text': '__None__'})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/test.json', 'w') as outfile:
    json.dump(output, outfile)


# In[8]:


get_ipython().system('cd /kaggle/input/pytorchtransformers/transformers-2.5.1; pip install .')


# In[9]:


get_ipython().system('mkdir results_roberta_large')


# In[10]:


get_ipython().system('python /kaggle/input/pytorchtransformers/transformers-2.5.1/examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --do_lower_case --do_train --do_eval --data_dir ./data --cache_dir /kaggle/input/cached-roberta-large-pretrained/cache --train_file train.json --predict_file test.json --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 192 --doc_stride 64 --output_dir results_roberta_large --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=16 --save_steps=100000')


# In[11]:


def post_process(selected):
    return " ".join(set(selected.lower().split()))


# In[12]:


# Copy predictions to submission file.
predictions = json.load(open('results_roberta_large/predictions_.json', 'r'))
submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if pd_test['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = pd_test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = post_process(predictions[id_])


# In[13]:


submission.head()


# In[14]:


# Save the submission file.
submission.to_csv('submission.csv', index=False)

