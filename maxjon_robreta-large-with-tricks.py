#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import json
import re


# In[2]:


pd_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


# In[3]:


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


# In[4]:


def remove_html(text):
    text = re.sub("&quot;", '"', text)
    text = re.sub("&gt;", ">", text)
    text = re.sub("&lt;", "<", text)
    text = re.sub("&le;", "≤", text)
    text = re.sub("&ge;", "≥", text)
    text = re.sub("&amp;", "&", text)
    return text

def add_html(text):
    text = re.sub("&", "&amp;", text)   
    text = re.sub('"', "&quot;",  text)
    text = re.sub(">", "&gt;", text)
    text = re.sub("<", "&lt;", text)
    text = re.sub("≤", "&le;", text)
    text = re.sub("≥", "&ge;", text)
    return text


# In[5]:


def f(selected):
     return " ".join(set(selected.lower().split()))


# In[6]:


get_ipython().system('mkdir data')
get_ipython().system('mkdir results_roberta_large')


# In[7]:


output = {}
output['version'] = 'v1.0'
output['data'] = []
train = np.array(pd_train)

converted = 0
for line in train:
    paragraphs = []
    context = line[1]
    qas = []
    qid = line[0]
    answers = []
    orig_answer = line[2]
    question = line[3]
    if type(orig_answer) != str or type(context) != str or type(question) != str:
        print(context, type(context))
        print(orig_answer, type(orig_answer))
        print(question, type(question))
        continue
    
    # get start index and then covert html-tag
    answer_starts = find_all(context, orig_answer)
    context = remove_html(context)

    for answer_start in answer_starts:
        # get new answer, if there are no html-tags answer will be the same as given in train.csv
        answer = context[answer_start:answer_start+len(orig_answer)]
        answers.append({'answer_start': answer_start, 'text': answer})
        if orig_answer != answer:
#             print("original:", orig_answer)
#             print("new:", answer)
            converted += 1
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/train.json', 'w') as outfile:
    json.dump(output, outfile)
    
print("converted:" , converted)


# In[8]:


# Convert pd_test data

output = {}
output['version'] = 'v1.0'
output['data'] = []

# covert html-tags
pd_test.text = pd_test.text.map(remove_html)
test_array = np.array(pd_test)

for line in test_array:
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


# In[9]:


get_ipython().system('cd /kaggle/input/pytorchtransformers/transformers-2.5.1; pip install .')


# In[10]:


get_ipython().system('python /kaggle/input/pytorchtransformers/transformers-2.5.1/examples/run_squad.py --model_type roberta --model_name_or_path /kaggle/input/roberta-large-model/results_roberta_large/ --do_lower_case --do_eval --data_dir ./data --cache_dir /kaggle/input/cached-roberta-large-pretrained/cache --train_file train.json --predict_file test.json --learning_rate 2.5e-5 --num_train_epochs 3 --max_seq_length 192 --doc_stride 64 --output_dir results_roberta_large --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=16 --save_steps=100000')


# In[11]:


predictions = json.load(open('results_roberta_large/predictions_.json', 'r'))

for i in range(len(pd_test)):
    id_ = pd_test['textID'][i]
    selected_text = predictions[id_]
    text = pd_test['text'][i]
    text = " ".join(text.split())
    starts = find_all(text, selected_text)
    
    # if none of (&><≤≥") exist before the end of the answer nothing will change by adding html-tags 
    # if there is more than one answer in the context we cannot know which to modify
    if len(starts) == 1 and any(c in text[:starts[0]+len(selected_text)] for c in list('&><≤≥"')):
        text = add_html(text)
        start = starts[0]
#         print("original:", selected_text)
        selected_text = text[start:start+len(selected_text)]
#         print("new:", selected_text)
    
    if pd_test['sentiment'][i] == 'neutral': # neutral postprocessing
        pd_test.loc[i, 'selected_text'] = pd_test['text'][i]
    else:
        pd_test.loc[i, 'selected_text'] = selected_text

pd_test.selected_text = pd_test.selected_text.map(f)


# In[12]:


pd_test.head()


# In[13]:


# Save the submission file.
pd_test[["textID", "selected_text"]].to_csv("submission.csv", index=False)


# In[ ]:




