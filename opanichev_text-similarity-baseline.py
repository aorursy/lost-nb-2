#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np 
import pandas as pd
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm
from Levenshtein import ratio as levenshtein_distance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from scipy import spatial


# In[2]:


n_answers = 1


# In[3]:


html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>',              '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'of', 'in', 'and', 'on',          'what', 'where', 'when', 'which'] + html_tags

def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, '')
    x = re.sub(' +', ' ', x)
    return x

bin_question_tokens = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can']
stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

def predict(json_data, annotated=False):
    # Parse JSON data
    candidates = json_data['long_answer_candidates']
    candidates = [c for c in candidates if c['top_level'] == True]
    doc_tokenized = json_data['document_text'].split(' ')
    question = json_data['question_text']
    question_s = question.split(' ') 
    if annotated:
        ann = json_data['annotations'][0]

    # TFIDF for the document
    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words)
    tfidf.fit([json_data['document_text']])
    q_tfidf = tfidf.transform([question]).todense()

    # Find the nearest answer from candidates
    distances = []
    scores = []
    i_ann = -1
    p_cnt = 1
    for i, c in enumerate(candidates):
        s, e = c['start_token'], c['end_token']
        t = ' '.join(doc_tokenized[s:e])
        distances.append(levenshtein_distance(clean(question), clean(t)))
        
        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        
        # See this kernel https://www.kaggle.com/petrov/first-long-paragraph
        if doc_tokenized[s] == '<P>':
            score += 0.25**p_cnt
            p_cnt += 1
        
#         score = 0
        
#         for w in doc_tokenized[s:e]:
#             if w in q_s:
#                 score += 0.1

        scores.append(score)
#     print(scores)
    # Format results
#     ans = candidates[np.argmin(distances)]
#     ans = candidates[np.argmax(scores)]
    ans = (np.array(candidates)[np.argsort(scores)])[-n_answers:].tolist()
    
    if np.max(scores) < 0.2:
        ans_long = ['-1:-1']
        ans = [{'start_token': 0, 'end_token': 0}]
    else:
#         ans_long = str(ans['start_token']) + ':' + str(ans['end_token'])
        ans_long = [str(a['start_token']) + ':' + str(a['end_token']) for a in ans]
    if question_s[0] in bin_question_tokens:
        ans_short = 'YES'
    else:
        ans_short = ''
        
    # Preparing data for debug
    if annotated:
        ann_long_text = ' '.join(doc_tokenized[ann['long_answer']['start_token']:ann['long_answer']['end_token']])
        if ann['yes_no_answer'] == 'NONE':
            if len(json_data['annotations'][0]['short_answers']) > 0:
                ann_short_text = ' '.join(doc_tokenized[ann['short_answers'][0]['start_token']:ann['short_answers'][0]['end_token']])
            else:
                ann_short_text = ''
        else:
            ann_short_text = ann['yes_no_answer']
    else:
        ann_long_text = ''
        ann_short_text = ''
        
    ans_long_text = [' '.join(doc_tokenized[a['start_token']:a['end_token']]) for a in ans]
    if len(ans_short) > 0 or ans_short == 'YES':
        ans_short_text = ans_short
    else:
        ans_short_text = '' # Fix when short answers will work
                    
    return ans_long, ans_short, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text


# In[4]:


get_ipython().run_cell_magic('time', '', "ids = []\nanns = []\npreds = []\n\n# Debug data\nquestions = []\nann_texts = []\nans_texts = []\n\nn_samples = 500\n\nwith open('/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl', 'r') as json_file:\n    cnt = 0\n    for line in tqdm(json_file):\n        json_data = json.loads(line)\n\n        l_ann = str(json_data['annotations'][0]['long_answer']['start_token']) + ':' + \\\n            str(json_data['annotations'][0]['long_answer']['end_token'])\n        if json_data['annotations'][0]['yes_no_answer'] == 'NONE':\n            if len(json_data['annotations'][0]['short_answers']) > 0:\n                s_ann = str(json_data['annotations'][0]['short_answers'][0]['start_token']) + ':' + \\\n                    str(json_data['annotations'][0]['short_answers'][0]['end_token'])\n            else:\n                s_ann = ''\n        else:\n            s_ann = json_data['annotations'][0]['yes_no_answer']\n\n        l_ans, s_ans, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text = predict(json_data, annotated=True)\n        \n        ids += [str(json_data['example_id']) + '_long']*len(l_ans)\n        ids.append(str(json_data['example_id']) + '_short')\n        \n        anns += [l_ann]*len(l_ans)\n        anns.append(s_ann)\n        \n        preds += l_ans\n        preds.append(s_ans)\n        questions += [question]*len(l_ans)\n        questions.append(question)\n        ann_texts += [ann_long_text]*len(l_ans)\n        ann_texts.append(ann_short_text)\n        ans_texts += ans_long_text\n        ans_texts.append(ans_short_text)\n        \n        cnt += 1\n        if cnt >= n_samples:\n            break\n        \ntrain_ann = pd.DataFrame()\ntrain_ann['example_id'] = ids\ntrain_ann['question'] = questions\ntrain_ann['CorrectString'] = anns\ntrain_ann['CorrectText'] = ann_texts\nif len(preds) > 0:\n    train_ann['PredictionString'] = preds\n    train_ann['PredictionText'] = ans_texts\n    \ntrain_ann.to_csv('train_data.csv', index=False)\ntrain_ann.head(10)")


# In[5]:


# Should be replaced by code from https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py
f1 = f1_score(train_ann['CorrectString'].values, train_ann['PredictionString'].values, average='micro')
print(f'F1-score: {f1:.4f}')


# In[6]:


get_ipython().run_cell_magic('time', '', "ids = []\nanns = []\npreds = []\n\n# Debug data\nquestions = []\nann_texts = []\nans_texts = []\n\nwith open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'r') as json_file:\n    cnt = 0\n    for line in tqdm(json_file):\n        json_data = json.loads(line)\n        \n        l_ans, s_ans, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text = predict(json_data)\n\n        ids += [str(json_data['example_id']) + '_long']*len(l_ans)\n        ids.append(str(json_data['example_id']) + '_short')\n        preds += l_ans\n        preds.append(s_ans)\n        questions += [question]*len(l_ans)\n        questions.append(question)\n        ans_texts += ans_long_text\n        ans_texts.append(ans_short_text)\n         \n#         cnt += 1\n#         if cnt >= n_samples:\n#             break\n        \nsubm = pd.DataFrame()\nsubm['example_id'] = ids\nsubm['question'] = questions\nsubm['PredictionString'] = preds\nsubm['PredictionText'] = ans_texts\nsubm.to_csv('test_data.csv', index=False)\n\ng = subm[['example_id', 'PredictionString']].groupby('example_id').agg(lambda x: ' '.join(x) if len(x) > 1 else x).reset_index()\ng.to_csv('submission.csv', index=False)\n\nsubm.head(10)")


# In[7]:


g.head()


# In[ ]:




