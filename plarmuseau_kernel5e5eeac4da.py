#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import json
import numpy as np 
import pandas as pd
import re
from tqdm import tqdm_notebook as tqdm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


def readjson(train,QorD):
    with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-'+train+'.jsonl', 'r') as json_file:
        cnt = 0
        temp=[]
        tempdoc=[]
        for line in tqdm(json_file,total=100):
            json_data = json.loads(line) 
            # Collect annotations
            if train=='train':
                start = json_data['annotations'][0]['long_answer']['start_token']
                end = json_data['annotations'][0]['long_answer']['end_token']

                # Collect short annotations
                if json_data['annotations'][0]['yes_no_answer'] == 'NONE':
                    if len(json_data['annotations'][0]['short_answers']) > 0:
                        sans = str(json_data['annotations'][0]['short_answers'][0]['start_token']) + ':' +                             str(json_data['annotations'][0]['short_answers'][0]['end_token'])
                    else:
                        sans = ''
                else:
                    sans = json_data['annotations'][0]['yes_no_answer']
        
                if QorD=='d':
                    tempdocline=[json_data['document_text'][:5000]]
                else:
                    templine=[start,end,sans,json_data['question_text']]

            else:
                if QorD=='d':
                    tempdocline=[json_data['document_text'] ]
                else:
                    templine=[json_data['question_text'] ]
            if QorD=='d':
                tempdoc.append(tempdocline)
            else:
                temp.append(templine)
        if QorD=='q':
            filenm=train+'_question_text.csv'
            pd.DataFrame(temp).to_csv(filenm)
            del temp
        else:
            filenm=train+'split_document_text.csv'        
            pd.DataFrame( wiki_tagsplit(pd.DataFrame(tempdoc)[0],0)).to_csv(filenm)
            
    return

#  find the start stop  token position of a sentence 
def find_start_end_token(txt,sent):
    pos=txt.find(np.str(sent) )
    start=len(txt[:pos-1].split(' '))
    end=start+len(np.str(sent).split(' '))
    return start-1,end-1

from bs4 import BeautifulSoup

def wiki_tagsplit(html,wi):
    temp=[]
    #html = trainW.iloc[wi]['wiki'] 
    for hi in html:
        
        soup = BeautifulSoup(hi, 'html.parser')
        for ti in ['h1','h2','p','table','tr']:  #splitting tags extracting this features
            allep=soup.find_all(ti) #p paragraph
            for pi in allep:
                start,stop=find_start_end_token(hi,pi.get_text())
                if start>1:
                    line=[wi,ti,start,stop,pi.get_text()]
                    temp.append(line)
                    
        wi=wi+1
    return pd.DataFrame(temp,columns=['id','tag','start','stop','txt'])

readjson('test','q')
readjson('test','d')

readjson('train','q')
readjson('train','d')

#readjson('train','document_text')


# In[3]:


subm=pd.read_csv('../input/tensorflow2-question-answering/sample_submission.csv')
subm.to_csv('submit.csv')

