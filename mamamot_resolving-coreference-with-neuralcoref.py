#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook


# In[2]:


MODEL_URL = "https://github.com/huggingface/neuralcoref-models/releases/"             "download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz"


# In[3]:


get_ipython().system('pip install spacy==2.0.12')


# In[4]:


get_ipython().system('pip install {MODEL_URL}')


# In[5]:


get_ipython().system('python -m spacy download en_core_web_md')


# In[6]:


import en_coref_md

nlp = en_coref_md.load()


# In[7]:


test_sent = "The doctor came in. She held a paper in her hand."


# In[8]:


doc = nlp(test_sent)


# In[9]:


doc._.has_coref


# In[10]:


doc._.coref_clusters


# In[11]:


doc._.coref_clusters[0].main


# In[12]:


doc._.coref_clusters[0].mentions


# In[13]:


def is_inside(offset, span):
    return offset >= span[0] and offset <= span[1]

def is_a_mention_of(sent, pron_offset, entity_offset_a, entity_offset_b):
    doc = nlp(sent)
    if doc._.has_coref:
        for cluster in doc._.coref_clusters:
            main = cluster.main
            main_span = main.start_char, main.end_char
            mentions_spans = [(m.start_char, m.end_char) for m in cluster.mentions                               if (m.start_char, m.end_char) != main_span]
            if is_inside(entity_offset_a, main_span) and                     np.any([is_inside(pron_offset, s) for s in mentions_spans]):
                return "A"
            elif is_inside(entity_offset_b, main_span) and                     np.any([is_inside(pron_offset, s) for s in mentions_spans]):
                return "B"
            else:
                return "NEITHER"
    else:
        return "NEITHER"


# In[14]:


# "The doctor came in. She held a paper in her hand."
entity_offset_a = test_sent.index("doctor")
entity_offset_b = test_sent.index("paper")
pron_offset = test_sent.index("She")

is_a_mention_of(test_sent, pron_offset, entity_offset_a, entity_offset_b)


# In[15]:


gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", 
                       delimiter='\t', index_col="ID")


# In[16]:


gap_train.head()


# In[17]:


def predict(df):
    pred = pd.DataFrame(index=df.index, columns=["A", "B", "NEITHER"]).fillna(False)
    for i, row in tqdm_notebook(df.iterrows()):
        pred.at[i, is_a_mention_of(row["Text"], row["Pronoun-offset"], row["A-offset"], row["B-offset"])] = True
    return pred


# In[18]:


train_preds = predict(gap_train)


# In[19]:


gap_train["NEITHER"] = np.logical_and(~gap_train["A-coref"], ~gap_train["B-coref"])


# In[20]:


gap_train[["A-coref", "B-coref", "NEITHER"]].describe()


# In[21]:


train_preds.describe()


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(gap_train[["A-coref", "B-coref", "NEITHER"]], train_preds[["A", "B", "NEITHER"]]))

