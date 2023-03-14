#!/usr/bin/env python
# coding: utf-8

# In[ ]:


RUNNING_ON_KAGGLE = True
if RUNNING_ON_KAGGLE:
    def spell(word):
        return word
else: 
    from autocorrect import spell
    
spell('horse')


# In[ ]:


from multiprocessing import Pool, cpu_count
import pandas as pd
import re


def get_known_words(word_embeddings_file):
    words = set()
    with open(word_embeddings_file,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            words.add(values[0].lower())
    return words


EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
if RUNNING_ON_KAGGLE:
    words = set()
else:
    words = get_known_words(EMBEDDING_FILE)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def spell_check(chunk):
    fixed_rows = []
    for i,row in chunk.iterrows():
        fxd_words = []
        comment = row['comment_text'].lower()
        comment = re.sub('[^a-zA-Z ]+', '', comment)
        for w in comment.split():
            if w is None:
                continue
            if w in words or len(w) > 24:
                fxd_words.append(w)
            else:
                fxd_words.append(spell(w).lower())
        sp_comment = ' '.join(fxd_words)
        fixed_rows.append((row[0],sp_comment))
    return fixed_rows


PROC_COUNT = cpu_count()
CHUNK_SIZE = 1024
pool = Pool(PROC_COUNT)

# Uncomment line below to run
for set_name in [] #['train', 'test']:
    source = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/'+set_name+'.csv') # remove [:100] to proces all examples
    source['comment_text'] = source['comment_text'].astype(str)
    result = source.copy()

    fixed_rows = pool.map(spell_check,chunker(source,CHUNK_SIZE))
    for fxd_row in fixed_rows:
        for index,fixed_comment in fxd_row:
            result.set_value(index,'comment_text',fixed_comment)

    if RUNNING_ON_KAGGLE:
        print(result)
    else:
        result.to_csv('sp_check_'+set_name+'.csv')


# In[ ]:




