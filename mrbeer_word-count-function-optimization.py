#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


train_X = pd.read_csv(
    '../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, 
    names=["ID","Text"], index_col=0)
train_y = pd.DataFrame.from_csv("../input/training_variants")
train_X = pd.concat([train_X, train_y], axis=1)
train_y = train_X["Class"] - 1
del train_X["Class"]


# In[3]:


test_X = pd.read_csv(
    '../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, 
    names=["ID","Text"], index_col=0)
test_y = pd.DataFrame.from_csv("../input/test_variants")
test_X = pd.concat([test_X, test_y], axis=1)
del test_y


# In[4]:


data = pd.concat([train_X, test_X], axis=0)
print(data.head())
print(data.tail())


# In[5]:


# remove punctuation
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def remove_punctuation(text):
    stemmed_list = tokenizer.tokenize(text)
    stemmed_text = ' '.join(stemmed_list)
    return stemmed_text

# stem words
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def stem_text(text):
    text_list = text.split()
    stemmed_list = [stemmer.stem(word) for word in text_list]
    stemmed_text = ' '.join(stemmed_list)
    return stemmed_text


# In[6]:


def preprocess_text(text):
    text = remove_punctuation(text)
    text = stem_text(text)
    return text


# In[7]:


demo_text = data["Text"].iat[0][:2000]
print(demo_text, '\n')
demo_text = preprocess_text(demo_text)
print(demo_text)


# In[8]:


def word_count_v1(text, word):
    count = 0
    text = text.split()
    for word_i in text:
        if word_i == word:
            count += 1
    return count

print('Normal word counting function in python')
print(word_count_v1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1(demo_text, "cdk10")')


# In[9]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[10]:


get_ipython().run_cell_magic('cython', '', 'def c_word_counts_v1(word_list, str word):\n    cdef int count = 0\n    for word_i in word_list:\n        if word_i == word:\n            count += 1\n    return count')


# In[11]:


def word_count_v1_1(text, word):
    text = text.split()
    return c_word_counts_v1(text, word)

print('Changing word counting from Python to Cython')
print(word_count_v1_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1_1(demo_text, "cdk10")')


# In[12]:


get_ipython().run_cell_magic('cython', '', 'def c_word_counts_v1_1(word_list, str word):\n    cdef int count = 0\n    cdef str word_i\n    for word_i in word_list:\n        if word_i == word:\n            count += 1\n    return count')


# In[13]:


def word_count_v1_1_1(text, word):
    text = text.split()
    return c_word_counts_v1_1(text, word)

print('Declaring in Cython word_count that word_i would also be str')
print(word_count_v1_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1_1(demo_text, "cdk10")')


# In[14]:


get_ipython().run_cell_magic('cython', '', 'def c_word_counts_v2(word_list, str word):\n    cdef int count = 0\n    cdef int i\n    for i in range(len(word_list)):\n        if word_list[i] == word:\n            count += 1\n    return count')


# In[15]:


def word_count_v1_2(text, word):
    text = text.split()
    return c_word_counts_v2(text, word)

print('Using for instead of \"foreach\" in cython word_count')
print(word_count_v1_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1_1(demo_text, "cdk10")')


# In[16]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v1_3(str text, str word):\n    cdef int count = 0\n    cdef str word_i\n    text_list = text.split()\n    for word_i in text_list:\n        if word_i == word:\n            count += 1\n    return count')


# In[17]:


print('Inserting the text.split() to the Cython function, it might be optimized as well')
print(word_count_v1_3(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1_3(demo_text, "cdk10")')


# In[18]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v1_4(str text, str word):\n    cdef int count = 0\n    cdef int i\n    text_list = text.split()\n    for i in range(len(text_list)):\n        if text_list[i] == word:\n            count += 1\n    return count')


# In[19]:


print('Using for instead of \"foreach\" in the unified cython word_count')
print(word_count_v1_4(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v1_4(demo_text, "cdk10")')


# In[20]:


def word_count_v2(text, word):
    count = 0
    n_chars = len(word)
    for i in range(len(text) - n_chars):
        if text[i:i+n_chars] == word:
            count += 1
    return count
print('Using rolling word comparison, removes the need for str.split() with python')
print(word_count_v2(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v2(demo_text, "cdk10")')


# In[21]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v2_1(str text, str word):\n    cdef int count = 0\n    cdef int n_chars = len(word)\n    cdef int i\n    for i in range(len(text) - n_chars):\n        if text[i:i+n_chars] == word:\n            count += 1\n    return count')


# In[22]:


print('Changing to Cython')
print(word_count_v2_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v2_1(demo_text, "cdk10")')


# In[23]:


def word_count_v3(text, word):
    count = 0
    n_chars = len(word)
    for i in range(len(text) - n_chars):
        for j in range(n_chars):
            if text[i+j] != word[j]:
                break
            if j == n_chars-1:
                count += 1
    return count
print('Instead of comparing words, using lazy evaluation. In python')
print(word_count_v3(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3(demo_text, "cdk10")')


# In[24]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_1(str text, str word):\n    cdef int count = 0\n    cdef int word_chars = len(word)\n    cdef int text_chars = len(text)\n    cdef int i\n    cdef int j\n    for i in range(text_chars - word_chars + 1):\n        for j in range(word_chars):\n            if text[i+j] != word[j]:\n                break\n            if j == word_chars-1:\n                count += 1\n    return count')


# In[25]:


print('Switching to Cython')
print(word_count_v3_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_1(demo_text, "cdk10")')


# In[26]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_2(str text, str word):\n    cdef int count = 0\n    cdef int i\n    cdef int j\n    cdef int word_chars = len(word)\n    for i in range(len(text) - word_chars + 1):\n        for j in range(word_chars):\n            if text[i+j] != word[j]:\n                break\n            if j == word_chars-1:\n                count += 1\n    return count')


# In[27]:


print('removing declaration for text_chars')
print(word_count_v3_2(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_2(demo_text, "cdk10")')


# In[28]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_3(str text, str word):\n    cdef int count = 0\n    cdef str text_cache\n    cdef int i\n    cdef int j\n    cdef int word_chars = len(word)\n    for i in range(len(text)):\n        text_cache = text[i:i+word_chars]\n        for j in range(word_chars):\n            if text_cache[j] != word[j]:\n                break\n            if j == word_chars-1:\n                count += 1\n    return count')


# In[29]:


print('Trying to cache the word for each iteration')
print(word_count_v3_3(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_3(demo_text, "cdk10")')


# In[30]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_4(str text, str word):\n    cdef int count = 0\n    cdef int i = 0\n    cdef int j\n    cdef int word_chars = len(word)\n    while i < (len(text) - word_chars + 1):\n        for j in range(word_chars):\n            if text[i+j] != word[j]:\n                break\n            if j == word_chars-1:\n                count += 1\n        i += 1\n    return count')


# In[31]:


print('Switching to while')
print(word_count_v3_4(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_4(demo_text, "cdk10")')


# In[32]:


get_ipython().run_cell_magic('cython', '', "def word_count_v3_5(str text, str word):\n    cdef int count = 0\n    cdef int i = 0\n    cdef int j\n    cdef int word_chars = len(word)\n    while i < (len(text) - word_chars + 1):\n        for j in range(word_chars):\n            if text[i+j] != word[j]:\n                break\n            if j == word_chars-1:\n                count += 1\n                i += word_chars  # after the word ends there's a space so it can move word_chars+1 chars\n        i += 1\n    return count")


# In[33]:


print('Caching len(word)')
print(word_count_v3_5(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_5(demo_text, "cdk10")')


# In[34]:


get_ipython().run_cell_magic('cython', '', "def word_count_v3_6(str text, str word):\n    cdef int count = 0\n    cdef int i = 0\n    cdef int j\n    cdef int word_chars = len(word)\n    cdef int text_scan_end = (len(text) - word_chars + 1)\n    cdef int word_scan_end = word_chars - 1\n    while i < text_scan_end:\n        for j in range(word_chars):\n            if text[i+j] != word[j]:\n                break\n            if j == word_scan_end:\n                count += 1\n                i += word_chars  # after the word ends there's a space so it can move word_chars+1 chars\n        i += 1\n    return count")


# In[35]:


print('Caching (len(text) - word_chars + 1), word_chars - 1')
print(word_count_v3_6(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_6(demo_text, "cdk10")')


# In[36]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_7(str text, str word):\n    cdef int count = 0\n    cdef int i = 0\n    cdef int j\n    cdef int word_chars = len(word)\n    cdef int text_scan_end = (len(text) - word_chars + 1)\n    cdef int word_scan_end = word_chars - 1\n    while i < text_scan_end:\n        if i:\n            if text[i-1] == " ":                \n                for j in range(word_chars):\n                    if text[i+j] != word[j]:\n                        break\n                    if j == word_scan_end:\n                        if i == text_scan_end + 1:\n                            count += 1\n                        else:\n                            if text[i+j+1] == " ":\n                                count += 1\n                        i += word_chars  # after the word ends there\'s a space so it can move word_chars+1 chars\n        else:\n            for j in range(word_chars):\n                if text[i+j] != word[j]:\n                    break\n                if j == word_scan_end:\n                    if i == text_scan_end + 1:\n                        count += 1\n                    else:\n                        if text[i+j+1] == " ":\n                            count += 1\n                    i += word_chars  # after the word ends there\'s a space so it can move word_chars+1 chars\n        i += 1\n    return count')


# In[37]:


print('Starting to check only if it is a start of a word, and count only when it is not a part of a longer word')
print(word_count_v3_7(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_7(demo_text, "cdk10")')


# In[38]:


get_ipython().run_cell_magic('cython', '', 'def word_count_v3_7_1(str text, str word):\n    cdef int count = 0\n    cdef int i\n    cdef int j\n    cdef int word_chars = len(word)\n    cdef int text_scan_end = (len(text) - word_chars + 1)\n    cdef int word_scan_end = word_chars - 1\n    cdef bint start_word_flag\n    \n    while i < text_scan_end:\n        start_word_flag = False\n        if i:\n            if text[i-1] == " ":\n                start_word_flag = True\n        else:\n            start_word_flag = True\n            \n        if start_word_flag:\n            for j in range(word_chars):\n                if text[i+j] != word[j]:\n                    break\n                if j == word_scan_end:\n                    if i == text_scan_end + 1:\n                        count += 1\n                    else:\n                        if text[i+j+1] == " ":\n                            count += 1\n                    i += word_chars  # after the word ends there\'s a space so it can move word_chars+1 chars\n        i += 1\n    return count')


# In[39]:


print('Refactioring the monstrosity')
print(word_count_v3_7_1(demo_text, "cdk10"))
get_ipython().run_line_magic('timeit', 'word_count_v3_7_1(demo_text, "cdk10")')

