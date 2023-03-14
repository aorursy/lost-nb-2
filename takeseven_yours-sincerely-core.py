#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', "\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('Q', {\n    help : 'run all cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);")


# In[2]:


gFirstTime = True
gNumEpochs = 8
gTrainingBatchSize = 256
gModelName = "GPULSTM"
gSelectedEmbeddings = ["GLOVE","PARAGRAM","WIKINEWS"]
#gSelectedEmbeddings = ["GLOVE"]
gTrainableEmbeddings = False
gWIP = False
gInspect = False
gLimit = False
gExternalData = False
gGenerateNewData = False
gExecuteMain = False


# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import string, time
import gc
from collections import defaultdict, Counter
import numpy as np # linear algebra
from IPython.core.display import display, HTML
import random
import re
import os, psutil, sys, pickle, operator
import csv
import multiprocessing as mpc
import textwrap
import nltk.data
from shutil import copyfile, rmtree

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import multi_gpu_model, Sequence
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.callbacks import Callback

from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
import spacy
from spacy import displacy
import en_core_web_sm
from nltk.corpus import stopwords

from tqdm.autonotebook import tqdm
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
process = psutil.Process(os.getpid())
gCurrentMemory = process.memory_info().rss
gStopWords = set(stopwords.words('english'))
SEED = 2019
np.random.seed(SEED)

def show_html(x):
    display(HTML(x))
    
show_html("<hr/><h1 align='center'>Levers of the Machine</h1><hr/>")
show_html(f"<big><big>Execute Main?   <b>{gExecuteMain}</b></big></big>")
show_html(f"<big><big>Logging Enabled?    <b>{gWIP}</b></big></big>")
show_html(f"<big><big>Data Limit Enabled?   <b>{gLimit}</b></big></big>")
show_html(f"<big><big>Inspection Enabled?    <b>{gInspect}</b></big></big>")
show_html(f"<big><big>Generate New Data?   <b>{gGenerateNewData}</b></big></big>")
show_html(f"<big><big>Trainable Embeddings?   <b>{gTrainableEmbeddings}</b></big></big>")
show_html("<hr/>")


# In[4]:


def log(x, *argv):
    if gWIP == False:
        return
    x = str(x)
    for arg in argv:
        x = x + str(arg)
    display(HTML(x))

def log_list(items, transpose=False, desc=None, limit=None, force=False):
    if gWIP == False and not force:
        return
    
    if items is None:
        if desc is None:
            desc = "<no desc>"
        if gWIP:
            log("For the list {}, items are None".format(desc))
        return
    
    html = []
    html.append("<table>")
    if desc is not None:
        if not transpose:
            html.append("<th>No.</th>")
        html.append("<th colspan='3'>" + desc + "</th>")
    if transpose:
        html.append("<tr>")
        
    count = 0
    if isinstance(items, dict):
        for key, value in items.items():
            if limit is not None and limit < count:
                break
            if transpose:
                html.append("<td>{} : {}</td>".format(key, value))
            else:
                html.append("<tr><td>{}</td>".format(count))
                html.append("<td>{}</td><td>{}</td></tr>".format(key, value))
            count += 1
    else:
        for item in items:
            if limit is not None and limit < count:
                break
            if transpose:
                html.append("<td>" + str(item) + "</td>")
            else:
                if isinstance(item, tuple) or isinstance(item, list):
                    html.append("<tr><td>{}</td>".format(count))
                    for col in item:
                        html.append("<td>{}</td>".format(col)) 
                else:
                    html.append("<tr><td>{}</td><td>{}</td></tr>".format(count, str(item)))
                count += 1
                
    if transpose:
        html.append("</tr>")            

    html.append("</table>")
    log(''.join(html))
    
def log_current_memory(caption):
    if gWIP == False:
        return
    
    global gCurrentMemory
    tmp = process.memory_info().rss
    log("MEMORY -- {} : {:.4f} MB  ; delta: <b>{:.2f}</b>".format(caption, tmp/(1024*1024), (tmp - gCurrentMemory)/(1024*1024)))
    gCurrentMemory = tmp

def log_dir(path):
    
    dir_list = os.listdir(path)
    pairs = []
    for file in dir_list:
        # Use join to get full file path.
        location = os.path.join(path, file)

        # Get size and add to list of tuples.
        size = os.path.getsize(location)
        modified = time.ctime(os.path.getmtime(location))
        pairs.append((file, str(size/(1024)) + " KB", str(modified)))

        # Sort list of tuples by the first element, size.
        pairs.sort(key=lambda s: s[1])
    
    log_list(pairs, desc="Files from '{}'".format(path))

def remove_file(file):
    
    if file is not None and os.path.isfile(file):
        #log("Removed file...{}".format(file))
        os.remove(file)

def has_gpu():
    return True
    #return len(K.tensorflow_backend._get_available_gpus()) > 0 

def move_files(source_dir, target_dir):
    files = os.listdir(source_dir)
    files.sort()
    for f in files:
        src = source_dir+f
        dst = target_dir+f
        copyfile(src,dst)
    
if False:
    log_dir("../input")
    log_dir("../working")
    log_current_memory("In the beginning")
    log("CPU Count:{}".format(mpc.cpu_count()))
    log(f"Has GPU: {has_gpu()}, count: {len(K.tensorflow_backend._get_available_gpus())}")
    #rmtree('../working/sincerely-gpu-lstm')
    
if gExternalData:
    if os.path.isfile('../working/MATRIX_GOOGLENEWS') == False:
        log("Moving files from input to working")
        move_files('../input/sincerely-gpu-lstm/', '../working/')


# In[5]:


import time
from functools import wraps

PROF_DATA = {}

class_extract_regex = re.compile('^.*(main__.)(.*)\'>$', re.IGNORECASE)

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.perf_counter()

        ret = fn(*args, **kwargs)
        
        elapsed_time = time.perf_counter() - start_time

        if len(args) > 0:
            profile_key = str(type(args[0])) + ":::" + fn.__name__
        else:
            profile_key = fn.__name__
                
        if profile_key not in PROF_DATA:
            PROF_DATA[profile_key] = [0, []]
        PROF_DATA[profile_key][0] += 1
        PROF_DATA[profile_key][1].append(elapsed_time)
    
        return ret
    
    with_profiling.__wrapped__ = fn
    
    return with_profiling

def log_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print ("Function %s called %d times. " % (fname, data[0]))
        print ('Execution time max: %.3f s, average: %.3f s' % (max_time, avg_time))

def get_prof_data():
    headers = ("Class", "Function", "Call Frequency", "Max Time (m)", "Avg Time (m)")
    contents = []
    
    for profile_key, data in PROF_DATA.items():
        max_time = round(1000*max(data[1])/(1000*60), 3)
        avg_time = round(1000*sum(data[1]) / (len(data[1])*1000*60), 3)
                    
        if ':::' in profile_key:
            src = profile_key.split(':::')
            if src is None:
                className = src
            elif class_extract_regex.match(src[0]) is None:
                className = "classmethod" + str(src[0])
            else:
                className = class_extract_regex.match(src[0])[2]
        else:
            className = "-"
            src = ['-', profile_key]
            
        contents.append((className, src[1], data[0], max_time, avg_time))
        
    return headers, contents

def pp_prof_data():
    headers, contents = get_prof_data()
    tmp = (headers,)
    tmp += tuple(contents)
    log_list(tmp)

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
    

class Profiler:
    def __init__(self):
        clear_prof_data()
    
    def start(self):
        clear_prof_data()
    
    def get_data(self):
        return get_prof_data()
    
gProfiler = Profiler()

@profile
def save_binary(obj, file):
    log(f"Saving file: {file}")
    pickle.dump(obj, open(file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

class TestProfiler:
    def __init__(self):
        pass
    
    @profile
    def test_method(self, hello=1):
        log("In Test Method:" + str(hello))
    
    @classmethod
    @profile
    def class_method(cls, hello="hi"):
        log("In Class Method: " + str(hello))


# In[6]:


if False:
    test = TestProfiler()
    clear_prof_data()
    test.test_method()
    test.class_method(hello='hi')
    pp_prof_data()


# In[7]:


class TestMPCWithFileCache:
    def __init__(self, cache=None):
        self.cache = cache
    
    def process(self, cpu_id, cpu_count, cache_file, v1, v2, kv3=[1, 2, 4]):
        self.cache = {"foo":"bar", "Hello" : v1, "Hey": cpu_id*cpu_count}
        save_binary(self.cache, cache_file)
        f = open(cache_file+".csv", "w")
        f.write("key,value\r\n")
        for k, v in self.cache.items():
            f.write("{},{}\r\n".format(k, v))
        f.close()
        
class MPHelper:
    def __init__(self, file_prefix, verbose=False):
        self.output_file = file_prefix
        self.file_prefix = file_prefix + "_{}"
        self.verbose = verbose
        
    @classmethod
    def range_partition(cls, total_count, cpu_id):
        cpu_count = mpc.cpu_count()
        start = 0
        end = 0
        
        n = int(total_count/cpu_count)
        start = cpu_id*n
        end = start + n
        if cpu_id + 1 == cpu_count:
            r = int(total_count % cpu_count)
            end += r
        
        return range(start, end)
    
    # All target methods must have a signature : func(self, cpu_id, cpu_count, cache_file, ...)
    def map_process(self, target, *args, **kwargs):
        cpu_count = mpc.cpu_count()
        jobs = []
        for i in range(cpu_count):
            mpc_args = ()
            mpc_args += tuple([i])
            mpc_args += tuple([cpu_count])
            mpc_args += tuple([self.file_prefix.format(i)])
            mpc_args += args
            if self.verbose:
                log_list(mpc_args, desc="Starting process on CPU {}".format(i))
            p = mpc.Process(target=target, args=mpc_args, kwargs=kwargs)
            jobs.append(p)
        
        def spawn():
            [j.start() for j in jobs]
            [j.join() for j in jobs]
            
        
        is_wrapped = hasattr(target, "__wrapped__")
        if is_wrapped == False:
            spawn()
        else:
            start_time = time.perf_counter()
            spawn()
            elapsed_time = time.perf_counter() - start_time
        
            class_name = str(target.__self__)
            profile_key = class_name + ":::" + target.__name__        
            if profile_key not in PROF_DATA:
                PROF_DATA[profile_key] = [0, []]
            PROF_DATA[profile_key][0] += len(jobs)
            PROF_DATA[profile_key][1].append(elapsed_time)

        
    # Helper function
    def cpu_cache_name(self, cpu_id=None, is_csv=False):
        
        count = cpu_id
        if count is None:
            count = mpc.cpu_count()
        for cpu_index in range(count):
            retval = self.file_prefix.format(cpu_index)
            if is_csv:
                retval += ".csv"
            if os.path.isfile(retval):
                yield retval
            else:
                #if gWIP:
                #    log(f"Not found {retval}")
                yield None
                break
    
    # Reduces output files from worker processes to single file.
    # Handles csv files, if they exist.
    @profile
    def reduce(self, clean=True):
        
        # Pickled binary object
        retval = []
        for cpu_cached_file in self.cpu_cache_name():
            if cpu_cached_file is None:
                continue
            tmp = pickle.load(open(cpu_cached_file, "rb"))
            if tmp is not None:
                retval.append(tmp)
        if len(retval) > 0:
            save_binary(retval, self.output_file)
        
        # If there is a CSV file, then reduce.
        merged_df = None
        for cpu_cached_file in self.cpu_cache_name(is_csv=True):
            if cpu_cached_file is None:
                continue
            cpu_df = pd.read_csv(cpu_cached_file)
            if merged_df is None:
                merged_df = cpu_df
            else:
                merged_df = merged_df.append(cpu_df, ignore_index=True)
            dest_file = self.output_file + ".csv"
            merged_df.to_csv(dest_file, index=False)
            
        if retval is None:
            retval = merged_df
            
        if clean:
            self.clean_cache(verbose=False)
            
        return retval
    
    def clean_cache(self, verbose=True, remove_output=False):
        if verbose:
            log("Before cleanup...")
            log_dir('../working')
            log("Starting Cleanup of MultiProcessing Cache")
            
        for cache_file in self.cpu_cache_name():
            remove_file(cache_file)
        
        for cache_file in self.cpu_cache_name(is_csv=True):
            remove_file(cache_file)
        
        if remove_output:
            remove_file(self.output_file)
            remove_file(self.output_file+".csv")
        
        if verbose:
            log("Finished cleanup of MultiProcessing Cache")
            log_dir('../working')
            
    def inspect(self):
        if not gInspect:
            return
        
        log("<h1>Inspect MPHelper</h1>")
        results = self.reduce(clean=False)
        log("Merged results from pickle")
        log_list(results)
        for item in results:
            log("Results from CPU {}".format(results.index(item)))
            log_list(item)
        
        dest_file = self.output_file + ".csv"
        if os.path.isfile(dest_file):
            log("CSV output from reduced file as pandas frame.")
            df = pd.read_csv(dest_file)
            log(df.to_html())
        
    @classmethod
    def testme(cls):
        
        clear_prof_data()
        tmp = []
        n = 23
        for cpu_id in range(mpc.cpu_count()):
            tmp.append(str(MPHelper.range_partition(n, cpu_id)))
        log_list(tmp, desc="Test range partitioning by CPU count for length 23")
    
        log_dir("../working")
        me = cls("Test_MPFileCache", verbose=True)
        test = TestMPCWithFileCache()
        me.map_process(test.process, "hello", "world", kv3=[100, 300, 400])
        me.inspect()
        me.clean_cache(remove_output=True)
        del me
        del test
        pp_prof_data()


# In[8]:


if False:
    compute = MPHelper.testme()


# In[9]:


gEmbeddingsSources = {
    "GLOVE" : {
        'path' : '../input/embeddings/glove.840B.300d/glove.840B.300d.txt',
        'mean' : -0.005838498938828707,
        'std' : 0.4878219664096832
    },
    "WIKINEWS" : {
        'path' : '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
        'mean' : -0.0033469984773546457,
        'std' : 0.10985549539327621
    },
    "PARAGRAM" : {
        'path' : '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt',
        'mean' : -0.005324783269315958,
        'std' : 0.4934646189212799
    },
    "GOOGLENEWS" : {
        'path' : '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
        'mean' : 0,
        'std' : 0
    },
}
        
class EmbeddingsControl():
    
    OUTPUT_CACHE_FILE_PREFIX = "MATRIX"
    OOV_CACHE_FILE_PREFIX = "OOV"
    
    @classmethod
    def output_cache_name(cls, prefix, source_name):
        return prefix + "_" + source_name
        
    @classmethod
    def load_embedding_index(cls, source_name):
        
        assert(source_name in gEmbeddingsSources)

        log("Loading embeddings index {}".format(source_name))
        
        def get_coefs(word,*arr): 
            return word, np.asarray(arr, dtype='float32')
    
        file = gEmbeddingsSources[source_name]['path']
        retval = None
        if source_name == "GOOGLENEWS":
            retval = KeyedVectors.load_word2vec_format(file, binary=True, limit=500000)
        elif source_name == "PARAGRAM":
            retval = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        elif source_name == "WIKINEWS" or source_name == "GLOVE":
            retval = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
            
        return retval
                    
    # Memory efficient loading, except for GOOGLENEWS (Word2Vec) which is loaded as a whole.
    # Returns embeddings matrix for one source.
    @classmethod
    @profile
    def load(cls, source_name, word2index, embed_size=300):   
        
        log_current_memory('Before {} processing, memory at:'.format(source_name))
        
        assert(source_name in gEmbeddingsSources)

        embedding_matrix = None
        oov = None

        cache = EmbeddingsControl.output_cache_name(EmbeddingsControl.OUTPUT_CACHE_FILE_PREFIX, source_name)
        if os.path.isfile(cache):
            embedding_matrix = pickle.load(open(cache, 'rb'))
        
        cache = EmbeddingsControl.output_cache_name(EmbeddingsControl.OOV_CACHE_FILE_PREFIX, source_name)
        if gWIP and os.path.isfile(cache):
            oov = pickle.load(open(cache, 'rb'))
            
        if (not gWIP and embedding_matrix is not None) or (gWIP and embedding_matrix is not None and oov is not None):
            return embedding_matrix, oov
        
        embeddings_params = gEmbeddingsSources[source_name]
        nb_words = len(word2index) + 1
        log(f"<h3>Embeddings for {source_name}<h3><br> Matrix size: {nb_words}; <br>word2index length: {len(word2index)};<br> max_words : {nb_words}")
        found_vecs = set()
        
        if source_name == "GOOGLENEWS":
            embeddings_index = KeyedVectors.load_word2vec_format(embeddings_params['path'], binary=True, limit=500000)
            embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5)/5.0
            
            for word, index in word2index.items():
                if len(found_vecs) >= nb_words:
                    break
                    
                if word not in embeddings_index: 
                    if word.lower() in embeddings_index:
                        # Upper-case word's index gets assigned to lower-case word's vector
                        word = word.lower()
                    else:
                        continue
                
                embedding_vector = embeddings_index.get_vector(word)
                if len(embedding_vector) == embed_size:
                    embedding_matrix[index] = embedding_vector
                    found_vecs.add(index)
            
            del embeddings_index
        else:
            # For paragram, let's give the vector for lowercase word to uppercase word
            lowercase_word2index = {}
            for w, index in word2index.items():
                l_w = w.lower()
                if l_w not in lowercase_word2index:
                    lowercase_word2index[l_w] = [index]
                else:
                    lowercase_word2index[l_w].append(index)
                    
            words = ['Quora','Trump','Indian','US','Would','Google']
            for w in words:
                indices = lowercase_word2index[w.lower()]
                log(f"{w} as lower {w.lower()} : {indices}")
            words = [w.lower() for w in words]
            log_list(words, desc="Samples of OOV")
            
            mean = embeddings_params['mean']
            std = embeddings_params['std']
            embedding_matrix = np.random.normal(mean, std, (nb_words, embed_size))

            def read_embeddings_file(f):
                for line in f:                    
                    if len(line) < 100:
                        continue
                    
                    embedded_word, vec = line.split(' ', 1)
                    
                    indices = []
                    # Allow any form of a word that is in the word2index,
                    # and if not present, then find the lowercase, and
                    # and if present, but PARAGRAM, then, find all indices
                    # of all forms of the word in word2index.
                    # PARAGRAM has all lowercase.
                    if embedded_word in word2index:
                        if source_name == "PARAGRAM":
                            if embedded_word in lowercase_word2index:
                                indices = lowercase_word2index[embedded_word]
                            else:
                                continue
                        else:
                            indices = [word2index[embedded_word]]
                    else:
                        embedded_word = embedded_word.lower()
                        if embedded_word in lowercase_word2index:
                            indices = lowercase_word2index[embedded_word]
                        else:
                            continue

                    if embedded_word in words:
                        log(f" >>>> Adding indices for {embedded_word} : {indices}")
                                
                    embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:embed_size]
                    if len(embedding_vector) == embed_size:
                        for index in indices:
                            embedding_matrix[index] = embedding_vector
                            found_vecs.add(index)                    
                    if len(found_vecs) == nb_words:
                        break
                                
            if source_name == "PARAGRAM":
                with open(embeddings_params['path'], 'r', encoding='utf8', errors='ignore') as f:
                    read_embeddings_file(f)
            else:
                with open(embeddings_params['path']) as f:
                    read_embeddings_file(f)
        
        gc.collect()
        
        if gWIP:
            log_current_memory('After {} processing, memory at'.format(source_name))
            oov = set()
            vocab_indices = set(word2index.values())
            oov = vocab_indices.difference(found_vecs)      
            log(f"Length of found vecs dictionary: {len(found_vecs)};\nLength of oov: {len(oov)}")
            cache = EmbeddingsControl.output_cache_name(EmbeddingsControl.OOV_CACHE_FILE_PREFIX, source_name)
            save_binary(oov, cache)
        
        cache = EmbeddingsControl.output_cache_name(EmbeddingsControl.OUTPUT_CACHE_FILE_PREFIX, source_name)
        save_binary(embedding_matrix, cache)
        
        return embedding_matrix, oov
    
    @classmethod 
    def clean_cache(cls):
        for source, _ in gEmbeddingsSources.items():
            remove_file(EmbeddingsControl.output_cache_name(EmbeddingsControl.OUTPUT_CACHE_FILE_PREFIX, source))
            remove_file(EmbeddingsControl.output_cache_name(EmbeddingsControl.OOV_CACHE_FILE_PREFIX, source))
    
    @classmethod
    def inspect(cls, source, index2word, word2count, w2v, oov=None):
        log(f"<b>Embeddings for {source}</b>")
        log("Loaded Embeddings  ", len(w2v))
        
        if w2v is not None:
            log("Shape of embeddings for {} is {}".format(source, w2v.shape))
        
        if oov is not None and index2word is not None:
            log("Number of words ", len(index2word))
            log(f"Number of OOV: {len(oov)}")
            oov_words = Counter()
            for index in oov:
                word = index2word[index]
                oov_words[word] = word2count[word]
            
            log_list(oov_words.most_common(100), desc="OOV for {}".format(source))

    @classmethod
    def testme(cls, source="GOOGLENEWS"):
        me = cls()
        
        use_test_samples = False
        index2word = None
        word2index = None
        vocab2count = None
        
        if use_test_samples:
            word2index = {'good': 0, 'bad': 1, 'sincere': 2, 'insincere': 3, 'lkjsdlfajsflkdj' : 4, 'sexual_intercourse':5, 'Donald_Trump' : 6, 'obama' : 7, 'f_**_king' : 8, '?' : 9, "LOVE":10, "Why":11, "What":12}
            index2word = {}
            vocab2count = Counter()
            for k, v in word2index.items():
                index2word[v] = k
                if k in vocab2count:
                    vocab2count[k] += 1
                else:
                    vocab2count[k] = 1
        else:
            word2index = pickle.load(open('w2index_BOTH', 'rb'))
            index2word = pickle.load(open('index2w_BOTH','rb'))
            vocab2count = pickle.load(open('vocab_BOTH', 'rb'))
        
        log_current_memory('Before embeddings processing')
        for a_source, _ in gEmbeddingsSources.items():
            if a_source == source or source == None:
                w2v, oov = EmbeddingsControl.load(a_source, word2index)
                EmbeddingsControl.inspect(a_source, index2word, vocab2count, w2v, oov)
        del w2v
        del oov
        del me
        gc.collect()
        log_current_memory('After embeddings processing')
                
    @classmethod
    def precompute_mean_std(cls):
        
        def calculate(source):
            embeddings = EmbeddingsControl.load_embedding_index(source)
            all_embs = None
            all_embs = np.stack(embeddings.values())
            emb_mean,emb_std = all_embs.mean(), all_embs.std()
            log("{} Embeddings... mean: {} std:{}".format(source, emb_mean, emb_std))
            del all_embs
            del embeddings
            gc.collect()
        
        calculate("GLOVE")
        calculate("WIKINEWS")
        calculate("PARAGRAM")
        # GOOGLENEWS is done differently; doesn't use mean/std (why?? XX)


# In[10]:


if False:
    log_current_memory('Start processing embeddings')
    EmbeddingsControl.clean_cache()
    log_dir("../working/")
    EmbeddingsControl.testme(source=None)
    gc.collect()
    log_current_memory('Finished processing embeddings')
    
if False:
    log_current_memory('Start processing embeddings')
    EmbeddingsControl.precompute_mean_std()
    log_current_memory('Finished processing embeddings')


# In[11]:


class DataManager:
    
    INPUT_TRAINING_DATA = '../input/train.csv'
    INPUT_TEST_DATA = '../input/test.csv'
    INPUT_GENERATED_DATA = 'gen_q.csv'
    OUTPUT_CACHE_FILE = 'orig_gen.csv'
    gInstance = None
    DEV_LIMIT = 5000
    
    def __init__(self):
        self.training_data = None 
        self.test_data = None        
        
        if gExternalData:
            DataManager.INPUT_TRAINING_DATA = '../input/quora-insincere-questions-classification/train.csv'
            DataManager.INPUT_TEST_DATA = '../input/quora-insincere-questions-classification/test.csv'
        
    @classmethod
    def instance(cls):
        if DataManager.gInstance == None:
            return cls()
        else:
            return gInstance
        
    def load(self, orig=True, combined=False, test=False, source_file=None):

        if source_file is not None:
            self.training_data = pd.read_csv(source_file)
        elif combined and os.path.isfile(DataManager.OUTPUT_CACHE_FILE):
            log("Found combined training data file")
            self.training_data = pd.read_csv(DataManager.OUTPUT_CACHE_FILE)
        elif combined or orig:
            self.training_data = pd.read_csv(DataManager.INPUT_TRAINING_DATA)

        if gLimit and self.training_data is not None and combined == False:
                self.training_data = self.training_data.sample(DataManager.DEV_LIMIT)
        
        if test:
            self.test_data = pd.read_csv(DataManager.INPUT_TEST_DATA)
            if gLimit:
                self.test_data = self.test_data[:min(len(self.test_data), DataManager.DEV_LIMIT)]
                
    def save_combined_data(self):
        
        if os.path.isfile(DataManager.INPUT_GENERATED_DATA):
            self.load(orig=True)
            more_training_data = pd.read_csv(DataManager.INPUT_GENERATED_DATA)
            combined = self.training_data.append(more_training_data, ignore_index=True)
            combined.to_csv(DataManager.OUTPUT_CACHE_FILE)
        
    def memclean(self):
        del self.training_data
        self.training_data = None
        del self.test_data
        self.test_data = None
        
    @classmethod
    def clean_cache(cls):
        log("Cleaning combined original and generated questions file.")
        remove_file(DataManager.OUTPUT_CACHE_FILE)
        
    def inspect(self):
        if not gInspect:
            return
        
        if self.training_data is not None:
            log("Shape of training data {}".format(self.training_data.shape))
            log("Length of unique q_id {}".format(len(self.training_data.iloc[0,:]['qid'])))
                  
        if self.test_data is not None:
            log("Shape of test data {}".format(self.test_data.shape))                  
    
    @classmethod
    @profile
    def testme(cls):
        DataManager.clean_cache()
        me = DataManager.instance()
        log("<hr>Original training nd test data")
        me.load(test=True)
        me.inspect()
        me.save_combined_data()
        log("<hr>Combined training data")
        me.load(combined=True)
        me.inspect()
        me.memclean()
        del me
        gc.collect()


# In[12]:


if False:
    DataManager.testme()


# In[13]:


class TextualDataControl:
    
    PUNCTUATIONS = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
        
    CONTRACTIONS_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
    
    MISPELL_MAP = {'to':None, 'a':None, 'of':None,'.':None,'and': 'plus','':None,'akistani':'Pakistani','Snapchat':'online social medium','WhatsApp':'messaging app','huminity':'humanity','motherfuckin':'motherfucking','motherfuckingg':'motherfucking','mujahiddin':'mujahideen','descpicable':'despicable','MOZLEMS':'Muslims','Quorans':'Quora users','Quoran':'Quora user','Niccaragua':'Nicaragua','Shivdharma':'Shiva dharma','colour': 'color', 'neighbour':'neighbor','behaviour':'behavior','favour':'favor','favoured':'favored','generalised':'generalized','realise':'realize','centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}
    
    SPECIAL_CHARS_MAP = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    
    SPACES_LIST = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']
    
    def __init__(self, data, target_column = 'target'):        
        self.rawdata = data
        self.label_column_name = target_column    
        
    def memclean(self):
        del self.rawdata
        self.rawdata = None
        
    @classmethod                
    def find_string(cls, src, target):
        
        retval = False
        if re.search(r"\b" + re.escape(src.lower()) + r"\b", target.lower()):
          retval = True
        
        return retval
    
    @classmethod
    def sent_capitalize(cls, src):
        
        retval = None
        sentences = sent_tokenizer.tokenize(src)
        sentences = [sent[:1].upper() + sent[1:] for sent in sentences]
        retval = ' '.join(sentences)
        
        return retval
    
    @classmethod
    def fix_special_chars(cls, phrase):
        
        if not re.match("^[a-zA-Z0-9 _]*$", phrase):
            all_special_chars = TextualDataControl.SPECIAL_CHARS_MAP.keys()
            for s in all_special_chars:
                if s in phrase:
                    phrase = phrase.replace(s, TextualDataControl.SPECIAL_CHARS_MAP[s])
                    
        return phrase
    
    @classmethod
    def decontract(cls, phrase):
        
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            if s in phrase:
                phrase = phrase.replace(s, "'")
        phrase = ' '.join([TextualDataControl.CONTRACTIONS_MAP[t.lower()] if t.lower() in TextualDataControl.CONTRACTIONS_MAP else t for t in phrase.split(" ")])
        possessive = "'s"
        if possessive in phrase:
            phrase = phrase.replace(possessive, " ")
        
        return phrase
    
    @classmethod
    def despace(cls, phrase):
        for space in TextualDataControl.SPACES_LIST:
            if space in phrase:
                phrase = phrase.replace(space, ' ')
        
        phrase = phrase.strip()
        phrase = re.sub('\s+', ' ', phrase)
        
        return phrase
        
        
    @classmethod
    def denumber(cls, phrase):
        
        if bool(re.search(r'\d', phrase)):
            phrase = re.sub('[0-9]{5,}', '#####', phrase)
            phrase = re.sub('[0-9]{4}', '####', phrase)
            phrase = re.sub('[0-9]{3}', '###', phrase)
            phrase = re.sub('[0-9]{2}', '##', phrase)
        
        return phrase
    
    @classmethod
    def fillna(cls, phrase, fill_value="_na_"):
        
        if phrase is None or phrase.strip() is None:
            phrase = "_na_"
        return phrase
    
    @classmethod
    def depunctuate(cls, phrase, remove=False):
        
        for punct in TextualDataControl.PUNCTUATIONS:
            if punct in phrase:
                if not remove:
                    # phrase = phrase.replace(punct, f' {punct} ')
                    phrase = re.sub('(['+punct+']{2,})', r' \1 ', phrase)
                else:
                    phrase = phrase.replace(punct, f' ')                    
        
        phrase = phrase.replace('  ', ' ').strip()
        
        return phrase
        
    @classmethod
    def respell(cls, a_word):

        retval = a_word
        if a_word in TextualDataControl.MISPELL_MAP:
            retval = TextualDataControl.MISPELL_MAP[a_word]    
        return retval
        
    def inspect(self):
        if not gInspect:
            return
        
        log("--------" + type(self).__name__ + "---------")
        log("Raw input data shape : ", self.rawdata.shape)
        log("Number of unique labels: ", self.rawdata[self.label_column_name].nunique())
        tmp = pd.DataFrame([self.rawdata["target"].unique(), self.rawdata["target"].value_counts()], index=["Label", "Count"])
        display(HTML("<h3>" + "Labels" + "</h3>" + tmp.to_html(max_rows=10)))
        #log_list(, desc="Raw Data Statistics")
        #log("Test shape : ",self.test.shape)
        display(HTML(self.rawdata.to_html(max_rows=5)))
        display(HTML(self.rawdata[self.rawdata[self.label_column_name] == 1].to_html(max_rows=5)))

        f,ax=plt.subplots(1,2,figsize=(18,8))
        self.rawdata[self.label_column_name].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
        ax[0].set_title('By ' + self.label_column_name)
        ax[0].set_ylabel('')

        sns.countplot(self.label_column_name,data=self.rawdata,ax=ax[1])
        ax[1].set_title('By ' + self.label_column_name)
        plt.show()
        
    @classmethod
    def testme(cls):
        app = DataManager.instance()
        app.load(orig=True)
        td = cls(app.training_data)
        td.inspect()
        td.memclean()
        del td
        app.memclean()
        del app
        gc.collect()


# In[14]:


if False:
    TextualDataControl.testme()

if False:
    log(TextualDataControl.decontract("shouldn't"))
    log(TextualDataControl.decontract("child's"))
    log(TextualDataControl.denumber("Is it advisable to take up 20 (4 credits and 4 non credits) courses at Harvard Summer School in 3 weeks?"))
    log(TextualDataControl.fix_special_chars("∞ करना `foo_bar` … right? f_**_king"))
    log(str(TextualDataControl.respell('colour')))
    log("Test \u200b Test")
    tmp = "Is it advisable to >>\u200b<< take up 20 (4 credits and 4 non credits) courses at Harvard Summer School in 3 weeks?"
    log(tmp)
    log(str(TextualDataControl.despace(tmp)))


# In[15]:


class QuoraPreprocessor(TextualDataControl):
    
    CLEANED_OUTPUT_FILE = "cleaned_q"
    VOCAB_OUTPUT_FILE = "vocab"
    W2INDEX_OUTPUT_FILE = "w2index"
    INDEX2W_OUTPUT_FILE = "index2w"
    SEQUENCES_OUTPUT_FILE = "sequences"
    MAX_WORDS_IN_QUESTION = 45
    MAX_VOCAB_SIZE = 60000
        
    def __init__(self, training_data=None, test_data=None, is_gen=True):
        TextualDataControl.__init__(self, None)
        
        self.datasource = None
        self.rawdata = None
        
        self.training_data = training_data
        self.test_data = test_data
        self.is_generated_data = is_gen

        self.word2count = None
        self.word2index = None
        self.index2word = None    
        self.sample2sequence = None
    
    def output_cache_name(self, file_prefix, datasource=None, is_csv=False):
        
        if datasource is None:
            datasource = self.datasource
        retval = file_prefix + "_"  + datasource
        if is_csv:
            retval += ".csv"
        return retval
    
    @classmethod
    @profile
    def make_tokens(cls, text):
    
        words = list(text.split())
        multi_words = {}
        punct_to_split = "[a-zA-Z0-9']+|[.,!?;]"
        for word_index in range(len(words)):
            words[word_index] = words[word_index].strip(string.punctuation)
            found_multi_words = re.findall(r"{}".format(punct_to_split), words[word_index])
            if len(found_multi_words) > 1:
                multi_words[word_index] = found_multi_words

        if len(multi_words) > 0:
            offset = 0
            for index, replacements in multi_words.items():
                before = words[:index+offset]
                after = words[index+offset+1:]
                #log("Before:{}<br>Replacement:{}<br>After:{}".format(before, replacements, after))
                words = before + replacements + after
                offset += len(replacements)-1

        # Spelling errors - tokenwise
        retval = []
        for w in words: 
            if w is not None and len(w) > 0:
                replacement = TextualDataControl.respell(w)
                if replacement is not None and len(replacement.strip()) > 0:
                    tmp = replacement.split()
                    for item in tmp:
                        if len(item.strip()) > 0 and item not in gStopWords:
                            retval.append(item)

        return retval
    
    # Does more than just split the phrase.
    # Returns an array of words
    @profile
    def tokenize(self, text):
        return QuoraPreprocessor.make_tokens(text)
        
    
    @profile
    def cleaning_proc(self, cpu_id, cpu_count, cachefile):
        
         # RawData
        q_range = MPHelper.range_partition(len(self.rawdata), cpu_id)
        rawdata = self.rawdata[q_range.start:q_range.stop]
        
        # Pre-processing
        preprocessed_text = rawdata["question_text"].apply(lambda x: self.fix_special_chars(x))
        preprocessed_text = preprocessed_text.apply(lambda x: self.decontract(x))
        preprocessed_text = preprocessed_text.apply(lambda x: self.denumber(x))
        preprocessed_text = preprocessed_text.apply(lambda x: self.despace(x))
        preprocessed_text = preprocessed_text.apply(lambda x: self.tokenize(x))
        counts = preprocessed_text.apply(lambda x: len(x))
        preprocessed_text = preprocessed_text.apply(lambda x: ','.join(x))
        
        rawdata = self.rawdata.iloc[q_range.start:q_range.stop].assign(cleaned_text=preprocessed_text,num_words=counts)
        rawdata.to_csv(cachefile+".csv")
    
    # Word:Count
    @classmethod
    def make_word2count(cls, results, words):
        words = words.split(",")
        for word in words:
            if word in results:
                results[word] += 1
            else:
                results[word] = 1
                
        return len(words)

    def vocab_proc(self, cpu_id, cpu_count, cachefile):
        
        # Raw, cleaned data
        cleaned_q_csv_file = self.output_cache_name(QuoraPreprocessor.CLEANED_OUTPUT_FILE, is_csv=True)
        rawdata = pd.read_csv(cleaned_q_csv_file)
        q_range = MPHelper.range_partition(len(rawdata), cpu_id)
        
        # Produce vocab dictionary
        results = {}
        maxlen = 0
        maxqid = -1
        num_words = []
        for i in q_range:
            cleaned_text = rawdata.loc[i]["cleaned_text"]
            word_count = QuoraPreprocessor.make_word2count(results, str(cleaned_text))
            num_words.append(word_count)
            maxlen = max(maxlen, word_count)
            if maxlen == word_count:
                maxqid = i
        
        results = {'maxlen' : maxlen, 'maxq' : maxqid, 'vocab' : results }
        save_binary(results, cachefile)
        
    # The maxlength of questions at 60 arrived at after counting words in training set.
    def sequences_proc(self, cpu_id, cpu_count, cachefile, maxlen=60):
        # Raw, cleaned data
        cleaned_q_csv_file = self.output_cache_name(QuoraPreprocessor.CLEANED_OUTPUT_FILE, is_csv=True)
        rawdata = pd.read_csv(cleaned_q_csv_file)
        q_range = MPHelper.range_partition(len(rawdata), cpu_id)
        
        word2index = pickle.load(open(self.output_cache_name(QuoraPreprocessor.W2INDEX_OUTPUT_FILE, datasource="BOTH"), "rb"))
        
        results = []
        for i in q_range:
            cleaned_text = str(rawdata.loc[i]["cleaned_text"])
            cleaned_tokens = cleaned_text.split(",")
            a_sequence = [word2index[token] for token in cleaned_tokens if token in word2index]
            results.append(a_sequence)
        
        results = pad_sequences(results, maxlen=maxlen)
        save_binary(results, cachefile)
        
    @profile
    def make_clean_questions(self):
        
        source_file = self.output_cache_name(QuoraPreprocessor.CLEANED_OUTPUT_FILE, is_csv=True)
        if os.path.isfile(source_file):
            self.rawdata = pd.read_csv(source_file)
            return
        
        # Clean Question Text
        mp = MPHelper(self.output_cache_name(QuoraPreprocessor.CLEANED_OUTPUT_FILE, is_csv=False))
        mp.map_process(self.cleaning_proc)
        mp.reduce(clean=True)

        # load reduced csv
        self.rawdata = pd.read_csv(source_file)
        
    @profile
    def make_vocab(self, lower=False):
        
        cachefile = self.output_cache_name(QuoraPreprocessor.VOCAB_OUTPUT_FILE)
        if os.path.isfile(cachefile):
            word2count = pickle.load(open(cachefile, 'rb'))
            return word2count
        
        # Collect unique words (vocab)
        mp = MPHelper(QuoraPreprocessor.VOCAB_OUTPUT_FILE)
        mp.map_process(self.vocab_proc)
        merged_results = mp.reduce(clean=True)
        
        word2count = Counter()
        for cpu_result in merged_results:
            cpu_vocab = cpu_result['vocab']            
            for word, count in cpu_vocab.items():
                if lower:
                    word = word.lower()
                if word in word2count:
                    word2count[word] += count
                else:
                    word2count[word] = count
        
        save_binary(word2count, cachefile)
        return word2count

    # Combined training and test vocab into 1.
    def combine_vocab(self):
        
        cachefile = self.output_cache_name(QuoraPreprocessor.VOCAB_OUTPUT_FILE, datasource="BOTH")
        if os.path.isfile(cachefile):
            self.word2count = pickle.load(open(cachefile, "rb"))
            return

        vocab_file = self.output_cache_name(QuoraPreprocessor.VOCAB_OUTPUT_FILE)
        test_file = self.output_cache_name(QuoraPreprocessor.VOCAB_OUTPUT_FILE, datasource='test')
        # word2count is a Counter dictionary
        training_vocab = pickle.load(open(vocab_file, 'rb'))
        test_vocab = pickle.load(open(test_file, 'rb'))
        
        for wrd, count in test_vocab.items():
            if wrd in training_vocab:
                training_vocab[wrd] += count
            else:
                training_vocab[wrd] = count
            
        self.word2count = training_vocab
        remove_file(vocab_file)
        remove_file(test_file)
        save_binary(self.word2count, cachefile)
        
    @profile
    def make_word2index(self):
        
        cachefile1 = self.output_cache_name(QuoraPreprocessor.W2INDEX_OUTPUT_FILE, datasource="BOTH")
        cachefile2 = self.output_cache_name(QuoraPreprocessor.INDEX2W_OUTPUT_FILE, datasource="BOTH")
        
        if os.path.isfile(cachefile1) and os.path.isfile(cachefile2):
            self.word2index = pickle.load(open(cachefile1, "rb"))
            self.index2word = pickle.load(open(cachefile2, "rb"))
            return
        
        if self.word2count is None:
            return

        self.word2index = {}
        self.index2word = {}
        
        # Starting index at 1, so that index=0 is reserved.
        index = 1
        items = self.word2count.most_common()
        
        for (word, _) in items:
            self.word2index[word] = index
            if gInspect:
                self.index2word[index] = word
            index += 1
        
        save_binary(self.word2index, cachefile1)
        if gInspect:
            save_binary(self.index2word, cachefile2)

    @profile
    def make_sequences(self):
        
        cachefile = self.output_cache_name(QuoraPreprocessor.SEQUENCES_OUTPUT_FILE)
        if os.path.isfile(cachefile):
            self.sample2sequence = pickle.load(open(cachefile, "rb"))
            return
        
        # Collect unique words (vocab)
        mp = MPHelper(QuoraPreprocessor.SEQUENCES_OUTPUT_FILE)
        mp.map_process(self.sequences_proc, maxlen=QuoraPreprocessor.MAX_WORDS_IN_QUESTION)
        merged_results = mp.reduce(clean=True)
        
        self.sample2sequence = []
        count = 0
        for cpu_result in merged_results:
            self.sample2sequence.extend(cpu_result)
        
        save_binary(self.sample2sequence, cachefile)

    # This class was originally written to process some rawdata and write the results to file without focus on 
    # training vs test.This method changes the underlying rawdata and file names for processing training vs test data.
    def set_mode(self, is_training=False, is_test=False):
        
        assert(is_training or is_test)
        
        if is_training:
            self.rawdata = self.training_data
            if self.is_generated_data:
                self.datasource = 'gen'
            else:
                self.datasource = 'orig'
        else:
            self.rawdata = self.test_data
            self.datasource = 'test'
        
    @profile
    def process(self, sequences=True):        

        self.set_mode(is_training=True)
        self.make_clean_questions()
        self.make_vocab()
        
        self.set_mode(is_test=True)
        self.make_clean_questions()
        self.make_vocab()
        
        # Important insight - in many published kernels, the vocabulary is created from 
        # training data only.  This isn't correct, because the model for training and predictions 
        # has the same embeddings matrix.
        self.set_mode(is_training=True)
        self.combine_vocab()
        self.make_word2index()
        
        if sequences:
            self.set_mode(is_test=True)
            self.make_sequences()                

            self.set_mode(is_training=True)
            self.make_sequences()
    
    @classmethod
    def clean_cache(cls, cleaned_q=False, vocab=False, word2index=False, sequences=False):
        
        # can't use member function output_cache_name
        def make_path(file_prefix, datasource):
            return file_prefix + "_" + datasource
        
        for datasource in ['orig', 'gen', 'test', 'BOTH']:
            if cleaned_q:
                remove_file(make_path(QuoraPreprocessor.CLEANED_OUTPUT_FILE, datasource)+".csv")
                remove_file(QuoraPreprocessor.CLEANED_OUTPUT_FILE+".csv")

            if vocab:
                remove_file(make_path(QuoraPreprocessor.VOCAB_OUTPUT_FILE, datasource))
                remove_file(QuoraPreprocessor.VOCAB_OUTPUT_FILE)

            if sequences:
                remove_file(make_path(QuoraPreprocessor.SEQUENCES_OUTPUT_FILE, datasource))

        if word2index:
            remove_file(make_path(QuoraPreprocessor.W2INDEX_OUTPUT_FILE, "BOTH"))
            remove_file(make_path(QuoraPreprocessor.INDEX2W_OUTPUT_FILE, "BOTH"))


    @classmethod
    def testme(cls):
        
        clear_prof_data()
        log_current_memory(caption="Start Quora Preprocessing...")
        app = DataManager.instance()
        app.load(orig=True, combined=False, test=True)
        
        me = QuoraPreprocessor(training_data=app.training_data, test_data=app.test_data, is_gen=False)
        me.process()
        texts = ["The fool of a Took went off on his own!"]
        for text in texts:
            log(f'{me.tokenize(text)}')
        me.inspect()
        log_current_memory(caption="End Quora Preprocessing...")
        me.memclean()        
        del me
        
        app.memclean()
        del app
        
        gc.collect()        
        pp_prof_data()
    
    @classmethod
    def testme2(cls):
        app = DataManager.instance()
        app.load(orig=True, combined=False, test=False)
        clear_prof_data()
        me = QuoraPreprocessor(training_data=app.training_data, test_data=None, is_gen=False)
        me.set_mode(is_training=True)
        me.make_clean_questions()
        word2count = me.make_vocab(lower=True)
        log(f"Number of words in uncleaned question_text {len(word2count)}")
        pp_prof_data()
        del word2count
        del me
        del app
        gc.collect()
    
    def inspect(self):
        super().inspect()
        
        if not gInspect:
            return

        log("--------" + type(self).__name__ + "---------")
        
        log_list(self.word2count.most_common(10), desc="Vocab2Count - Top 10")
        log_list(self.word2index, limit=10, desc="Word2Index - Limit 10")
        log_list(self.sample2sequence, limit=10, desc="Question Sequences - Limit 10")
        log(f"Number of sequences: {len(self.sample2sequence)}")
        log(f"Number of questions:{len(self.rawdata)}")


# In[16]:


if False:
    QuoraPreprocessor.clean_cache(cleaned_q=True, vocab=True, word2index=True, sequences=True)
    QuoraPreprocessor.testme2()
    log_dir("../working")


# In[17]:


if False:
    questions = ['If men are roughly 40% of all domestic violence victims, why are there no resources to help them? Why is there so much resistance from feminists/women when this topic/topical/tear is brought up? Why are men!women not able to freely speak about these real problems?',
             'Why don\'t liberal progressive_voices oppose Amazon.com\'s potential takeover of state authority as the company ponders 2345 locations for its new HQ?', '"All countries support Indian Army to occupy Chinese land in Doklan," Indian FM claimed. "Can you name one," asked a reporter. "Hiiimmmmmm," replied the Indian FM. Why has India been good at nothing, but false claiming for the past 70 years?']
    qp = QuoraPreprocessor()
    for q in questions:
        tq = TextualDataControl.decontract(q)
        tq = TextualDataControl.denumber(tq)
        tq = TextualDataControl.fillna(tq)
        tq = qp.lowering(tq)
        log_list(qp.tokenize(tq), desc=q)


# In[18]:


class TopicalWords: 
     
    embeddings_index = None
    vocab2count = None
    _nlp = None
    punctuations = ''.join(TextualDataControl.PUNCTUATIONS)
    
    MIN_SIMILARITY_SCORE = 0.5

    def __init__(self, topical_word):
        self.source_word = topical_word
        self.substitutes = None
        self.related_sources = None
                
        self.my_NER_type = TopicalWords.get_NER_label(self.source_word)
        self.my_POS_tag = TopicalWords.get_tag(self.source_word.replace("_", " "))

    @classmethod
    def get_NER_label(cls, word):
        retval = None
        word = TextualDataControl.depunctuate(word, remove=True)
        nlp = TopicalWords.nlp()
        ents = nlp(word).ents
        if len(ents) > 0:
            retval = ents[0].label_
        
        return retval

    @classmethod
    def get_tag(cls, phrase):
    
        retval = None
        nlp = TopicalWords.nlp()
        
        doc = nlp(u'{}'.format(phrase))
        last_tag = None
        for token in doc:
            last_tag = token.tag_
            if last_tag in ["NNS", "NN", "NNP"]:
                retval = last_tag
                break
        
        return retval
    
    
    @classmethod
    def nlp(cls):
        
        if TopicalWords._nlp is None:
            TopicalWords._nlp = en_core_web_sm.load()

        return TopicalWords._nlp    
    
    @classmethod
    @profile
    def embeddings(cls):
        
        if TopicalWords.embeddings_index is None:
            glove_file = datapath(gEmbeddingsSources['GLOVE']['path'])
            tmp_file = get_tmpfile("glove_word2vec.txt")
            _ = glove2word2vec(glove_file, tmp_file)
            TopicalWords.embeddings_index = KeyedVectors.load_word2vec_format(tmp_file, binary=True, limit=500000)

        return TopicalWords.embeddings_index
    
    @classmethod
    @profile
    def word2count(cls):
        
        if TopicalWords.vocab2count is None:
            app = DataManager.instance()
            app.load(orig=True, test=False)
            orig_quora = QuoraPreprocessor(training_data=app.training_data[app.training_data.target == 1], test_data=None, is_gen=False)
            orig_quora.set_mode(is_training=True)
            orig_quora.make_clean_questions()
            TopicalWords.vocab2count = orig_quora.make_vocab(lower=True)
            log(f"Number of words in original input's vocabulary:{len(TopicalWords.vocab2count)}")
            del orig_quora
            del app
            gc.collect()
        
        return TopicalWords.vocab2count
    
    @classmethod
    def contains(cls, word1, word2):
        return (word1 in word2 or word2 in word1) and len(word1) != len(word2)   
    
    @classmethod
    def multi_word_in_vocab(cls, phrase):
        # because GOOGLENEWS has compound words with underscore
        # Allow underscore in single word, and check if each component is also
        # in the vocab.
        lower_phrase = phrase.lower()
        word2count = TopicalWords.word2count()
        retval = False
        if lower_phrase in word2count:
            retval = True
        elif '_' in phrase:
            punct_to_split = "[a-zA-Z0-9']+|[.,!?;]"
            found_multi_words = re.findall(r"{}".format(punct_to_split), phrase)
            if found_multi_words is not None and len(found_multi_words) > 1:
                found = True
                for w in found_multi_words:
                    if  w.lower() not in word2count:
                        found = False
                        break
                if found:
                    phrase = " ".join(found_multi_words)
                #if not found and verbose:
                #    log_list(found_multi_words, desc="Words of MultiWord not found.")
                retval = found
                
        return retval, phrase
    
    @classmethod
    def is_vocab(cls, phrase, verbose=False):
        retval, phrase = TopicalWords.multi_word_in_vocab(phrase)
        if retval == False and verbose:
            log(phrase + " is not in vocab")
        return retval
        
    
    def find_matching_phrases(self, word_list):
        tmp = self.source_word.lower()
        retval = []
        
        for (similar, _) in word_list:
            if TopicalWords.contains(similar.lower(), tmp.lower()) == False:
                continue
            
            found = False
            for existing_word in retval:
                if existing_word.lower() == similar.lower():
                    found = True
                    break

            if not found:
                if self.my_POS_tag is None or TopicalWords.get_tag(similar) == self.my_POS_tag:
                    retval.append(similar)
                
        retval.insert(0, self.source_word)
        
        return retval
    
    def find_substitute_phrases(self, matching_word_list, similars):
         # Combine all words that vertically belong to the source word (which is the topic)
        retval = [(w, 1) for w in matching_word_list]
        
        matching_lower = set()
        for w in matching_word_list:
            matching_lower.add(w.lower())
        # Topics that are more similar to textually matching words. They are "vertically" matching.
        if len(matching_word_list) > 2:
            substitutes = TopicalWords.embeddings().most_similar(positive=matching_word_list, topn=30)
            substitutes = [(w, score) for (w, score) in substitutes if score > TopicalWords.MIN_SIMILARITY_SCORE  and TopicalWords.is_vocab(w) and w.lower() not in matching_lower]
            retval.extend(substitutes)      
        else:
            for (w, score) in similars: 
                if score > 0.6 and TopicalWords.is_vocab(w):
                    w_NER = TopicalWords.get_NER_label(w)
                    if ((self.my_NER_type is None) or (self.my_NER_type == w_NER)):
                        if self.my_POS_tag is None or TopicalWords.get_tag(w) == self.my_POS_tag:
                            retval.append((w, score))
            
        return retval
    
    def filter_related_sources(self, similar_phrases):        
        
        unique_subs = set([w.lower() for (w, _) in self.substitutes])
        retval = [(w, score) for (w, score) in similar_phrases if w.lower() not in unique_subs and score > TopicalWords.MIN_SIMILARITY_SCORE and TopicalWords.is_vocab(w) and ((self.my_NER_type is None) or (self.my_NER_type == TopicalWords.get_NER_label(w)))]
        retval = (sorted(retval, key=lambda t:t[1], reverse=True))

        return retval
    
    def filter_substitutes(self):
        retval = []

        for (w, score) in self.substitutes:
            if score > TopicalWords.MIN_SIMILARITY_SCORE and TopicalWords.is_vocab(w):
                NER_type = TopicalWords.get_NER_label(w)
                if NER_type == self.my_NER_type:
                    if self.my_POS_tag is None or TopicalWords.get_tag(w) == self.my_POS_tag:
                        w = w.strip(TopicalWords.punctuations).strip()
                        if len(w) > 2 and w not in ['ing']:
                            retval.append((w, score, NER_type))

        retval = (sorted(retval, key=lambda t:t[1], reverse=True))    
        return retval
    
    @profile
    def process(self, num_top = 10, verbose=True):
        
        if verbose:
            log("Processing topical words for <b>{}</b>".format(self.source_word))
        
        # horizontally relevant to topical word
        other_topical_words = TopicalWords.embeddings().similar_by_word(self.source_word, topn=50)
        if verbose:
            log("Found other topical words: {}".format(len(other_topical_words)))
            
        # Textually matching words
        matching = self.find_matching_phrases(other_topical_words)        
        if verbose and matching is not None and len(matching) > 0:
            log("Found matching words: {}".format(len(matching)))

        # Words that could be substituted for each other, and the source word.
        self.substitutes = self.find_substitute_phrases(matching, other_topical_words)
        if verbose and self.substitutes is not None and len(self.substitutes) > 0:
            log("Found initial set of substitutes: {}".format(len(self.substitutes)))
                            
        # cleanup
        # related words that are not in word_list are newly discovered topics.
        other_topical_words = self.filter_related_sources(other_topical_words)
        
        self.substitutes = self.filter_substitutes()[:min(num_top, len(self.substitutes))]        
        self.related_sources = other_topical_words[:min(num_top, len(other_topical_words))]
    
    @classmethod
    def memclean(cls):
        del TopicalWords.embeddings_index
        TopicalWords.embeddings_index = None
        del TopicalWords._nlp
        TopicalWords._nlp = None
        del TopicalWords.vocab2count
        TopicalWords.vocab2count = None
    
    def inspect(self):
        if not gInspect:
            return
        
        log_list(self.substitutes, desc="Substitutes to {} with NER type:{}".format(self.source_word, self.my_NER_type))
        log_list(self.related_sources, transpose=True, desc="Source Words to {} with NER type:{}".format(self.source_word, self.my_NER_type))
        log(f"Number of words in vocabulary:{len(TopicalWords.word2count())}")
    
    @classmethod 
    def testme(cls):
        sources = ['f_ing','Pakistanis','ass', "Orthodox_Jews", "Hinduism", 'Muslim', 'Islamic_fundamentalist', 'Muslims', 'fuck', 'Barack_Obama', 'Christian', 'blacks']

        for source in sources:
            me = cls(source)
            me.process(num_top=10, verbose=True)
            me.inspect()
            if TopicalWords.is_vocab(source, verbose=True) == False:
                log(f"This word NOT found in data:{source}")
            del me
        
        TopicalWords.memclean()
        gc.collect() 


# In[19]:


if False:
    clear_prof_data()
    InsincereVocabGenerator.clean_cache(bow=True, vocab=True)
    TopicalWords.testme()
    pp_prof_data()


# In[20]:


# Find related words to source word.
# Find matching words in related words to source word
# Add all related words that are not matching words to insincere_vocab category
class InsincereVocabGenerator:
    
    InsincereBoW = "InsincereBoW"
    InsincereVocab = "InsincereVocab"
    MAX_DERIVED_WORDS_PER_TOPIC = 25
    INSINCERE_VOCAB = {
                "PEOPLES" : {"Hindus" : None, "Muslims" : None, "Christians" : None, "Jews" : None, "Latinos" : None, 'women':None, "gay" : None, "blacks" : None},
                "RELIGIONS" : {"Hinduism":None, "Islam" : None, "Christianity" : None},
                "NATIONALITIES" : {"Arabs":None, "Chinese":None, "Americans":None, "Europeans": None, "Pakistanis":None, 'Americans' : None},
                "PERSONS" : {"Donald_Trump" : None, "Bill_Clinton" : None, "Trump" : None, "Barack_Obama" : None},
                "PLACES"  : {"India" : None, "Pakistan": None, "Korea" : None, "China" : None, "Florida":None, 'Israel' : None, 'Iran':None},
                "SEXUAL"  : {"fuck" : None, "shit" : None, "asshole" : None, "ass":None, "pussy" : None, "penis": None, "rape": None},
                "NEGATIVE" : {"bitch" : None,"terrorists" : None, "hate" : None, "stupid":None },
                "POLITICAL_GROUPS" : {"Democrats" : None, "Republicans" : None, "liberals" : None, "conservatives" : None}
    }
    SINCERE_VOCAB = ["latina","c'mon","outta","fanny","love", "sex" 'ear_lobes', 'ear_lobe', 'inner_thigh','thighs', 'lobe', 'gland','African', 'funny','woman','religions','undocumented_aliens','Hindu_statesman_Rajan_Zed','religious', 'dude', 'cuz','wanna','shiz','hey','lol','pee','short_sighted']
    
    def __init__(self):
    
        self.already_cached = False
        if os.path.isfile(InsincereVocabGenerator.InsincereVocab) and os.path.isfile(InsincereVocabGenerator.InsincereBoW):
            self.insincere_vocab = pickle.load(open(InsincereVocabGenerator.InsincereVocab, "rb"))
            self.insincere_bow = pickle.load(open(InsincereVocabGenerator.InsincereBoW, "rb")) 
            self.already_cached = True
        else:
            # seed topical words by category
            self.insincere_vocab = InsincereVocabGenerator.INSINCERE_VOCAB
            self.insincere_bow = {}
        
    # Insincere_bow is an dictionary of categories. 
    # Each category is mapped to an array of dictionaries of words.
    @profile
    def generate_bow(self):
        for category, category_topics in self.insincere_vocab.items():
            self.insincere_bow[category] = []
            for source, topical_words in category_topics.items():
                if source in InsincereVocabGenerator.SINCERE_VOCAB:
                    continue
                
                (first_word, score, _) = topical_words[0]
                if first_word.lower() == source.lower() and score < 1.0:
                    continue
                    
                word_dict = {}
                for (phrase, score, _) in topical_words:                    
                    _, phrase = TopicalWords.multi_word_in_vocab(phrase)
                    word_dict[phrase] = score

                if len(word_dict) > 1:
                    self.insincere_bow[category].append(word_dict)
        
        if len(self.insincere_bow.items()) > 0:
            save_binary(self.insincere_bow, InsincereVocabGenerator.InsincereBoW)
    
    def get_topical_words(self, source, verbose):

        retval = ([], [])
        try:
            if verbose:
                log("Processing source word: {}".format(source))
            derived_words = TopicalWords(source)
            derived_words.process(num_top=InsincereVocabGenerator.MAX_DERIVED_WORDS_PER_TOPIC, verbose=verbose)
            if verbose:
                log_list(derived_words.substitutes, desc="Strongly Matched to {}".format(source))
                log_list(derived_words.related_sources, transpose=True, desc="Newly discovered based on {}".format(source))
            retval = (derived_words.substitutes, derived_words.related_sources)
            del derived_words
        except:
            if verbose:
                log("Source word {} not found".format(source))
        
        return retval
    
    # XXX culling requires more attention.
    def clean(self):

        for category, category_topics in self.insincere_vocab.items():
            to_delete = []
            for source, topical_words in category_topics.items():
                if topical_words is None or len(topical_words) < 3:
                    to_delete.append(source)
            for source in to_delete:
                del category_topics[source] 
            
    
    @profile
    def generate_similar_words(self, verbose=False):
        
        # Twice for newly discovered topical words
        twice = 0
        while twice < 2: 
            # Iterate categories
            # "PEOPLES":{ "Hindu":None ... }
            for category, category_topics in self.insincere_vocab.items():            
                if verbose:
                    log("<h1>{}</h1>".format(category))
                discovered_source_words = []
                # Iterate topical words in each category
                # "Hindu" : "Sikhs", ...
                for source, topical_words in category_topics.items():
                    if TopicalWords.is_vocab(source) == False:
                        continue
                        
                    if topical_words is None:
                        (substitutes, more_source_words) = self.get_topical_words(source, verbose)
                        if len(substitutes) > 0:
                            category_topics[source] = substitutes
                        if len(more_source_words) > 0:
                            discovered_source_words.extend(more_source_words)
                
                # Add newly discovered words to topical words for this category
                if twice == 0:                    
                    for (w, score) in discovered_source_words:
                        found = False
                        for _, category_topics_2 in self.insincere_vocab.items():
                            if w in category_topics_2 or w.lower() in category_topics_2 or w in InsincereVocabGenerator.SINCERE_VOCAB:
                                found = True
                                break
                        if not found:
                            category_topics[w] = None
                            
            twice = twice + 1
        
        self.clean()
        if len(self.insincere_vocab.items()) > 0:
            save_binary(self.insincere_vocab, InsincereVocabGenerator.InsincereVocab)
                            
    def mislabeled_sincerity(self, selected_categories=['SEXUAL', 'NEGATIVE']):
        
        app = DataManager.instance()
        app.load(orig=True)
        raw_data = app.training_data
        self.sincere = raw_data.index[raw_data.target == 0]        
            
        for i in random.sample(range(0, len(self.sincere)), 10000):
            q_text = raw_data.iloc[ self.sincere[i],:]['question_text']
            for category, category_lists in self.insincere_bow.items():
                if category not in selected_categories:
                    continue
                    
                # Find multiple word lists with the largest phrase 
                for words_list in category_lists:
                    largest_phrase = None
                    size = 0
                    words_list = list(words_list.keys())

                    for phrase in words_list:
                        if TextualDataControl.find_string(phrase, q_text):
                            if len(phrase) > size:
                                size = len(phrase)
                                largest_phrase = phrase

                    if largest_phrase is not None:
                        highlighted_q_text = re.sub(r"\b" + re.escape(largest_phrase) + r"\b", "<b>{} ({})</b>".format(largest_phrase, "??"), q_text, flags=re.IGNORECASE)
                        log(highlighted_q_text)
    @profile
    def process(self, verbose=False):
        if self.already_cached:
            if verbose:
                log("Similar words are already cached.")
        else:
            self.generate_similar_words(verbose=verbose)
            self.generate_bow()
                
    @classmethod
    def clean_cache(cls, vocab=False, bow=False):        
        try:
            if vocab:
                log("Deleting insincere vocabulary cache files...")
                remove_file(InsincereVocabGenerator.InsincereBoW)
                remove_file(InsincereVocabGenerator.InsincereVocab)

            if bow:
                log("Deleting insincere vocabulaty BoW cache file...")
                remove_file(InsincereVocabGenerator.InsincereBoW)
        except:
            log("Some error during cleaning insincere vocab.")

    
    def inspect(self, bow=True, vocab=False):        
        if bow:
            log("<h1>Insincere, Categorized Bag of Words</h1>")
            for category, category_dicts in self.insincere_bow.items():
                word_lists = []
                for a_dict in category_dicts:
                    substitutes = []
                    for word, score in a_dict.items():
                        substitutes.append("{} - {}".format(word, score))
                    substitutes[0] = "<b>{}</b>".format(substitutes[0])
                    word_lists.append(', '.join(substitutes))
                log_list(word_lists, desc="<b>{}</b>".format(category))
        
        if vocab:
            log("<h1>Insincere Vocabulary - Basis of BoW</h1>")
            for category, category_topics in self.insincere_vocab.items():
                log("<h2>{}</h2>".format(category))
                for source, topical_words in category_topics.items():
                    log_list(topical_words, desc="Words within topic {}".format(source))
    
    @classmethod
    def testme(cls):
        me = cls() 
        me.process(verbose=False)
        me.inspect(vocab=False)
        del me
        gc.collect()
        
    @classmethod
    def testme2(cls):
        oov = []
        for category, category_topics in InsincereVocabGenerator.INSINCERE_VOCAB.items():
            for source, _ in category_topics.items():
                if TopicalWords.is_vocab(source, verbose=True) == False:
                    oov.append([category, source])
        
        log_list(oov, desc="Category Topics not in Vocab")
        
        words = ['Barack', 'Obama', 'Donald', 'Trump','Bill','Clinton', "Hinduism"]
        tmp = []
        for w in words:
            tmp.append([w, TopicalWords.is_vocab(w, verbose=True)])
        
        log_list(tmp, desc="Words in Vocab ??")


# In[21]:


if False:
    clear_prof_data()
    #InsincereVocabGenerator.testme2()
    InsincereVocabGenerator.testme()
    pp_prof_data()
    log_dir("../working")

if False:
    log_dir("../working")
    InsincereVocabGenerator.clean_cache(vocab=True, bow=True)
    log_dir("../working")


# In[22]:


class InsincerityWithNER:
    
    CACHED_FILE_PREFIX = "NER"
    OUTPUT_FILE = "NER"
    
    def __init__(self):
        self.q_NER = None        
    
    def gen_NER(self, cpu_id, cpu_count, cachefile, insincere=True):     
        if os.path.isfile(cachefile):
            return

        app = DataManager.instance()
        app.load(orig=True)
        raw_data = app.training_data
        if insincere:
            raw_data = raw_data[raw_data.target == 1]

        q_range = MPHelper.range_partition(cpu_id=cpu_id, total_count=len(raw_data))
        NER_dict = Counter()
        for i in q_range:
            q_text = raw_data.iloc[i,:]['question_text']

            ents = TopicalWords.nlp()(q_text).ents
            if len(ents) == 0:
                continue
            else:
                for X in ents:
                    if X.label_ in ['GPE', 'NORP', 'PERSON', 'ORG']:
                        NER_dict[X.text + "/" + X.label_] += 1    

        save_binary(NER_dict, cachefile)
    
    @profile
    def process(self):
        app = DataManager.instance()
        app.load(orig=True)
        raw_data = app.training_data
        my_insincere_q = raw_data.index[raw_data.target == 1]
        
        log("Generating NER for {} insincere questions.".format(len(my_insincere_q)))

        mp = MPHelper(InsincerityWithNER.CACHED_FILE_PREFIX, verbose=True)
        mp.map_process(self.gen_NER)
        results = mp.reduce()
        merged = Counter()
        for NER_dict in results:
            merged += NER_dict
        
        merged = merged.most_common(len(merged.items()))
        self.q_NER = merged
        save_binary(self.q_NER, InsincerityWithNER.OUTPUT_FILE)
        del mp
        gc.collect()

    def inspect(self):
        if not gInspect:
            return
        
        log_list(self.q_NER[:200], desc="Most frequent 200 NER in 20K questions")        
        

    @classmethod
    def testme(cls):
        me = cls()
        clear_prof_data()
        me.process()
        pp_prof_data()
        me.inspect()
        del me
        
    @classmethod
    def display_cache(cls):
        me = cls()
        me.q_NER = pickle.load(open(InsincerityWithNER.OUTPUT_FILE, "rb"))
        me.inspect()


# In[23]:


if False:
    InsincerityWithNER.testme()
    
if False:
    InsincerityWithNER.display_cache()


# In[24]:


class QuestionsGenerator:
    
    CACHED_FILE_PREFIX = "gen_q"
    NOT_GEN_FILE_PREFIX = "no_gen_q"
    
    def __init__(self):
        self.gen_q = {}
        self.non_gen_q = []
        
        if os.path.isfile(QuestionsGenerator.CACHED_FILE_PREFIX):
            self.gen_q = pickle.load(open(QuestionsGenerator.CACHED_FILE_PREFIX, 'rb'))
        if os.path.isfile(QuestionsGenerator.NOT_GEN_FILE_PREFIX):
            self.non_gen_q = pickle.load(open(QuestionsGenerator.NOT_GEN_FILE_PREFIX, 'rb'))
            
        self.rawdata = None
        self.word_substitutes = None
        self.substitutes_set = None
    
    @classmethod
    def find_substitute_candidates(cls, q_text, substitutions, limit=10):
        retval = {}
        
        # Search for substitutes by matching the longest phrase in the question
        # within each category.
        for category, category_dicts in substitutions.insincere_bow.items():            
            for a_dict in category_dicts:
                largest_phrase = None
                size = 0
                for phrase, _ in a_dict.items():
                    if TextualDataControl.find_string(phrase, q_text):
                        if len(phrase) > size:
                            size = len(phrase)
                            largest_phrase = phrase
                            
                if largest_phrase is not None:
                    if largest_phrase not in retval:
                        retval[largest_phrase] = []
                    retval[largest_phrase].append(a_dict)
            
            '''
            substitutes_count = 1
            for _, subs_dict_list in retval.items():
                maxlen = max([len(d) for d in subs_dict_list])
                substitutes_count *= maxlen
            if substitutes_count > limit:
                break
            '''

        # For each matching phrase, there could be multiple lists of substitutions.
        return retval

    @classmethod
    def pos_phrase(cls, phrase, q_text):
        tmp = q_text.split()
        pos = nltk.pos_tag(tmp)
        
        retval = set()
        retval.add(nltk.pos_tag([phrase])[0][1])
        
        for i in range(len(tmp)): 
            if re.search(r"\b" + re.escape(phrase) + r"\b", tmp[i]):
                retval.add(pos[i][1])
                break
        
        return retval
        
    @classmethod
    def find_substitutes_list(cls, matching_phrase, substitutes_list):
         # Find the right list of substitutes
        retval = substitutes_list[0]
        
        if len(substitutes_list) > 1:
            highest_score = 0
            for a_dict in substitutes_list:
                if (highest_score < a_dict[matching_phrase]) or ((highest_score == a_dict[matching_phrase]) and (len(a_dict) > len(retval))):
                    retval = a_dict
                    highest_score = a_dict[matching_phrase]

        retval = list(retval.keys())
        
        return retval
    
    @classmethod
    def do_substitutes(cls, q_text, phrase, all_substitutes, verbose=False):
        retval = {}
        
        num_generated = 0
        pos = QuestionsGenerator.pos_phrase(phrase, q_text)
        q_tokens = QuoraPreprocessor.make_tokens(q_text)
        q_tokens = set([t.lower() for t in q_tokens])
        
        for substitute in all_substitutes:
            subs_pos = nltk.pos_tag([substitute])[0][1]
            if substitute.lower() != phrase.lower() and subs_pos in pos and substitute.lower() not in q_tokens:
                highlighted_q_text = re.sub(r"\b" + re.escape(phrase) + r"\b", "__{}__".format(phrase, pos), q_text, flags=re.IGNORECASE)
                    
                if highlighted_q_text not in retval:
                    retval[highlighted_q_text] = set()
                if len(retval[highlighted_q_text]) < 10:
                    num_generated += 1
                    
                    if verbose:
                        gen_q = re.sub(r"\b" + re.escape(phrase) + r"\b", "<b>{} ({})</b>".format(substitute, subs_pos), q_text, flags=re.IGNORECASE)
                    else:
                        gen_q = re.sub(r"\b" + re.escape(phrase) + r"\b", substitute, q_text, flags=re.IGNORECASE)                            
                    
                    gen_q = TextualDataControl.sent_capitalize(gen_q)
                    retval[highlighted_q_text].add(gen_q)
                    
        return retval, num_generated
    
    @classmethod
    @profile
    def gen_from_one_q(cls, q_text, substitutions, limit=20, verbose=False):
        
        results = {}
        total_generated = 0
        
        # For each substitutable phrase
        for largest_phrase, subs_dict_list in substitutions.items():
            if verbose:
                log("Number of substitutable lists for phrase <b>{}</b> found:<b>{}</b>".format(largest_phrase, len(subs_dict_list)))

            substitutes = QuestionsGenerator.find_substitutes_list(largest_phrase, subs_dict_list)
            new_q_dict, num_generated = QuestionsGenerator.do_substitutes(q_text, largest_phrase, substitutes, verbose=verbose)
            total_generated += num_generated
            results.update(new_q_dict)
        
        if verbose:
            log_list(results, desc="New Questions Prior to Mixing")
        
        mix_gen_set = set()
        [mix_gen_set.update(q) for _, q in results.items()]
        more_q = {}
        for largest_phrase, subs_dict_list in substitutions.items():
            for orig_q_text, new_q_list in results.items():
                if not re.search(re.escape("__" + largest_phrase + "__"), orig_q_text, flags=re.IGNORECASE):
                    for new_q in new_q_list:
                        if total_generated > limit:
                            break
                        substitutes = QuestionsGenerator.find_substitutes_list(largest_phrase, subs_dict_list)
                        new_q_dict, num_generated = QuestionsGenerator.do_substitutes(new_q, largest_phrase, substitutes)
                        found = False
                        for _, q in new_q_dict.items():
                            if q in mix_gen_set:
                                found = True
                                break
                        if not found:
                            more_q.update(new_q_dict)
                            [mix_gen_set.update(q) for _, q in new_q_dict.items()]
                            total_generated += len(new_q_dict.values())

        results = {q_text : list(mix_gen_set)[-max(limit, len(mix_gen_set)):]}
                                        
        if verbose: 
            if total_generated == 0:
                results["REPORT: " + q_text] = set(["<b><big>NO</big></b> substitutable word found."])
            else:
                results["REPORT: " + q_text] = set(["Number of questions generated: <b>{}</b><br>".format(total_generated)])
                
        return results
    
    @profile
    def gen_q_proc(self, cpu_id, cpu_count, cache_file):
        
        if os.path.isfile(cache_file):
            return        
        
        q_range = MPHelper.range_partition(cpu_id=cpu_id, total_count=len(self.rawdata))
        # Output        
        results = {}    
        max_len = q_range.stop - q_range.start
        total_gen = 0
        for i in q_range:
            q_text = self.rawdata.iloc[i,:]['question_text']
            if self.is_substitutable(q_text):
                # Find multiple word lists with the largest phrase
                substitutions = QuestionsGenerator.find_substitute_candidates(q_text, self.word_substitutes)            
                new_q_dict = QuestionsGenerator.gen_from_one_q(q_text, substitutions, verbose=False, limit=10)
                num_new_q = len(list(new_q_dict.values())[0])
                total_gen += num_new_q
                results.update(new_q_dict)
                if total_gen > max_len:
                    break

        for q, a_set in results.items():
            results[q] = list(a_set)
        
        self.save([results], cache_file)
        
    @profile
    def nogen_q_proc(self, cpu_id, cpu_count, cache_file):
        if os.path.isfile(cache_file):
            return        
        
        q_range = MPHelper.range_partition(cpu_id=cpu_id, total_count=len(self.rawdata))
        results = []
        for i in q_range:
            q_text = self.rawdata.iloc[i,:]['question_text']
            if self.is_substitutable(q_text) == False:
                results.append(q_text)
                    
        save_binary(results, cache_file)

    def save(self, results, cache_file):
        if gWIP:
            save_binary(results[0], cache_file)
        
        # Create generated questions CSV that matches the train.csv format.
        # XXX what assumptions can be made about train.csv format.
        cache_file_csv = cache_file + ".csv"
        headers = ['qid', 'question_text', 'target']
            
        dest_f = open(cache_file_csv, 'w')
        csv_writer = csv.writer(dest_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(headers)
        for cpu_result in results:
            for orig_q, gen_q_list in cpu_result.items():
                for gen_q in gen_q_list:
                    qid = ''.join(random.choice(string.hexdigits) for x in range(20)).lower()
                    row = [qid, gen_q, str(1)]
                    csv_writer.writerow(row)
            
        dest_f.close()
        
    def make_substitutes_bow(self):
        self.substitutes_set = set()
        for category, category_dicts in self.word_substitutes.insincere_bow.items():            
            for a_dict in category_dicts:
                for k, _ in a_dict.items():
                    self.substitutes_set.add(k.lower())
        
        log(f"Number of substitutes: {len(self.substitutes_set)}")
    
    def is_substitutable(self, q_text):
        tokens = QuoraPreprocessor.make_tokens(q_text)
        tokens = set([t.lower() for t in tokens])
        
        retval = len(tokens & self.substitutes_set) > 0  
        
        return retval
    
    @profile
    def process(self, debug_df=None):
        
        if len(self.gen_q) > 0:
            return
        
        # RawData
        if debug_df is None:
            app = DataManager.instance()
            app.load(orig=True)
            self.rawdata = app.training_data[app.training_data.target == 1]        
        else:
            self.rawdata = debug_df
        
        # word substitutes
        insincere_vocab = InsincereVocabGenerator()
        insincere_vocab.process()
        self.word_substitutes = insincere_vocab
        self.make_substitutes_bow()
        
        if gWIP:
            mp = MPHelper(QuestionsGenerator.NOT_GEN_FILE_PREFIX, verbose=True)
            mp.map_process(self.nogen_q_proc)
            results = mp.reduce(clean=True)
            self.non_gen_q = [q for result in results for q in result]
        
        # Multi-process q generation
        mp = MPHelper(QuestionsGenerator.CACHED_FILE_PREFIX, verbose=False)
        mp.map_process(self.gen_q_proc)
        self.gen_q = mp.reduce(clean=True)
        
                    
    def inspect(self, limit=1000):
        # To help improve generation algorithm.
        q_count = self.rawdata.count()['qid']

        gen_q_count = 0
        for gen_q_i in self.gen_q:
            for q_text, new_q in gen_q_i.items():
                gen_q_count += len(new_q)

        cpu_index = 0
        gen_text = set()
        for gen_q_i in self.gen_q:
            log("<h2>Random question sets From CPU {}</h2>".format(cpu_index))
            q_range = MPHelper.range_partition(cpu_id=cpu_index, total_count=len(self.rawdata))

            cpu_index += 1
            num_iter = min(int(limit/mpc.cpu_count()), len(gen_q_i))
            keys = list(gen_q_i.keys())
            samples = np.random.permutation(keys)
            for i in range(num_iter):
                q_text = samples[i]
                q_list = gen_q_i[q_text]
                gen_text.update(q_list)
                log_list(q_list, desc=q_text)

        log("<h1>Number of new questions generated: {} from total {} insincere questions.".format(gen_q_count, q_count))

        log("<h2>Number of original, insincere questions <b>NOT</b> used for generating new questions: {}</h2>".format(len(self.non_gen_q)))
        log_list(self.non_gen_q[0:min(50, len(self.non_gen_q))], desc="50 insincere questions with no substitutable words")

        log("<h3>Prior to generated data</h3>")
        self.rawdata.at[0,'target'] = 0
        
        tdc = TextualDataControl(self.rawdata)
        tdc.inspect()
        del tdc
        log("<h3>POST data generation</h3>")
        generated_data = pd.DataFrame()
        log(f"For limited display purposes, adding {len(gen_text)} new questionst to training data.")
        generated_data = generated_data.assign(qid=np.arange(len(gen_text)), question_text=gen_text, target=np.ones(len(gen_text), dtype=np.int8))
        self.rawdata = self.rawdata.append(generated_data, ignore_index=False)
        log(f"<h4>Total training data size: {len(self.rawdata)}</h4>")
        tdc = TextualDataControl(self.rawdata)
        tdc.inspect()
        del tdc
        gc.collect()
        
    @classmethod
    def clean_cache(cls, remove_output=False):
        try:
            if remove_output:
                remove_file(QuestionsGenerator.CACHED_FILE_PREFIX)
                remove_file(QuestionsGenerator.NOT_GEN_FILE_PREFIX)
                remove_file(QuestionsGenerator.CACHED_FILE_PREFIX + ".csv")
            
            for index in range(mpc.cpu_count()):
                remove_file(QuestionsGenerator.CACHED_FILE_PREFIX + "_{}".format(index))
            for index in range(mpc.cpu_count()):
                remove_file(QuestionsGenerator.CACHED_FILE_PREFIX + "_{}.csv".format(index))
                
        except:
            log("Got an exception while cleaning gen_q cache.")
    
    @classmethod
    def testme(cls):
       
        use_test_data = False
        debug_df = None
        if use_test_data:
            q_texts = ["I would like to punish my son by legally changing his name to Rapist. How do I do this?", "This is not an insincere question.","Is there anything wrong with ageism?", "What percentage of Americans are stupid, and why?", "Do moms have sex with their sons?", "Why women in Tinder are shit?","If blacks support school choice and mandatory sentencing for criminals why don't they vote Republican?", "Why do Christians in India hate Brahmins so much?", "Why do Europeans say they're the superior race, when in fact it took them over 2,000 years until mid 19th century to surpass China's largest economy?"]

            debug_df = pd.DataFrame()
            debug_df = debug_df.assign(qid=np.arange(len(q_texts)), question_text=q_texts, target=np.ones(len(q_texts), dtype=np.int8))
            debug_df.at[1,'target'] = 0
            debug_df.head()
            display(debug_df.head())
        
        me = cls()
        me.process(debug_df=debug_df)
        me.inspect(limit=20)
        QuestionsGenerator.clean_cache()
        del me
        gc.collect()
        
        if False:
            insincere_vocab = InsincereVocabGenerator()
            insincere_vocab.process()
            for q_text in q_texts:
                log("Generating new q from <h1>{}</h1>".format(q_text))
                gen_q = {}
                results = QuestionsGenerator.find_substitute_candidates(q_text=q_text, substitutions=insincere_vocab)
                log_list(results)
                results = QuestionsGenerator.gen_from_one_q(q_text, results, verbose=False)
                gen_q.update(results)
                log_list(gen_q)


# In[25]:


if False:
    clear_prof_data()
    #TopicalWords.memclean()
    #InsincereVocabGenerator.clean_cache(vocab=True, bow=True)
    QuestionsGenerator.clean_cache(remove_output=True)
    QuestionsGenerator.testme()
    log_dir('../working')
    pp_prof_data()


# In[26]:


class AttentionLayer(Layer):
    def __init__(self, step_dim=QuoraPreprocessor.MAX_WORDS_IN_QUESTION,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight('{}_W'.format(self.name), shape=(input_shape[-1].value,),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name), shape=(input_shape[1].value,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint
                                    )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[27]:


class ModelFactory:
    def __init__(self, embeddings_matrix):
        self.embeddings_matrix = embeddings_matrix
        self.max_features = self.embeddings_matrix.shape[0]
        self.embed_size = self.embeddings_matrix.shape[1]
        self.maxlen = QuoraPreprocessor.MAX_WORDS_IN_QUESTION
        
        self.mask_zero = False
        self.models = {}
    
    @classmethod
    def multi_gpu_model(cls, model):        
        retval = model
        return retval
        
    def create_GPU_LSTM_Model(self, units=64):
        
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        x = Bidirectional(CuDNNLSTM(units*2, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)
        x = AttentionLayer(self.maxlen)(x)
        x = Dense(units, activation='relu')(x)
        outp= Dense(1, activation='sigmoid')(x)
        model = ModelFactory.multi_gpu_model(Model(inputs=inp, outputs=outp))
        model = ModelFactory.compilation(model)
        
        return model
    
    def create_LSTMGRU_Model(self, units=64):
        
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        x = SpatialDropout1D(0.1)(x)
        x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)
        y = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)

        atten_1 = AttentionLayer(self.maxlen)(x)
        atten_2 = AttentionLayer(self.maxlen)(y)
        avg_pool = GlobalAveragePooling1D()(y)
        max_pool = GlobalMaxPool1D()(y)
        
        conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
        conc = Dense(int(units/4), activation='relu')(conc)
        conc = Dropout(0.1)(conc)
        outp= Dense(1, activation='sigmoid')(conc)
        
        model = ModelFactory.multi_gpu_model(Model(inputs=inp, outputs=outp))
        model = ModelFactory.compilation(model)
        
        return model
    
    
    def create_CPU_LSTM_Model(self, units=64):
        
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        x = Bidirectional(LSTM(units*2, return_sequences=True))(x)
        x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = AttentionLayer(self.maxlen)(x)
        x = Dense(units, activation='relu')(x)
        outp= Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=outp)
        model = ModelFactory.compilation(model)
        
        return model
    
    def create_GPU_GRU3_Model(self, units=64):
        
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features,  self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        x = Bidirectional(CuDNNGRU(units*2, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(int(1.5*units), return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
        x = AttentionLayer(self.maxlen)(x)
        outp = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inp, outputs=outp)
        model = ModelFactory.compilation(model)
    
        return model

    def create_2DCNN_Model(self, num_filters=64, filter_size=3):
        
        inp = Input(shape=(self.maxlen, ))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        #    x = SpatialDropout1D(0.4)(x)
        x = Reshape((self.maxlen, self.embed_size, 1))(x)

        x = Conv2D(num_filters, kernel_size=(filter_size, filter_size), 
                   kernel_initializer='he_normal', activation='elu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        # BatchNorm XXX
        x = Conv2D(num_filters, kernel_size=(filter_size, filter_size),
                   kernel_initializer='he_normal', activation='elu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        
        z = Flatten()(x)
        z = BatchNormalization()(z)

        outp = Dense(1, activation="sigmoid")(z)

        model = ModelFactory.multi_gpu_model(Model(inputs=inp, outputs=outp))
        model = ModelFactory.compilation(model)

        return model
    
    def create_2DCNN_ConcatModel(self, num_filters=36, filter_sizes=[1,2,3,5]):
        
        inp = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inp)
        x = Reshape((self.maxlen, self.embed_size, 1))(x)

        maxpool_pool = []
        for i in range(len(filter_sizes)):
            conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], self.embed_size),
                                         kernel_initializer='he_normal', activation='elu')(x)
            maxpool_pool.append(MaxPool2D(pool_size=(self.maxlen - filter_sizes[i] + 1, 1))(conv))

        z = Concatenate(axis=1)(maxpool_pool)   
        z = Flatten()(z)
        z = Dropout(0.1)(z)

        outp = Dense(1, activation="sigmoid")(z)

        model = ModelFactory.multi_gpu_model(Model(inputs=inp, outputs=outp))
        model = ModelFactory.compilation(model)

        return model
    
    def create_1DCNNNgram_Model(self, num_filters=32, kernel_sizes=[2,4,8], units=None):
        
        # channel 1
        inputs1 = Input(shape=(self.maxlen,))
        embedding1 = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inputs1)
        conv1 = Conv1D(filters=num_filters, kernel_size=kernel_sizes[0], activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(self.maxlen,))
        embedding2 = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inputs2)
        conv2 = Conv1D(filters=num_filters, kernel_size=kernel_sizes[1], activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(self.maxlen,))
        embedding3 = Embedding(self.max_features, self.embed_size, weights=[self.embeddings_matrix], trainable=gTrainableEmbeddings, mask_zero=self.mask_zero)(inputs3)
        conv3 = Conv1D(filters=num_filters, kernel_size=kernel_sizes[2], activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = ModelFactory.multi_gpu_model(Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs))
        model = ModelFactory.compilation(model)
        
        return model
        

    @classmethod
    def compilation(cls, model):
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])        
        return model
    
    def get_model(self, model_name, **kwargs):
        retval = None
        if model_name in self.models:
            retval = self.models[model_name]
        else:            
            if model_name in ['2DCNNConcat']:
                retval = self.create_2DCNN_ConcatModel(**kwargs)
            elif model_name in ['GPUGRU3']:
                retval = self.create_GPU_GRU3_Model(**kwargs)
            elif model_name in ['GPULSTM']:
                retval = self.create_GPU_LSTM_Model(**kwargs)
            elif model_name in ['LSTMGRU']:
                retval = self.create_LSTMGRU_Model(**kwargs)
            elif model_name in ['CPULSTM']:
                retval = self.create_CPU_LSTM_Model(**kwargs)
            elif model_name in ['2DCNN']:
                retval = self.create_2DCNN_Model(**kwargs)
            elif model_name in ['1DCNNNgram']:
                retval = self.create_1DCNNNgram_Model(**kwargs)
            else:
                assert True, "No model of this name found."
        
            self.models[model_name] = retval
        
        return retval
                
    
    def inspect(self):
        if not gInspect:
            return
        
        for name, model in self.models.items():
            log(f"<h1>Summary for {name}</h1>")
            model.summary()        
    
    @classmethod
    def testme(cls):
        sample_word2index = {'good': 0, 'bad': 1, 'sincere': 2, 'insincere': 3, 'sexual_intercourse':5, 'Donald_Trump' : 6}
        index2word = {}
        vocab2count = Counter()
        for k, v in sample_word2index.items():
            index2word[v] = k
            if k in vocab2count:
                vocab2count[k] += 1
            else:
                vocab2count[k] =1
        
        w2v, oov = EmbeddingsControl.load(gSelectedEmbeddings[0], sample_word2index)
        me = cls(w2v)
        me.get_model('GPULSTM')
        me.get_model('1DCNNNgram')
        me.get_model('GPUGRU3')
        tmp = me.get_model('LSTMGRU')
        tmp2 = clone_model(tmp)
        tmp2.summary()
        me.inspect()
        del w2v
        del oov
        del me
        gc.collect()


# In[28]:


if False:
    ModelFactory.testme()


# In[29]:


class QuoraSequence(Sequence):
    
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1, verbose=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=self.verbose)
            score, threshold = NLPPatterns.find_prediction_threshold(self.y_val, y_pred)
            #log("F1 Score - epoch: %d - threshold: %f - <b>score: %.6f</b>" % (epoch+1, best_threshold, best_score))
            log("\n F1 Score - epoch: %d - score: %.6f \n" % (epoch+1, score))

class NLPPatterns:
    
    MODEL_CACHE_FILE = 'bestmodel.h5'
    
    def __init__(self, model, X, y, epochs=2):
        assert(model is not None)
        self.model = model
        self.X_train = X
        self.y_train = y
        self.X_val = None
        self.y_val = None
        self.threshold = -1
        self.score = 0
        self.epochs = epochs

        self.thresholds = []
        self.scores = []        

    @classmethod
    def make_callbacks(cls, X_val, y_val, model_cachefile):
            
        # evaluator_cb = F1Evaluation(validation_data=(X_val, y_val), interval=1, verbose=1)
        checkpoint_cb = ModelCheckpoint(filepath=model_cachefile, monitor='val_loss', save_best_only=True, verbose=2, mode='min')
        earlystopping_cb = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
        reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
        
        callbacks = [earlystopping_cb, checkpoint_cb, reduce_lr_cb]

        return callbacks
    
    def data_partition(self, train_to_val_ratio=0.9, strategy="interleave"):
        
        if strategy == "interleave":            
            tmp_insincere = np.where(self.y_train == 1)[0]
            tmp_sincere = np.where(self.y_train == 0)[0]
            np.random.shuffle(tmp_sincere)
            np.random.shuffle(tmp_insincere)

            sincere_count = 0
            insincere_count = 0
            modulo = int(len(tmp_sincere)/len(tmp_insincere))+1
            indices = np.zeros(len(self.y_train), dtype=np.int32)
            
            for i in range(len(indices)):
                if i % modulo == 0 and insincere_count < len(tmp_insincere):
                    indices[i] = tmp_insincere[insincere_count]
                    insincere_count += 1
                elif sincere_count < len(tmp_sincere):
                    indices[i] = tmp_sincere[sincere_count]
                    sincere_count += 1
            
            self.X_train = np.array(self.X_train)
            indices = np.array(indices)
            self.X_train = self.X_train[indices]
            self.y_train = self.y_train[indices]
            
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, train_size=train_to_val_ratio, random_state=SEED, shuffle=True)
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)
        self.X_val = np.asarray(self.X_val)
        self.y_val = np.asarray(self.y_val)
        
    def data_generator(self, train=False, val=False, batch_size=256):
        X = None
        y = None
        
        if train:
            X = self.X_train
            y = self.y_train
        
        if val:
            X = self.X_val
            y = self.y_val
        
        # Initialize a counter
        counter = 0
        num_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        while True:
            for i in range(num_batches):
                yield  X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

    
    @profile
    def train(self, train_batch_size=256, validation_batch_size=1024, use_generator=False):
        
        callbacks = NLPPatterns.make_callbacks(self.X_val, self.y_val, NLPPatterns.MODEL_CACHE_FILE)
        
        if use_generator:
            num_batches = int(np.ceil(self.X_train.shape[0] / float(train_batch_size)))
            training_generator = self.data_generator(train=True, batch_size=train_batch_size)
            
            self.model.fit_generator(generator=training_generator, validation_data=(self.X_val, self.y_val), steps_per_epoch=num_batches, epochs=self.epochs, use_multiprocessing=True, workers=mpc.cpu_count(), callbacks=callbacks,verbose=1)
        elif gModelName == "1DCNNNgram":
            self.model.fit([self.X_train, self.X_train, self.X_train], self.y_train, validation_data=([self.X_val, self.X_val, self.X_val], self.y_val), batch_size=train_batch_size, epochs=self.epochs, callbacks=callbacks, verbose=1)
        else:
            self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), batch_size=train_batch_size, epochs=self.epochs, callbacks=callbacks, verbose=1)
            
        self.load()
        self.validate(batch_size=validation_batch_size)
        
    @profile
    def validate(self, batch_size=1024):        
        
        y_val_pred = None
        if gModelName == "1DCNNNgram":
            y_val_pred = self.model.predict([self.X_val, self.X_val, self.X_val], batch_size=batch_size)
        else:
            y_val_pred = self.model.predict(self.X_val, batch_size=batch_size)
        
        self.score, self.threshold = NLPPatterns.find_prediction_threshold(self.y_val, y_val_pred)      
        log('optimal F1: {:.4f} at threshold: {:.4f}'.format(self.score, self.threshold))
        
    @profile
    def predict(self, X_test, batch_size=1024):
        if gModelName == "1DCNNNgram":
            retval = self.model.predict([X_test, X_test, X_test], batch_size=batch_size, verbose=1)
        else:
            retval = self.model.predict(X_test, batch_size=batch_size, verbose=1)
            
        return retval
    
    def save(self):
        if os.path.isfile(NLPPatterns.MODEL_CACHE_FILE):
            return
        
        self.model.save(NLPPatterns.MODEL_CACHE_FILE)        
        log("Saved model to disk")   
            
    def load(self):
        self.model = load_model(NLPPatterns.MODEL_CACHE_FILE, custom_objects={'AttentionLayer': AttentionLayer})
        log("Loaded model from disk")
    
    @classmethod
    def find_prediction_threshold(cls, y_true, y_proba):
        
        best_threshold = 0.01
        best_score = 0.0
        for threshold in [i * 0.01 for i in range(1,100)]:
            tmp = (y_proba > threshold).astype(int)
            if np.count_nonzero(tmp) == 0:
                continue
            score = f1_score(y_true=y_true, y_pred=tmp)
            if score > best_score:
                best_threshold = threshold
                best_score = score

        return best_score, best_threshold
    
    @classmethod
    def clean_cache(cls):
        remove_file(NLPPatterns.MODEL_CACHE_FILE)
    
    def process(self):
        
        self.data_partition(strategy="interleave")

        if os.path.isfile(NLPPatterns.MODEL_CACHE_FILE):
            self.load()
            self.validate()
            return
        
        if has_gpu():
            self.train(train_batch_size=gTrainingBatchSize, use_generator=False)
        else:
            #self.train(use_sequence=True)
            self.train(train_batch_size=gTrainingBatchSize, use_generator=True)
    
    def inspect(self):
        if not gInspect:
            return
        
        log(f"Shape of Training input set {self.X_train.shape}")
        log(f"Shape of Training output set {self.y_train.shape}")
        log(f"Shape of Validation input set {self.X_val.shape}")
        log(f"Shape of Validation output {self.y_val.shape}")
        log_list(self.y_val[0:25], desc="y validation samples (25)")
        log(f"At threshold {self.threshold}, best F1 score: {self.score}")


# In[30]:


if False:
    NLPPatterns.testme()


# In[31]:


def initialize_cache():
    if gFirstTime:
        log("<hr><h2 align='center'>Initialize Cache</h2><hr>")
        log("Before cleanup...")
        log_dir('../working')
        log(f"Inspection is {gInspect}")

        DataManager.clean_cache()
        InsincereVocabGenerator.clean_cache(bow=True, vocab=True)
        QuestionsGenerator.clean_cache(remove_output=True)
        for source, _ in gEmbeddingsSources.items():
            QuoraPreprocessor.clean_cache(cleaned_q=True, vocab=True, word2index=True, sequences=True)
        
        EmbeddingsControl.clean_cache() 
        NLPPatterns.clean_cache()
        log("After cleanup, cache contents:")
        log_dir('../working')


# In[32]:


def synthesize_data():    
    if gGenerateNewData == False:
        log("<hr><h2 align='center'>Using Original Training Data ONLY</h2><hr>")
        return
    
    log("<hr><h2 align='center'>Generate Insincere Vocab</h2><hr>")

    gen_vocab = InsincereVocabGenerator()
    gen_vocab.process(verbose=False)
    if gInspect:
        gen_vocab.inspect(vocab=False)

    log("<hr><h2 align='center'>Generate Questions</h2><hr>")
    
    gen_q = QuestionsGenerator()
    gen_q.process()
    
    # Combined generated cached file with input to create big training data cache.
    app = DataManager.instance()
    app.save_combined_data()
    if gInspect:
        gen_q.inspect(limit=10)
        
    if gLimit == False:
        # Since we have created a combined file.
        QuestionsGenerator.clean_cache(remove_output=True)
        TopicalWords.memclean()
        del gen_q
        del gen_vocab
        gc.collect()


# In[33]:


def massage_data():
    log("<hr><h2 align='center'>Pre-Processing Question Text</h2><hr>")

    app = DataManager.instance()
    app.load(combined=(gGenerateNewData==True), orig=(gGenerateNewData == False), test=True)
    
    if gWIP:
        stats = []
        stats.append(["Training Data Size", str(app.training_data.shape)])
        stats.append(["Test Data Size", str(app.test_data.shape)])
        log_list(stats, desc="<big>Data Sizes</big>")
    
    data2numbers = QuoraPreprocessor(training_data=app.training_data, test_data=app.test_data, is_gen=gGenerateNewData)
    data2numbers.process()
    data2numbers.inspect()
    
    all_w2v = {}
    for source, _ in gEmbeddingsSources.items():        
        if source not in gSelectedEmbeddings:
            continue
        a_w2v, a_oov = EmbeddingsControl.load(source, data2numbers.word2index)
        all_w2v[source] = a_w2v
        EmbeddingsControl.inspect(index2word=data2numbers.index2word, word2count=data2numbers.word2count, oov=a_oov, source=source, w2v=a_w2v)
        del a_oov
    
    i=0
    embeddings_matrix = None
    for source in gSelectedEmbeddings:
        if i == 0:
            embeddings_matrix = all_w2v[source]
        else:
            embeddings_matrix += all_w2v[source]
        i += 1
            
    embeddings_matrix = embeddings_matrix/len(gSelectedEmbeddings)
    log("Combined embedding matrix shape:" + str(embeddings_matrix.shape))

    EmbeddingsControl.clean_cache()
    save_binary(embeddings_matrix, "MATRIX_ALL")
    del embeddings_matrix
    
    if gLimit == False:
        for source, _ in gEmbeddingsSources.items():
            QuoraPreprocessor.clean_cache(cleaned_q=False, vocab=True, word2index=True, sequences=False)
    
    del app
    del data2numbers
    del all_w2v
    gc.collect()


# In[34]:


@profile
def find_patterns():
    log("<hr><h2 align='center'>Training</h2><hr>")    
    retval = {}
    
    gen_suffix = "orig"
    if gGenerateNewData:
        gen_suffix = "gen"

    # Get X
    sequences_file = QuoraPreprocessor.SEQUENCES_OUTPUT_FILE + "_" + gen_suffix
    X = np.squeeze(pickle.load(open(sequences_file, 'rb')))

    # Get y
    cleaned_data_file = QuoraPreprocessor.CLEANED_OUTPUT_FILE + "_" + gen_suffix + ".csv"
    data = pd.read_csv(cleaned_data_file)
    y = np.squeeze(data['target'].values)

    # Get embeddings matrix
    embeddings_matrix = pickle.load(open("MATRIX_ALL", 'rb'))

    # Select Model
    model_factory = ModelFactory(embeddings_matrix=embeddings_matrix)
    model = model_factory.get_model(gModelName, units=64)
    model_factory.inspect()
        
    # Train
    patterns = NLPPatterns(X=X,y=y,model=model,epochs=gNumEpochs)
    patterns.process()
    patterns.inspect()

    return patterns


# In[35]:


def apply_patterns(patterns):
    log("<hr><h2 align='center'>Predict</h2><hr>")
    
    y_predictions = {}
    sequences = QuoraPreprocessor.SEQUENCES_OUTPUT_FILE + "_" + "test"
    X_test = pickle.load(open(sequences, 'rb'))
    log(f"Number of test samples: {len(X_test)}")
    X_test = np.squeeze(X_test)
    y_predictions = patterns.predict(X_test)
    
    y_predictions = y_predictions > patterns.threshold
    
    return y_predictions


# In[36]:


def save(output):
    log("<hr><h2 align='center'>Submission of Test Predictions</h2><hr>")
    
    y_pred = output
    y_pred = y_pred.reshape(-1).astype(int)
    
    app = DataManager.instance()
    app.load(orig=False,combined=False, test=True)
    test_data = app.test_data
    log(f"Shapes<br>test_data {test_data.shape}")
    log(f"output {y_pred.shape}")
    
    if gInspect:
        display_frame = pd.DataFrame()
        display_frame = display_frame.assign(qid=test_data['qid'],question_text=test_data['question_text'],prediction=y_pred)
        display(display_frame[display_frame.prediction == 1].head(100))

    submission = pd.DataFrame()
    submission = submission.assign(qid=test_data['qid'],prediction=y_pred)
    submission.to_csv('submission.csv', index=False)


# In[37]:


@profile
def main():
    global gFirstTime
    initialize_cache()
    synthesize_data()
    massage_data()
    NLPPatterns.clean_cache()
    gFirstTime = False
    patterns = find_patterns()
    output = apply_patterns(patterns)
    save(output)


# In[38]:


if gExecuteMain:
    log_current_memory(caption="Memory before starting Main")
    clear_prof_data()
    main()
    pp_prof_data()
    log_current_memory(caption="Memory after Main")


# In[39]:


if False:
    # initialize_cache()
    # remove_file('sequences')
    # remove_file('submission.csv')
    log_dir('../working')
    log_current_memory(caption="The End")


# In[40]:


if False:
    app = DataManager.instance()
    app.load(source_file="cleaned_q.csv")
    data = app.training_data
    log = []
    for i in range(0, 200, 10):
        df = data[(data.target==1) & (data.num_words > i)]
        log.append([str(i), len(df)])
    log_list(log, desc="Question Lengths Count")
    
    word2index = pickle.load(open(QuoraPreprocessor.W2INDEX_OUTPUT_FILE+"_GOOGLENEWS", "rb"))
    log(f'vocab_size: {len(word2index)}')


# In[41]:


if False:
    tmp = pickle.load(open('no_gen_q', 'rb'))
    tmp = [q for result in tmp for q in result]
    log_list(tmp[0:20])


# In[42]:


if False:
    y_train = np.asarray([0,0,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1, 0, 0])
    tmp_X = np.repeat(["Hello"], y_train.shape[0])
    X = []
    for i in range(y_train.shape[0]):
        tmp = []
        for j in range(10):
            tmp.append(tmp_X[i] + str(i) + "--" + str(y_train[i]))
        X.append(tmp)
        
    print(X)
    
    tmp_insincere = np.where(y_train == 1)[0]
    tmp_sincere = np.where(y_train == 0)[0]
    
    np.random.shuffle(tmp_sincere)
    np.random.shuffle(tmp_insincere)
    
    print(tmp_insincere)
    print(tmp_sincere)
    
    print(str(len(tmp_insincere)))
    print(str(len(tmp_sincere)))
    modulo = int(len(tmp_sincere)/len(tmp_insincere))+1
    print(modulo)
    
    sincere_count = 0
    insincere_count = 0
    indices = np.zeros(len(y_train),dtype=np.int8)
    for i in range(len(indices)):
        if i % modulo == 0 and insincere_count < len(tmp_insincere):
            indices[i] = tmp_insincere[insincere_count]
            insincere_count += 1
        elif sincere_count < len(tmp_sincere):
            indices[i] = tmp_sincere[sincere_count]
            sincere_count += 1
    
    print("Before:\n", y_train)
    indices = np.array(indices)
    print(y_train[indices])
    X = np.array(X)
    print(X[indices])


# In[43]:





# In[43]:




