#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd 
import os
import zipfile

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv')
nrows = 10
data = pd.read_csv("gap-development.tsv", sep = '\t', nrows = nrows)


# In[3]:


#downloading weights and cofiguration file for bert
get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip')
with zipfile.ZipFile("uncased_L-24_H-1024_A-16.zip","r") as zip_ref:
    zip_ref.extractall()
get_ipython().system('rm "uncased_L-24_H-1024_A-16.zip"')


# In[4]:


get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/extract_features.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py')

import modeling
import extract_features
import tokenization
import tensorflow as tf


# In[5]:


def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count

def count_chars_no_special(text):
	count = 0
	special_char_list = ["#"]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def count_length_no_special(text):
	count = 0
	special_char_list = ["#", " "]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count


# In[6]:


text = data["Text"]
text.to_csv("input.txt", index = False, header = False)

os.system("python3 extract_features.py   --input_file=input.txt   --output_file=output.jsonl   --vocab_file=uncased_L-24_H-1024_A-16/vocab.txt   --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json   --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt   --layers=-1,-5   --max_seq_length=256   --batch_size=8")

bert_output = pd.read_json("output.jsonl", lines = True)
os.system("rm output.jsonl")
os.system("rm input.txt")


# In[7]:


emb_2d = {}
for row in range(nrows):
    P = data.loc[row,"Pronoun"].lower()
    A = data.loc[row,"A"].lower()
    B = data.loc[row,"B"].lower()
    P_offset = compute_offset_no_spaces(data.loc[row,"Text"], data.loc[row,"Pronoun-offset"])
    A_offset = compute_offset_no_spaces(data.loc[row,"Text"], data.loc[row,"A-offset"])
    B_offset = compute_offset_no_spaces(data.loc[row,"Text"], data.loc[row,"B-offset"])
    # Figure out the length of A, B, not counting spaces or special characters
    A_length = count_length_no_special(A)
    B_length = count_length_no_special(B)
    
    # Get the BERT embeddings for the current line in the data file
    features = pd.DataFrame(bert_output.loc[row,"features"]) 
    
    span = range(2,len(features)-2)
    emb1, emb5 = {}, {}
    count_chars = 0
    
    # Make a list with the text of each token, to be used in the plots
    texts = []

    for j in span:
        token = features.loc[j,'token']
        texts.append(token)
        emb1[j] = np.array(features.loc[j,'layers'][0]['values'])
        emb5[j] = np.array(features.loc[j,'layers'][1]['values'])
        if count_chars == P_offset:
            texts.pop()
            texts.append("@P" + token)
        if count_chars in range(A_offset, A_offset + A_length): 
            texts.pop()
            if data.loc[row,"A-coref"]:
                texts.append("@G" + token)
            else:
                texts.append("@R" + token)
        if count_chars in range(B_offset, B_offset + B_length): 
            texts.pop()
            if data.loc[row,"B-coref"]:
                texts.append("@G" + token)
            else:
                texts.append("@R" + token)
        count_chars += count_length_no_special(token)
    
    X1 = np.array(list(emb1.values()))
    X5 = np.array(list(emb5.values()))
    if row == 0: print("Shape of embedding matrix: ", X1.shape)

    # Use PCA to reduce dimensions to a number that's manageable for t-SNE
    pca = PCA(n_components = 50, random_state = 7)
    X1 = pca.fit_transform(X1)
    X5 = pca.fit_transform(X5)
    if row == 0: print("Shape after PCA: ", X1.shape)

    # Reduce dimensionality to 2 with t-SNE.
    # Perplexity is roughly the number of close neighbors you expect a
    # point to have. Our data is sparse, so we chose a small value, 10.
    # The KL divergence objective is non-convex, so the result is different
    # depending on the seed used.
    tsne = TSNE(n_components = 2, perplexity = 10, random_state = 6, 
                learning_rate = 1000, n_iter = 1500)
    X1 = tsne.fit_transform(X1)
    X5 = tsne.fit_transform(X5)
    if row == 0: print("Shape after t-SNE: ", X1.shape)
    
    # Recording the position of the tokens, to be used in the plot
    position = np.array(list(span)) 
    position = position.reshape(-1,1)
    
    X = pd.DataFrame(np.concatenate([X1, X5, position, np.array(texts).reshape(-1,1)], axis = 1), 
                     columns = ["x1", "y1", "x5", "y5", "position", "texts"])
    X = X.astype({"x1": float, "y1": float, "x5": float, "y5": float, "position": float, "texts": object})

    # Remove a few outliers based on zscore
    X = X[(np.abs(stats.zscore(X[["x1", "y1", "x5", "y5"]])) < 3).all(axis=1)]
    emb_2d[row] = X


# In[8]:


for row in range(nrows):
    X = emb_2d[row]
    
    # Plot for layer -1
    plt.figure(figsize = (20,15))
    p1 = sns.scatterplot(x = X["x1"], y = X["y1"], hue = X["position"], palette = "coolwarm")
    p1.set_title("development-"+str(row+1)+", layer -1")
    
    # Label each datapoint with the word it corresponds to
    for line in X.index:
        text = X.loc[line,"texts"]
        if "@P" in text:
            p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text[2:], horizontalalignment='left', 
                    size='medium', color='blue', weight='semibold')
        elif "@G" in text:
            p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text[2:], horizontalalignment='left', 
                    size='medium', color='green', weight='semibold')
        elif "@R" in text:
            p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text[2:], horizontalalignment='left', 
                    size='medium', color='red', weight='semibold')
        else:
            p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text, horizontalalignment='left', 
                    size='medium', color='black', weight='semibold')
    
    # Plot for layer -5
    plt.figure(figsize = (20,15))
    p1 = sns.scatterplot(x = X["x5"], y = X["y5"], hue = X["position"], palette = "coolwarm")
    p1.set_title("development-"+str(row+1)+", layer -5")
    
    for line in X.index:
        text = X.loc[line,"texts"]
        if "@P" in text:
            p1.text(X.loc[line,"x5"]+0.2, X.loc[line,"y5"], text[2:], horizontalalignment='left', 
                    size='medium', color='blue', weight='semibold')
        elif "@G" in text:
            p1.text(X.loc[line,"x5"]+0.2, X.loc[line,"y5"], text[2:], horizontalalignment='left', 
                    size='medium', color='green', weight='semibold')
        elif "@R" in text:
            p1.text(X.loc[line,"x5"]+0.2, X.loc[line,"y5"], text[2:], horizontalalignment='left', 
                    size='medium', color='red', weight='semibold')
        else:
            p1.text(X.loc[line,"x5"]+0.2, X.loc[line,"y5"], text, horizontalalignment='left', 
                    size='medium', color='black', weight='semibold') 


# In[9]:




