#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import string

DATA_FOLDER = "../input"
SEED = 9

def read_data(train_data_perc=0.8):
    train_data_file = DATA_FOLDER + "/" + "train.csv"

    all_data = pd.read_csv(train_data_file)
    X = all_data[["qid", "question_text"]]
    y = all_data[["target"]]
    
    X["num_words"] = X["question_text"].apply(lambda x: len(str(x).split()))
    X["num_unique_words"] = X["question_text"].apply(lambda x: len(set(str(x).split())))
    X["num_chars"] = X["question_text"].apply(lambda x: len(str(x)))
    X["num_punctuations"] = X["num_unique_words"].apply(lambda x: len([c for c in str(x)
                                                                       if c in string.punctuation]) )    

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=1 - train_data_perc,
                                                        random_state=SEED)

    return X_train, y_train, X_test, y_test

def read_test_data():
    test_data_file = DATA_FOLDER + "/" + "test.csv"

    all_data = pd.read_csv(test_data_file)
    X = all_data[["qid", "question_text"]]

    X["num_words"] = X["question_text"].apply(lambda x: math.log(len(str(x).split())))
    X["num_chars"] = X["question_text"].apply(lambda x: math.log(len(str(x))))

    return X


# In[2]:


X_train, y_train, X_test, y_test = read_data(train_data_perc=0.6)

print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

number_of_ones = (y_train["target"] == 1).sum()
number_of_zeros = len(y_train) - number_of_ones

print("Number of zeros in train set (sincere questions)", number_of_zeros)
print("Number of ones in train set (insincere questions)", number_of_ones)
"""
insincere_stats = {"num_words": 0, 
                  "num_unique_words": 0,
                  "num_chars": 0,
                  "num_punctuations": 0}
sincere_stats = {"num_words": 0, 
                  "num_unique_words": 0,
                  "num_chars": 0,
                  "num_punctuations": 0}
for i, X_val in X_train.iterrows():
    if y_train["target"][i] == 0:
        sincere_stats["num_words"] += X_val["num_words"]
        sincere_stats["num_unique_words"] += X_val["num_unique_words"] 
        sincere_stats["num_chars"] += X_val["num_chars"]
        sincere_stats["num_punctuations"] += X_val["num_punctuations"]   
    else:
        insincere_stats["num_words"] += X_val["num_words"]
        insincere_stats["num_unique_words"] += X_val["num_unique_words"] 
        insincere_stats["num_chars"] += X_val["num_chars"]
        insincere_stats["num_punctuations"] += X_val["num_punctuations"]           

# calculate averages instead of sums
for key, value in sincere_stats.items():
    sincere_stats[key] = sincere_stats[key] / number_of_zeros
    
for key, value in insincere_stats.items():
    insincere_stats[key] = insincere_stats[key] / number_of_ones
    
print("insincere_stats", insincere_stats)
print("sincere_stats", sincere_stats)
"""


# In[3]:


"""
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


print(X_train["question_text"].head())

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_non_ascii(word):
    # Remove non-ASCII characters from list of tokenized words
    new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_word

def to_lowercase(word):
    # Convert all characters to lowercase from list of tokenized words
    return word.lower()

def remove_punctuation(word):
    # Remove punctuation from list of tokenized words
    new_word = re.sub(r'[^\w\s]', '', word)
    return new_word

def remove_stopwords(word):
    # Remove stop words from list of tokenized words
    if word not in stopwords.words('english'):
        return word
    return ''

def stem_words(word):
    # Stem words in list of tokenized words
    return stemmer.stem(word)

def lemmatize_verbs(word):
    # Lemmatize verbs in list of tokenized words
    return lemmatizer.lemmatize(word, pos='v')

def normalize(word):
    word = remove_non_ascii(word)
    word = to_lowercase(word)
    word = remove_punctuation(word)
    # word = remove_stopwords(word)
    #word = lemmatize_verbs(word)
    return word

def get_processed_text(string):
    words = nltk.word_tokenize(string)
    new_words = []
    for word in words:
        new_word = normalize(word)
        if new_word != '':
            new_words.append(new_word)
    return ' '.join(new_words)

X_train["question_text"] = X_train["question_text"].apply(lambda x: get_processed_text(x))
print(X_train["question_text"].head())
X_test["question_text"] = X_test["question_text"].apply(lambda x: get_processed_text(x))
"""

vectorizer = CountVectorizer(ngram_range=(1, 1),
                               min_df=1)


X_final_test = read_test_data()

prepared_text = vectorizer.fit_transform(X_train["question_text"].values.tolist() +
                                          X_test["question_text"].values.tolist() +
                                        X_final_test["question_text"].values.tolist())
unique_words = set(vectorizer.get_feature_names())
print("Number of unique words: ", len(unique_words))


# In[4]:


def load_glove_with_vocabulary(vocabulary_map):
    gloveFile = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    f = open(gloveFile,'r')
    emb_list = [None] * len(vocabulary_map)
    print("Length of vocabulary: ", len(vocabulary_map))
    found_words = 0
    for i, line in enumerate(f):
        splitLine = line.split(" ")
        word = splitLine[0]
        if word in vocabulary_map:
            found_words += 1
            embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
            emb_list[vocabulary_map[word]] = embedding
    print("Loaded GloVe vectors for %i words. Generating random vectors for the rest." % found_words)
    full_emb_list = []
    glove_mean = -0.00584
    glove_std = 0.452
    for emb in emb_list:
        if emb is None:
            full_emb_list.append(np.random.normal(glove_mean, glove_std, (1, 300)))
        else:
            full_emb_list.append(emb)
    print("Done. Vectors loaded :", len(full_emb_list))
    embs = np.vstack(full_emb_list)
    return embs


import gc
gc.collect()
#unique_words = load_unique_words(X_train, X_test)
print("len(unique_words)", len(unique_words))
embs = load_glove_with_vocabulary(vectorizer.vocabulary_)
print("embs.shape", embs.shape)


# In[5]:


import torch
import torch.nn as nn

train_prepared_data = vectorizer.transform(X_train["question_text"])
test_prepared_data = vectorizer.transform(X_test["question_text"])
final_test_prepared_data = vectorizer.transform(X_final_test["question_text"])

print("Vectorized train and test data")


train_targets = torch.from_numpy(np.array(y_train["target"])).float()
print("Train targets", train_targets)
test_targets = torch.from_numpy(np.array(y_test["target"])).float()
print("Test targets", test_targets)

train_word_indices_list = []
train_offsets_list = []

offset = 0
for i in range(0, train_prepared_data.shape[0]):
    train_offsets_list.append(offset)
    row, col = train_prepared_data.getrow(i).nonzero()
    train_word_indices_list.append(torch.tensor(torch.from_numpy(col), dtype=torch.int64))
    offset += len(row)
    
train_words = torch.cat(train_word_indices_list)
train_offsets = torch.tensor(train_offsets_list, dtype=torch.int64)

print("Created words and offsets for train data")

test_word_indices_list = []
test_offsets_list = []

offset = 0
for i in range(0, test_prepared_data.shape[0]):
    test_offsets_list.append(offset)
    row, col = test_prepared_data.getrow(i).nonzero()
    test_word_indices_list.append(torch.tensor(torch.from_numpy(col), dtype=torch.int64))
    offset += len(row)
    
test_words = torch.cat(test_word_indices_list)
test_offsets = torch.tensor(test_offsets_list, dtype=torch.int64)


print("Created words and offsets for test data")

final_test_word_indices_list = []
final_test_offsets_list = []

offset = 0
for i in range(0, final_test_prepared_data.shape[0]):
    final_test_offsets_list.append(offset)
    row, col = final_test_prepared_data.getrow(i).nonzero()
    final_test_word_indices_list.append(torch.tensor(torch.from_numpy(col), dtype=torch.int64))
    offset += len(row)
    
final_test_words = torch.cat(final_test_word_indices_list)
final_test_offsets = torch.tensor(final_test_offsets_list, dtype=torch.int64)
print("final_test_words.shape", final_test_words.shape)
print("final_test_offsets.shape", final_test_offsets.shape)

print("Created words and offsets for final test data")


# In[6]:


class BagOfEmbeddings(nn.Module):
    def __init__(self, embedding_weights, hidden_dim=100, dropout=0.5, embedding_mode='mean'):
        super(BagOfEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        embedding_size = embedding_weights.shape[1]
        self.embedding = nn.EmbeddingBag(embedding_weights.shape[0], embedding_size, mode=embedding_mode)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float())
        self.embedding.weight.requires_grad = False
        self.final_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, words, offsets):
        x = self.embedding(words, offsets)
        return self.final_layer(x)

    
model = BagOfEmbeddings(embs, dropout=0.1, hidden_dim=75, embedding_mode='mean')
print("model", model)


# In[7]:


Run bag of embeddings model training


# In[8]:


loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
torch.manual_seed(SEED)

def get_batch(words, offsets, targets, start_index, size):
    first_word_index = offsets[start_index]
    offsets_end_index = start_index + size
    if offsets_end_index > offsets.shape[0]:
        offsets_end_index = offsets.shape[0]
        last_word_index = words.shape[0]
    else:
        last_word_index = offsets[offsets_end_index]
    if targets is not None:
        return words[first_word_index:last_word_index], offsets[start_index:offsets_end_index] - offsets[start_index], targets[start_index:offsets_end_index]
    else:
        return words[first_word_index:last_word_index], offsets[start_index:offsets_end_index] - offsets[start_index], None
    
def run_training(epochs, model, optimizer, loss_fn, 
                 all_words, all_offsets, 
                 all_targets, batch_size=32):
    
    print("Training samples: ", all_offsets.shape[0])
    batch_losses = []
    for e in range(epochs):
        model.train()
        start_index = 0
        batch_nr = 0
        print("Starting epoch %i" % (e + 1))
        while start_index < all_offsets.shape[0]:
            batch_nr += 1
            words, offsets, target = get_batch(all_words, all_offsets, all_targets, start_index, batch_size)            
            start_index += batch_size
            optimizer.zero_grad()
            output = model(words, offsets)
            loss = loss_fn(output.squeeze(), target)
            loss.backward()
            optimizer.step()            
            if batch_nr % 1000 == 0:
                batch_losses.append(loss.item())
                print("Epoch %i, batch: %i, loss: %.5f" % (e + 1, batch_nr, loss.item()))
    return batch_losses
            

losses = run_training(epochs=1, model=model, optimizer=optimizer, loss_fn=loss,
                     all_words=train_words, all_offsets=train_offsets, all_targets=train_targets,
                     batch_size=32)
print("Training avg model complete!")

                


# In[9]:


def run_test(model, loss_fn, 
                 all_words, all_offsets, 
                 all_targets, batch_size=256):
    
    print("Test samples: ", all_offsets.shape[0])
    batch_losses = []
    outputs = []
    model.eval()
    start_index = 0
    batch_nr = 0
    print("Starting testing")
    while start_index < all_offsets.shape[0]:
        batch_nr += 1
        words, offsets, target = get_batch(all_words, all_offsets, all_targets, start_index, batch_size)            
        start_index += batch_size
        output = model(words, offsets)
        outputs.append(torch.sigmoid(output))
        if loss_fn:
            loss = loss_fn(output.squeeze(), target)        
            if batch_nr % 100 == 0:
                batch_losses.append(loss.item())
                print("Batch: %i, loss: %.5f" % (batch_nr, loss.item()))
    return batch_losses, torch.cat(outputs)

print("Evaluating test set")
batch_losses, outputs = run_test(model=model, loss_fn=loss,
                     all_words=test_words, all_offsets=test_offsets, all_targets=train_targets,
                     batch_size=256)

print("outputs.shape", outputs.shape)

def calculate_best_threshold(y_test, predictions):
    best_threshold = -1
    best_score = -1
    for threshold in np.arange(0.01, 0.801, 0.01):
        threshold = np.round(threshold, 2)
        model_f1_score = f1_score(y_true=y_test["target"],
                                  y_pred=(predictions > threshold).astype(int))
        if model_f1_score > best_score:
            best_score = model_f1_score
            best_threshold = threshold
        print("F1 score at threshold %s: %s" % (threshold, model_f1_score))
    print("F1 score at best threshold %s: %s" % (best_threshold, best_score))
    return best_threshold, best_score

boe_pred = outputs.detach().numpy()
best_threshold_boe, best_score_boe = calculate_best_threshold(y_test[:400000], boe_pred[:400000])

print("boe_pred", boe_pred)
print("Evaluating final test outputs")
_, final_test_outputs = run_test(model=model, loss_fn=None,
                     all_words=final_test_words, all_offsets=final_test_offsets, all_targets=None,
                     batch_size=256)

boe_final_pred = final_test_outputs.detach().numpy()
print("boe_final_pred", boe_final_pred)
print("Final test outputs done")


# In[10]:


logistic_regression_model = LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1, random_state=SEED)
logistic_regression_model.fit(train_prepared_data, y_train)
logreg_predictions = logistic_regression_model.predict_proba(test_prepared_data)[:, 1]
best_threshold_logreg, best_score_logreg = calculate_best_threshold(y_test[:400000], logreg_predictions[:400000])

logreg_final_predictions = logistic_regression_model.predict_proba(final_test_prepared_data)[:, 1]


# In[11]:


#final_classifier = lgbm.sklearn.LGBMClassifier(learning_rate=0.05,
#                                           n_estimators=25,
#                                           n_jobs=-1,
 #                                          random_state=SEED)

final_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1, random_state=SEED)

X_final_augmented_train = pd.DataFrame({"logreg_pred": logreg_predictions[:400000], "boe_pred": boe_pred[:400000].squeeze(), 
                             "num_words": X_test["num_words"].values[:400000], "num_chars": X_test["num_chars"].values[:400000]})

y_final_augmented_train = y_test[:400000]

X_final_augmented_test = pd.DataFrame({"logreg_pred": logreg_predictions[400000:], "boe_pred": boe_pred[400000:].squeeze(), 
                             "num_words": X_test["num_words"].values[400000:], "num_chars": X_test["num_chars"].values[400000:]})
y_final_augmented_test = y_test[400000:]

final_classifier.fit(X_final_augmented_train, y_final_augmented_train)

final_predictions = final_classifier.predict_proba(X_final_augmented_test)[:,1]
best_threshold_final, best_score_final = calculate_best_threshold(y_final_augmented_test, final_predictions)


# In[12]:




threshold = best_threshold_final
print(X_final_test["qid"].head())
print(X_final_test["question_text"].head())
X_final_classifier_test = pd.DataFrame({"logreg_pred": logreg_final_predictions, "boe_pred": boe_final_pred.squeeze(), 
                             "num_words": X_final_test["num_words"].values, "num_chars": X_final_test["num_chars"].values})
print(X_final_classifier_test[:10])

predictions_proba = final_classifier.predict_proba(X_final_classifier_test)[:,1]
print("predictions_proba[:10]", predictions_proba[:10])
labels = (predictions_proba > threshold).astype(int)
print("labels[:10]", labels[:10])
preds = pd.DataFrame({
    "qid": X_final_test["qid"],
    "prediction": labels
})
preds.to_csv("submission.csv", index=False)

