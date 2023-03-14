#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
import gc
import os
import operator
import keras
from keras import backend as K
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import CuDNNGRU, CuDNNLSTM, Dense, Embedding, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout, Conv1D, SpatialDropout1D
from keras.optimizers import Adam
print(os.listdir("../input"))


# In[2]:


train_df = pd.read_csv('../input/train.csv')
print(train_df.shape)


# In[3]:


train_df.head()


# In[4]:


def load_emb(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding="latin"))
    
    return embedding_index


# In[5]:


glove = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"


# In[6]:


print('Loading Glove Embeddings')
embed_glove = load_emb(glove)


# In[7]:


def build_vocab(sentences, verbose=True):
    
    vocab = {}
    
    for sentence in tqdm(sentences):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
                
    return vocab    


# In[8]:


sentences = train_df["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)


# In[9]:


print({k: vocab[k] for k in list(vocab)[234:245]})


# In[10]:


print(sentences[:2])


# In[11]:


def check_coverage(vocab, embedding_index):
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words = 0
    
    for word in tqdm(vocab):
        try:
            known_words[word] = embedding_index[word]
            num_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            num_unknown_words += vocab[word]
            pass
    print("Found embeddings for {:.2%} of the Vocab".format(len(known_words)/len(vocab)))
    print("Found embeddings for {:.2%} of all text".format(num_known_words/(num_known_words+num_unknown_words)))
        
    sorted_x = sorted(unknown_words.items(), key = operator.itemgetter(1))[::-1]
        
    return sorted_x


# In[12]:


oov = check_coverage(vocab, embed_glove)


# In[13]:


oov[:20]


# In[14]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


# In[15]:


def known_contractions(embedding_index):
    known= []
    for contraction in contraction_mapping:
        if contraction in embedding_index:
            known.append(contraction)
    return known


# In[16]:


print("Known contractions in Glove embedding:")
print(known_contractions(embed_glove))


# In[17]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[18]:


train_df['treated_question'] = train_df['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))


# In[19]:


sentences = train_df["treated_question"].progress_apply(lambda x: x.split()).values


# In[20]:


vocab = build_vocab(sentences)


# In[21]:


oov = check_coverage(vocab, embed_glove)


# In[22]:


oov[:20]


# In[23]:


punc = "/-'?!.#$%\'()*+-/:;<=>,@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# In[24]:


def unknown_punct(embed, punct):
    unknown = ''
    count = 0
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


# In[25]:


print("Unknown punctuations in Glove:")
print(unknown_punct(embed_glove, punc))


# In[26]:


'rupee' in embed_glove


# In[27]:


def clean_text(text):
    text = str(text)
    
    for p in punc:
        text = text.replace(p, f' {p} ')
    for pun in "₹":
        text = text.replace(pun, "rupee")
            
    return text    


# In[28]:


train_df["cleaned_question"] = train_df["treated_question"].progress_apply(lambda x: clean_text(x))


# In[29]:


train_df.head()


# In[30]:


sentences = train_df["cleaned_question"].progress_apply(lambda x: x.split())


# In[31]:


vocab = build_vocab(sentences)


# In[32]:


oov = check_coverage(vocab, embed_glove)


# In[33]:


oov[:20]


# In[34]:


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}


# In[35]:


def correct_spelling(text, dic):
    for m in dic.keys():
        if m in text:
            text = text.replace(m, dic[m])
            
    return text


# In[36]:


train_df["cleaned_question"] = train_df["cleaned_question"].progress_apply(lambda x: correct_spelling(x, mispell_dict))


# In[37]:


sentences = train_df["cleaned_question"].progress_apply(lambda x: x.split())


# In[38]:


vocab = build_vocab(sentences)


# In[39]:


oov = check_coverage(vocab, embed_glove)


# In[40]:


oov[:20]


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


train_x, val_x = train_test_split(train_df[["cleaned_question", "target"]], test_size = 0.2, random_state=2019)


# In[43]:


train_x.head()


# In[44]:


embed_size = 300 #dimension of word embedding
vocab_len = 70000 #length of vocabulary
max_len = 100 #maximum number of words in a sentence

train_X = train_x["cleaned_question"].values
train_Y = train_x["target"].values
val_X = val_x["cleaned_question"].values
val_Y = val_x["target"].values


# In[45]:


train_X[:2]


# In[46]:


tokenizer = Tokenizer(num_words=vocab_len)
tokenizer.fit_on_texts(list(train_X))
train_sentences = tokenizer.texts_to_sequences(list(train_X))
train_sentences = pad_sequences(train_sentences, maxlen=max_len)


# In[47]:


train_sentences[:2]


# In[48]:


val_sentences = tokenizer.texts_to_sequences(val_X)
val_sentences = pad_sequences(val_sentences, maxlen=max_len)


# In[49]:


word_index = tokenizer.word_index


# In[50]:


val_sentences[:2]


# In[51]:


del train_df, sentences, vocab, oov, train_x, train_X, val_x, val_X
gc.collect()


# In[52]:


def make_embed_matrix(embedding_index, word_index, len_voc):
    all_emb = np.stack(embedding_index.values())
    mean_emb = all_emb.mean()
    std_emb = all_emb.std()
    embed_sz = all_emb.shape[1]
    word_index = word_index
    embedding_matrix = np.random.normal(mean_emb, std_emb, (len_voc, embed_sz))
    
    for word, i in word_index.items():
        if i>= len_voc:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector 
        return embedding_matrix


# In[53]:


embed_matrix = make_embed_matrix(embed_glove, word_index, vocab_len)


# In[54]:


del embed_glove
gc.collect()


# In[55]:


embed_matrix.shape


# In[56]:


inp = Input(shape=(max_len, ))
x = Embedding(vocab_len, embed_size, weights=[embed_matrix], trainable=False)(inp)
x = SpatialDropout1D(0.125)(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Conv1D(64, kernel_size=1, activation="relu")(x)
y = GlobalMaxPooling1D()(x)
z = GlobalAveragePooling1D()(x)
x = concatenate([y, z])
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss = 'binary_crossentropy', optimizer="adam", metrics = ["accuracy"])


# In[57]:


model.summary()


# In[58]:


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) *                 np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) *                 np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


# In[59]:


clr =  CyclicLR(base_lr=0.0005,
                max_lr=0.005,
                step_size = 300,
                mode="exp_range",
               gamma = 0.99994)


# In[60]:


model.fit(train_sentences, train_Y, batch_size=1024, epochs=5, validation_data=(val_sentences, val_Y), callbacks = [clr])


# In[61]:


pred_val_glove = model.predict([val_sentences], batch_size=512, verbose=1)


# In[62]:


from sklearn.metrics import f1_score

def tweak_threshold(pred, truth):
    thresholds = []
    scores = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        thresholds.append(thresh)
        score = f1_score(truth, (pred>thresh).astype(int))
        scores.append(score)
    return np.max(scores), thresholds[np.argmax(scores)]


# In[63]:


score_val, threshold_val = tweak_threshold(pred_val_glove, val_Y)

print(f"Scored {round(score_val, 4)} for threshold {threshold_val} on glove embedding on validation data")


# In[64]:


test_df = pd.read_csv('../input/test.csv')
print(test_df.shape)


# In[65]:


test_df.head()


# In[66]:


test_df['treated_question'] = test_df['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))


# In[67]:


test_df["cleaned_question"] = test_df["treated_question"].progress_apply(lambda x: clean_text(x))


# In[68]:


test_df["cleaned_question"] = test_df["cleaned_question"].progress_apply(lambda x: correct_spelling(x, mispell_dict))


# In[69]:


test_x = test_df["cleaned_question"].values


# In[70]:


test_x[:2]


# In[71]:


test_X = tokenizer.texts_to_sequences(list(test_x))
test_X = pad_sequences(test_X, maxlen=max_len)


# In[72]:


test_X[:2]


# In[73]:


del test_x
gc.collect()


# In[74]:


pred_test_y_glove = model.predict([test_X], batch_size=512, verbose=1)


# In[75]:


del embed_matrix, model, inp, x
gc.collect()


# In[76]:


paragram = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"


# In[77]:


embed_paragram = load_emb(paragram)


# In[78]:


embed_matrix = make_embed_matrix(embed_paragram, word_index, vocab_len)


# In[79]:


inp = Input(shape=(max_len, ))
x = Embedding(vocab_len, embed_size, weights=[embed_matrix], trainable=False)(inp)
x = SpatialDropout1D(0.125)(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = Conv1D(64, kernel_size=1, activation="relu")(x)
y = GlobalMaxPooling1D()(x)
z = GlobalAveragePooling1D()(x)
x = concatenate([y, z])
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss = 'binary_crossentropy', optimizer="adam", metrics = ["accuracy"])


# In[80]:


model.fit(train_sentences, train_Y, batch_size=512, epochs=5, validation_data=(val_sentences, val_Y), callbacks=[clr])


# In[81]:


pred_val_paragram = model.predict([val_sentences], batch_size=512, verbose=1)


# In[82]:


score_val, threshold_val = tweak_threshold(pred_val_paragram, val_Y)

print(f"Scored {round(score_val, 4)} for threshold {threshold_val} on paragram embedding on validation data")


# In[83]:


pred_val_y = 0.5*pred_val_glove + 0.5*pred_val_paragram

score_val, threshold_val = tweak_threshold(pred_val_y, val_Y)

print(f"Scored {round(score_val, 4)} for threshold {threshold_val} on glove and paragram embedding on validation data")


# In[84]:


pred_test_y_paragram = model.predict([test_X], batch_size=512, verbose=1)


# In[85]:


pred_test_y = 0.5*pred_test_y_glove + 0.5*pred_test_y_paragram
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

