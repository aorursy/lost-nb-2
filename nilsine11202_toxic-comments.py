#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[2]:


get_ipython().system(' pip uninstall tensorflow -y')


# In[ ]:





# In[3]:


get_ipython().system(' pip install ')


# In[4]:


import tensorflow as tf
tf.__version__


# In[5]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[6]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=f'{path}sdasdaa/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'


# In[7]:


embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


# In[8]:


# 데이터 호출

train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

train[train["comment_text"].isna()]


# In[9]:


# 문장 데이터의 결측치를 "_na_"로 채워넣은 문장 데이터를 list_sentences_train에 저장

list_sentences_train = train["comment_text"].fillna("_na_").values


# In[10]:


# y값 설정 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


# In[11]:


# max_feature(20000) 개수의 유니크한 단어를 갖는 vocab을 만든다
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))


# In[12]:


# tokenizer 인스턴스에 만들어진 vocab에 맞추어 text_to_sequences 메서드를 통해 각 문장의 단어를 vocab의 인덱스로 mapping

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)


# In[13]:


# 그 결과는 다음과 같음
list_tokenized_train[:1]

# 테스트도 똑같이 해주고
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[14]:


# 각 문장마다 token의 갯수가 다르기때문에, 균일한 column의 matrix 형태로 만들어주기 위해 padding을 해줌
# 이때, 가장 긴 길이를 가지는 sequence의 길이(maxlen)에 맞추어 줌.
X_t = pad_sequences(list_tokenized_train, maxlen = maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen = maxlen)


# In[15]:


# 기학습한 GloVe 모델의 embedding 값을 가져오기
# get_coefs 함수를 정의해 단어마다 저장된 계수 가져오기
# dict로 "단어": 임베딩 벡터 형태로 저장

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# In[16]:


embeddings_index


# In[17]:


# np.stack 함수를 통해 각 임베딩 인덱스의 값을 차곡차곡 쌓아올림
# np.stack 함수의 axis = -1으로 놓으면 아웃풋의 첫번째 차원이 인풋의 마지막 차원을 따른다고 함
all_embs = np.stack(embeddings_index.values())


# In[18]:


emb_mean, emb_std = all_embs.mean(), all_embs.std()


# In[19]:


a = np.array([1, 2, 3,1,1])
b = np.array([2, 3, 4 ,5 ,6])
np.stack((a, b), axis=-1)


# In[20]:


emb_mean, emb_std


# In[21]:


# 토크나이저의 vocab에 저장된 단어의 index를 word_index에 저장
word_index = tokenizer.word_index


# In[22]:


word_index


# In[23]:


# 단어 vocab의 길이와 내가 지정한 최대 단어수의 갯수 중 더 짧은 것을 nb_words로 취함
nb_words = min(max_features, len(word_index))

# emb_mean, emb_std를 따르고, nb_words X embed_size 사이즈의 정규분포 matrix를 embedding_matrix로 생성
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))


# In[24]:


embedding_matrix


# In[25]:


word_index.items()


# In[26]:


for word, i in word_index.items():
    # vocab(word_index.items()) 딕셔너리에서, max_features(20000)을 넘어가는 단어,
    # 즉 빈도수 기준 20000위 이상의 단어의 경우 그냥 패스. 
    if i>= max_features: continue
    
    # 그렇지 않을 경우, .get(word)로 embedding_vector에 저장.
    embedding_vector = embeddings_index.get(word)
    
    # 값이 있을 경우, embedding_matrix에 하나씩 차곡차곡 쌓아줌
    # 어느정도 임의로 생성한 20000*50의 임베딩 매트릭스를, GloVe 임베딩 메트릭스의 값으로 갱신해주는 것
    
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[27]:


embedding_matrix


# In[28]:


embedding_matrix.shape

# inp의 shape는 2차원일 경우 (column,)
# embedding layer에는 아까 만들어 놓은 GloVe기반 웨이트를 씌워줌, 앞의 두 arg로 임베딩 메트릭스의 차원을 지정해줌
# BiLSTM - CNN으로 샇아주고, dropout 옵션
# binary 분류로 레이어 마무리하고, fit
# 끝


# In[29]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size,trainable=True, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50,activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[30]:


model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1)


# In[ ]:





# In[ ]:





# In[31]:


import pandas as pd
import os
import numpy as np
import time
import tensorflow.keras.initializers
import statistics
import tensorflow.keras
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import LeakyReLU,PReLU
from tensorflow.keras.optimizers import Adam


# In[32]:


def generate_model(dropout, neuronPct, neuronShrink):
    # We start with some percent of 5000 starting neurons on the first hidden layer.
    neuronCount = int(neuronPct * 5000)
    
    # Construct neural network
    # kernel_initializer = tensorflow.keras.initializers.he_uniform(seed=None)
    model = Sequential()

    # So long as there would have been at least 25 neurons and fewer than 10
    # layers, create a new layer.
    layer = 0
    while neuronCount>25 and layer<5:
        # The first (0th) layer needs an input input_dim(neuronCount)
        if layer==0:
            model.add(Dense(neuronCount, 
                input_dim=X_t.shape[1], 
                activation=PReLU()))
        else:
            model.add(Dense(neuronCount, activation=PReLU())) 
        layer += 1

        # Add dropout after each hidden layer
        model.add(Dropout(dropout))

        # Shrink neuron count for each layer
        neuronCount = neuronCount * neuronShrink

    model.add(Dense(1,activation='sigmoid')) # Output
    return model


# In[33]:


aa =[np.where(r.sum()>=1 , 1, 0) for r in y]
index_aa=np.where([np.where(a==1,1,0) for a in aa])


# In[34]:


to_modify_np = np.zeros(159571)
indexes = index_aa[0]
replacements = np.ones(16225)

for (index, replacement) in zip(indexes, replacements):
    to_modify_np[index] = replacement


# In[35]:


to_modify=pd.Series(to_modify_np).astype("object")


# In[36]:


start_time = time.time()

# Split train and test
x_train = X_t[:1000]
y_train = to_modify[:1000]
x_test = X_t[1000:]
y_test = to_modify[1000:]


# In[37]:


dropout=0.2
lr=1e-6
neuronPct=0.2
neuronShrink=0.2


# In[38]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model = generate_model(dropout, neuronPct, neuronShrink)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
patience=100, verbose=0, mode='auto', restore_best_weights=True)


# In[39]:


import tensorflow as tf
tf.__version__


# In[40]:


import keras
keras.__version__


# In[41]:


type(x_train)


# In[42]:


x_train.shape


# In[43]:


type(pd.DataFrame(x_train))


# In[44]:


x_train


# In[45]:


type(x_train)


# In[46]:


# Train on the bootstrap sample
model.fit(x_train,y_train,validation_data=(x_test,npy_test)),callbacks=[monitor],verbose=0,epochs=1)


# In[47]:


epochs = monitor.stopped_epoch
epochs_needed.append(epochs)


# In[48]:


pred = model.predict(x_test)

y_test1=np.array(pd.get_dummies(pd.Series(y_test).astype("object")))

#         pred=pd.Series(pred.flatten()).astype("int")

y_test=y_test.astype("int")

flatten=pd.Series(pred.flatten())

aa=pd.DataFrame(flatten)

aa['1']=1-flatten

#         pred=list(pred)
#         y_test=list(y_test)

# Measure this bootstrap's log loss
#         y_compare = np.argmax(y_test,axis=0) # For log loss calculation
score = metrics.log_loss(y_test1, np.array(aa))
mean_benchmark.append(score)
m1 = statistics.mean(mean_benchmark)
m2 = statistics.mean(epochs_needed)
mdev = statistics.pstdev(mean_benchmark)

# Record this iteration
time_took = time.time() - start_time
#print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")


# In[49]:


def evaluate_network(dropout,lr,neuronPct,neuronShrink):
    SPLITS = 2

    # Bootstrap
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.3)

    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0
    

    # Loop through samples
    for train, test in boot.split(X_t,to_modify):
        start_time = time.time()
        num+=1

        # Split train and test
        x_train = X_t[train]
        y_train = to_modify[train]
        x_test = X_t[test]
        y_test = to_modify[test]
        
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        
        model = generate_model(dropout, neuronPct, neuronShrink)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=100, verbose=0, mode='auto', restore_best_weights=True)

        # Train on the bootstrap sample
        model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1)
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)


        pred = model.predict(x_test)

        y_test1=np.array(pd.get_dummies(pd.Series(y_test).astype("object")))

#         pred=pd.Series(pred.flatten()).astype("int")

        y_test=y_test.astype("int")

        flatten=pd.Series(pred.flatten())

        aa=pd.DataFrame(flatten)

        aa['1']=1-flatten
        
#         pred=list(pred)
#         y_test=list(y_test)
        
        # Measure this bootstrap's log loss
#         y_compare = np.argmax(y_test,axis=0) # For log loss calculation
        score = metrics.log_loss(y_test1, np.array(aa))
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        m2 = statistics.mean(epochs_needed)
        mdev = statistics.pstdev(mean_benchmark)

        # Record this iteration
        time_took = time.time() - start_time
        #print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")

    tensorflow.keras.backend.clear_session()
    return (-m1)

print(evaluate_network(
    dropout=0.2,
    lr=1e-6,
    neuronPct=0.2,
    neuronShrink=0.2))


# In[50]:


from bayes_opt import BayesianOptimization
import time


# In[51]:


# Supress NaN warnings, see: https://stackoverflow.com/questions/34955158/what-might-be-the-cause-of-invalid-value-encountered-in-less-equal-in-numpy
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

# Bounded region of parameter space
pbounds = {'dropout': (0.0, 0.499),
           'lr': (0.0, 0.1),
           'neuronPct': (0.01, 1),
           'neuronShrink': (0.01, 1)
          }

optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

start_time = time.time()
optimizer.maximize(init_points=10, n_iter=100,)
time_took = time.time() - start_time

print(f"Total runtime: {hms_string(time_took)}")
print(optimizer.max)

