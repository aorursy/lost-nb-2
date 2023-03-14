#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
print(os.listdir("../input"))


# In[2]:


# 读取需要处理的数据
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)


# In[3]:


train_df.head()


# In[4]:


train_df.target.value_counts()


# In[5]:


train_df.target.value_counts() / train_df.target.count()


# In[6]:


# 查看虚假问题的前5例是什么类型
train_df.loc[train_df['target']==1, 'question_text'][:5].tolist()


# In[7]:


# 查看问题中是否存在缺失值
train_df["question_text"].isnull().sum().sum()


# In[8]:


test_df["question_text"].isnull().sum().sum()


# In[9]:


# 如果存在缺失值，需要对缺失值进行处理
train_df['question_text'].fillna("__na__", inplace=True)
test_df['question_text'].fillna("__na__", inplace=True)


# In[10]:


# 导入英文数据清洗时常用的库
import re
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
# 对单词的形式进行转换
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


# In[11]:


# 添加需要去除的标点符号集
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', 
          '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', 
          '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', 
          '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', 
          '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', 
          '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', 
          '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

# 定义一些常见的缩写
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                        (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), 
                        (r'(\w+)n\'t', '\g<1> not'),(r'(\w+)\'ve', '\g<1> have'), 
                        (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), 
                        (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'),
                        (r'dont', 'do not'), (r'wont', 'will not') ]


# In[12]:


# 定义清洗文本的函数
def clean_text(text):
    # 去除对分类没什么作用的数字
    text = re.sub('[0-9]+', '', text)
    # 对重复出现的标点进行替换
    text = re.sub(r'(\!)\1+', 'multiExclamation', text)
    text = re.sub(r'(\?)\1+', 'multiQuestion', text)
    text = re.sub(r'(\.)\1+', 'multiStop', text)
    
    # 在标点前后加空格
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    # 对缩写进行替换
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    
    # 对文本段进行分词操作
    text_split = tokenize.word_tokenize(text)
    text = [word for word in text_split if word not in stoplist]
    text = [wnl.lemmatize(word) for word in text]
    
    return " ".join(text)


# In[13]:


train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text(x))


# In[14]:


# 添加计算处理之后一段话长度的列
train_df['text_len'] = train_df['question_text'].apply(lambda x: len(x.split()))
test_df['text_len'] = test_df['question_text'].apply(lambda x: len(x.split()))


# In[15]:


train_df['text_len'].describe()


# In[16]:


train_df['text_len'].value_counts()


# In[17]:


test_df['text_len'].value_counts()


# In[18]:


# 将训练集数据划分为训练集和验证集
from sklearn.model_selection import train_test_split
# 保留5%的数据集作为验证集
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=2018)


# In[19]:


train_X = train_df['question_text'].values
val_X = val_df['question_text'].values
test_X = test_df['question_text'].values


# In[20]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[21]:


# 设置一些参数
embed_size = 300 # 设置词向量的长度
max_features = 95000  # 设置保留频度为前95000个的单词
maxlen = 70     # 设置问题保留的最大单词长度（超过70个的只取前70）


# In[22]:


train_X.shape, val_X.shape, test_X.shape


# In[23]:


# 设置词典的长度，保留出现频度在前max_features个的单词
tokenizer = Tokenizer(num_words = max_features)
# 将训练集，验证集和测试集的句子结合
text = np.concatenate((np.concatenate((train_X,val_X)),test_X))
# 基于已有的单词建立词典
tokenizer.fit_on_texts(list(text))


# In[24]:


# 基于建立的词典将句子中的每个单词序列化为在词典中对应的下标
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)


# In[25]:


# 对上面得到的二维列表进行对齐，将长度大于70的在后面截断，对长度小于70的在后面补0
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)


# In[26]:


# 获取target的值
train_y = train_df['target'].values
val_y = val_df['target'].values

# 打乱数据
np.random.seed(2018)
train_idx = np.random.permutation(len(train_y))
val_idx = np.random.permutation(len(val_y))

# 获取打乱之后的训练集特征，训练集目标以及验证集特征，验证集目标
train_X = train_X[train_idx]
val_X = val_X[val_idx]
train_y = train_y[train_idx]
val_y = val_y[val_idx]


# In[27]:


print(os.listdir("../input/embeddings"))


# In[28]:


# 对embedding文件进行处理的函数
# 对于文件中的每一行，通过空格分割
# 返回的第一个元素的单词，第二个元素是对应的词向量
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# In[29]:


# 加载glove模型的词向量
def load_glove(word_index=None):
    EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
        
    # 将所有的词向量进行堆叠
    all_embs = np.stack(embeddings_index.values())
    # 设置随机生成词向量时的均值和方差
    emb_mean, emb_std = -0.005838499, 0.48782197
    # 获取词向量的长度
    embed_size = all_embs.shape[1]
    
    # 获取之前词典中单词的总数量
    nb_words = min(max_features, len(word_index))
    # 随机生成词向量矩阵
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    # 对于字典中存在的单词，将对应的词向量替换为已有的词向量
    for word, i in word_index.items():
        if i>= max_features: 
            continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix       


# In[30]:


# 加载para模型的词向量
def load_para(word_index=None):
    EMBEDDING_FILE = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,
                                                                  encoding='utf8',
                                                                  errors='ignore')
                           if len(o)>100)
        
    # 将所有的词向量进行堆叠
    all_embs = np.stack(embeddings_index.values())
    # 设置随机生成词向量时的均值和方差
    emb_mean, emb_std = -0.0053247833, 0.49346462
    # 获取词向量的长度
    embed_size = all_embs.shape[1]
    
    # 获取之前词典中单词的总数量
    nb_words = min(max_features, len(word_index))
    # 随机生成词向量矩阵
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    # 对于字典中存在的单词，将对应的词向量替换为已有的词向量
    for word, i in word_index.items():
        if i>= max_features: 
            continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix  


# In[31]:


# 获取之前生成字典的键值对
# 键对应单词，值对应索引
word_index = tokenizer.word_index
embedding_matrix_1 = load_glove(word_index)
embedding_matrix_3 = load_para(word_index)
# 将由两个不同的语料库得到的词向量平均
embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis=0)


# In[32]:


# 对删除的垃圾进行回收
import gc
del embedding_matrix_1, embedding_matrix_3
gc.collect()


# In[33]:


from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU
from keras.layers import Conv1D, Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPool2D, MaxPooling1D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *


# In[34]:


from sklearn.metrics import f1_score

# 定义训练预测函数
def train_pred(model, epochs=2):
    # 对于每一个epoch
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, 
                  validation_data=(val_X, val_y))
        # 对于验证集结果进行预测
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        search_result = threshold_search(val_y, pred_val_y)
        print(search_result)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

# 寻找预测概率的最佳分割阈值
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i*0.001 for i in range(250, 450)]:
        # 计算每一个阈值的f1得分
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


# In[35]:


# 定义CNN模型
def model_cnn(embedding_matrix):
    # 定义滤波器大小的搜索值
    filter_sizes = [1,2,3,5]
    num_filters = 36
    
    inp = Input(shape=(maxlen,))
    # Embedding的第一个参数是词汇量的大小，第二个参数是词向量的大小
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    # 将输入转换成CNN2D需要的形式，长、宽以及通道数
    x = Reshape((maxlen,embed_size, 1))(x)
    
    maxpool_pool = []
    # 使用不同大小的滤波器
    for i in range(len(filter_sizes)):
        # 卷积层
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                     kernel_initializer='he_normal', activation='elu')(x)
        # 池化层，将不同滤波器得到的卷积层结果转化为相同大小
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen-filter_sizes[i]+1, 1))(conv))
    
    # 将不同卷积核得到的结果进行连接
    z = Concatenate(axis=1)(maxpool_pool)
    # 展开
    z = Flatten()(z)
    z = Dropout(0.1)(z)
    
    # 全连接
    outp = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[36]:


# 由于使用的标准是准确率而不是f1度量，所以得到的结果不是很好
pred_val_y1, pred_test_y1 = train_pred(model_cnn(embedding_matrix), epochs=2)


# In[37]:


# 定义f1度量
def f1(y_true, y_pred):
    # 先计算召回率
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    # 再计算准确率
    def precision(y_true, y_pred):
        # 真正例
        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        # 预测正例数
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        # 计算准确率
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[38]:


# 定义LSTM模型
def model_lstm(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=False))(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model


# In[39]:


# 计算得到LSTM预测的结果
pred_val_y2, pred_test_y2 = train_pred(model_lstm(embedding_matrix), epochs=4)


# In[40]:


# 定义CNN和LSTM结合的网络
def model_cnn_lstm(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    # 卷积
    x = Conv1D(64, 3, activation='relu')(x)
    # 池化
    x = Bidirectional(CuDNNLSTM(50, return_sequences=False))(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    
    return model


# In[41]:


# 结合CNN和LSTM得到的预测结果
pred_val_y3, pred_test_y3 = train_pred(model_cnn_lstm(embedding_matrix), epochs=4)


# In[42]:


class Attention(Layer):
    def __init__(self, step_dim,
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
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
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


# In[43]:


def model_lstm_atten(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x)
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    
    return model


# In[44]:


# 将lstm和attention结合
pred_val_y4, pred_test_y4 = train_pred(model_lstm_atten(embedding_matrix), epochs=5)


# In[45]:


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[46]:



import tensorflow as tf
from keras.layers import BatchNormalization
def model_capsule(embedding_matrix):
    K.clear_session()
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True, 
                               kernel_initializer=initializers.glorot_normal(seed=12300),
                               recurrent_initializer=initializers.orthogonal(gain=1.0, seed=10000)))(x)

    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x = Flatten()(x)

    x = Dense(100, activation="relu", kernel_initializer=initializers.glorot_normal(seed=12300))(x)
    x = Dropout(0.12)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(),)
    return model


# In[47]:


pred_val_y5, pred_test_y5 = train_pred(model_capsule(embedding_matrix), epochs=5)


# In[48]:



class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
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
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

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

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


# In[49]:



clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300., mode='exp_range',
               gamma=0.99994)
def train_pred2(model, epochs=2):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks=[clr])
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        search_result = threshold_search(val_y, pred_val_y)
        print(search_result)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y


# In[50]:



def model_3gru_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    
    return model


# In[51]:


pred_val_y6, pred_test_y6 = train_pred2(model_3gru_atten(embedding_matrix), epochs=3)


# In[52]:


pred_val_y7, pred_test_y7 = train_pred2(model_lstm_atten(embedding_matrix), epochs=4)


# In[53]:


# 这里进行简单加权融合
pred_val_y = 0.09*pred_val_y1 + 0.15*pred_val_y2 + 0.08*pred_val_y3 + 0.22*pred_val_y4 +             0.18*pred_val_y5 + 0.16*pred_val_y7 + 0.12*pred_val_y7
search_result = threshold_search(val_y, pred_val_y)
print(search_result)


# In[54]:


pred_test_y = 0.09*pred_test_y1 + 0.15*pred_test_y2 + 0.08*pred_test_y3 + 0.22*pred_test_y4 +             0.18*pred_test_y5 + 0.16*pred_test_y6 + 0.12*pred_test_y7
pred_test_y = (pred_test_y > search_result['threshold']).astype(int)
sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = pred_test_y
sub.to_csv("submission.csv", index=False)


# In[55]:




