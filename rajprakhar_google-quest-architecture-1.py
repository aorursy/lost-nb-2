#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Statements

import pandas as pd
import numpy as np

import tensorflow as tf
print(tf.__version__)

import re
from tqdm import tqdm

from scipy.stats import spearmanr

import warnings
warnings.simplefilter('ignore')


# In[2]:


PATH = '../input/google-quest-challenge/'
PATH_w2vec_300d = '../input/glove-300d/'

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('Train Shape =', df_train.shape)
print('Test Shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\nOutput Categories:\n\t', output_categories)
print('\nInput Categories:\n\t', input_categories)


# In[3]:


#Preprocessing Text Data

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]

#Utility Methods
def decontracted(phrase): # https://stackoverflow.com/a/47091490/4084039
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text

def perform_preprocessing(text_array):
    #Changing all characters to lower case
    lower_text_array = pd.Series(text_array).str.lower()
    #Calling utility method preprocess_text to perform some more preprocessing
    preprocessed_text_array = preprocess_text(lower_text_array)

    return pd.Series(preprocessed_text_array)


#Preprocessing Train Input Columns
df_train['Preproc_Question_Title'] = perform_preprocessing(df_train['question_title'].values)
df_train['Preproc_Question_Body'] = perform_preprocessing(df_train['question_body'].values)
df_train['Preproc_Answer'] = perform_preprocessing(df_train['answer'].values)
  
#Preprocessing Test Input Columns
df_test['Preproc_Question_Title'] = perform_preprocessing(df_test['question_title'].values)
df_test['Preproc_Question_Body'] = perform_preprocessing(df_test['question_body'].values)
df_test['Preproc_Answer'] = perform_preprocessing(df_test['answer'].values)

print("\n")
print("="*70 + "Question Title" + "="*70)
print("Before Preprocessing:\n", df_train['question_title'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Question_Title'][0])

print("="*70 + "Question Body" + "="*70)
print("Before Preprocessing:\n", df_train['question_body'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Question_Body'][0])

print("="*70 + "Answer" + "="*70)
print("Before Preprocessing:\n", df_train['answer'][0])
print("\nAfter Preprocessing:\n", df_train['Preproc_Answer'][0])


# In[4]:


#Prepare Embedding for 3 inputs together

def prepare_embedding(input_series_train, input_series_test, column_name):

    print("="*70 + column_name + "="*70)

    #Tokenization of text data to numbers
    tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_obj.fit_on_texts(input_series_train.values)

    word_index = tokenizer_obj.word_index
    print('Found %s unique tokens.' % len(word_index))

    #Encoded inputs
    train_sequences = tokenizer_obj.texts_to_sequences(input_series_train.values)
    test_sequences = tokenizer_obj.texts_to_sequences(input_series_test.values)
    print("Train Sequences Length", len(train_sequences))
    print("Test Sequences Length", len(test_sequences))

    #Selecting max_length of words in an essay
    MAX_SEQUENCE_LENGTH = int(np.percentile(pd.Series(train_sequences).apply(lambda x: len(x)), 98))
    print("Around 96 percentile of " + column_name + " have length of words less than ", MAX_SEQUENCE_LENGTH)

    #Padding of Word Sequences
    vocab_size = len(word_index)+1
    train_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print("Shape of padded train sequences: ", train_sequences_pad.shape)
    print("Shape of padded test sequences: ", test_sequences_pad.shape)

    #Preparing Embedding Layer using Glove vector (300 dimension)

    # Loading Glove embedding layer
    embeddings_index = {}
    f = open(PATH_w2vec_300d+'glove-840B-300d-char_embed.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    #*--*create a weight matrix for words in training docs*--*
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, train_sequences_pad, test_sequences_pad


# In[5]:


#Merge all the text columns into 1 as 'Total_Text'

df_train['Total_Text'] = df_train["Preproc_Question_Title"].map(str) + df_train["Preproc_Question_Body"].map(str) + df_train['Preproc_Answer'].map(str)
df_test['Total_Text'] = df_test["Preproc_Question_Title"].map(str) + df_test["Preproc_Question_Body"].map(str) + df_test['Preproc_Answer'].map(str)

#Calling prepare_embedding method to generate embedding for 3 inputs for both train and test
vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, train_sequences_pad_qt = prepare_embedding(df_train['Total_Text'], df_train['question_title'], 'Question Title Train')
vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, test_sequences_pad_qt = prepare_embedding(df_train['Total_Text'], df_test['question_title'], 'Question Title Test')

vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, train_sequences_pad_qb = prepare_embedding(df_train['Total_Text'], df_train['question_body'], 'Question Body Train')
vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, test_sequences_pad_qb = prepare_embedding(df_train['Total_Text'], df_test['question_body'], 'Question Body Test')

vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, train_sequences_pad_ans = prepare_embedding(df_train['Total_Text'], df_train['answer'], 'Answer Train')
vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH, _, test_sequences_pad_ans = prepare_embedding(df_train['Total_Text'], df_test['answer'], 'Answer Test')


# In[6]:


#Splitting into train and validation
validation_sequences_pad_qt = train_sequences_pad_qt[5000:]
train_sequences_pad_qt = train_sequences_pad_qt[:5000]

validation_sequences_pad_qb = train_sequences_pad_qb[5000:]
train_sequences_pad_qb = train_sequences_pad_qb[:5000]

validation_sequences_pad_ans = train_sequences_pad_ans[5000:]
train_sequences_pad_ans = train_sequences_pad_ans[:5000]


# In[7]:


#Creating Embedding Layers for Total_Text:

embedding_layer_total_text = tf.keras.layers.Embedding(vocab_size,
                                            300,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            name = 'Shared_Embedding_Layer',
                                            trainable=False)


# In[8]:


def create_model(): 

    #Path 1 
    input_question_title = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name = 'IP_Question_Title')
    embedded_question_title = embedding_layer_total_text(input_question_title)
    LSTM_layer_question_title = tf.keras.layers.LSTM(100, kernel_initializer='glorot_uniform', name = 'Question_Title_LSTM')(embedded_question_title) #32
    flatten = tf.keras.layers.Flatten(name='flatten_Question_Title')(LSTM_layer_question_title)

    #Path 2
    input_question_body = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name = 'IP_Question_Body')
    embedded_question_body = embedding_layer_total_text(input_question_body)
    LSTM_layer_question_body = tf.keras.layers.LSTM(100, kernel_initializer='glorot_uniform', name = 'Question_Body_LSTM')(embedded_question_body) #32
    flatten_1 = tf.keras.layers.Flatten(name='flatten_Question_Body')(LSTM_layer_question_body)

    #Path 3
    input_answer = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name = 'IP_Answer')
    embedded_answer = embedding_layer_total_text(input_answer)
    LSTM_layer_answer = tf.keras.layers.LSTM(100, kernel_initializer='glorot_uniform', name = 'Answer_LSTM')(embedded_answer) #32
    flatten_2 = tf.keras.layers.Flatten(name='flatten_Answer')(LSTM_layer_answer)

    #Concat
    concat = tf.keras.layers.concatenate([flatten, flatten_1, flatten_2], axis=1, name='concatenate')

    #Dense & Dropout - 1
    dense_1 = tf.keras.layers.Dense(64, activation='relu', name='Dense_1')(concat) #64 #tanh
    dropout_1 = tf.keras.layers.Dropout(0.5, name='Dropout_1')(dense_1) #Taking Dropout Rate = 0.5

    #Dense & Dropout - 2
    dense_2 = tf.keras.layers.Dense(32, activation='relu', name='Dense_2')(dropout_1) #64 #tanh
    dropout_2 = tf.keras.layers.Dropout(0.5, name='Dropout_2')(dense_2) #Taking Dropout Rate = 0.5

    # -----------------------------------------------------TRY--------------------------------------------------------------
    #Dense & Dropout - 3
    dense_3 = tf.keras.layers.Dense(64, activation='relu', name='Dense_3')(dropout_2) #64 #tanh
    dropout_3 = tf.keras.layers.Dropout(0.2, name='Dropout_3')(dense_3) #Taking Dropout Rate = 0.5
    # -----------------------------------------------------TRY--------------------------------------------------------------

    #Output Layer
    dense_4 = tf.keras.layers.Dense(4, activation='relu', name='Dense_4')(dropout_3) #64 #tanh
    preds = tf.keras.layers.Dense(30, activation='sigmoid', name='Output')(dense_4)

    model_created = tf.keras.models.Model(inputs = [input_question_title, input_question_body, input_answer], outputs = [preds], name='Model_Google_QUEST')

    return model_created

#Calling create_model method and printing summary of model
model_Google_QUEST = create_model()
print(model_Google_QUEST.summary())


# In[9]:


#Plotting Architecture_1 of Google QUEST:
tf.keras.utils.plot_model(model_Google_QUEST, to_file='Arch1_v2.png')


# In[10]:


#Metrics and Callbacks  

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.nanmean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.train_data = {'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans}
        self.train_target = df_train[output_categories].values[:5000]

        self.validation_data = {'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans}
        self.validation_target = df_train[output_categories].values[5000:]

        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.validation_data))
        
        rho_val = compute_spearmanr(
            self.validation_target, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        # if self.fold is not None:
        #     self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        # self.test_predictions.append(
        #     self.model.predict(self.test_inputs, batch_size=self.batch_size)

custom_callback = CustomCallback()


# In[11]:


#Compile and fit Model:

train_data = {'IP_Question_Title': train_sequences_pad_qt, 'IP_Question_Body': train_sequences_pad_qb, 'IP_Answer': train_sequences_pad_ans}
train_target = df_train[output_categories].values[:5000]

test_data = {'IP_Question_Title': validation_sequences_pad_qt, 'IP_Question_Body': validation_sequences_pad_qb, 'IP_Answer': validation_sequences_pad_ans}
test_target = df_train[output_categories].values[5000:]

optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model_Google_QUEST.compile(loss='mean_squared_error', optimizer=optimizer_adam)
model_Google_QUEST.fit(train_data, train_target, validation_data = (test_data, test_target),
           epochs=100, batch_size=64, verbose=1, callbacks=[custom_callback])


# In[12]:


#Prepare submission file:
test_prediction = model_Google_QUEST.predict({'IP_Question_Title': test_sequences_pad_qt, 'IP_Question_Body': test_sequences_pad_qb, 'IP_Answer': test_sequences_pad_ans})
submission_df = pd.concat([pd.DataFrame(df_test['qa_id']), pd.DataFrame(test_prediction, columns=output_categories)], axis=1)
submission_df.to_csv('submission.csv', index=False)

