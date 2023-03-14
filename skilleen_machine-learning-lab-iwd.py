#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
import time


# In[2]:


# capture start time for metrics
start = time.time()
# show all columns when printing out tables
pd.options.display.max_columns = None


# In[3]:


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', nrows=1000000)
#EXERCISE:Let's look at some rows of the data. You can see this with the pandas .head() function
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html


# In[4]:


test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', nrows=1000000)
#EXERCISE:Let's use the head function again to see how the test set differs


# In[5]:


cols_for_pairwise_correlation = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white', 'funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']
target_col_for_pairwise_correlation = 'target'
pairwise_correlation = []

for col in cols_for_pairwise_correlation:
    # Compute pairwise correlation of columns, excluding NA/null values.
    corr = train_df[col].corr(train_df[target_col_for_pairwise_correlation])
    #print("Correlation between {} and {}: {:f}".format(target_col, col, corr))
    pairwise_correlation.append({'Score': corr, 'Feature': col})

correlation_df = pd.DataFrame(pairwise_correlation)
correlation_df.sort_values(by=['Score'], ascending=False)


# In[6]:


print("Average comment length:", train_df.comment_text.str.len().mean())
print("Max comment length:", train_df.comment_text.str.len().max())


# In[7]:


# Full data set takes a long time to plot 
train_df['target'].head(10000).plot.kde()


# In[8]:


# Let's select the data that we're going to use in our experiment

x_train = train_df['comment_text'].astype(str)
x_test = test_df['comment_text'].astype(str)
y_train = train_df['target'].values

y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values


# In[9]:


# Convert values for most common identity labels to boolean
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
for column in IDENTITY_COLUMNS + ['target']:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False) 
    
#EXERCISE:Use the head function again to see the updated dataframe


# In[10]:



CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π' 

# Let's first break down comment texts
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

#http://faroit.com/keras-docs/1.2.2/preprocessing/text/#tokenizer
#EXERCISE: call the tokenizer fit_on_texts method that fits on x_train and x_test, then display word_index dictionary for tokenizer
#what can you observe?

# fit_on_texts Updates internal vocabulary based on a list of texts. 
# It creates the vocabulary index based on word frequency, the lower value the more frequent the word is (value 0 is reserved for padding). 
# eg "The cat sat on the mat." => will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 ..

#EXERCISE: display the dictionary of words created by tokenizer


# In[11]:


#EXERCISE:Use the tokenizer texts_to_sequences method to transform each comment_text to a sequence of integers. Each word in the comment should be replaced 
# with its corresponding integer value from the word_index dictionary. Then, display a sample comment
x_train = # 


# In[12]:


#EXERCISE do the same for the test data

x_test = # 


# In[13]:


# We will be processing comment vectors so it's useful to make them the same length.
# pad_sequences is used to ensure that all sequences in a list have the same length. 
# By default this is done by padding 0 in the beginning of each sequence 
MAX_LEN = 297
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
#EXERCISE display the vector above


# In[14]:


#EXERCISE Call pad_sequences for the test data
x_test = #
#EXERCISE display the vector above


# In[15]:


# We converted comments to vectors of numbers 
#
# What about the meaning of words used in the comments?
# 
EMBEDDING_FILES = [
    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',
    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'
]

# Lets load one of the embedding files to see what's there
embeddings = KeyedVectors.load('../input/gensim-embeddings-dataset/crawl-300d-2M.gensim', mmap='r')
#
# How are words represented in a numerical space?

#https://radimrehurek.com/gensim/models/keyedvectors.html
#EXERCISE Use word_vec method to display a word vector for a word 'apple'


# In[16]:


#EXERCISE What's the length of that vector? Use size property


# In[17]:


# What is the "distance" between two words?
#
w1 = embeddings.word_vec('king')
w2 = embeddings.word_vec('queen')

dist = np.linalg.norm(w1-w2)

print(dist)
#EXERCISE Try different words and rerun the cell to see their distances


# In[18]:


# Calculate 'king' - 'man' + 'woman' = ?

embeddings.most_similar(positive=['woman', 'king'], negative=['man'])

#EXERCISE try something different embeddings.most_similar(positive=['beans', 'stew', 'Texas'])


# In[19]:


# What else can we do using embeddings?
#
embeddings.doesnt_match("cat mouse rose dog".split())

#EXERCISE Try differnet set of words and doesnt_match method to check which word doesn't match the rest


# In[20]:


embeddings.similarity('cat', 'dog')


# In[21]:


# It indicates similarity of meaning of words not their spelling:
embeddings.similarity('cat', 'car')


# In[22]:


# Our model we're be building will require embedding information for the words used in the comments
# We will construct an embedding_matrix for the words in word_index
# using pre-trained embedding word vectors from resource in path
def build_matrix(word_index, path):
    # we've seen the embeddings already
    embedding_index = KeyedVectors.load(path, mmap='r')
    
    # embedding_matrix is a matrix of len(word_index)+1  x 300
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    
    # use word_index to get the index of each word 
    # and place its embedding in the matrix at that index
    for word, i in word_index.items():
        for candidate_word in [word, word.lower()]:
            if candidate_word in embedding_index:
                embedding_matrix[i] = embedding_index[candidate_word]
                break
    return embedding_matrix

# concatenate results for each type of embedding
embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)


# In[23]:


# To address the bias aspect we will create a set of weights that will reduce significance of samples containing identity words
# Build an array to provide a weight for each training sample. 
# (we'll indicating the weight for each of those samples during the training)
sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df['target'] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df['target']) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights /= sample_weights.mean()


# In[24]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D


#units parameters in Keras.layers.LSTM/cuDNNLSTM
#it it the dimension of the output vector of each LSTM cell.
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

# embedding_matrix: The matrix we created in the preprocessing stage
# num_aux_targets: The number of auxiliary target variables we have.
def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    
    #The Keras Embedding will turn our matrix of indexes into dense vectors of fixed size
    #The first parameter represents our max index integer 
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    #Randomlly drop 0.2% of input values to help prevent overfitting (This prevents units from coadapting too much)
    x = SpatialDropout1D(0.2)(x)
    
    #CuDNNLSTM is a LSTM implementation using the Nvidia cuDNN library 
    #The LSTM units is the dimension of the output vector of each LSTM cell
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        #Global max pooling has a pool size equal to the size of the input.
        GlobalMaxPooling1D()(x),
        GlobalMaxPooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# In[25]:


#EXCERCISE: Lets call the above method to build our model 
model = #


# In[26]:


#EXCERCISE: Use the .summary() function on the model we created above to help see a summary of the different layers in the model


# In[27]:


from keras.utils import plot_model

#EXCERCISE: Use the plot_model function with our model as a parameter for another visual 


# In[28]:


#this is the number of training samples to put in the model each step
BATCH_SIZE = 512

EPOCHS = 4
checkpoint_predictions = []
weights = []
for global_epoch in range(EPOCHS):
    model.fit(
        x_train,
        [y_train, y_aux_train],
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=2,
        sample_weight=[sample_weights.values, np.ones_like(sample_weights)]
        )
    checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
    weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
print(predictions)
end = time.time()
print("Total time taken:", end - start)


# In[29]:


##EXCERCISE: Write a for loop to loop over each entry in our predictions and explore our toxicity predictions
#For example, output the index of each prediction > 80 and compare it with the original text

for prediction in predictions:


#print(test_df.comment_text[index])

