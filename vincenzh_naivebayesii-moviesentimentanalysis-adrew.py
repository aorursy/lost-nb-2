#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../input/labeledTrainData.tsv",sep = '\t')
test = pd.read_csv("../input/testData.tsv", sep = '\t')
#unlabeld_test = pd.read_csv("unlabeledTrainData.tsv", sep = '\t')


# In[ ]:


print(train.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

#pd.DataFrame(train.loc[:,['id','review']])

X = train.loc[:, 'review']
y = train.loc[:,'sentiment']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


def GetVocabulary(data):
   vocab_dict = {}
   wid = 0
   for document in data:
       words = document.split()
       for word in words:
           word = word.lower()
           if word not in vocab_dict:
               vocab_dict[word] = wid
               wid += 1
   return vocab_dict

#print(len(vocab_dict.keys()))
#truncated_X = xtrain[0:18000]  # rason for this is when I run it on jupyter, it crashes - out of memeory - if it's the entire xtrain matrix
truncated_X = xtrain           # but for submitting the homework I set it to the entire xtrain, assuming 老师的 machine can handle it
#print(len(truncated_X))

vocab_dict = GetVocabulary(truncated_X)
print('Number of all the unique words: ' + str(len(vocab_dict.keys())))


# In[ ]:


def Document2Vector(vocab_dict, data):
    word_vector = np.zeros(len(vocab_dict.keys()))
    words = data.split()
    for word in words:
        word = word.lower()
        if word in vocab_dict:
            word_vector[vocab_dict[word]] += 1
    return word_vector

example = Document2Vector(vocab_dict, 'we are good good')
print(example)
print(example[vocab_dict['we']], example[vocab_dict['are']], example[vocab_dict['good']])


# In[ ]:


train_matrix = []
for document in truncated_X.values:
    word_vector = Document2Vector(vocab_dict, document)
    train_matrix.append(word_vector)


# In[ ]:


def NaiveBayes_train(train_matrix, labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    spam_word_counter = np.ones(num_words)
    ham_word_counter = np.ones(num_words)
    
    ham_total_count = 0
    spam_total_count = 0
    
    spam_count = 0 
    ham_count = 0
    
    for i in range(num_docs):
        if i % 500 == 0:
            print('Train on the doc id' + str(i))
            
        if labels_train[i] == 1:   # 1 is positive, or 'ham'
            ham_word_counter += train_matrix[i]
            ham_total_count += sum(train_matrix[i])
            ham_count += 1
        else:                      # 0 is negative, or 'spam'
            spam_word_counter += train_matrix[i]
            spam_total_count += sum(train_matrix[i])
            spam_count += 1
    
    # spam_word_counter => 没个词的计数
    # spam_total_count => Spam的总词数
    # spam_count => spam邮件的计数
    
    p_spam_vector = np.log(spam_word_counter / (spam_total_count + num_words))
    p_ham_vector = np.log(ham_word_counter / (ham_total_count + num_words))
    
    return p_spam_vector, np.log(spam_count/num_docs),spam_total_count, p_ham_vector, np.log(ham_count/num_docs), ham_total_count

# p_spam_vetor/p_ham_vector 的每一维分别是一个单词再spam/ham分类下的概率
# p_spam / p_ham 分别是两个分类的概率

p_spam_vector, p_spam, spam_total_count, p_ham_vector, p_ham, ham_total_count = NaiveBayes_train(train_matrix, ytrain.values)


# In[ ]:


def Test2Vector(vocab_dict, data):
    word_vector = np.zeros(len(vocab_dict.keys()))
    words = data.split()
    
    out_of_voc = 0
    for word in words:
        word = word.lower()
        if word in vocab_dict:
            word_vector[vocab_dict[word]] += 1
        #else:
            #out_of_voc +=
    return word_vector

def Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):
    spam = sum(test_word_vector * p_spam_vector) + p_spam
    ham = sum(test_word_vector * p_ham_vector) + p_ham
    if spam > ham:
        return 0
    else:
        return 1

predictions = []
i = 0
for document in xtest.values:
    if i % 200 == 0:
        print('Test on the doc id: ', str(i))
    i += 1
    test_word_vector = Document2Vector(vocab_dict, document)
    ans = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    predictions.append(ans)
    
print(len(predictions))


# In[ ]:


predictions


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

print(accuracy_score(ytest, predictions))
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))


# In[ ]:


test_X = test.loc[:, 'review']
test_X_id = test.loc[:,'id']

results = []
i = 0
for document in test_X.values:
    if i % 200 == 0:
        print('Test on the doc id: ', str(i))
    i += 1
    result_word_vector = Document2Vector(vocab_dict, document)
    ans = Predict(result_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    results.append(ans)


# In[ ]:


df = pd.DataFrame({"id": test.loc[:,'id'], "sentiment":results})
df.to_csv('submission.csv', header = True, index = False)

