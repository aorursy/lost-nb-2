#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Chargement des données depuis Kaggle
train_path = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
test_path = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'
train_data = pd.read_csv(train_path, sep="\t")
test_data = pd.read_csv(test_path, sep="\t")
train_data.head()


# In[2]:


test_data.head()


# In[3]:


train_phrase_count = train_data['Phrase'].count()
test_phrase_count = test_data['Phrase'].count()

print(f'Il y a {train_phrase_count} phrases dans le jeu d\'entraînement.')
print(f'Il y a {test_phrase_count} phrases dans le jeu de test.\n')

vcounts = train_data['Sentiment'].value_counts()
vpcts = []
for i in range(5):
    vpcts.append(round(vcounts[i] / train_phrase_count * 100, 2))
print(f'Répartition des phrases selon leur catégorie :\n 0: {vcounts[0]} ({vpcts[0]}%) \n 1: {vcounts[1]} ({vpcts[1]}%) \n 2: {vcounts[2]} ({vpcts[2]}%)       \n 3: {vcounts[3]} ({vpcts[3]}%) \n 4: {vcounts[4]} ({vpcts[4]}%) \n')

avg_number_words = round(train_data['Phrase'].apply(lambda phrase : len(phrase.split(" "))).mean())
print(f'Chaque phrase est constituée de {avg_number_words} mots en moyenne.')
min_number_words = train_data['Phrase'].apply(lambda phrase : len(phrase.split(" "))).min()
max_number_words = train_data['Phrase'].apply(lambda phrase : len(phrase.split(" "))).max()
print(f'La phrase la plus courte a {min_number_words} mot(s).')
print(f'La phrase la plus longue a {max_number_words} mots.\n')


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Vectorisation des critiques
vectorizer = CountVectorizer(max_features=256)
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']

# Séparation des données en 2 parties : entraînement / validation
train_X, val_X, train_y, val_y = train_test_split(X, y)

# Entraînement du modèle
decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
baseline_accuracy = round(accuracy_score(val_y, predictions) * 100, 2)
print('Score (=accuracy) du modèle simple sur le jeu de validation : ', baseline_accuracy, '%')


# In[5]:


# Modèle 1
vectorizer = CountVectorizer(max_features=256)
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 1 sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[6]:


# Modèle 2
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=256)
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 2 sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[7]:


import matplotlib.pyplot as plt

# Scikit learn n'est pas multi-threadé, mais on peut quand même accélérer les choses en entraînant plusieurs modèles en même temps
from joblib import Parallel, delayed

vocab_sizes = [i for i in range(1000, 10_000, 1000)]

def test_vocab(vocab_size):
    vectorizer = CountVectorizer(max_features=vocab_size)
    X = vectorizer.fit_transform(train_data['Phrase'])
    y = train_data['Sentiment']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree = decision_tree.fit(train_X, train_y)
    return accuracy_score(val_y, decision_tree.predict(val_X))

val_scores = Parallel(n_jobs=-1)(delayed(test_vocab)(vocab_size) for vocab_size in vocab_sizes)
    
plt.plot(vocab_sizes, val_scores, 'g', label='Jeu de validation')
plt.xlabel('Taille du vocabulaire')
plt.ylabel('Score (accuracy)')
plt.show()


# In[8]:


vectorizer = CountVectorizer()
vectorizer.fit(train_data['Phrase'])
vocab_size = len(vectorizer.vocabulary_)
print(f'Il y a {vocab_size} mots différents dans notre corpus.')


# In[9]:


import nltk

tokens = [nltk.word_tokenize(phrase) for phrase in train_data['Phrase']]
tokens = [token for token_list in tokens for token in token_list]
fd = nltk.FreqDist(tokens)
fd.most_common(20)


# In[10]:


from nltk.corpus import stopwords 

# nos stopwords / (+ ponctuation), trouvés en analysant le résultat du code en dessous
custom_stopwords = [',', '.', '\'s', 'n\'t', '--', '\'', '-rrb-', '-lrb-', '`', '...', '``', '\'\'', '-', ':', '\'re', 'ca', '\'ll', ';', '?', '\'d', '!', '\'m']

# On ajoute les stopwords de nltk
stop_words = set(stopwords.words('english') + custom_stopwords)
tokens = [nltk.word_tokenize(phrase) for phrase in train_data['Phrase']]
tokens = [token.lower() for token_list in tokens for token in token_list if not (token.lower() in stop_words)]
fd = nltk.FreqDist(tokens)
fd.most_common(20)


# In[11]:


# Modèle 3
vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 3 sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[12]:


# Modèle 4
vectorizer = TfidfVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 4 sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[13]:


# modèle 5
vectorizer = CountVectorizer(max_features=256, ngram_range=(2,2), vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 5 (2-grams) sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[14]:


# Pour chaque phrase, calculer un embedding "moyen"
def average_embeddings(phrases, wordvectors, vec_dim):
    avg_embeddings = []
    for phrase in phrases:
        phrase_embeddings = []
        for word in phrase.split(' '):
            if word.lower() in wordvectors.vocab:
                phrase_embeddings.append(wordvectors[word.lower()])
        # gestion des mots ignorés
        if len(phrase_embeddings) == 0:
            avg_embeddings.append(np.zeros(vec_dim))
        else:
            avg_embeddings.append(np.mean(phrase_embeddings, axis=0))
    return avg_embeddings

# Pour chaque phrase, calculer la somme des embedddings
def embeddings_sum(phrases, wordvectors, vec_dim):
    avg_embeddings = []
    for phrase in phrases:
        phrase_embeddings = []
        for word in phrase.split(' '):
            if word.lower() in wordvectors.vocab:
                phrase_embeddings.append(wordvectors[word.lower()])
        # gestion des mots ignorés
        if len(phrase_embeddings) == 0:
            avg_embeddings.append(np.zeros(vec_dim))
        else:
            avg_embeddings.append(np.sum(phrase_embeddings, axis=0))
    return avg_embeddings


# In[15]:


from gensim.models import Word2Vec

# Entraînement de Word2Vec sur nos phrases
def train_embeddings(phrases, mincount):
    phrases_list = []
    for phrase in phrases:
        words = phrase.split(' ')
        phrases_list.append(words)
    w2v = Word2Vec(phrases_list, size = 300, min_count=mincount, workers=4)
    return w2v
    
# On peut maintenant retourner les mots les plus similaires à "movie" par exemple
w2v = train_embeddings(train_data['Phrase'], 5)
w2v.wv.most_similar('movie')


# In[16]:


# Modèle 6 : avec des embeddings entraînés sur le corpus
X1 = average_embeddings(train_data['Phrase'], w2v.wv, 300)
X2 = embeddings_sum(train_data['Phrase'], w2v.wv, 300)
y = train_data['Sentiment']
train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y, random_state=0)
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y, random_state=0)
dt1 = DecisionTreeClassifier(random_state=0)
dt1 = decision_tree.fit(train_X1, train_y1)
dt2 = DecisionTreeClassifier(random_state=0)
dt2 = decision_tree.fit(train_X2, train_y2)
predictions1 = decision_tree.predict(val_X1)
predictions2 = decision_tree.predict(val_X2)

print('Score (=accuracy) du modèle 6 - embedding moyen - sur le jeu de validation : ', round(accuracy_score(val_y1, predictions1) * 100, 2), '%')
print('Score (=accuracy) du modèle 6 - embedding somme - sur le jeu de validation : ', round(accuracy_score(val_y2, predictions2) * 100, 2), '%')


# In[17]:


from gensim.models import KeyedVectors

# Chargement des embeddings pré-entraînés dans la RAM
ft_path = '../input/fasttext-wikinews/wiki-news-300d-1M.vec'
ft_wv = KeyedVectors.load_word2vec_format(ft_path)


# In[18]:


# Pour chaque phrase, calculer un embedding "moyen" (fastText)
avg_embeddings = average_embeddings(train_data['Phrase'], ft_wv, 300)
sum_embeddings = embeddings_sum(train_data['Phrase'], ft_wv, 300)


# In[19]:


# Modèle 7
X1 = avg_embeddings
X2 = sum_embeddings
y = train_data['Sentiment']
train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y, random_state=0)
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y, random_state=0)
dt1 = DecisionTreeClassifier(random_state=0)
dt2 = decision_tree.fit(train_X1, train_y)
dt1 = DecisionTreeClassifier(random_state=0)
dt2 = decision_tree.fit(train_X2, train_y)
predictions1 = decision_tree.predict(val_X1)
predictions2 = decision_tree.predict(val_X2)
print('Score (=accuracy) du modèle 7 - embedding moyen - sur le jeu de validation : ', round(accuracy_score(val_y1, predictions1) * 100, 2), '%')
print('Score (=accuracy) du modèle 7 - embedding somme - sur le jeu de validation : ', round(accuracy_score(val_y2, predictions2) * 100, 2), '%')


# In[20]:


import graphviz
from sklearn import tree

vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)

dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                      feature_names=vectorizer.get_feature_names(),
                      class_names=['0', '1', '2', '3', '4'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph 


# In[21]:


import matplotlib.pyplot as plt

# Scikit learn n'est pas multi-threadé, mais on peut quand même accélérer les choses en entraînant plusieurs modèles en même temps
from joblib import Parallel, delayed

depths = [10, 100, 500, 1000]

def test_depths(depth):
    vectorizer = CountVectorizer(max_features=256)
    X = vectorizer.fit_transform(train_data['Phrase'])
    y = train_data['Sentiment']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    decision_tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    decision_tree = decision_tree.fit(train_X, train_y)
    return accuracy_score(val_y, decision_tree.predict(val_X))

val_scores = Parallel(n_jobs=-1)(delayed(test_depths)(depth) for depth in depths)
    
plt.plot(depths, val_scores, 'g', label='Jeu de validation')
plt.xlabel('Profondeur de l\'arbre')
plt.ylabel('Score (accuracy)')
plt.show()


# In[22]:


vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10000, random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)

dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                      feature_names=vectorizer.get_feature_names(),
                      class_names=['0', '1', '2', '3', '4'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph 


# In[23]:


import matplotlib.pyplot as plt

# Scikit learn n'est pas multi-threadé, mais on peut quand même accélérer les choses en entraînant plusieurs modèles en même temps
from joblib import Parallel, delayed

mss_list = [0.1, 0.2, 0.5, 1.0] # en % du nombre total d'exemples

def test_mss(mss):
    vectorizer = CountVectorizer(max_features=256)
    X = vectorizer.fit_transform(train_data['Phrase'])
    y = train_data['Sentiment']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    decision_tree = DecisionTreeClassifier(min_samples_split=mss, random_state=0)
    decision_tree = decision_tree.fit(train_X, train_y)
    return accuracy_score(val_y, decision_tree.predict(val_X))

val_scores = Parallel(n_jobs=-1)(delayed(test_mss)(mss) for mss in mss_list)
    
plt.plot(depths, val_scores, 'g', label='Jeu de validation')
plt.xlabel('min_samples_split')
plt.ylabel('Score (accuracy)')
plt.show()


# In[24]:


vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1000, random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)

dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                      feature_names=vectorizer.get_feature_names(),
                      class_names=['0', '1', '2', '3', '4'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph 


# In[25]:


import matplotlib.pyplot as plt

# Scikit learn n'est pas multi-threadé, mais on peut quand même accélérer les choses en entraînant plusieurs modèles en même temps
from joblib import Parallel, delayed

msl_list = [10, 100, 1000, 10000]

def test_msl(msl):
    vectorizer = CountVectorizer(max_features=256)
    X = vectorizer.fit_transform(train_data['Phrase'])
    y = train_data['Sentiment']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    decision_tree = DecisionTreeClassifier(min_samples_leaf=msl, random_state=0)
    decision_tree = decision_tree.fit(train_X, train_y)
    return accuracy_score(val_y, decision_tree.predict(val_X))

val_scores = Parallel(n_jobs=-1)(delayed(test_msl)(msl) for msl in msl_list)
    
plt.plot(depths, val_scores, 'g', label='Jeu de validation')
plt.xlabel('min_samples_leaf')
plt.ylabel('Score (accuracy)')
plt.show()


# In[26]:


sst_phrases_path = '../input/sst-phrases-sentiments/sst_phrases_sentiments.csv'
sst_phrases = pd.read_csv(sst_phrases_path)

sst_phrases_count = sst_phrases['Phrase'].count()
difference = sst_phrases_count - train_phrase_count
print(f'Il y a {sst_phrases_count} phrases dans le jeu de données SST, soit {difference} phrases de plus que dans les données Kaggle')

sst_phrases.head()


# In[27]:


merged_data = pd.merge(train_data, sst_phrases, on=['Phrase', 'Sentiment'])
merged_count = merged_data.shape[0]
count_difference = sst_phrases_count - merged_count
print(f'Il y a {merged_count} phrases identiques entres les 2 jeux de données.')
print(f'On gagne donc {count_difference} phrases supplémentaires grâce à SST.')

merged_data.head()


# In[28]:


# Modèle 3 avec plus de données
vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(merged_data['Phrase'])
y = merged_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(train_X, train_y)
predictions = decision_tree.predict(val_X)
print('Score (=accuracy) du modèle 3 (+ de données) sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[29]:


from sklearn.ensemble import RandomForestClassifier

vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(merged_data['Phrase'])
y = merged_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
rf = RandomForestClassifier(n_estimators=10)
rf = rf.fit(train_X, train_y)
predictions = rf.predict(val_X)
print('Score (=accuracy) de la forêt aléatoire sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[30]:


from sklearn.ensemble import AdaBoostClassifier

vectorizer = CountVectorizer(max_features=256, vocabulary=set(tokens))
X = vectorizer.fit_transform(merged_data['Phrase'])
y = merged_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
ab = AdaBoostClassifier(n_estimators=50)
ab = ab.fit(train_X, train_y)
predictions = ab.predict(val_X)
print('Score (=accuracy) de AdaBoost sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[31]:


vectorizer = CountVectorizer(vocabulary=set(tokens))
X = vectorizer.fit_transform(train_data['Phrase'])
y = train_data['Sentiment']
train_X, val_X, train_y, val_y = train_test_split(X, y)
rf = RandomForestClassifier(n_estimators=50)
rf = rf.fit(train_X, train_y)
predictions = rf.predict(val_X)
print('Score (=accuracy) du modèle final sur le jeu de validation : ', round(accuracy_score(val_y, predictions) * 100, 2), '%')


# In[32]:


# Soumission
test_X = vectorizer.transform(test_data['Phrase'])
test_preds = rf.predict(test_X)
submission = pd.DataFrame(test_data['PhraseId'])
submission['Sentiment'] = pd.Series(test_preds)
submission.to_csv('submission.csv', index=False)

