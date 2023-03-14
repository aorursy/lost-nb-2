#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


# 출력이 너무 길어지지 않게하기 위해 찍지 않도록 했으나 
# 실제 학습 할 때는 아래 두 줄을 주석처리 하는 것을 권장한다.
# import warnings
# warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd

train = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv', 
                    header=0, delimiter='\t', quoting=3)


# In[ ]:


train.head()


# In[ ]:


train.iloc[0,2][:700]


# In[ ]:


train['review'][0][:700]


# In[ ]:


test = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv', 
                   header=0, delimiter='\t', quoting=3)
unlabeled_train = pd.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv', 
                              header=0, delimiter='\t', quoting=3)


# In[ ]:


print(train.shape)
print(test.shape)
print(unlabeled_train.shape)

print(train['review'].size)


# In[ ]:


import missingno as msno


# In[ ]:


train['review'].isna().sum()


# In[ ]:


print(train.shape)
print(test.shape)
print(unlabeled_train.shape)

print(train['review'].size)
print(test['review'].size)
print(unlabeled_train['review'].size)


# In[ ]:


train.head()


# In[ ]:


# train에 있는 평점정보인 sentiment가 없다.
test.head()


# In[ ]:


# import module we'll need to import our custom module
from shutil import copyfile


# In[ ]:


# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/kaggleword2vecutility/KaggleWord2VecUtility.py", dst = "../working/KaggleWord2VecUtility.py")

# import all our functions
#from my_functions import *


# In[ ]:


from KaggleWord2VecUtility import KaggleWord2VecUtility


# In[ ]:


# train['review'][0]
# mj 리뷰
# 'review의 첫번째 데이터'


# In[ ]:


"""import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from multiprocessing import Pool

class KaggleWord2VecUtility(object):

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # 1. HTML 제거
        review_text = BeautifulSoup(review, "html.parser").get_text()
        # 2. 특수문자를 공백으로 바꿔줌
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자로 변환 후 나눈다.
        words = review_text.lower().split()
        # 4. 불용어 제거
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 6. 리스트 형태로 반환
        return(words)

    @staticmethod
    def review_to_join_words( review, remove_stopwords=False ):
        words = KaggleWord2VecUtility.review_to_wordlist(\
            review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words

    @staticmethod
    def review_to_sentences( review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
   
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. nltk tokenizer를 사용해서 단어로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = tokenizer.tokenize(review.strip())
        #review.()을 하면 띄어 쓰기 단위로 자름
        #그것을 to
        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                sentences.append(\
                    KaggleWord2VecUtility.review_to_wordlist(\
                    raw_sentence, remove_stopwords))
        return sentences


    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록
    @staticmethod
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)

    @staticmethod
    def apply_by_multiprocessing(df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(KaggleWord2VecUtility._apply_df, [(d, func, kwargs)
                for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(result)
"""


# In[ ]:


KaggleWord2VecUtility.review_to_wordlist(train['review'][0])[:10]
#html tag 지우고
#특수문자 지우고
#소문자로 변환하고
#기본적인 영어 불용어 지우고
#어간추출(Stemming)하고
#리스트 형태로 반환


# In[ ]:


sentences = []
for review in train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[ ]:


for review in unlabeled_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[ ]:


len(sentences)


# In[ ]:


sentences[0][:10]


# In[ ]:


sentences[1][:10]


# In[ ]:


import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)


# In[ ]:


# 파라메터값 지정
num_features = 300 # 문자 벡터 차원 수
min_word_count = 40 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 10 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도 수 Downsample

# 초기화 및 모델 학습
from gensim.models import word2vec

# 모델 학습
model = word2vec.Word2Vec(sentences, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)
model


# In[ ]:


# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
# model_name = '300features_50minwords_20text'
model.save(model_name)


# In[ ]:


# 유사도가 없는 단어 추출
model.wv.doesnt_match('man woman child kitchen'.split())


# In[ ]:


model.wv.doesnt_match("france england germany berlin".split())


# In[ ]:


# 가장 유사한 단어를 추출
model.wv.most_similar("man")


# In[ ]:


model.wv.most_similar("queen")


# In[ ]:


# model.wv.most_similar("awful")


# In[ ]:


model.wv.most_similar("film")


# In[ ]:


# model.wv.most_similar("happy")
model.wv.most_similar("happi") # stemming 처리 시 


# In[ ]:


# 참고 https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

model_name = '300features_40minwords_10text'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])
# X_tsne = tsne.fit_transform(X)


# In[ ]:


df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
df.shape


# In[ ]:


df.head(10)


# In[ ]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()


# In[ ]:


import numpy as np

def makeFeatureVec(words, model, num_features):
    """
    주어진 문장에서 단어 벡터의 평균을 구하는 함수
    """
    # 속도를 위해 0으로 채운 배열로 초기화 한다.
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화 한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 결과를 단어수로 나누어 평균을 구한다.
    featureVec = np.divide(featureVec,nwords)
    return featureVec


# In[ ]:


def getAvgFeatureVecs(reviews, model, num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고 
    # 2D numpy 배열을 반환한다.
    
    # 카운터를 초기화 한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews),num_features),dtype="float32")
    
    for review in reviews:
       # 매 1000개 리뷰마다 상태를 출력
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,            num_features)
       # 카운터를 증가시킨다.
       counter = counter + 1.
    return reviewFeatureVecs


# In[ ]:


# 멀티스레드로 4개의 워커를 사용해 처리한다.
def getCleanReviews(reviews):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(        reviews["review"], KaggleWord2VecUtility.review_to_wordlist,        workers=4)
    return clean_reviews


# In[ ]:


get_ipython().run_line_magic('time', 'trainDataVecs = getAvgFeatureVecs(    getCleanReviews(train), model, num_features )')


# In[ ]:


get_ipython().run_line_magic('time', 'testDataVecs = getAvgFeatureVecs(        getCleanReviews(test), model, num_features )')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)


# In[ ]:


get_ipython().run_line_magic('time', 'forest = forest.fit( trainDataVecs, train["sentiment"] )')


# In[ ]:


from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('time', "score = np.mean(cross_val_score(    forest, trainDataVecs,     train['sentiment'], cv=10, scoring='roc_auc'))")


# In[ ]:


score


# In[ ]:


result = forest.predict( testDataVecs )


# In[ ]:


output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv('data/Word2Vec_AverageVectors_{0:.5f}.csv'.format(score), 
              index=False, quoting=3 )


# In[ ]:


output_sentiment = output['sentiment'].value_counts()
print(output_sentiment[0] - output_sentiment[1])
output_sentiment


# In[ ]:


import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])


# In[ ]:


544/578


# In[ ]:




