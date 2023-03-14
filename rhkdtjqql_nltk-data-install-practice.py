#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import nltk


# In[ ]:


get_ipython().system('pip3 show nltk ')


# In[ ]:


nltk.download('punkt')


# In[ ]:


sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""


# In[ ]:


tokens = nltk.word_tokenize(sentence)
tokens


# In[ ]:


tagged=nltk.pos_tag(tokens)
#토큰화 하는 예제
#토큰은 한 단어 수준? 보통 띄어쓰기 단위


# In[ ]:


tagged[0:6]


# In[ ]:


import pandas as pd
"""
header=0 은 파일의 첫번째 줄에  이름이 있음을 나타내며
delimiter =\t 는 필드가 탭으로 구분 (in r sep="\t")
quoting = 3 옵션 3은 쌍따옴표 무시
QUOTE_MINIMAL (0), 구분자 같은 특별한 문자가 포함된 필드만 적용
QUOTE_ALL (1), 모든 필드에 적용
QUOTE_NONNUMERIC (2) or 숫자가 아닌 값에만 적용
QUOTE_NONE (3). 값을 둘러싸지 않음
기본은 QUOTE_MINIMAL (0)
"""
train0=pd.read_csv("../input/labeledTrainData.tsv",header=0,delimiter='\t',quoting=0)
train1=pd.read_csv("../input/labeledTrainData.tsv",header=0,delimiter='\t',quoting=1)
train2=pd.read_csv("../input/labeledTrainData.tsv",header=0,delimiter='\t',quoting=2)
train=pd.read_csv("../input/labeledTrainData.tsv",header=0,delimiter='\t',quoting=3)
test=pd.read_csv("../input/testData.tsv",header=0,delimiter='\t',quoting=3)


# In[ ]:


train.shape


# In[ ]:


train0.head()


# In[ ]:


train1.head()


# In[ ]:


train2.head()


# In[ ]:


train.head()


# In[ ]:


test.tail()


# In[ ]:


train.columns.values


# In[ ]:


test.columns.values


# In[ ]:


train.info()


# In[ ]:


train.describe()
#sentiment에 대해 분석해줌


# In[ ]:


train['sentiment'].value_counts()


# In[ ]:


train['review'][1]


# In[ ]:


# html 태그 때문에 텍스트 전처리 필요
train['review'][0][:700]
#첫번째 데이터에 대해 700자 까지만 본다


# In[ ]:


from bs4 import BeautifulSoup

example1 = BeautifulSoup(train['review'][0], "html5lib")

#html5lib parser을 통해 태그 삭제
#"html.parser" : 빠르지만 유연하지 않기 때문에 단순한 HTML문서에 사용합니다.
#"lxml" : 매우 빠르고 유연합니다.
#"xml" : XML 파일에만 사용합니다.
#"html5lib" : 복잡한 구조의 HTML에 대해서 사용합니다.
#출처: 
#https://jungwoon.github.io/python/2018/03/20/Data-Analysis-With-Python-3/


# In[ ]:


print(type(example1))


# In[ ]:


print(train['review'][0][:700])


# In[ ]:


print(type(example1.get_text()))
#클래스가 str변환됨


# In[ ]:


example1.get_text()[:700]


# In[ ]:


import re #정규표현식 호출-> 특수문제 제거 용도
#소문자와 대문자가 아닌 것은 공백으로 대체
letters_only=re.sub('[^a-zA-Z]', ' ', example1.get_text())
#^은 아닌거 선택, a-z까지 소문자, A-Z 대문자 아닌거는 ' '으로 치환 
#대상: example1.get_text()
letters_only


# In[ ]:


#모두 소문자로 변환
lower_case=letters_only.lower()
lower_case


# In[ ]:


#토큰화
words=lower_case.split()
#단순히 공백(스페이스, 탭 )기준으로 짜른 후 리스트로 반환
print(type(words))
print(len(words))
words[:10]


# In[ ]:


#words


# In[ ]:


import nltk
#로컬 환경에서 nltk 설치는 잘되지만 nltk 데이터가 설치가 잘안된다.
from nltk.corpus import stopwords
#nltk의 corpus에서 stopwords 호출
stopwords.words('english')[:10]


# In[ ]:


#stopword를 제거한 토큰들
words=[w for w in words if not w in stopwords.words('english')]
# w에 대해 w가 words에 있을 때 w가 stopwords에 없다면 words에 w를 넣어라 
# 3개의 변수가 같아야함
print(len(words))
words[:10]


# In[ ]:


stemmer=nltk.stem.PorterStemmer()
print(stemmer.stem('maximum'))
print("The stemmed form of running is:",stemmer.stem("running"))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))


# In[ ]:


# 랭커스터 스태머의 사용 예
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))
print("The stemmed form of running is: {}".format(lancaster_stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(lancaster_stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(lancaster_stemmer.stem("run")))


# In[ ]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
#snowball stemmer 한국어는 지원안함
words = [stemmer.stem(w) for w in words]
#for문을 돌면서 words에서 w를 뽑아 stemmer를 적용하여 words에 저장


# In[ ]:


#words


# In[ ]:


words[1]


# In[ ]:


stemmer.stem("cats")


# In[ ]:


words[:10]


# In[ ]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
a="the boy's cars are different colors"
b='she is meeting friends'
a1=a.split()
b1=b.split()
print(wordnet_lemmatizer.lemmatize('fly'))
print(wordnet_lemmatizer.lemmatize('flies'))
print(wordnet_lemmatizer.lemmatize('a'))
a2= [wordnet_lemmatizer.lemmatize(w) for w in a1]
b2= [wordnet_lemmatizer.lemmatize(w) for w in b1]
# 처리 후 단어
b2[:10]


# In[ ]:


def review_to_words(raw_review):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review,'lxml').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]',' ',review_text)
    # 3. 소문자 변환 및 띄어쓰기 단위로 토큰화
    words=letters_only.lower().split()
    type(words)
    # 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops=set(stopwords.words('english'))
    type(stops)
    # 5. stopwords 불용어 제거
    meaningful_words=[w for w in words if not w in stops]
    # 6. 어간추출
    stemming_words=[stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열을 결합하여 결과를 반환
    return(' '.join(stemming_words))

clean_review=review_to_words(train['review'][0])
clean_review[:700]


# In[ ]:


def review_to_words2( raw_review ):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는 게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return( ' '.join(stemming_words) )
clean_review = review_to_words(train['review'][0])
clean_review


# In[ ]:


#전체 데이터 리뷰를 대상으로 전처리
#전체 리뷰 데이터 수
num_reviews=train['review'].size
num_reviews


# In[ ]:


#for i in range(0, num_reviews):
#    if (i + 1)%5000 == 0:
#        print('Review {} of {} '.format(i+1, num_reviews))    
#    clean_train_reviews.append(review_to_words(train['review'][i]))


# In[ ]:


# %time train['review_clean']=train['review'].apply(review_to_words)


# In[ ]:





# In[ ]:


train['review']


# In[ ]:


#train데이터에서 review컬럼에 대해 review_to_words 함수를 적용
#코드를 한줄로 만들었지만 여전히 오래걸림
from multiprocessing import Pool
import numpy as np

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))

clean_train_reviews=apply_by_multiprocessing(                                                  train['review'],review_to_words,workers=4)
clean_test_reviews = apply_by_multiprocessing(    test['review'], review_to_words2, workers=4)


# In[ ]:


clean_train_reviews[:10]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %matplotlib inline 설정을 해주어야지만 노트북 안에 그래프가 디스플레이 된다.
get_ipython().run_line_magic('matplotlib', 'inline')

def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS, #불용어 처리
                          background_color = backgroundcolor, # 배경색
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 


# In[ ]:


' '.join(clean_train_reviews[:3])


# In[ ]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(''.join(clean_train_reviews))")

# 왜 ' '.join을 쓰는가?
# 공백으로 구분된 문자열로 결합?
# 데이터를 공백으로 구분하여 통째로 넣는 것


# In[ ]:


train.head()


# In[ ]:


# 단어 수
train['num_words'] = clean_train_reviews.apply(lambda x: len(str(x).split()))
#apply에다가 lambda 식 적용
#str(x) 문자화
#str(x).split() tokenize
#len(str(x).split()) 단어 수 count


# In[ ]:


len(str(clean_train_reviews[1]).split())


# In[ ]:


# 중복을 제거한 단어 수
train['num_uniq_words'] = clean_train_reviews.apply(lambda x: len(set(str(x).split())))
#set을 통해 리스트를 set으로 변환하여 중복 제거


# In[ ]:


# 첫 번째 리뷰에 대해 tokenize
x = clean_train_reviews[0]
x = str(x).split()
print(x)


# In[ ]:


import seaborn as sns

fig, axes = plt.subplots(ncols=2)
#가로로 2개의 그래프 구현
fig.set_size_inches(18, 6)
print('리뷰 별 단어 평균값 :', train['num_words'].mean())
print('리뷰 별 단어 중간값', train['num_words'].median())
sns.distplot(train['num_words'], bins=100, ax=axes[0])
axes[0].axvline(train['num_words'].median(), linestyle='dashed')
axes[0].set_title('리뷰 별 단어 수 분포')
print('리뷰 별 고유 단어 평균값 :', train['num_uniq_words'].mean())
print('리뷰 별 고유 단어 중간값', train['num_uniq_words'].median())
sns.distplot(train['num_uniq_words'], bins=100, color='g', ax=axes[1])
axes[1].axvline(train['num_uniq_words'].median(), linestyle='dashed')
axes[1].set_title('리뷰 별 고유한 단어 수 분포')


# In[ ]:





# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라메터 값을 수정 
# 파라메터 값만 수정해도 캐글 스코어 차이가 크게 남
vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,#tokenizer 설정
                             preprocessor = None, 
                             stop_words = None,#불용어
                             min_df = 2, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 3),#ngram 갯수
                             max_features = 20000 #최대 토큰 갯수
                            )
vectorizer


# In[ ]:


# fit_transform의 속도 개선을 위해 파이프라인을 사용하도록 개선
# 참고 : https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph
pipeline = Pipeline([
    ('vect', vectorizer),# 'vectorizer'자리에 tf - idf 사용 가능
])  


# In[ ]:


get_ipython().run_line_magic('time', 'train_data_features = pipeline.fit_transform(clean_train_reviews)')


# In[ ]:


train_data_features.shape


# In[ ]:


train_data_features.shape


# In[ ]:


vocab = vectorizer.get_feature_names()
#행렬의 feature name을 확인 가능

vocab[:10]


# In[ ]:


train_data_features


# In[ ]:


# 벡터화된 피처를 확인해 봄
import numpy as np
dist = np.sum(train_data_features, axis=0)
dist
dist.shape
#http://taewan.kim/post/numpy_sum_axis/


# In[ ]:


for tag, count in zip(vocab, dist):
    print(count)
    print(tag)
#


# In[ ]:


pd.DataFrame(dist, columns=vocab)


# In[ ]:


pd.DataFrame(train_data_features[:100].toarray(), columns=vocab).head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# 랜덤포레스트 분류기를 사용
forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1,#모든 코어 사용 -1
    random_state=2018 #파라미터 튜닝을 위해 회차마다 결과 동일하게 만듬    
    )
forest


# In[ ]:


get_ipython().run_line_magic('time', "forest = forest.fit(train_data_features ,train['sentiment']) #행렬 데이터,벡터 데이터")
                        
                         


# In[ ]:


forest


# In[ ]:


from sklearn.cross_validation import cross_val_score
get_ipython().run_line_magic('time', "np.mean(cross_val_score(forest, train_data_features, train['sentiment'], cv=10,scoring='roc_auc'))#cross validation #roc 커브 사용")
                               


# In[ ]:


# 위에서 정제해준 리뷰의 첫 번째 데이터를 확인
clean_test_reviews[0]


# In[ ]:


# 테스트 데이터를 벡터화 함
get_ipython().run_line_magic('time', 'test_data_features = pipeline.transform(clean_test_reviews)#파이프 라인을 통해 여러개의 쓰래드 사용하여 벡터화')
test_data_features = test_data_features.toarray()


# In[ ]:


test_data_features


# In[ ]:


# 벡터화된 단어로 숫자가 문서에서 등장하는 횟수를 나타낸다
test_data_features[5][:100]


# In[ ]:


# 벡터화하며 만든 사전에서 해당 단어가 무엇인지 찾아볼 수 있다.
# vocab = vectorizer.get_feature_names()
vocab[8], vocab[2558], vocab[2559], vocab[2560]


# In[ ]:


# 테스트 데이터를 넣고 예측한다.
result = forest.predict(test_data_features)
result[:10]


# In[ ]:


# 예측 결과를 저장하기 위해 데이터프레임에 담아 준다.
output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
output.head()


# In[ ]:


output.to_csv('data/tutorial_1_BOW_model.csv', index=False, quoting=3)
output_sentiment = output['sentiment'].value_counts()
print(output_sentiment[0] - output_sentiment[1])
output_sentiment


# In[ ]:


fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])


# In[ ]:


# 파라메터를 조정해 가며 점수를 조금씩 올려본다.

# uni-gram 사용 시 캐글 점수 0.84476
print(436/578)
# tri-gram 사용 시 캐글 점수 0.84608
print(388/578)
# 어간추출 후 캐글 점수 0.84780
print(339/578)
# 랜덤포레스트의 max_depth = 5 로 지정하고
# CountVectorizer의 tokenizer=nltk.word_tokenize 를 지정 후 캐글 점수 0.81460
print(546/578)
# 랜덤포레스트의 max_depth = 5 는 다시 None으로 변경
# CountVectorizer max_features = 10000개로 변경 후 캐글 점수 0.85272
print(321/578)
# CountVectorizer의 tokenizer=nltk.word_tokenize 를 지정 후 캐글 점수 0.85044
print(326/578)
# CountVectorizer max_features = 10000개로 변경 후 캐글 점수 0.85612
print(305/578)
# 0.85884
print(296/578)

