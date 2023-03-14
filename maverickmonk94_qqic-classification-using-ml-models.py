#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import re
import os
import nltk
import time
import scipy
import string
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as pt
import plotly.graph_objects as go
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RandomizedSearchCV, train_test_split


# In[3]:


"""Loading the train and test data"""
quora_train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
quora_test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')


# In[4]:


punctuation = "=∈•→≈≠(∗☹／¼؟・̑℃×¦\\⊂⎠…∫̴̀）☁|´{∂′˜¥⅔▾∠®ு¢⅓✔≡↑│΅̶̊/∅¬–☉∨（̸̪＄،！，∞˂≥&)✏∩̉-↓”?≅̂÷″⟨̃+̵✓♡̳;̈[ா¾*͂:＞⋅°]_£♨✌☝#।♏€＝±⎛℅∴⟩ி—̎≤ْ̣̅⎝‘«@̌⋯△̱∪†、̲%̕◦¿－∘$̡̓♣⎞^∆<⊆̗.❤；’∼„❓·™,̖»½⁻♀¸‛©¡̷̐‰∑√⇒>：˚✅்̾`⊥∇͡!̄➡~̿\x92¶\
☺§−♭。▒“？∝⬇＾¯◌}\'∀∧\x02্ു̧̤͆ีॄ\xad̋ः̘《͜\u200cি̫\u2060ุ͗ూ⦁͑\u202a\x7fូ้ାู̰̼̺͒₦\x06ٌ͋ះం͚ె្⌚్₱㏑া́ा़̩͌₩\uf0d8〖ͅ⊨ֿ〗ਾ̍️「ॣ\x8d่ी㏒͐োோ̦〇」̥\
\u200bં\x13∛͊\x9d⚧》͈̝̜ে̓∖ీా̭̔\x8f͖ৃ\u200eَ̒̈́͛ੰ\ufeffા₹\u200fॉைី︡￼਼͕̯͎ា\ue019\x1b͉ँ\x10\x01̮\u2061्ೋ\u202c\
‑ിे્͝ិ͘ै\x17ੀ\uf02d\x1ă∡ীಿੁ⧽ៃ̽\x03̛̙ंి᠌ਿ≱⧼ั್ِ͠ू̀̚₊\ue01bಾौើំ়͇̞͔⃗ोृ̬ొು്̻︠ు̟ਂ̢ാॢिௌ̹ुّ‐"

contractions = {"'aight": 'alright', "ain't": 'am not', "amn't": 'am not', "aren't": 'are not', 
                "can't": 'can not', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', 
                "couldn't've": 'could not have', "daren't": 'dare not', "daresn't": 'dare not', 
                "dasn't": 'dare not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', 
                'dunno': "don't know", "d'ye": 'do you', "e'er": 'ever', "everybody's": 'everybody is', 
                "everyone's": 'everyone is', 'finna': 'fixing to', "g'day": 'good day', 'gimme': 'give me', 
                "giv'n": 'given', 'gonna': 'going to', "gon't": 'go not', 'gotta': 'got to', 
                "hadn't": 'had not', "had've": 'had have', "hasn't": 'has not', "haven't": 'have not', 
                "he'd": 'he had', "he'll": 'he will', "he's": 'he is', "he've": 'he have', "how'd": 'how did',
                'howdy': 'how do you do', "how'll": 'how will', "how're": 'how are', "how's": 'how is', 
                "I'd": 'I had', "I'd've": 'I would have', "I'll": 'I will', "I'm": 'I am', 
                "I'm'a": 'I am about to', "I'm'o": 'I am going to', 'innit': 'is it not', "I've": 'I have', 
                "isn't": 'is not', "it'd ": 'it would', "it'll": 'it will', "it's ": 'it is', 
                'iunno': "I don't know", "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', 
                "may've": 'may have', 'methinks': 'me thinks', "mightn't": 'might not', 
                "might've": 'might have', "mustn't": 'must not', "mustn't've": 'must not have', 
                "must've": 'must have', "needn't": 'need not', 'nal': 'and all', "ne'er": 'never', 
                "o'clock": 'of the clock', "o'er": 'over',"ol'": 'old', "oughtn't": 'ought not', "'s": 'is',
                "shalln't": 'shall not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', 
                "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "somebody's": 'somebody has', 
                "someone's": 'someone has', "something's": 'something has', "so're": 'so are', "that'll": 'that will', 
                "that're": 'that are', "that's": 'that is', "that'd": 'that would', "there'd": 'there would', 
                "there'll": 'there will', "there're": 'there are', "there's": 'there is', "these're": 'these are', 
                "they've": 'they have', "this's": 'this is', "those're": 'those are', "those've": 'those have', "'tis": 'it is', 
                "to've": 'to have', "'twas": 'it was', 'wanna': 'want to', "wasn't": 'was not', "we'd": 'we would', 
                "we'd've": 'we would have', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', 
                "what'd": 'what did', "what'll": 'what will', "what're": 'what are', "what's": 'what does', "what've": 'what have',
                "when's": 'when is', "where'd": 'where did', "where'll": 'where will', "where're": 'where are',
                "where's": 'where is',"where've": 'where have', "which'd": 'which would', "which'll": 'which will', 
                "which're": 'which are',"which's": 'which is', "which've": 'which have', "who'd": 'who would',
                "who'd've": 'who would have', "who'll": 'who will', "who're": 'who are', "who'ves": 'who is', "who'": 'who have',
                "why'd": 'why did', "why're": 'why are', "why's": 'why does', "willn't": 'will not', "won't": 'will not',
                'wonnot': 'will not', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have',
                "y'all": 'you all', "y'all'd've": 'you all would have', "y'all'd'n've": 'you all would not have',
                "y'all're": 'you all are', "cause":"because","have't":"have not","cann't":"can not","ain't":"am not",
                "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', 'cannot': 'can not', 
                'wont': 'will not', "You'": 'Am not', "Ain'": 'Am not', "Amn'": 'Am not', "Aren'": 'Are not',
                "Can'": 'Because', "Could'": 'Could have', "Couldn'": 'Could not have', "Daren'": 'Dare not', 
                "Daresn'": 'Dare not', "Dasn'": 'Dare not', "Didn'": 'Did not', "Doesn'": 'Does not', "Don'": "Don't know", 
                "D'": 'Do you', "E'": 'Ever', "Everybody'": 'Everybody is', "Everyone'": 'Fixing to', "G'": 'Give me', 
                "Giv'": 'Going to', "Gon'": 'Got to', "Hadn'": 'Had not', "Had'": 'Had have', "Hasn'": 'Has not', 
                "Haven'": 'Have not', "He'": 'He have', "How'": 'How is', "I'": 'I have', "Isn'": 'Is not', "It'": "I don't know", 
                "Let'": 'Let us', "Ma'": 'Madam', "Mayn'": 'May not', "May'": 'Me thinks', "Mightn'": 'Might not', 
                "Might'": 'Might have', "Mustn'": 'Must not have', "Must'": 'Must have', "Needn'": 'And all', "Ne'": 'Never',
                "O'": 'Old', "Oughtn'": 'Is', "Shalln'": 'Shall not', "Shan'": 'Shall not', "She'": 'She is', 
                "Should'": 'Should have', "Shouldn'": 'Should not have', "Somebody'": 'Somebody has', "Someone'": 'Someone has', 
                "Something'": 'Something has', "So'": 'So are', "That'": 'That would', "There'": 'There is',
                "They'": 'They have', "This'": 'This is', "Those'": 'It is', "To'": 'Want to', "Wasn'": 'Was not',
                "Weren'": 'Were not', "What'": 'What have', "When'": 'When is', "Where'": 'Where have', "Which'": 'Which have', 
                "Who'": 'Who have', "Why'": 'Why does', "Willn'": 'Will not', "Won'": 'Will not', "Would'": 'Would have',
                "Wouldn'": 'Would not have', "Y'": 'You all are',"What's":"What is","What're":"What are","what's":"what is",
                "what're":"what are", "Who're":"Who are", "your're":"you are","you're":"you are", "You're":"You are",
                "We're":"We are", "These'": 'These have', "we're":"we are","Why're":"Why are","How're":"How are ",
                "how're ":"how are ","they're ":"they are ", "befo're":"before","'re ":" are ",'don"t ':"do not", 
                "Won't ":"Will not ","could't":"could not", "would't":"would not", "We'": 'We have',"Hasn't":"Has not",
                "n't":"not", 'who"s':"who is"}

correct_words = dict({"√":" sqrt ","π":" pi ","α":" alpha ","θ":" theta ","∞":" infinity ","∝":" proportional to ","sinx":" sin x ",
                "cosx":" cos x ", "tanx":" tan x ","cotx":" cot x ", "secx":" sec x ", "cosecx":" cosec x ", "£":" pound ", "β":" beta ", 
                "σ": " theta ", "∆":" delta ","μ":" mu ",'∫': " integration ", "ρ":" rho ", "λ":" lambda ","∩":" intersection ",
                "Δ":" delta ", "φ":" phi ", "℃":" centigrade ","≠":" does not equal to ","Ω":" omega ","∑":" summation ","∪":" union ",
                "ψ":" psi ", "Γ":" gamma ","⇒":" implies ","∈":" is an element of ", "≡":" is congruent to ","xⁿ":" x power n",
                "≈":" is approximately equal to ", "~":" is distributed as ","≅":" is isomorphic to ","⩽":" is less than or equal to ",
                "≥":" is greater than or equal to ","⇐":" is implied by ","⇔":" is equivalent to ", "∉":" is not an element of ",
                "∅" : " empty set ", "∛":" cbrt ","÷":" division ","㏒":" log ","∇":" del ","⊆":" is a subset of ","±":" plus–minus ",
                "⊂":" is a proper subset of ","€":" euro ","㏑":" ln ","₹":" rupee ","∀":" there exists ","∆":" delta ","∑":" summation",
                "=":" equal to ","₹":" rupee ","≤":" less than or equal to ", "±":" plus or minus ", "£":" pound ","∝":" propertional to ",
                "¼":" one by four ","&":" and ","™":" trade mark ","½":" one by two ","＄":" dollar ","quorans":"quora",
                "cryptocurrencies":"cryptocurrency","haveheard":"have heard","amafraid":"am afraid","amplanning":"am planning",
                "demonetisation":"demonetization","pokémon":"pokemon","havegot":"have got","amscared":"am scared","qoura":"quora",
                "haveread":"have heard","fiancé":"fiance", "amworried":"am worried","amfeeling":"am feeling","havetried":"have tried", 
                "amwriting":"am writing","havealways":"have always","amconfused":"am confused", "havejust":"have just","amgay":"am gay",
                "amstudying":"am studying","amtalking":"am talking","amdepressed":"am depressed","havenoticed":"have noticed",
                "amdating":"am dating", "x²":"x square","quoras":"quora","amcurious":"am curious","havelost":"have lost",
                "amunable":"am unable", "haverecently":"have recently", "amasking":"am asking", "amsick":"am sick", "clickbait":"click bait", 
                "haveever": "have ever", "amapplying":"am applying", "haveknown":"have known","ampregnant":"am pregnant",
                "haveonly":"have only","amalone":"am alone","havestarted":"have started", "²":"square", "amlearning":"am learning",
                "amconstantly":"am constantly", "amugly":"am ugly", "amstruggling":"am struggling", "amready":"am ready", "são":"sao", 
                "amturning":"am turning", "genderfluid":"gender fluid", "wouldrather":"would rather", "chapterwise":"chapter wise", 
                "undergraduation":"under graduation", "blockchains":"blockchain", "amwondering":"am wondering","havecompleted":"have completed", 
                "amextremely":"am extremely", "amattracted":"am attracted", "amlosing":"am losing", "fiancée":"fiance",
                "amangry":"am angry", "amaddicted":"am addicted", "havegotten":"have gotten", "makaut":"make out", "havegotten":"have gotten", 
                "amyoung":"am young", "amfalling":"am falling", "clichés":"cliches", "beyoncé":"beyonce",
                "erdoğan":"erdogan", "atatürk":"ataturk", "amfinding":"am finding", "ampreparing":"am preparing", "whyis":"why is", 
                "haveused":"have used", "ammarried":"am married",  "2k17":"2017", "cos2x":"cos 2x", "flipcart":"flipkart", 
                "brexit":"britain exit", "havefallen":"have fallen","demonitisation":"demonetization", "microservices":"micro services",
                "amallergic":"am allergic", "amskinny":"am skinny", "amaware":"am aware","amdoing":"am doing","amtired":"am tired",
                "p0rnographic":"pornographic","1st":"first","2nd":"second","3rd":"third","ww2":"www","ps4":"play station four"})

defined_stopwords = ['i','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                    'himself', 'she', "she's", 'her', 'hers','herself', 'it', "it's", 'its', 'itself', 
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                    'who', 'whom', 'this', 'why','that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                    'was', 'were', 'be', 'been', 'the', 'and', 'but', 'if', 'through', 'during', 'before', 
                    'after', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
                    'an', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
                    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                    't', 'u', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
                    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                    "won't", 'wouldn', "wouldn't","would", "could", 'the']

"""mapping all questions text into string and making it lowercase"""
def preprocessing(text,stopwords,contractions,correct_words,punctuation):

    start_time = time.time()
    #removing punctuation
    translate_table = dict((ord(char), None) for char in punctuation) 

    #convert all the words to lower case first and then remove the stopwords
    text = text.map(str)
    for line in range(len(text.values)):
        text.values[line] = text.values[line].lower()
        
    #decontraction
    for idx,val in enumerate(text.values):
        val = ' '.join(word.replace(word,contractions[word]) if word in contractions else word for 
                       word in val.split())
        #generic one
        val = re.sub(r"\'s", " ", val); val = re.sub(r"\''s", " ", val); val = re.sub(r"\"s", " ", val);
        val = re.sub(r"n\''t", " not ", val); val = re.sub(r"n\"t", " not ", val); 
        val = re.sub(r"\'re ", " are ", val); val = re.sub(r"\'d ", " would", val); 
        val = re.sub(r"\''d ", " would", val); val = re.sub(r"\"d ", " would", val);
        val = re.sub(r"\'ll ", " will", val); val = re.sub(r"\''ll ", " will", val); 
        val = re.sub(r"\"ll ", " will", val);val = re.sub(r"\'ve ", " have", val); 
        val = re.sub(r"\''ve ", " have", val); val = re.sub(r"\"ve ", " have", val);
        val = re.sub(r"\'m ", " am", val); val = re.sub(r"\''m "," am", val); 
        val = re.sub(r"\"m "," am", val); val = re.sub("''","",val); val = re.sub("``","",val);
        val = re.sub('"','',val); val = re.sub("̇",'',val); val = re.sub("\s{2}"," ",val)
        
        #replacing correct word with incorrect one
        val = ' '.join(word.replace(word,correct_words[word]) if word in correct_words else word 
                       for word in val.split())
        
        #removing stopwords
        val = ' '.join(e.lower() for e in val.split() if e.lower() not in stopwords)
        
        #Removing special characters
        val = val.translate(translate_table)
        
        #mapping text after above steps
        text.values[idx] = val.strip() 
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    for idx,val in enumerate(text.values):
        sent = ''
        for word in word_tokenize(val):
            sent += lemmatizer.lemmatize(word) + ' '
        text.values[idx] = sent.strip()
    
    hours, rem = divmod(time.time()-start_time, 3600)
    print("Time to process the whole function : {:02}.{:0>2}.{:02} minutes.".format(int(hours),
                                                                            int(divmod(rem, 60)[0]), 
                                                                            int(divmod(rem, 60)[1])))

    return text 


# In[5]:


quora_train['question_text_preprocessed'] = preprocessing(quora_train['question_text'],defined_stopwords,
                                                          contractions,correct_words,punctuation)
quora_test['question_text_preprocessed'] = preprocessing(quora_test['question_text'],defined_stopwords,
                                                         contractions,correct_words,punctuation)


# In[6]:


"""Here I am omitting diversity_score, total_punctuations and total_digit as these features 
    are not kind of indentical for both class."""

start_time = time.time()

"""number of words in the question text"""
quora_train["total_words"] = quora_train["question_text"].apply(lambda sent: len(str(sent).split()))
quora_test["total_words"] = quora_test["question_text"].apply(lambda sent: len(str(sent).split()))

"""number of characters in the question text"""
quora_train["total_chars"] = quora_train["question_text"].apply(lambda sent: len(str(sent)))
quora_test["total_chars"] = quora_test["question_text"].apply(lambda sent: len(str(sent)))

"""total unique words in the question text"""
quora_train["total_unique_words"] = quora_train["question_text"].apply(lambda sent: len(set(str(sent)
                                                                                            .split())))
quora_test["total_unique_words"] = quora_test["question_text"].apply(lambda sent: len(set(str(sent)
                                                                                          .split())))

"""word score in the question text"""
quora_train['word_score'] = quora_train["total_unique_words"]/quora_train["total_words"]
quora_test['word_score'] = quora_test["total_unique_words"]/quora_test["total_words"]

"""total number of stopwords in the question text"""
#gettings stopwords from nltk library
Stopwords = stopwords.words('english')
quora_train["total_stopwords"] = quora_train["question_text"].apply(lambda sent: len([s for s in str(sent)
                                                                                      .lower().split() if s 
                                                                                      in Stopwords]))
quora_test["total_stopwords"] = quora_test["question_text"].apply(lambda sent: len([s for s in str(sent)
                                                                                    .lower().split() if s 
                                                                                    in Stopwords]))

"""total number of UPPERcase words in the question text"""
quora_train["total_upper"] = quora_train["question_text"].apply(lambda sent: len([u for u in str(sent)
                                                                                  .split() if u.isupper()]))
quora_test["total_upper"] = quora_test["question_text"].apply(lambda sent: len([u for u in str(sent)
                                                                                .split() if u.isupper()]))

"""total number of lowercase words in the question text"""
quora_train["total_lower"] = quora_train["question_text"].apply(lambda sent: len([l for l in str(sent)
                                                                                  .split() if l.islower()]))
quora_test["total_lower"] = quora_test["question_text"].apply(lambda sent: len([l for l in str(sent)
                                                                                .split() if l.islower()]))

"""total number of word title in the question text"""
quora_train["total_word_title"] = quora_train["question_text"].apply(lambda sent: len([u for u in 
                                                                                       str(sent).split() 
                                                                                       if u.istitle()]))
quora_test["total_word_title"] = quora_test["question_text"].apply(lambda sent: len([u for u in 
                                                                                     str(sent).split() 
                                                                                     if u.istitle()]))


"""median word length of the question text"""
quora_train["median_word_len"] = quora_train["question_text"].apply(lambda sent: np.median([len(w) 
                                                                                            for w in 
                                                                                            str(sent)
                                                                                            .split()]))
quora_test["median_word_len"] = quora_test["question_text"].apply(lambda sent: np.median([len(w) 
                                                                                          for w in 
                                                                                          str(sent)
                                                                                          .split()]))

"""Truncating Outliers"""
#Total number of words
quora_train['total_words'].loc[quora_train["total_words"] > 60] = 60
quora_test['total_words'].loc[quora_test["total_words"] > 60] = 60
#Total number of characters
quora_train['total_chars'].loc[quora_train["total_chars"] > 250] = 250
quora_test['total_chars'].loc[quora_test["total_chars"] > 250] = 250
#Total number of unique words
quora_train['total_unique_words'].loc[quora_train["total_unique_words"] > 60] = 60
quora_test['total_unique_words'].loc[quora_test["total_unique_words"] > 60] = 60
#Word Score
quora_train['word_score'].loc[quora_train["word_score"] < 0.6] = 0.6
quora_test['word_score'].loc[quora_test["word_score"] < 0.6] = 0.6
#Total number of stopwords
quora_train['total_stopwords'].loc[quora_train["total_stopwords"] > 30] = 30
quora_test['total_stopwords'].loc[quora_test["total_stopwords"] > 30] = 30
#Total number of uppercase word
quora_train['total_upper'].loc[quora_train["total_upper"] > 6] = 6
quora_test['total_upper'].loc[quora_test["total_upper"] > 6] = 6
#Total number of lowercase word
quora_train['total_lower'].loc[quora_train["total_lower"] > 45] = 45
quora_test['total_lower'].loc[quora_test["total_lower"] > 45] = 45
#Total number of word title
quora_train['total_word_title'].loc[quora_train["total_word_title"] > 15] = 15
quora_test['total_word_title'].loc[quora_test["total_word_title"] > 15] = 15
#median_word_length
quora_train['median_word_len'].loc[quora_train["median_word_len"] > 10] = 10
quora_test['median_word_len'].loc[quora_test["median_word_len"] > 10] = 10

#time calculation
hours, rem = divmod(time.time()-start_time, 3600)
print("Time to process the whole function : {:02}.{:0>2}.{:02} minutes.".format(int(hours),
                                                                                int(divmod(rem, 60)[0]), 
                                                                                int(divmod(rem, 60)[1])))


# In[7]:


"""splitting into train and test data"""
y = quora_train['target']
X = quora_train.drop(columns = ['target'])

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.20, stratify=y)
#Printing the shape of train,cv and test dataset
print("The shape of train,cv & test dataset before conversion into vector")
print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(quora_test.shape)


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
start = time.time()

#tokenization function
def tokenize(sentence): 
    tokens = re.sub('[^a-zA-Z0-9]'," ",sentence).split()
    return tokens

#tfidf vectorizer
tfidfvec = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.9, strip_accents='unicode', 
                           tokenizer=tokenize,use_idf=True, smooth_idf=True, sublinear_tf=True)

Xtrain_text = tfidfvec.fit_transform(X_train['question_text_preprocessed'].values.astype(str))
Xcv_text = tfidfvec.transform(X_cv['question_text_preprocessed'].values.astype(str))
Xtest_text = tfidfvec.transform(quora_test['question_text_preprocessed'].values.astype(str))

print("Shape of matrix after one hot encoding:")
print(Xtrain_text.shape, y_train.shape)
print(Xcv_text.shape, y_cv.shape)
print(Xtest_text.shape)

hours, rem = divmod(time.time()-start, 3600)
print("Time to process the whole function : {:02}.{:0>2}.{:02} minutes.".format(int(hours),
                                                                                int(divmod(rem, 60)[0]), 
                                                                                int(divmod(rem, 60)[1])))


# In[9]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
#total_words
normalizer.fit(X_train['total_words'].values.reshape(-1,1))
Xtrain_total_words = normalizer.transform(X_train['total_words'].values.reshape(-1,1))
Xcv_total_words = normalizer.transform(X_cv['total_words'].values.reshape(-1,1))
Xtest_total_words = normalizer.transform(quora_test['total_words'].values.reshape(-1,1))

#total_chars
normalizer.fit(X_train['total_chars'].values.reshape(-1,1))
Xtrain_total_chars = normalizer.transform(X_train['total_chars'].values.reshape(-1,1))
Xcv_total_chars = normalizer.transform(X_cv['total_chars'].values.reshape(-1,1))
Xtest_total_chars = normalizer.transform(quora_test['total_chars'].values.reshape(-1,1))

#total_unique_words
normalizer.fit(X_train['total_unique_words'].values.reshape(-1,1))
Xtrain_total_unique_words = normalizer.transform(X_train['total_unique_words'].values.reshape(-1,1))
Xcv_total_unique_words = normalizer.transform(X_cv['total_unique_words'].values.reshape(-1,1))
Xtest_total_unique_words = normalizer.transform(quora_test['total_unique_words'].values.reshape(-1,1))

#word_score
normalizer.fit(X_train['word_score'].values.reshape(-1,1))
Xtrain_word_score = normalizer.transform(X_train['word_score'].values.reshape(-1,1))
Xcv_word_score = normalizer.transform(X_cv['word_score'].values.reshape(-1,1))
Xtest_word_score = normalizer.transform(quora_test['word_score'].values.reshape(-1,1))

#total_stopwords
normalizer.fit(X_train['total_stopwords'].values.reshape(-1,1))
Xtrain_total_stopwords = normalizer.transform(X_train['total_stopwords'].values.reshape(-1,1))
Xcv_total_stopwords = normalizer.transform(X_cv['total_stopwords'].values.reshape(-1,1))
Xtest_total_stopwords = normalizer.transform(quora_test['total_stopwords'].values.reshape(-1,1))

#total_upper
normalizer.fit(X_train['total_upper'].values.reshape(-1,1))
Xtrain_total_upper = normalizer.transform(X_train['total_upper'].values.reshape(-1,1))
Xcv_total_upper = normalizer.transform(X_cv['total_upper'].values.reshape(-1,1))
Xtest_total_upper = normalizer.transform(quora_test['total_upper'].values.reshape(-1,1))

#total_lower
normalizer.fit(X_train['total_lower'].values.reshape(-1,1))
Xtrain_total_lower = normalizer.transform(X_train['total_lower'].values.reshape(-1,1))
Xcv_total_lower = normalizer.transform(X_cv['total_lower'].values.reshape(-1,1))
Xtest_total_lower = normalizer.transform(quora_test['total_lower'].values.reshape(-1,1))

#total_word_title
normalizer.fit(X_train['total_word_title'].values.reshape(-1,1))
Xtrain_total_word_title = normalizer.transform(X_train['total_word_title'].values.reshape(-1,1))
Xcv_total_word_title = normalizer.transform(X_cv['total_word_title'].values.reshape(-1,1))
Xtest_total_word_title = normalizer.transform(quora_test['total_word_title'].values.reshape(-1,1))

#median_word_len
normalizer.fit(X_train['median_word_len'].values.reshape(-1,1))
Xtrain_median_word_len = normalizer.transform(X_train['median_word_len'].values.reshape(-1,1))
Xcv_median_word_len = normalizer.transform(X_cv['median_word_len'].values.reshape(-1,1))
Xtest_median_word_len = normalizer.transform(quora_test['median_word_len'].values.reshape(-1,1))


# In[10]:


from scipy.sparse import hstack
#stacking all features
Xtrain_quora = hstack((Xtrain_total_words, Xtrain_total_chars, Xtrain_total_unique_words, Xtrain_word_score,
                    Xtrain_total_stopwords, Xtrain_total_upper, Xtrain_total_lower, Xtrain_total_word_title,
                    Xtrain_median_word_len, Xtrain_text)).tocsr()

Xcv_quora = hstack((Xcv_total_words, Xcv_total_chars, Xcv_total_unique_words, Xcv_word_score,
                    Xcv_total_stopwords, Xcv_total_upper, Xcv_total_lower, Xcv_total_word_title,
                    Xcv_median_word_len, Xcv_text)).tocsr()

Xtest_quora =hstack((Xtest_total_words, Xtest_total_chars, Xtest_total_unique_words, Xtest_word_score,
                    Xtest_total_stopwords, Xtest_total_upper, Xtest_total_lower, Xtest_total_word_title,
                    Xtest_median_word_len, Xtest_text)).tocsr()


# In[11]:


#https://lightgbm.readthedocs.io/en/latest/Parameters.html
start = time.time()
import lightgbm as lgbm
#LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, 
#objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, 
#colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1, silent=True, importance_type='split', **kwargs)
lgb = lgbm.LGBMClassifier(boosting_type='gbdt', objective="binary", metric="auc", boost_from_average=False,max_depth=-1,
                          learning_rate=0.3, max_bin=100, num_leaves=31, bagging_fraction = 0.8, feature_fraction = 0.8,
                          scale_pos_weight = 1, num_threads = 32, verbosity = 0, n_jobs=- 1)

values = [round(0.2 * x,2) for x in range(1, 6)]
param_grid = {'min_gain_to_split': values, 'lambda_l1':values, 'lambda_l2':values,
              "n_estimators":[100,200,500,1000]}
clf = RandomizedSearchCV(lgb, param_grid, scoring='f1',return_train_score=True, verbose=5, n_jobs=-1)
clf.fit(Xtrain_quora, y_train)

print("Best cross-validation score: {:.2f}".format(clf.best_score_))
print("Best parameters: ", clf.best_params_)


lgb = lgbm.LGBMClassifier(boosting_type='gbdt', objective="binary", metric="auc", boost_from_average=False, 
                          max_depth = -1,learning_rate=0.3, max_bin=100, num_leaves=31, bagging_fraction = 0.8, 
                          feature_fraction = 0.8,min_gain_to_split = clf.best_params_['min_gain_to_split'],
                          lambda_l1 = clf.best_params_['lambda_l1'],lambda_l2 = clf.best_params_['lambda_l2'],
                          n_estimators = clf.best_params_['n_estimators'], scale_pos_weight = 1, num_threads = 32,
                          verbosity = 0)
lgb.fit(Xtrain_quora,y_train)
y_pred=lgb.predict(Xcv_quora)

print("----LightGBM----")
print("Overall f1 score:",round((metrics.f1_score(y_cv,y_pred)),2))
print("Overall Precision:",round((metrics.precision_score(y_cv,y_pred)),2))
print("Overall Recall:",round((metrics.recall_score(y_cv,y_pred)),2))
print("Classification Report:\n",metrics.classification_report(y_cv,y_pred))

fig, ax = plot_confusion_matrix(conf_mat=metrics.confusion_matrix(y_cv,y_pred), figsize=(5, 5))
pt.show()

#predicting output
ytestPred = lgb.predict(Xtest_quora)
ytestPred = (ytestPred>0.25).astype(int)
quora_test = pd.DataFrame({'qid':quora_test['qid'].values})
quora_test['prediction'] = ytestPred
print("Quora Test Output:\n",quora_test['prediction'].value_counts())

hours, rem = divmod(time.time()-start, 3600)
print("Time to process the whole function : {:02}.{:0>2}.{:02} minutes.".format(int(hours),
                                                                                int(divmod(rem, 60)[0]), 
                                                                                int(divmod(rem, 60)[1])))


# In[12]:


#Submitting values
quora_test.to_csv('submission.csv', index=False)

