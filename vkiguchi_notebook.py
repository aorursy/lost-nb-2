#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from scipy import stats as st
import os


# In[ ]:


os.listdir("../input/spamham")


# In[ ]:


filepath = "../input/spamham/train_data.csv"


# In[ ]:


bag = pd.read_csv(filepath, sep=r'\s*,\s*', engine='python', na_values='?')


# In[ ]:


bag.head()


# In[ ]:


pearson = bag.drop("Id", axis="columns").corr()["ham"].drop('ham')


# In[ ]:


pearson.plot(kind='bar')


# In[ ]:


pearson.sort_values()


# In[ ]:


bfeats = [x for x in pearson.index if abs(pearson[x])>0.16]
bfeats


# In[ ]:


pearson = pearson.drop(['capital_run_length_average','capital_run_length_longest', 'capital_run_length_total'])
psons = pd.DataFrame(pearson ,columns=['ham', 'ham_self']).copy()
psons = psons.apply(lambda x:x)

for label in pearson.axes[0]:
    psons["ham_self"][label] = bag[bag[label].apply(lambda x: abs(x)>0)].drop("Id", axis="columns").corr()[label]['ham']
    #correlation given that the value is non-zero


# In[ ]:


psons.index


# In[ ]:


print(psons.ham_self.sort_values())
features = [x for x in psons.index if abs(psons.ham_self[x])>0.15]
print(features)
print(len(features))


# In[ ]:


#bag["word_freq_cs_bool"] = bag["word_freq_cs"].transform(bool)
#bfeats.append("word_freq_cs_bool")


# In[ ]:


spam = bag.apply(lambda x: x)
spam['spam'] = spam['ham'].apply(lambda x: not x)
ham = bag.apply(lambda x: x)
#for label in ['capital_run_length_average','capital_run_length_longest', 'capital_run_length_total']:
#    ham[label] = bag[label] * spam['ham']
#    spam[label] = bag[label] * spam['spam']


# In[ ]:


spam = spam[spam['spam']]
ham = ham[ham['ham']]


# In[ ]:


spam.head()


# In[ ]:


distSpam = pd.DataFrame()
distSpam['u'] = spam.drop(['Id', 'ham', 'spam'], axis='columns').mean()
distSpam['s'] = spam.drop(['Id', 'ham', 'spam'], axis='columns').std()
distSpam['k'] = distSpam['u']*distSpam['u']/(distSpam['s']*distSpam['s'])
distSpam['theta'] = (distSpam['s']*distSpam['s'])/distSpam['u']
distSpam


# In[ ]:


ham.head()


# In[ ]:


distHam = pd.DataFrame()
distHam['u'] = ham.drop(['Id', 'ham'], axis='columns').mean()
distHam['s'] = ham.drop(['Id', 'ham'], axis='columns').std()
distHam['k'] = distHam['u']*distHam['u']/(distHam['s']*distHam['s'])
distHam['theta'] = (distHam['s']*distHam['s'])/distHam['u']
distHam


# In[ ]:


def _histo(dataHam, dataSpam, n_bins):
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].set_xlabel("Ham")
    axs[0].hist(dataHam, bins=n_bins)
    axs[1].set_xlabel("Spam")
    axs[1].hist(dataSpam, bins=n_bins)
def histo(name, n_bins):
    _histo(ham[name], spam[name], n_bins)
def histon(name, n_bins):
    if(not histon.ham_given_n):
        histon.ham_given_n = ham[ham.apply(lambda x: abs(x)>0)]
    if(not histon.spam_given_n):
        histon.spam_given_n = spam[spam.apply(lambda x: abs(x)>0)]
    _histo(histon.ham_given_n[name], histon.spam_given_n[name], n_bins)
histon.ham_given_n = []
histon.spam_given_n = []


# In[ ]:


histo("capital_run_length_total", 100)
histo("capital_run_length_average", 100)
histo("capital_run_length_longest", 100)
histo("char_freq_$", 50)
#histon("char_freq_$", 50)
histo("word_freq_re", 50)
histo("word_freq_table", 50)


# In[ ]:


diffDistrib = ((distSpam['u']-distHam['u'])/distHam['s'])
print(diffDistrib.sort_values())
cutoff = 0.02
bigDiff = [label for label in bag.columns if abs(diffDist[label])>cutoff and label not in ["capital_run_length_total", "capital_run_length_average", "capital_run_length_longest"]
diffDist = diffDistrib/diffDistrib.apply(abs)


# In[ ]:


#z = 0 #0.50
#z = st.norm.ppf(0.75)
#z = 0.84 #0.80
#z = 1.28 #0.90
z = 1.645 #0.95
#zlist = [st.norm.ppf(0.85), st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.99)]
#zlist = [st.norm.ppf(0.9), st.norm.ppf(0.95),st.norm.ppf(0.975) , st.norm.ppf(0.99)]
zlist = [0.9, 0.95, 0.975, 0.99]


# In[ ]:


#li = 0 #0.5
#li = 0.25 #0.6
#li = 1.28 #0.90
#li = 1.645 #0.95
#li = 1.96 #0.975
#llist = [st.norm.ppf(0.85), st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.99)]
###llist = [st.norm.ppf(0.855), st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.99), st.norm.ppf(0.995)]
#llist = [st.norm.ppf(0.9), st.norm.ppf(0.95), st.norm.ppf(0.975), st.norm.ppf(0.99)]
llist = [0.855, 0.9, 0.95, 0.99, 0.995]
#testlist = [st.norm.ppf(0.80), st.norm.ppf(0.825), st.norm.ppf(0.85), st.norm.ppf(0.875), st.norm.ppf(0.9), st.norm.ppf(0.925), st.norm.ppf(0.95), st.norm.ppf(0.975), st.norm.ppf(0.99)]


# In[ ]:


Xbag = bag.drop(["Id", "ham"], axis='columns')


# In[ ]:


XbagCapCut = bag.drop(["Id"], axis='columns')

capital_run = ["capital_run_length_total", "capital_run_length_average", "capital_run_length_longest"]
print("Hello")
print(capital_run)
appendee = []
for label in capital_run:
    for li in llist:
        #XbagCapCut[label+str(int(li*100))] = XbagCapCut[label].transform(lambda x, u, s: (x-u)/s > li, u=distHam['u'][label], s = distHam['s'][label])
        #XbagCapCut[label+str(int(st.norm.ppf(li)*100))] = XbagCapCut[label].transform(lambda x, u, s: (x-u)/s > st.norm.ppf(li), u=distHam['u'][label], s = distHam['s'][label])
        #XbagCapCut[label+str(int(li*1000))] = XbagCapCut[label].transform(lambda x, g: x>g ,g=st.gamma.ppf(li, distHam['k'][label], loc=distHam['u'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        XbagCapCut[label+str(int(li*1000))] = XbagCapCut[label].transform(lambda x, g: x>g ,g=st.gamma.ppf(li, distHam['k'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        #appendee.append(label+str(int(st.norm.ppf(li)*100)))
        appendee.append(label+str(int(li*1000)))
capital_run.extend(appendee)
print(capital_run)
XbagCapCut = XbagCapCut.drop(["ham"], axis='columns')


# In[ ]:


#for label in XbagCapCut.drop(capital_run, axis='columns').columns:
for label in bigDiff:
    for z in zlist:
        #XbagCapCut[label+"null_hip"+str(int(z*100))] = XbagCapCut[label].transform(lambda x, u, s: (x-u)/s > z, u=distHam['u'][label], s=distHam['s'][label]*diffDist[label])
        #XbagCapCut[label+"null_hip"+str(int(z*1000))] = XbagCapCut[label].transform(lambda x, g: x>g, g=st.gamma.ppf(z, distHam['k'][label], loc=distHam['u'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        XbagCapCut[label+"null_hip"+str(int(z*1000))] = XbagCapCut[label].transform(lambda x, g: x>g, g=st.gamma.ppf(z, distHam['k'][label], scale=distHam['theta'][label]**distHam['k'][label]))
    #XbagCapCut[label+"p_val"] = XbagCapCut[label].transform(lambda x: st.norm.cdf(-1*(x-distHam['u'][label])/distHam['s'][label]*diffDist[label]) > st.norm.cdf((x-distSpam['u'][label])/distSpam['s'][label]*diffDist[label]))
    #XbagCapCut[label+"p_val"] = XbagCapCut[label].transform(lambda x, uHam, sHam, uSpam, sSpam, diffD: st.norm.cdf(-1*(x-uHam)/sHam*diffD) > st.norm.cdf((x-uSpam)/sSpam*diffD), uHam=distHam['u'][label], sHam=distHam['s'][label], uSpam=distSpam['u'][label], sSpam=distSpam['s'][label], diffD=diffDist[label])
    XbagCapCut[label+"p_val"] = XbagCapCut[label].transform(lambda x, kHam, thetaHam, kSpam, thetaSpam: 1-st.gamma.cdf(x, kHam, scale=thetaHam**kHam) > st.gamma.cdf(x, kSpam, scale=thetaSpam**kSpam), kHam=distHam['k'][label], thetaHam=distHam['theta'][label], kSpam=distSpam['k'][label], thetaSpam=distSpam['theta'][label])


# In[ ]:


#XbagCapCut["word_freq_cs_bool"] = bag["word_freq_cs"].transform(bool)
#bfeats.append("word_freq_cs_bool")


# In[ ]:


XbagCapCut


# In[ ]:


Ybag = bag['ham']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


# In[ ]:


testpath = "../input/spamham/test_features.csv"


# In[ ]:


testBag = pd.read_csv(testpath, sep=r'\s*,\s*', engine='python', na_values='?')


# In[ ]:


testBag


# In[ ]:


XtestBagCapCut = testBag.drop("Id", axis='columns')

capital_run = ["capital_run_length_total", "capital_run_length_average", "capital_run_length_longest"]
for label in capital_run:
    for li in llist:
        #XtestBagCapCut[label+str(int(li*100))] = XtestBagCapCut[label].transform(lambda x, u, s: (x-u)/s > li, u=distHam['u'][label], s = distHam['s'][label])
        #XtestBagCapCut[label+str(int(st.norm.ppf(li)*100))] = XtestBagCapCut[label].transform(lambda x, u, s: (x-u)/s > st.norm.ppf(li), u=distHam['u'][label], s = distHam['s'][label])
        #XtestBagCapCut[label+str(int(li*1000))] = XtestBagCapCut[label].transform(lambda x, g: x>g ,g=st.gamma.ppf(li, distHam['k'][label], loc=distHam['u'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        XtestBagCapCut[label+str(int(li*1000))] = XtestBagCapCut[label].transform(lambda x, g: x>g ,g=st.gamma.ppf(li, distHam['k'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        #appendee.append(label+str(int(st.norm.ppf(li)*100)))
        appendee.append(label+str(int(li*1000)))
capital_run.extend(appendee)
#for label in XtestBagCapCut.drop(capital_run, axis='columns').columns:
for label in bigDiff:
    for z in zlist:
        #XtestBagCapCut[label+"null_hip"+str(int(z*100))] = XtestBagCapCut[label].transform(lambda x, u, s: (x-u)/s > z, u=distHam['u'][label], s=distHam['s'][label]*diffDist[label])
        #XtestBagCapCut[label+"null_hip"+str(int(z*1000))] = XtestBagCapCut[label].transform(lambda x, g: x>g, g=st.gamma.ppf(z, distHam['k'][label], loc=distHam['u'][label], scale=distHam['theta'][label]**distHam['k'][label]))
        XtestBagCapCut[label+"null_hip"+str(int(z*1000))] = XtestBagCapCut[label].transform(lambda x, g: x>g, g=st.gamma.ppf(z, distHam['k'][label], scale=distHam['theta'][label]**distHam['k'][label]))
    #XtestBagCapCut[label+'p_val'] = XtestBagCapCut[label].transform(lambda x: st.norm.cdf(-1*(x-distHam['u'][label])/distHam['s'][label]*diffDist[label]) > st.norm.cdf((x-distSpam['u'][label])/distSpam['s'][label]*diffDist[label]))
    XtestBagCapCut[label+"p_val"] = XtestBagCapCut[label].transform(lambda x, kHam, thetaHam, kSpam, thetaSpam: 1-st.gamma.cdf(x, kHam, scale=thetaHam**kHam) > st.gamma.cdf(x, kSpam, scale=thetaSpam**kSpam), kHam=distHam['k'][label], thetaHam=distHam['theta'][label], kSpam=distSpam['k'][label], thetaSpam=distSpam['theta'][label])


# In[ ]:


#XtestBagCapCut["word_freq_cs_bool"] = XtestBagCapCut["word_freq_cs"].transform(bool)


# In[ ]:


XtestBagCapCut


# In[ ]:


def f3(estimador, x, y):
    y_pred = estimador.predict(x)
    return fbeta_score(y, y_pred, 3)
#f3 = (lambda estimator, x, y: fbeta_score(y, estimador.predict(x), 3))


# In[ ]:


gnb = GaussianNB()
#scores = cross_val_score(gnb, Xbag, Ybag, cv=10, scoring='f1')
scores = cross_val_score(gnb, Xbag, Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())
#gnb.fit(Xbag, Ybag)
#predictBag = testBag.apply(lambda x:x)
#predictBag['ham'] = gnb.predict(testBag.drop('Id', axis='columns'))
#predictBag


# In[ ]:


#features = bfeats
knn = KNeighborsClassifier(n_neighbors=3)
#scores = cross_val_score(knn, XbagCapCut[bfeats], Ybag, cv=10, scoring='f1')
scores = cross_val_score(knn, XbagCapCut[bfeats], Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())
#knn.fit(Xbag,Ybag)
#predictBag = testBag.apply(lambda x:x)
#predictBag['ham'] = knn.predict(testBag.drop('Id', axis='columns'))
#predictBag


# In[ ]:


mnnb = MultinomialNB()
#scores = cross_val_score(mnnb, XbagCapCut, Ybag, cv=10, scoring='f1')
scores = cross_val_score(mnnb, XbagCapCut, Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())
#mnnb.fit(XbagCapCut, Ybag)
#predictBag = testBag.apply(lambda x:x)
#predictBag['ham'] = mnnb.predict(testBag.drop('Id', axis='columns'))
#predictBag['ham'] = mnnb.predict(XtestBagCapCut)
#predictBag


# In[ ]:


bnnb = BernoulliNB()
#scores = cross_val_score(bnnb, Xbag, Ybag, cv=10, scoring='f1')
scores = cross_val_score(bnnb, Xbag, Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())
#bnnb.fit(Xbag, Ybag)
#predictBag = testBag.apply(lambda x:x)
#predictBag['ham'] = bnnb.predict(testBag.drop('Id', axis='columns'))
#predictBag


# In[ ]:


#maxscorei = 0
#maxscore = 0
#for i in range(50, 100):
#    i=i/100
#    bnnbprior = BernoulliNB(class_prior = [1-i, i])
#    score = cross_val_score(bnnbprior, XbagCapCut, Ybag, cv=10, scoring=f3).mean()
#    if(score>maxscore):
#        maxscore = score
#        maxscorei = i
#print(maxscore, maxscorei)
#for j in range(-99,100):
#    j = maxscorei+j/10000
#    bnnbprior = BernoulliNB(class_prior = [1-j, j])
#    score = cross_val_score(bnnbprior, XbagCapCut, Ybag, cv=10, scoring=f3).mean()
#    if(score>maxscore):
#        maxscore = score
#        maxscorej = j
#print(maxscore, maxscorej)


# In[ ]:


bnnb = BernoulliNB()
XbagCapCut['dummy'] = bag['ham']
XtestBagCapCut['dummy'] = bag['ham'].apply(lambda x: False)
XbagCapCut['dummy2'] = bag['ham']
XtestBagCapCut['dummy2'] = bag['ham'].apply(lambda x: False)

#scores = cross_val_score(bnnb, XbagCapCut, Ybag, cv=10, scoring='f1')
scores = cross_val_score(bnnb, XbagCapCut, Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())

prop = np.count_nonzero(bag['ham'])/bag['ham'].count()
print(prop)
#bnnbprior = BernoulliNB(class_prior = [1-prop, prop])
bnnbprior = BernoulliNB(class_prior = [1-(0.999999999999), 0.999999999999])
bnnbprior2 = BernoulliNB(class_prior = [1-(0.9999999999), 0.9999999999])
scores = cross_val_score(bnnbprior, XbagCapCut.drop(['dummy', 'dummy2'], axis='columns'), Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())
scores = cross_val_score(bnnbprior2, XbagCapCut, Ybag, cv=10, scoring=f3)
print(scores)
print(scores.mean())

bnnb.fit(XbagCapCut, Ybag)
predictBag = testBag.apply(lambda x:x)
predictBag['ham'] = bnnb.predict(XtestBagCapCut)

bnnbprior.fit(XbagCapCut.drop(['dummy', 'dummy2'], axis='columns'), Ybag)
predictBag1 = testBag.apply(lambda x:x)
predictBag1['ham'] = bnnbprior.predict(XtestBagCapCut.drop(['dummy', 'dummy2'], axis='columns'))
print(np.count_nonzero(predictBag1['ham'])/predictBag1['ham'].count())
print(predictBag1.head()['ham'])

bnnbprior2.fit(XbagCapCut, Ybag)
predictBag2 = testBag.apply(lambda x:x)
predictBag2['ham'] = bnnbprior2.predict(XtestBagCapCut)
print(np.count_nonzero(predictBag2['ham'])/predictBag2['ham'].count())
print(predictBag2.head()['ham'])


# In[ ]:


savepath = "bagPredict.csv"
savepath1 = "bagPredict1.csv"
savepath2 = "bagPredict2.csv"


# In[ ]:


predictBag.to_csv(savepath, index=False, columns=['Id', 'ham'])


# In[ ]:


predictBag1.to_csv(savepath1, index=False, columns=['Id', 'ham'])


# In[ ]:


predictBag2.to_csv(savepath2, index=False, columns=['Id', 'ham'])

