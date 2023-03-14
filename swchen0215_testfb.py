# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv( '../input/train.csv' )
print( train.head() )

test = pd.read_csv( '../input/test.csv' )
print( test.head() )

X_train = train[['x', 'y', 'time']].values[:100000]
y_train = train['place_id'].values[:100000]
weights_train = train['accuracy'].values[:100000]

X_test = test[['x', 'y', 'time']].values
weights_test = test['accuracy'].values


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors"] 
classifiers = [
    KNeighborsClassifier(3, n_jobs = -1 )
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
#    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    AdaBoostClassifier(),
#    GaussianNB(),
#    LinearDiscriminantAnalysis(),
#    QuadraticDiscriminantAnalysis()
    ]

#figure = plt.figure(figsize=(27, 9))

# iterate over classifiers
for name, clf in zip(names, classifiers):
#    clf.fit(X_train, y_train, weights_train )
    clf.fit(X_train, y_train )
    score = clf.score(X_train, y_train)

    print '\n\n===== ' + name + ' ======'
    print 'score: ', score
    
    pred_prob = clf.predict_proba(X_train)
    predicts = [ np.argsort( row )[::-1][:3] for row in pred_prob ]

  

