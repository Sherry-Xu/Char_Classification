#   author  :   Shawn
#   data    :   2017.2.12

import numpy as np
from sklearn.linear_model import LogisticRegression
from DataPreprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale


trainX, trainY, testX, testY = loadData(trainPerc=0.7)
trainX, testX = scale(trainX, axis=0), scale(testX, axis=0)

parameters = {
    'multi_class': ['ovr'], #'multinomial', 'ovr'
    'C': [1],   #100, 10, 1, 0.1, 1e-2, 1e-3
    'penalty': ['l2'],
    'solver': ['sag']
}

LR = LogisticRegression( n_jobs=-1, tol=1e-4, max_iter=2000)
clf = GridSearchCV(LR, parameters, cv=4, verbose=True, n_jobs=-1)
clf.fit(trainX, trainY)

print clf.score(trainX, trainY)
print clf.score(testX, testY)

print clf.best_score_
print clf.best_params_

'''
0.781970865467
0.755338718708
0.75969151671
{'penalty': 'l2', 'multi_class': 'ovr', 'C': 1, 'solver': 'sag'}
'''
