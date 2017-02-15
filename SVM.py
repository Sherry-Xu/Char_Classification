#   author  :   Shawn
#   data    :   2017.2.12

import numpy as np
from DataPreprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn import svm


trainX, trainY, testX, testY = loadData(trainPerc=0.7)


parameters = {
    'kernel': ('linear', 'rbf'),    #'linear', 'rbf'
    'C': [1],   #100, 10, 1, 0.1, 1e-2, 1e-3
    'decision_function_shape': ['ovo', 'ovr']
}

#svr = svm.SVC()
#clf = GridSearchCV(svr, parameters, cv=3, verbose=True)
#clf.fit(trainX, trainY)

clf = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovr', verbose=True)
clf.fit(trainX, trainY)

print clf.score(trainX, trainY)
print clf.score(testX, testY)

# print clf.best_score_
# print clf.best_params_


'''
0.84205655527
0.826841558026
'''