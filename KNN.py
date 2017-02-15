#   author  :   Shawn
#   data    :   2017.2.12

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from DataPreprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale


trainX, trainY, testX, testY = loadData(trainPerc=0.3)
trainX, testX = scale(trainX, axis=0), scale(testX, axis=0)

parameters = {
    'n_neighbors': [3, 5, 10, 20, 50], #3, 5, 10, 20
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], #'auto', 'ball_tree', 'kd_tree', 'brute'
}


neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters, cv=4, verbose=True, n_jobs=-1)
clf.fit(trainX, trainY)

print clf.score(trainX, trainY)
print clf.score(testX, testY)

print clf.best_score_
print clf.best_params_


'''
0.840358314005
0.760754070266
0.749180196753
{'n_neighbors': 5, 'algorithm': 'auto'}
'''