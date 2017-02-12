import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from DataPreprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale


trainX, trainY, testX, testY = loadData(trainPerc=0.3)
trainX, testX = scale(trainX, axis=0), scale(testX, axis=0)

parameters = {
    'n_neighbors': [3],
    'algorithm': ['auto'], #'auto', 'ball_tree', 'kd_tree', 'brute'
}


neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters, cv=3, verbose=True, n_jobs=-1)
clf.fit(trainX, trainY)

print clf.score(trainX, trainY)
print clf.score(testX, testY)

print clf.best_score_
print clf.best_params_


'''
0.762096516459
0.575357475483
0.560562480026
{'n_neighbors': 3, 'algorithm': 'auto'}
'''