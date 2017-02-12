'''
Cai Shaofeng - 2017.2
Implementation of Data Preprocessing Utilities
'''

import numpy as np
import scipy.io as sio

''' package for data preprocessing
    suppose input is np matrix with shape (m, n)
    m rows -> m samples
    n columes -> n features
'''

# load the training & testing data
def loadData(trainPerc=0.7):
    samples1 = np.loadtxt('data/train.csv', dtype='int32', delimiter=',', skiprows=1,
                          converters={1: lambda char: ord(char) - ord('a')})
    samples2 = np.loadtxt('data/test.csv', dtype='int32', delimiter=',', skiprows=1,
                          converters={1: lambda char: ord(char) - ord('a')})
    samples = np.vstack((samples1, samples2))
    testIndex = np.arange(0, samples.shape[0] - 1)
    np.random.shuffle(testIndex)
    trainNum = int(trainPerc * samples.shape[0])  # 70 percent train number
    trainX, trainY = samples[testIndex[:trainNum], 4:], samples[testIndex[:trainNum], 1]
    testX, testY = samples[testIndex[trainNum:], 4:], samples[testIndex[trainNum:], 1]
    return trainX, trainY, testX, testY

# data transformation routines
def zNormarlization(matrix):
    mean, std = np.mean(matrix, axis=0), np.std(matrix, axis=0)
    return (matrix - mean) / std

def logTransform(matrix):
    return np.log(matrix + 0.1)

def binarization(matrix):
    matrix[matrix!=0.0] = 1
    return matrix

def toFullIndices(matrix, lenght):
    fullMatrix = np.zeros(shape=(matrix.shape[0], lenght))
    for row in xrange(matrix.shape[0]):
        fullMatrix[row][matrix[row]] = 1
    return fullMatrix

