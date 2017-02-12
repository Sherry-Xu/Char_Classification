#   author  :   Shawn
#   data    :   2016.10.4

import tensorflow as tf
from DataPreprocessing import *

trainX, trainY, testX, testY = loadData(trainPerc=0.8)
trainY, testY = toFullIndices(trainY, 26), toFullIndices(testY, 26)

batchSize = 200
imageSize, outputSize, imageShape = 128, 26, [16, 8]
convSize1, convSize2, kernelNum1, kernelNum2, fcSize1, fcSize2 = 5, 3, 6, 16, 128, 84


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, imageSize])
y_ = tf.placeholder(tf.float32, shape=[None, outputSize])

# weight initialization for weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# conv & pooling layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# weight and bias for 1st conv layer
W_conv1 = weight_variable([convSize1, convSize1, 1, kernelNum1])    #1, 2, patch size, 3 # of input channels, 4, # of output channels
b_conv1 = bias_variable([kernelNum1])

x_image = tf.reshape(x, [-1, imageShape[0], imageShape[1], 1])     #1, ## 2, 3 shape of image, 4, # of color channels

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# weight and bias for 2nd conv layer
W_conv2 = weight_variable([convSize2, convSize2, kernelNum1, kernelNum2])   #1, 2, patch size, 3 # of input channels, 4, # of output channels
b_conv2 = bias_variable([kernelNum2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 1st FC layer
W_fc1 = weight_variable([(imageShape[0] / 4) * (imageShape[1] / 4) * kernelNum2, fcSize1])
b_fc1 = bias_variable([fcSize1])

h_pool2_flat = tf.reshape(h_pool2, [-1, (imageShape[0] / 4) * (imageShape[1] / 4) * kernelNum2])    # Fc size 512 * 128, 128 * 26
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 2nd FC layer
W_fc2 = weight_variable([fcSize1, fcSize2])
b_fc2 = bias_variable([fcSize2])

h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# dropout
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


# 3nd (output) FC layer
W_fc3 = weight_variable([fcSize2, outputSize])
b_fc3 = bias_variable([outputSize])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

try:
    ## traing part
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for step in xrange(50000):   #5000
        randomIndex = np.random.random_integers(0, trainX.shape[0] - 1, size=(batchSize,))  # mini batch size 100
        batch_xs, batch_ys = trainX[randomIndex], trainY[randomIndex]
        if step % 100 == 0:
            train_accurary = accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0})
            print("epoc: %3d\tstep: %8d\ttraining accuracy: %.6f" %(int(step*batchSize/trainX.shape[0]), step, train_accurary))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
except:
    pass

print("train accuracy %.5f" %(accuracy.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0})))
print("test accuracy %.5f" %(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0})))




'''
train accuracy 0.71942
test accuracy 0.70796

batchSize = 200
imageSize, outputSize, imageShape = 128, 26, [16, 8]
convSize1, convSize2, kernelNum1, kernelNum2, fcSize1, fcSize2 = 5, 3, 6, 16, 128, 84
train accuracy 0.80190
test accuracy 0.71390
'''