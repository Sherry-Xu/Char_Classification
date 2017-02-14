#   author  :   Shawn
#   data    :   2017.2.12


# Import data
import numpy as np
from  DataPreprocessing import *
import tensorflow as tf

def main(_):
    trainX, trainY, testX, testY = loadData(trainPerc=0.7)
    #trainX, testX = zNormarlization(trainX), zNormarlization(testX)
    trainY, testY = toFullIndices(trainY, 26), toFullIndices(testY, 26)
    #print trainX.shape, trainY.shape (15645, 128) (15645,)

    imageSize, hiddenSize, outputSize = 128, 50, 26
    batchSize = 100

    # Create the model
    x = tf.placeholder(tf.float32, [None, imageSize])
    W1 = tf.Variable(tf.random_normal([imageSize, hiddenSize]))
    b1 = tf.Variable(tf.random_normal([hiddenSize]))
    h1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([hiddenSize, outputSize]))
    b1 = tf.Variable(tf.random_normal([outputSize]))
    y = tf.matmul(h1, W2) + b1

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 26])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()
    try:
        for step in range(20000):
            randomIndex = np.random.random_integers(0, trainX.shape[0]-1, size=(batchSize, )) # mini batch size 100
            batch_xs, batch_ys = trainX[randomIndex], trainY[randomIndex]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if step % 100 == 0:
            # Test trained model
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("epoc: %3d\tstep: %8d\ttraining accuracy: %.6f" % (int(step * batchSize / trainX.shape[0]),
                    step, sess.run(accuracy, feed_dict={x: testX, y_: testY})))
    except:
        pass

    print("training accuracy: %.6f" % (sess.run(accuracy, feed_dict={x: trainX, y_: trainY})))
    print("testing accuracy: %.6f" % (sess.run(accuracy, feed_dict={x: testX, y_: testY})))

if __name__ == '__main__':
  tf.app.run()


'''
training accuracy: 0.861937
testing accuracy: 0.787811
'''