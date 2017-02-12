#   author  :   Shawn
#   data    :   2016.10.4

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

# mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print type(mnist)
print mnist.test.images

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

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
W_conv1 = weight_variable([5, 5, 1, 32])    #1, 2, patch size, 3 # of input channels, 4, # of output channels
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])     #1, ## 2, 3 shape of image, 4, # of color channels

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# weight and bias for 2nd conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 1st FC layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd (output) FC layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

## traing part
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for step in xrange(1000):   #5000
    batch = mnist.train.next_batch(50)
    if step % 100 == 0:
        train_accurary = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %.5f" %(step, train_accurary))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %.5f" %(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


# saving model parameters
def model_saving(file_name, paras):
    modelFile = open(file_name, 'wb')
    print('saving model parameters into %s...\n' %file_name)

    for para in paras:
        cPickle.dump(para, modelFile, -1)

    modelFile.close()
# saving model parameters into cnnMnist.pickle
# model_saving('cnnMnist.pickle', [W_conv1])


