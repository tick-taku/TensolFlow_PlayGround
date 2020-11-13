import input_data as idata
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = idata.read_data_sets('MNIST_data', one_hot = True)
session = tf.InteractiveSession()

x = tf.placeholder("float", [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float", [None, 10])


cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tff.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(session.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.testlabels}))
