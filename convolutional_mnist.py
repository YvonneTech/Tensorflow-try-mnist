import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import array2image

# get data from mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28 x 28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # (tensor, shape)

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])                    # patch 5 x 5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                             # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])                   # patch 5 x 5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                             # output size 7x7x64

## conv3 layer ##
W_conv3 = weight_variable([5, 5, 64, 128])                  # patch 5 x 5, in size 64, out size 128
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)    # output size 7x7x128
h_pool3 = max_pool_2x2(h_conv3)                             # output size 4x4x128


W_fc1 = weight_variable([4*4*128, 10])
b_fc1 = bias_variable([10])
h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
prediction = tf.nn.softmax(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
