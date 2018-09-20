#coding:utf_8

'''在mnist1的基础上加上正则项'''

import tensorflow as tf

'''run this code to download the mnist code'''
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#difine variable and graph

x_input = tf.placeholder(tf.float32, shape=[None,784])
weight = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10]))
y_output = tf.nn.softmax(tf.matmul(x_input,weight) + bias)

#train loss function and gradient decent
y_real = tf.placeholder(tf.float32,shape=[None,10])

#loss
#修改了失真函数，天添加了ｌ2正则，没有效果呀
cross_entropy = -tf.reduce_sum( y_real* tf.log(y_output)) + tf.contrib.layers.l2_regularizer(0.01)(weight)
#train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#init all variable
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#call session
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x_input:batch_xs,y_real:batch_ys})

correct_prediction = tf.equal(tf.argmax(y_output,1), tf.argmax(y_real,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_real: mnist.test.labels})

