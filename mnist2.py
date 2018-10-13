#coding:utf_8

'''参考 http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html'''
#使用了多层次的神经网络架构
import tensorflow as tf


'''run this code to download the mnist code'''
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''difine variable and graph'''

#定义卷积和池化  padding='SAME'表示要进行补0操作，Padding不是0，也就是说卷积后，
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#定义生成权重和偏置的网络
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #有限制的正态分布
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义训练数据，label
x_input = tf.placeholder(tf.float32,shape=[None,784])
y_target = tf.placeholder(tf.float32,shape=[None,10])

#定义网络结构
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#reshape 中的-1表示，自动算出剩下这个维度的值，这个值和输入的x_input有关系。 参考资料：https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/reshape?hl=de
x_image = tf.reshape(x_input, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全卷积成
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drouout层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


'''train loss function and gradient decent'''

#loss and train step
cross_entropy = -tf.reduce_sum(y_target*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


'''评估手段 test'''
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


#init all variable

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

#call session to train

for i in range(20000):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={ x_input:batch[0], y_target: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    if i==0:
        print h_conv1;
        print h_pool1;
        print h_conv2;
        print h_pool2;
  train_step.run(feed_dict={x_input: batch[0], y_target: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={x_input: mnist.test.images, y_target: mnist.test.labels, keep_prob: 1.0})


