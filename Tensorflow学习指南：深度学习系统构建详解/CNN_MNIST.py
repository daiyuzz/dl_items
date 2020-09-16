import tensorflow as tf



x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))


tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

