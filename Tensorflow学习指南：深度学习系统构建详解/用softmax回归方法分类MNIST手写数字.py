import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/home/daiyu/data/dl/MNIST'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)


# 变量是由计算所操控的对象，占位符是触发该计算时需要的对象。
# 图像本身（x）是一个占位符，因为当运算计算图时，我们需要提供它。
# [None,784]表示每张图的维度大小是784（将唯独为28×28的图像展开为一个向量），None表示当前不指定每次使用的图片数量

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)


# 在这个模型中使用的测量相似度的方法叫做交叉熵cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true
))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})
    print("Accuracy:{:.4}%".format(ans * 100))
