import tensorflow as tf


def batch_norm(x, name, training=True, epsilon=1e-5, momentum=0.9, reuse=False):
    return tf.layers.batch_normalization(x, name=name+"batch_norm", training=training, epsilon=epsilon, momentum=momentum, reuse=reuse)


def conv2d(name, x, out_dim, reuse=False, h=4, w=4, s=2):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w",
                            [h, w, x.get_shape().as_list()[-1], out_dim],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        return tf.nn.conv2d(x, w, [1, s, s, 1], padding='SAME')


def dconv2d(name, x, out, h=4, w=4, s=2):
    with tf.variable_scope(name):
        w = tf.get_variable("w",
                            [h, w, out[-1], x.get_shape().as_list()[-1]],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        dconv = tf.nn.conv2d_transpose(x,
                                       w,
                                       output_shape=out,
                                       strides=[1, s, s, 1],
                                       padding='SAME')
        return dconv


def conv2d_bn(name, x, out_dim, reuse=False, h=5, w=5, s=2, training=True, epsilon=1e-5, momentum=0.9):
    conv = conv2d(name, x, out_dim, reuse, h, w, s)
    return batch_norm(conv, name, training, epsilon, momentum, reuse)


def dconv2d_bn(name, x, out, h=5, w=5, s=2, training=True, epsilon=1e-5, momentum=0.9):
    dconv = dconv2d(name, x, out, h, w, s)
    return batch_norm(dconv, name, training, epsilon, momentum)


def linear(name, x, out_size, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w",
                            shape=[x.get_shape().as_list()[-1], out_size],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([out_size]))
        z = tf.matmul(x, w)+b
        return z


def lrelu(x):
    return tf.nn.leaky_relu(x)


def relu(x):
    return tf.nn.relu(x)
