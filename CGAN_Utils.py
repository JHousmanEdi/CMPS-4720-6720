import tensorflow as tf



def conv(batch_input, out_channels, ksize, stride, padding):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        kernel = tf.get_variable("filter", [ksize, in_channels, out_channels],
                                 dtype=tf.float32, initializer = tf.random_normal_initializer)
        padded_input = tf.pad(batch_input, padding, mode="CONSTANT")

        conv = tf.nn.conv2d(padded_input, kernel, [1, stride, stride, 1], padding="VALID")
        return conv


def leaky_relu(x, a):
    with tf.name_scope("leaky-relu"):
        x = tf.identity(x)
        return (0.5 * (1+a)) * x + (0.5 * (1-a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batch-normalize"):
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer = tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)
        variance_epislon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epislon)
        return normalized


def deconv(batch_input, out_channels, ksize, stride, padding):
    with tf.variable_scope("transposed_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        kernel = tf.get_variable("filter", [ksize,ksize, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer)
        padded_input = tf.pad(batch_input, padding, mode="CONSTANT")
        tranconv = tf.nn.conv2d_transpose(padded_input, kernel, [batch, in_height*2, in_width*2, out_channels],
                                          [1, stride, stride, 1], padding="VALID")
        return tranconv

