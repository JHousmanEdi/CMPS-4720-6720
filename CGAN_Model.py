import tensorflow as tf
from CGAN_Utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
a = parser.parse_args()

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    #encoder_1: [batch, 256, 256, in_channels]
    with tf.variable_scope("encoder_1"):
        output = conv(batch_input=generator_inputs, out_channels=a.ngf, ksize=4, stride=2, padding=1)
        layers.append(output)

        #ENCODER Model
    layer_specs = [
        a.ngf * 2,
        a.ngf * 4,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" %(len(layers) + 1)):
            rectified = leaky_relu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, ksize=4,stride=2, padding=1)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.5),
        (a.ngf * 8, 0.0),
        (a.ngf * 4, 0.0),
        (a.ngf * 2, 0.0),
    ]
    for decode_layer, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("decoder_{}".format(decode_layer)):
            input = tf.concat([layers[-1], layers[decode_layer]], axis=3)

        rectified = tf.nn.relu(input)
        output = deconv(rectified, out_channels=out_channels, ksize=4, stride=2, padding=1)
        output = batchnorm(output)

        if dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=1 - dropout)
        layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels,ksize=4,  stride=2, padding=1)
        output = tf.tanh(output)
        layers.append(output)
    return layers[-1]





