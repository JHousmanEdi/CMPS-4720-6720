from argsource import *
from cgan_utils import layer_wrapper
import tensorflow as tf
import collections

Model_Outs = collections.namedtuple("Model", "outputs, predict_real, predict_fake, D_loss, "
                                        "discrim_grad_vars, gen_l2_loss, G_loss, gen_grads_vars, train")


class EncoderLayers:
    def __init__(self, layers=None):
        if layers is None:
            self.layers =[]
        if layers is not None:
            self.layers = layers
        self.encoder_utils = layer_wrapper("encoder")

    def build(self, gen_input, gen_output_chan):

        with tf.variable_scope("encoder_1"):
            output = self.encoder_utils.conv(batch_input=gen_input, out_channels=gen_output_chan, ksize=4, stride=2, padding=1)
            self.layers.append(output)

        layer_specs = [
            gen_output_chan * 2,
            gen_output_chan * 4,  # 3
            gen_output_chan * 8,  # 4
            gen_output_chan * 8,  # 5
            gen_output_chan * 8,  # 6
            gen_output_chan * 8,  # 7
            gen_output_chan * 8,  # 8
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(self.layers) + 1)):
                rectified = self.encoder_utils.leaky_relu(self.layers[-1])
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.encoder_utils.conv(rectified, out_channels, ksize=4, stride=2, padding=1)
                output = self.encoder_utils.batchnorm(convolved)
                self.layers.append(output)
        return self.layers


class DecoderLayers:
    def __init__(self,gen_input, gen_output_chan, final_output_channels, layers=None):
        if layers is None:
            self.layers = []
        if layers is not None:
            self.layers = layers
        self.gen_input = gen_input
        self.gen_output_chan = gen_output_chan
        self.final_output_channels = final_output_channels
        self.decoder_utils = layer_wrapper("decoder")

    def build(self):
        layer_specs = [
            (self.gen_output_chan * 8, 0.5),  # 8
            (self.gen_output_chan * 8, 0.5),  # 7
            (self.gen_output_chan * 8, 0.5),  # 6
            (self.gen_output_chan * 8, 0.0),  # 5
            (self.gen_output_chan * 4, 0.0),  # 4
            (self.gen_output_chan * 2, 0.0),  # 3
            (self.gen_output_chan, 0.0)  # 2
        ]
        for index, (out_channels, dropout) in reversed(list(enumerate(layer_specs))):
            with tf.variable_scope("decoder_{}".format(index + 2)):
                if index == 6:
                    input_img = self.gen_input
                else:
                    input_img = tf.concat([self.layers[-1], self.layers[index + 1]], axis=3)

                rectified = tf.nn.relu(input_img)
                output = self.decoder_utils.deconv(rectified, out_channels=out_channels, ksize=4, stride=2, padding=0)
                output = self.decoder_utils.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                self.layers.append(output)

        with tf.variable_scope("decoder_1"):
            input_img = tf.concat([self.layers[-1], self.layers[0]], axis=3)
            rectified = tf.nn.relu(input_img)
            output = self.decoder_utils.deconv(rectified, self. final_output_channels, ksize=4, stride=2, padding=0)
            output = tf.tanh(output)
            self.layers.append(output)
        return self.layers


class Generator:
    def __init__(self, gen_input, final_output_channels, gen_output_chan = args['ngf']):
        self.gen_input = gen_input
        self.gen_output_chan = gen_output_chan
        self.final_output_channels = final_output_channels

    def build(self):
        encoder = EncoderLayers()
        enc_output = encoder.build(self.gen_input, self.gen_output_chan)
        layers = enc_output
        decoder = DecoderLayers(layers[-1], self.gen_output_chan, self.final_output_channels, layers)
        output = decoder.build()

        return output[-1]


class Discriminator:
    def __init__(self,discrim_inputs, discrim_targets, discrim_channels, n_layers=3, ):
        self.discrim_inputs = discrim_inputs
        self.discrim_targets = discrim_targets
        self.discrim_channels = discrim_channels
        self.n_layers = n_layers
        self.layers = []
        self.discriminator_utils = layer_wrapper("discriminator")

    def build(self):
        input = tf.concat([self.discrim_inputs, self.discrim_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolution = self.discriminator_utils.conv(input, self.discrim_channels, ksize=4, stride=2, padding=1)
            activated = self.discriminator_utils.leaky_relu(convolution)
            self.layers.append(activated)

        for i in range(self.n_layers):
            with tf.variable_scope("layer_{}".format(len(self.layers) + 1)):
                out_channels = self.discrim_channels * min(2**(i+1), 8)
                convolution = self.discriminator_utils.conv(self.layers[-1], out_channels, stride=2, ksize=4, padding=1)
                normalized = self.discriminator_utils.batchnorm(convolution)
                activated = self.discriminator_utils.leaky_relu(normalized)
                self.layers.append(activated)
        with tf.variable_scope("layer_{}".format(len(self.layers) + 1)):
            convolution = self.discriminator_utils.conv(self.layers[-1], out_channels=1, stride=1, ksize=4, padding=1)
            score = tf.sigmoid(convolution)
        return score


class Model():
    def __init__(self, inputs, targets, final_output_channels = args['ngf'], discrim_filters = args['ndf'], n_layers=3):
        self.inputs = inputs
        self.targets = targets
        self.n_layers = n_layers
        self.final_output_channels = final_output_channels
        self.discrim_filters = discrim_filters
        self.scores = {}

    def build(self):
        with tf.variable_scope("generator") as scope:
            out_channels = int(self.targets.get_shape()[-1])
            generator = Generator(self.inputs, out_channels, self.final_output_channels)
            output = generator.build()

        with tf.name_scope('real_discriminator'):
            with tf.variable_scope('discriminator'):
                real_discrim = Discriminator(self.inputs, self.targets, self.discrim_filters)
                real_discrim_score = real_discrim.build()
                self.scores['real'] = real_discrim_score

        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope("discriminator", reuse=True):
                gen_discrim = Discriminator(self.inputs, output, self.discrim_filters)
                gen_discrim_score = gen_discrim.build()
                self.scores['fake'] = gen_discrim_score
        return output

    def optimize(self):
        output = self.build()
        real_discrim = self.scores['real']
        gen_discrim = self.scores['fake']
        with tf.name_scope("discriminator_loss"):
            D_loss = tf.reduce_mean(-(tf.log(real_discrim) + tf.log(1-gen_discrim)))

        with tf.name_scope("generator_loss"):
            gan_loss = tf.reduce_mean(-tf.log(gen_discrim))
            G_l2_loss = tf.nn.l2_loss(tf.abs(self.targets - output))
            G_loss = (1- args['l2_weight']) * gan_loss + (args['l2_weight'] * G_l2_loss)

        with tf.name_scope("discriminator_train"):
            D_train_vars= [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            D_adam = tf.train.AdamOptimizer(args['lr_d'], args['beta1'])
            D_gradients = D_adam.compute_gradients(D_loss, var_list=D_train_vars)
            D_train = D_adam.apply_gradients(D_gradients)
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([D_train]):
                G_train_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                G_adam = tf.train.AdamOptimizer(args['lr_g'], args['beta1'])
                G_gradients = G_adam.compute_gradients(G_loss, var_list=G_train_vars)
                G_train = G_adam.apply_gradients(G_gradients)
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([D_loss, G_loss, G_l2_loss])

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return Model_Outs(
            outputs=output,
            predict_real=real_discrim,
            predict_fake=gen_discrim,
            D_loss=ema.average(D_loss),
            discrim_grad_vars=D_gradients,
            gen_l2_loss=ema.average(G_l2_loss),
            G_loss=ema.average(G_loss),
            gen_grads_vars=G_gradients,
            train=tf.group(update_losses, incr_global_step, G_train),
        )
