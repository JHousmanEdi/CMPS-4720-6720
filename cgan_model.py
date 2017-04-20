from argsource import *
from cgan_utils import layer_wrapper
import tensorflow as tf
import collections

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, D_loss, "
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
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        if layers is not None:
            self.layers = self.layers
        self.decoder_utils = layer_wrapper("decoder")

    def build(self, gen_input, gen_output_chan, final_output_channels):
        layer_specs = [
            (gen_output_chan * 8, 0.5),  # 8
            (gen_output_chan * 8, 0.5),  # 7
            (gen_output_chan * 8, 0.5),  # 6
            (gen_output_chan * 8, 0.0),  # 5
            (gen_output_chan * 4, 0.0),  # 4
            (gen_output_chan * 2, 0.0),  # 3
            (gen_output_chan, 0.0)  # 2
        ]
        for decode_layer, (out_channels, dropout) in reversed(list(enumerate(layer_specs))):
            with tf.variable_scope("decoder_{}".format(decode_layer + 2)):
                if decode_layer == 6:
                    input = self.layers[-1]
                else:
                    input = tf.concat([gen_input, self.layers[decode_layer + 1]], axis=3)

                rectified = tf.nn.relu(input)
                output = self.decoder_utils.deconv(rectified, out_channels=out_channels, ksize=4, stride=2, padding=0)
                output = self.decoder_utils.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                self.layers.append(output)

        with tf.variable_scope("decoder_1"):
            input = tf.concat([self.layers[-1], self.layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = self.decoder_utils.deconv(rectified, final_output_channels, ksize=4, stride=2, padding=0)
            output = tf.tanh(output)
            self.layers.append(output)
        return self.layers


class Generator:
    def __init__(self, gen_input, gen_output_chan, final_output_channels):
        self.gen_input = gen_input
        self.gen_output_chan = gen_output_chan
        self.final_output_channels = final_output_channels

    def build(self):
        encoder = EncoderLayers()
        enc_output = encoder.build(self.gen_input, self.gen_output_chan)
        layers = enc_output
        decoder = DecoderLayers(layers)
        output = decoder.build(layers[-1], self.gen_output_chan, self.final_output_channels)

        return output

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
            convolution = self.discriminator_utils.conv(self.layers[-1], out_channels=1, stride=1, ksize=4)
            score = tf.sigmoid(convolution)
        return score


class model():
    def __init__(self, inputs, targets, final_output_channels, n_layers=3, discrim_filters):
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
                real_discrim = create_discriminator(self.inputs, self.targets)
                self.scores['real'] = real_discrim

        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope("discriminator", reuse=True):
                gen_discrim = Discriminator(self.inputs, output, self.discrim_filters)
                self.scores['fake'] = gen_discrim
        return output

    def optimize(self):
        real_discrim = self.scores['real']
        gen_discrim = self.scores['fake']
        output = self.build()
        with tf.name_scope("discriminator_loss"):
            D_loss = tf.reduce_mean(-(tf.log(real_discrim) + tf.log(1-gen_discrim)))

        with tf.name_scope("generator_loss"):
            gan_loss = tf.reduce_mean(-tf.log(gen_discrim))
            G_l2_loss = tf.nn.l2_loss(tf.abs(self.targets - output))
            G_loss = (1- args['l2_weight']) * gan_loss + (args['l2_weight'] * G_l2_loss)

        with tf.name_scope("discriminator_train"):
            D_train_vars= [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            D_adam = tf.train.AdamOptimizer(args['lr_d'], args['beta1'])
            D_gradients = D_adam.compute_gradients(discrim_loss, var_list=D_train_vars)
            D_train = D_adam.apply_gradients(D_gradients)
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([D_train]):
                G_train_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                G_adam = tf.train.AdamOptimizer(args['lr_g'], args['beta1'])
                G_gradients = G_adam.compute_gradients(gan_loss, var_list=G_train_vars)
                G_train = G_adam.apply_gradients(G_gradients)
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([D_loss, G_loss, G_l2_loss])

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return Model(
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

"""
TODO Remove and update main method
"""
def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    #encoder_1: [batch, 256, 256, in_channels]
    with tf.variable_scope("encoder_1"):
        output = conv(batch_input=generator_inputs, out_channels=args['ngf'], ksize=4, stride=2, padding=1)
        layers.append(output)


        #ENCODER Model
    layer_specs = [
        args['ngf'] * 2,
        args['ngf'] * 4, #3
        args['ngf'] * 8, #4
        args['ngf'] * 8, #5
        args['ngf'] * 8, #6
        args['ngf'] * 8, #7
        args['ngf'] * 8, #8
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" %(len(layers) + 1)):
            rectified = leaky_relu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, ksize=4,stride=2, padding=1)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (args['ngf'] * 8, 0.5), #8
        (args['ngf'] * 8, 0.5), #7
        (args['ngf'] * 8, 0.5), #6
        (args['ngf'] * 8, 0.0), #5
        (args['ngf'] * 4, 0.0), #4
        (args['ngf'] * 2, 0.0), #3
        (args['ngf'], 0.0) #2
    ]
    for decode_layer, (out_channels, dropout) in reversed(list(enumerate(layer_specs))):
        with tf.variable_scope("decoder_{}".format(decode_layer+2)):
            if decode_layer == 6:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[decode_layer+1]], axis=3)

            rectified = tf.nn.relu(input)
            output = deconv(rectified, out_channels=out_channels, ksize=4, stride=2, padding=0)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels,ksize=4,  stride=2, padding=0)
        output = tf.tanh(output)
        layers.append(output)
    return layers[-1]

def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolved = conv(input, args['ndf'], ksize=4,stride=2,padding=1)
            rectified = leaky_relu(convolved,0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_{}" .format(len(layers) + 1)):
                out_channels = args['ndf'] * min(2**(i+1),8)
                convolved = conv(layers[-1], out_channels, stride=2, ksize=4, padding=1)
                normalized = batchnorm(convolved)
                rectified = leaky_relu(normalized, 0.2)
                layers.append(rectified)
        with tf.variable_scope("layer_{}".format(len(layers)+1)):
            convolved = conv(rectified, out_channels=1, stride =1, padding=1, ksize=4)
            output = tf.sigmoid(convolved)
            layers.append(output)
        return layers[-1]


def create_model(inputs, targets):
    with tf.variable_scope("generator") as scope: ##Load Generator Model
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    with tf.name_scope('real_discriminator'): #Load discriminator that compares touched up and untouched images
        with tf.variable_scope('discriminator'):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope('fake_discriminator'): #Loads discriminator that compares original and generated images
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"): #Calculates loss of generator
        D_loss = tf.reduce_mean(-(tf.log(predict_real) + tf.log(1 - predict_fake)))

    with tf.name_scope("generator_loss"): #Calculates loss of generator
        gan_loss = tf.reduce_mean(-tf.log(predict_fake))
        gen_l2_loss = tf.nn.l2_loss(tf.abs(targets - outputs))
        G_loss = (1-args['l2_weight'])*gan_loss + (args['l2_weight'] * gen_l2_loss) #GP-GAN LOSS

    with tf.name_scope("discriminator_train"): #Trains the discriminator with ADAM optimizer
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(args['lr_d'], args['beta1'])
        discrim_grad_vars = discrim_optim.compute_gradients(D_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grad_vars)

    with tf.name_scope("generator_train"): #Trains the generator with the ADAM optimizer
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(args['lr_g'], args['beta1'])
            gen_grads_vars = gen_optim.compute_gradients(gan_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([D_loss, G_loss, gen_l2_loss])

    global_step=tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        outputs=outputs,
        predict_real=predict_real,
        predict_fake = predict_fake,
        D_loss=ema.average(D_loss),
        discrim_grad_vars=discrim_grad_vars,
        gen_l2_loss=ema.average(gen_l2_loss),
        G_loss=ema.average(G_loss),
        gen_grads_vars = gen_grads_vars,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )



