from argsource import *
from cgan_utils import *
import tensorflow as tf
import collections

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, D_loss, "
                                        "discrim_grad_vars, gen_l2_loss, G_loss, gen_grads_vars, train")

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



