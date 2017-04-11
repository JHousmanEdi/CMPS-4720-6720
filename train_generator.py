from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import Dataset as ds
from cgan_model import *

import tensorflow as tf
import numpy as np
import random
import os
import json
import time
import math



def main():
    # noinspection PyUnresolvedReferences
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")


    if args['seed'] is None:
        args['seed'] = random.randint(0, 2**31 -1)

    tf.set_random_seed(args['seed'])
    # noinspection PyUnresolvedReferences
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if not os.path.exists(a['results__dir']):

        if args['mode'] == "test" or a.mode == "export":
            if args['checkpoint'] is None:
                raise Exception("Checkpoint required for test mode")

        options = {"which_direction","ngf", "ndf", "lab_clorization"}
        with open(os.path.join(args['checkpoint'],  "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("Resuming with {} = {}".format(key, val))
                    args[key] = val
        args['scaling'] = args['cropping']
        args['flip'] = False

        for key, val in args:
            print("{} = {}".format(key, val))

        with open(os.path.join(args['checkpoint'], "options.json"), "w") as f:
            f.write(json.dumps(args, sort_keys=True, indent=4))

        if args['mode'] == "export":

            input = tf.placeholder(tf.string, shape=[1])
            input_data = tf.decode_base64(input[0])
            input_image = tf.image.decode_png(input_data)
            input_image = input_image[:,:,:3]
            input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
            input_image.set_shape([args['cropping'], args['cropping'], 3])
            batch_input = tf.expand_dims(input_image, axis=0)

            with tf.variable_scope('generator') as scope:
                batch_output = ds.deprocess(create_generator(ds.preprocess(batch_input)))













