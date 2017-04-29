from argsource import *
import tensorflow as tf
import glob
import random
import os
import collections
import math


Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")


def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image+1)/2


def image_loader():
    if args['images'] is None or not os.path.exists(args['images']):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(args['images'], "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        raise Exception("There were no images in the folder")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args['mode'] == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="not enough color channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        width = tf.shape(raw_input)[1] #split image pair
        original = preprocess(raw_input[:,:width//2,:])  #Original image
        modified = preprocess(raw_input[:,width//2:,:])  #Modified image

        inputs, targets = [original, modified]

        #Transforming images for robustness
        seed = random.randint(0, 2**31 - 1)
        def transform(image):
            r = image
            if args['flip']:
                r = tf.image.random_flip_left_right(r, seed=seed)
            #scale down photo
            r = tf.image.resize_images(r, [args['scaling'], args['scaling']], method =tf.image.ResizeMethod.AREA)
            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, args['scaling'] - args['cropping'] + 1, seed=seed)), dtype=tf.int32)
            if args['scaling'] > args['cropping']:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], args['cropping'], args['cropping'])
            elif args['scaling'] < args['cropping']:
                raise Exception("scale size is less than crop size")
            return r

        with tf.name_scope("original"):
            original_images = transform(inputs)

        with tf.name_scope("retouched"):
            target_images = transform(targets)

        paths_batch, input_batch, targets_batch = tf.train.batch([paths, original_images, target_images],
                                                                 batch_size=args['batch'])
        steps_per_epoch = int(math.ceil(len(input_paths) / args['batch']))

        return Examples(
            paths=paths_batch, inputs=input_batch,targets=targets_batch,
            count=len(input_paths), steps_per_epoch=steps_per_epoch
        )


def convert(image):
    if args['aspect_ratio'] != 1.0:
        size = [args['cropping'], int(round(args['cropping'] * args['aspect_ratio']))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def save_image(fetches, dataset, step=None):
    image_dir = os.path.join(args['results_dir'], dataset)
    if args['mode'] == "test":
        image_dir = os.path.join(args['results_dir'], dataset, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name" : name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".jpg"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
        return filesets


def export_image(image, dataset):
    exp_dir = '/home/jason/Documents/CMPS-4720-6720/results/TestResults'
    out_path = os.path.join(exp_dir, "edited.jpg")
    with open(out_path, "wb") as f:
        f.write(image)


def results_to_html(filesets, step=False):
    index_path = os.path.join(args['results_dir'],"test", "index.html")
    if not os.path.exists(index_path):
        with open('index.html', 'w'):
            pass
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path










