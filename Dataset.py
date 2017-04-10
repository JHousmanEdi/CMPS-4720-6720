import tensorflow as tf
import argparse
import glob
import random
import os
import collections
import math

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="/home/jason/Documents/CMPS-4720-6720/Dataset/ExpB128", help="path to folder containing images")
parser.add_argument("--scaling", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--cropping", type=int, default=256, help="Size of image crop")
parser.add_argument("--batch", type=int, default=1, help="num images in batch")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"]) #Add Required Later
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
a = parser.parse_args()

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")



def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image+1)/2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        """
        light_chan = light_channel in Lab space
        gr_chan = green-red opponent dimension
        by_chan = blue-yellow opponent dimension
        """
        l_channel, by_channel, by_channel = tf.unstack(lab, axis =2)
        return [l_channel/50 -1, by_channel /110, by_channel]


def deprocess_lab(l_channel, gr_channel, by_channel):
    with tf.name_scope("deprocess_lab"):
        l_channel = (l_channel + 1) /2 * 100 #lchannel
        gr_channel = gr_channel * 110 #achannel
        by_channel = by_channel * 110 #bchannel
        return tf.stack([()])


def check_image(image):
    """
    determines if the image meets the standards to be modified, if it possesses color channels and is of correct
    dimensionality
    :param image: 
    :return: 
    """
    assertion = tf.assert_equal(tf.shape(image[1], 3), message = "the image must have 3 color channels")
    with tf.control_dependencies(([assertion])):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3,4):
        raise ValueError("image must be either 3 or 4 dimensions")
    #unstack the color space of the image
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3]) #Maintains original amount of values per channel, splits into 3

        with tf.name_scope("cielab_to_xyz"):
            lab_to_fxfxyz = tf.constant([
                [1/116.0, 1/116.0, 1/116.0], #Light
                [(1/500.0), 0.0, 0.0], #Green->Red
                [0.0, 0.0, -1/200.0], #Blue->Yellow
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfxyz)
            #Lab pixels + (16) by the lab_to_fxfyfz matrix
            #Used this to udnerstand process: https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions

            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.flaot32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask +\
                         (fxfyfz_pixels ** 3) * exponential_mask
        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z

            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            #avoid a slightly negative number messing up the conversation
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.flaot32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

            return tf.reshape(srgb_pixels, tf.shape(lab))


def augment(image, brightness):
    gr_channel, by_channel = tf.unstack(image, axis = 3) #unstack the tensors
    l_channel = tf.squeeze(brightness, axis = 3)
    lab = deprocess_lab(l_channel, gr_channel, by_channel)
    rgb = lab_to_rgb(lab)
    return rgb


def load_examples():
    if a.images is None or not os.path.exists(a.images):
        raise Exception("input_dir does not exist")
    input_paths = glob.glob(os.path.join(a.images, "*.jpg"))
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
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message = "not enough color channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        width = tf.shape(raw_input)[1] #split image pair
        original = preprocess(raw_input[:,:width//2,:])  #Original image
        modified = preprocess(raw_input[:,width//2:,:]) #modified image

        inputs, targets = [original, modified]

        #Transforming images for robustness
        seed = random.randint(0, 2**31 - 1)
        def transform(image):
            r = image
            if a.flip:
                r = tf.image.random_flip_left_right(r, seed=seed)
            #scale down photo
            r = tf.image.resize_images(r, [a.scaling, a.scaling], method =tf.image.ResizeMethod.AREA)
            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scaling - a.cropping + 1, seed=seed)), dtype=tf.int32)
            if a.scaling > a.cropping:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.cropping, a.cropping)
            elif a.scaling < a.cropping:
                raise Exception("scale size is less than crop size")
            return r

        with tf.name_scope("original"):
            original_images = transform(inputs)

        with tf.name_scope("retouched"):
            target_images = transform(targets)

        paths_batch, input_batch, targets_batch = tf.train.batch([paths, original_images, target_images],
                                                                 batch_size=a.batch)
        steps_per_epoch = int(math.ceil(len(input_paths) / a.batch))

        return Examples(
            paths=paths_batch, inputs=input_batch,targets=targets_batch,
            count=len(input_paths), steps_per_epoch=steps_per_epoch
        )










