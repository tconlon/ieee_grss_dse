import tensorflow as tf
import glob
import numpy as np
import os


def parse_example(example_proto):
    '''
    Helper function converts .tfrecords into input and output images
    '''

    features = {
        "image/image_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "target/target_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "target/height": tf.io.FixedLenFeature([], tf.int64),
        "target/width": tf.io.FixedLenFeature([], tf.int64),
        "target/channels": tf.io.FixedLenFeature([], tf.int64),
    }

    image_features = tf.io.parse_single_example(example_proto, features)

    img_height = tf.cast(image_features["image/height"], tf.int32)
    img_width = tf.cast(image_features["image/width"], tf.int32)
    img_channels = tf.cast(image_features["image/channels"], tf.int32)

    target_height = tf.cast(image_features["target/height"], tf.int32)
    target_width = tf.cast(image_features["target/width"], tf.int32)
    target_channels = tf.cast(image_features["target/channels"], tf.int32)

    image_raw = tf.reshape(
        tf.squeeze(image_features["image/image_data"]),
        tf.stack([img_height, img_width, img_channels]),
    )

    target_raw = tf.reshape(
        tf.squeeze(image_features["target/target_data"]),
        tf.stack([target_height, target_width, target_channels]),
    )

    return image_raw, target_raw


def use_select_input_bands(input_image, target_image):
    '''
    These bands are determined by the layers saved in the .tfrecord file;
    The .tfrecord layers are determined via:
    `data_processing/convert_tif_to_tfrecord.py`

    '''

    # Most recent data is in Log space
    input_image_list = []
    select_bands = range(98)

    for i in select_bands:
        input_image_list.extend([input_image[..., i]])

    input_image = tf.stack(input_image_list, axis = -1)

    return input_image, target_image

def type_transform(input_image, target_image):
    '''
    Ensuring the input images are of type float 32
    and the target images are of type uint 8
    '''

    input_image  = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.uint8)

    return input_image, target_image

def list_imagery_by_ts(input_image, target_image):

    imagery_list_by_ts = [input_image[..., 0:12],
                          input_image[..., 12:24],
                          input_image[..., 24:36],
                          input_image[..., 36:48]]


    return imagery_list_by_ts, target_image


def one_hot_encoding_target(input_image, target_image):

    target_image = tf.one_hot(target_image, depth=4, axis=-1)
    target_image = tf.squeeze(target_image, axis=-2)

    return input_image, target_image


def apply_band_normalization(args, input_image, target_image):
    '''
    Apply the band normalization so that mean = 0, std = 1 for all bands in
    input images. Adjust class ranges on target image
    '''

    # Save band normalization mean + std
    input_mean = tf.constant(args.INPUT_BANDS_MEAN, dtype=tf.float32)
    input_std = tf.constant(args.INPUT_BANDS_STD, dtype=tf.float32)

    # Normalize input band
    input_image = tf.divide((input_image - input_mean), input_std)

    # Adjust target image classes
    target_image = target_image - 1

    return input_image, target_image