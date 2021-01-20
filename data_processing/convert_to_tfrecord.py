import tensorflow as tf
import numpy as np
import json
import os
import datetime
import glob
from shapely.geometry import shape
from typing import Sequence
from tqdm import tqdm
import tqdm.notebook as tq
import findpeaks
from ieee_grss_dse.data_processing.load_raw_images_by_type import *
import matplotlib.pyplot as plt
from ieee_grss_dse.data_processing.data_processing_utils import *


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):

    """Wrapper for inserting float64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(img_data, target_data, img_shape, target_shape):
    """ Converts image and target data into TFRecords example.

    Parameters
    ----------
    img_data: ndarray
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)

    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    if len(target_shape) == 2:
        target_shape = (*target_shape, 1)

    features = {
        "image/image_data": _float64_feature(img_data),
        "image/height": _int64_feature(img_shape[0]),
        "image/width": _int64_feature(img_shape[1]),
        "image/channels": _int64_feature(img_shape[2]),
        "target/target_data": _float64_feature(target_data),
        "target/height": _int64_feature(target_shape[0]),
        "target/width": _int64_feature(target_shape[1]),
        "target/channels": _int64_feature(target_shape[2]),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def process_and_select_sar_bands(args, sar_array):
    '''
    Function that processes incoming channel-flattened SAR array and
    selects bands for output

    :param sar_array: input array of shape (timesteps*bands, height, width)
    :return: Processed sar_array with select bands + band combinations

    The channel flattened  SAR array has the following layers
    0: image 20200723, band vh
    1: image 20200723, band vv
    2: image 20200804, band vh
    3: image 20200804, band vv
    4: image 20200816, band vh
    5: image 20200816, band vv
    6: image 20200828, band vh
    7: image 20200828, band vv
    '''


    if args.apply_enh_lee_filter:
        sar_array = enhanced_lee_filtering(sar_array)

    ### APPLY OTHER PROCESSING FUNCTIONS HERE

    return sar_array


def process_and_select_s2_bands(s2_array):
    '''
    Function that processes incoming channel-flattened Sentinel-2 array and
    selects bands for output

    :param s2_array: input array of shape (timesteps*bands, height, width)
    :return: Processed s2_array with select bands + band combinations

    The channel flattened  S2 array has the following layers
    0: image 20200811, band B01
    1: image 20200811, band B02
    2: image 20200811, band B03
    3: image 20200811, band B04
    4: image 20200811, band B05
    5: image 20200811, band B06
    6: image 20200811, band B07
    7: image 20200811, band B08
    8: image 20200811, band B09
    9: image 20200811, band B11
    10: image 20200811, band B12
    11: image 20200811, band B8A

    12: image 20200816, band B01
    13: image 20200816, band B02
    14: image 20200816, band B03
    15: image 20200816, band B04
    16: image 20200816, band B05
    17: image 20200816, band B06
    18: image 20200816, band B07
    19: image 20200816, band B08
    20: image 20200816, band B09
    21: image 20200816, band B11
    22: image 20200816, band B12
    23: image 20200816, band B8A

    24: image 20200826, band B01
    25: image 20200826, band B02
    26: image 20200826, band B03
    27: image 20200826, band B04
    28: image 20200826, band B05
    29: image 20200826, band B06
    30: image 20200826, band B07
    31: image 20200826, band B08
    32: image 20200826, band B09
    33: image 20200826, band B11
    34: image 20200826, band B12
    35: image 20200826, band B8A

    36: image 20200831, band B01
    37: image 20200831, band B02
    38: image 20200831, band B03
    39: image 20200831, band B04
    40: image 20200831, band B05
    41: image 20200831, band B06
    42: image 20200831, band B07
    43: image 20200831, band B08
    44: image 20200831, band B09
    45: image 20200831, band B11
    46: image 20200831, band B12
    47: image 20200831, band B8A

    '''

    ### APPLY OTHER PROCESSING FUNCTIONS HERE

    return s2_array

def process_and_select_l8_bands(l8_array):
    '''
    Function that processes incoming channel-flattened Landsat8 array
    and selects bands for output

    :param l8_array: input array of shape (timesteps*bands, height, width)
    :return: Processed l8_array with select bands + band combinations

    The channel-flattened L8 array has the following layers
    0: image 20200729, band B01
    1: image 20200729, band B02
    2: image 20200729, band B03
    3: image 20200729, band B04
    4: image 20200729, band B05
    5: image 20200729, band B06
    6: image 20200729, band B07
    7: image 20200729, band B08
    8: image 20200729, band B09
    9: image 20200729, band B10
    10: image 20200729, band B11

    11: image 20200814, band B01
    12: image 20200814, band B02
    13: image 20200814, band B03
    14: image 20200814, band B04
    15: image 20200814, band B05
    16: image 20200814, band B06
    17: image 20200814, band B07
    18: image 20200814, band B08
    19: image 20200814, band B09
    20: image 20200814, band B10
    21: image 20200814, band B11

    22: image 20200830, band B01
    23: image 20200830, band B02
    24: image 20200830, band B03
    25: image 20200830, band B04
    26: image 20200830, band B05
    27: image 20200830, band B06
    28: image 20200830, band B07
    29: image 20200830, band B08
    30: image 20200830, band B09
    31: image 20200830, band B10
    32: image 20200830, band B11
    '''

    ### APPLY OTHER PROCESSING FUNCTIONS HERE

    return l8_array


def process_and_select_dnb_bands(dnb_array):
    '''
    Function that processes incoming DNB nightlights array and selects bands for output

    :param dnb_array: input array of shape (timesteps, height, width,)
    :return: Processed dnb_array with select bands + band combinations

    After the dnb array has the following layers
    0: image DNB_VNP46A1_A2020221
    1: image DNB_VNP46A1_A2020224
    2: image DNB_VNP46A1_A2020225.tif
    3: image DNB_VNP46A1_A2020226.tif
    4: image DNB_VNP46A1_A2020227.tif
    5: image DNB_VNP46A1_A2020231.tif
    6: image DNB_VNP46A1_A2020235.tif
    7: image DNB_VNP46A1_A2020236.tif
    8: image DNB_VNP46A1_A2020237.tif
    '''

    ### APPLY OTHER PROCESSING FUNCTIONS HERE

    return dnb_array



def make_tfrecord_dataset(args, tiles, config='train'):
    '''
    Function for loading in processed input data stacks and saving out as a .tfrecord
    The function writes input/output combinations to the .tfrecord for every tile listed in `tiles`.

    It is recommended that the user trains models on .tfrecords with as little processing applied to the .tfrecord
    as possible. Therefore, the user should do most of the image processing before saving the .tfrecord,
     then load in the .tfrecord and train on it as-is.

    Before windowing the images:
    input stacks are of size (800, 800, bands)
    target stacks are of size (16, 16, bands)

    After windowing:
    input stacks are of size (200, 200, bands)
    target stacks are of size (200, 200, bands)

    :return: Nothing; saves output .tfrecord
    '''


    window_mult = 49
    out_file = f"{args.data_dir}/tfrecords/{config}/{config}_nimgs_{len(tiles)*window_mult}.tfrecords"

    writer = tf.io.TFRecordWriter(out_file)
    print(tiles)

    for i, tile in tqdm(enumerate(tiles)):
        print(f'Uploading {tile} to tfrecord')

        # sar_array = load_sar_images(args, tile)
        s2_array = load_s2_images(args, tile)
        # l8_array = load_l8_images(args, tile)
        # dnb_array = load_dnb_images(args, tile)

        gt_array = load_groundtruth(args, tile)

        ## ADD CROPPING FUNCTION HERE

        input_stack = np.concatenate((
                                      # sar_array,
                                      s2_array,
                                      # l8_array,
                                      # dnb_array
                                     ),
                                     axis=0).astype(np.float32)

        ## Convert to bands last
        input_stack = np.transpose(input_stack, (1, 2, 0))
        gt_array = np.transpose(gt_array, (1, 2, 0))

        ## Take windowed arrays of both input stack and groundtruth
        input_stack = windowed_image_reading(input_stack)
        gt_array = windowed_image_reading(gt_array)

        if i == 0:
            print(f'Shape of input stack {input_stack.shape}')
            print(f'Shape of input stack {gt_array.shape}')

        # Write each window to the tfrecord
        for ix in tqdm(range(input_stack.shape[0]), position=0, leave=True):
            example = convert_to_example(
                input_stack[ix], gt_array[ix], input_stack[ix].shape, gt_array[ix].shape
            )

            writer.write(example.SerializeToString())


if __name__ == '__main__':

    features = {
        "image/image_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "target/target_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.uint8, allow_missing=True
        ),
        "target/height": tf.io.FixedLenFeature([], tf.int64),
        "target/width": tf.io.FixedLenFeature([], tf.int64),
        "target/channels": tf.io.FixedLenFeature([], tf.int64),
    }

    args = get_args()
    tiles = load_tile_names(args, config='val')[0:1]

    make_tfrecord_dataset(args, tiles, config='val')


