import tensorflow as tf
import numpy as np
import pandas as pd
from ieee_grss_dse.learning.datagenerator import parse_example, type_transform


def calculate_normalization(args, dir_time):
    '''
    Calculate band means and standard deviations for normalization. This only needs
    to be run once for a given 'num_images_for_norm': The results will save in a
    .npy file that can be loaded for future normalization.
    '''

    print('Calculating normalizations')

    train_funcs = [type_transform]
    num_images_for_norm = args.IMGS_FOR_NORMALIZATION

    # Load in training dataset. This dataset is separate from the dataset used for training,
    # as I'm not sure I can reset the status of the Dataset after pulling num_images_for_norm
    # in order to calculate the band normalization.
    norm_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Apply all training funcs in train_funcs. At this point, train_funcs does not include the
    # apply_band_normalization
    for func in train_funcs:
        norm_dataset = norm_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Create numpy arrays to hold images taken for normalization
    input_mean_array = np.zeros((num_images_for_norm, args.INPUT_BANDS))
    input_std_array  = np.zeros((num_images_for_norm, args.INPUT_BANDS))

    # Load and assign
    norm_dataset = norm_dataset.take(num_images_for_norm)
    for ix, (input_image, output_image) in enumerate(norm_dataset.as_numpy_iterator()):
        input_mean_array[ix] = np.mean(input_image, axis=(0, 1))
        input_std_array[ix]  = np.std(input_image, axis=(0, 1))

    # Calculate means + standards deviations
    input_mean = np.expand_dims(np.mean(input_mean_array, axis=0), -1)
    input_std = np.expand_dims(np.mean(input_std_array, axis=0), -1)

    norm_df = pd.DataFrame(np.concatenate((input_mean, input_std), axis=-1),
                           columns=['band_means', 'band_stds'])

    ## Save!
    norm_df.to_csv(f'{args.NORMALIZATION_DIR}/model_{dir_time}_nimgs_' \
            f'{args.IMGS_FOR_NORMALIZATION}.csv' )


def load_normalization_arrays(args, norm_dir):
    '''
    This function loads previously saved normalization arrays.
    Normalization arrays are distinguished by args.IMGS_FOR_NORMALIZATION, or the number
    of images used to calculate the terms required for band normalization.
    '''

    # Load in mean and standard deviation
    input_norm = pd.read_csv(f'{args.NORMALIZATION_DIR}/model_{norm_dir}_nimgs_' \
                         f'{args.IMGS_FOR_NORMALIZATION}.csv')

    args.INPUT_BANDS_MEAN = np.array(input_norm['band_means'])
    args.INPUT_BANDS_STD  = np.array(input_norm['band_stds'])

