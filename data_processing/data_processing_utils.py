import pandas
import numpy as np
import findpeaks
import argparse, yaml
from skimage.util.shape import view_as_windows

def get_args():
    parser = argparse.ArgumentParser(
        description='Predict irrigation presence using Sentinel imagery')

    parser.add_argument('--params_filename',
                        type=str,
                        default='../params.yaml',
                        help='Filename defining repo configuration')

    args = parser.parse_args()
    config = yaml.load(open(args.params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args

def windowed_image_reading(array):
    '''
    Function for spatially splitting the input and target images

    input image size: (1, 800, 800, channels)
    target imager size: (1, 16, 16, 1)

    :param array: array to split along spatial dimensions
    :return: array: split array, stacked along axis=0
    '''

    window_size = int(0.25 * array.shape[1]) # fraction of spatial length
    stride_size = int(0.125 * array.shape[1]) # fraction of spatial length

    windowed_array = view_as_windows(array, window_shape=(window_size, window_size, array.shape[-1]),
                                     step=stride_size)

    windowed_array = np.reshape(windowed_array, newshape=(windowed_array.shape[0]*windowed_array.shape[1],
                                                          windowed_array.shape[3], windowed_array.shape[4],
                                                          windowed_array.shape[5]))

    '''
    As is, returns input array of size: (49, 200, 200, channels)
    and target array of size: (49, 4, 4, 1)
    '''

    return windowed_array

def enhanced_lee_filtering(sar_array):
    '''
    Function that applies an enhanced lee filter on an input SAR array

    :param array:
    :return: Lee-filtered array
    '''

    ## Filters parameters
    # window size
    winsize = 15
    # damping factor for lee enhanced
    k_value2 = 1.0
    # coefficient of variation for lee enhanced of noise
    cu_lee_enhanced = 0.523
    # max coefficient of variation for lee enhanced
    cmax_value = 1.73

    for ix in range(sar_array.shape[0]):
        sar_layer = sar_array[ix]
        image_lee_enhanced = findpeaks.lee_enhanced_filter(sar_layer, win_size=winsize,
                                                           k=k_value2, cu=cu_lee_enhanced,
                                                           cmax=cmax_value)
        sar_array[ix] = image_lee_enhanced

    return sar_array

if __name__ == '__main__':
    array = np.arange(16*16*2).reshape((1,16,16,2))
    windowed_array = split_full_sized_images(array)