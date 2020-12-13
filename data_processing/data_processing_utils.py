import numpy as np
import findpeaks
import argparse, yaml

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