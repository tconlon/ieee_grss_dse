import numpy as np
import tensorflow as tf
from glob import glob
import rasterio
from ieee_grss_dse.data_processing.data_processing_utils import get_args


def channel_flatten_array(array):
    '''
    Function that flattens an array along its timesteps and channels

    :param array: Input array of shape (timesteps, height, width, bands)
    :return: channel_flattened_array of shape (timesteps*bands, height, width)
    '''
    array = np.moveaxis(array, [0, 1, 2, 3], [0, 2, 3, 1])
    channel_flattened_array = np.reshape(array, (array.shape[0] * array.shape[1],
                                            array.shape[2], array.shape[3]))

    return channel_flattened_array

def load_tile_names(args, config='train'):
    tile_paths = glob(f'{args.data_dir}/raw/{config}/*/')
    tiles = [i.split('/')[-2] for i in tile_paths]

    return tiles

def stack_images_by_date_and_band(image_files, dates, dtype):
    stack_list_dates_first = []

    for date in dates:
        image_bands_for_date = [i for i in image_files if date in i]
        image_band_list = []

        for band in image_bands_for_date:
            image_band = rasterio.open(band, 'r').read()
            image_band_list.append(image_band)

        image_stack_for_date = np.stack(image_band_list, -1).astype(dtype)
        stack_list_dates_first.append(image_stack_for_date)

    image_stack = np.concatenate(stack_list_dates_first, axis = 0)
    image_stack = channel_flatten_array(image_stack)

    return image_stack

def load_sar_images(args, tile, config='train'):
    dtype = np.float32

    sar_image_files = glob(f'{args.data_dir}/raw/{config}/{tile}/S1A*.tif')
    dates = np.unique([i.split('_')[-2] for i in sar_image_files])
    image_stack = stack_images_by_date_and_band(sar_image_files, dates, dtype)

    return image_stack

def load_s2_images(args, tile, config='train'):
    dtype = np.uint16

    s2_image_files = glob(f'{args.data_dir}/raw/{config}/{tile}/L2A*.tif')

    dates = np.unique([i.split('_')[-2] for i in s2_image_files])
    image_stack = stack_images_by_date_and_band(s2_image_files, dates, dtype)

    return image_stack

def load_l8_images(args, tile, config='train'):
    dtype = np.float32

    l8_image_files = glob(f'{args.data_dir}/raw/{config}/{tile}/LC08*.tif')
    dates = np.unique([i.split('_')[-2] for i in l8_image_files])
    image_stack = stack_images_by_date_and_band(l8_image_files, dates, dtype)

    return image_stack

def load_dnb_images(args, tile, config='train'):
    dtype = np.uint16

    dnb_image_files = glob(f'{args.data_dir}/raw/{config}/{tile}/DNB*.tif')
    dates = np.unique([i.split('_')[-1] for i in dnb_image_files])
    image_stack = stack_images_by_date_and_band(dnb_image_files, dates, dtype)

    return image_stack

def load_groundtruth(args, tile, config='train'):

    gt_file = f'{args.data_dir}/raw/{config}/{tile}/groundTruth.tif'
    gt_image = rasterio.open(gt_file, 'r').read()

    return gt_image.astype(np.uint8)


if __name__ == '__main__':
    args = get_args()
    load_s2_images(args, tile='Tile1')