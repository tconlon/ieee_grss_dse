import tensorflow as tf
import rasterio
import numpy as np

from ieee_grss_dse.data_processing.data_processing_utils import get_args
from ieee_grss_dse.data_processing.calculate_normalizations import calculate_normalization, \
    load_normalization_arrays
from ieee_grss_dse.learning.model import s2_convlstm_network, xception_model
from ieee_grss_dse.learning.datagenerator import (type_transform, use_select_input_bands,
                                                  parse_example, list_imagery_by_ts,
                                                  one_hot_encoding_target, apply_band_normalization,
                                                  )

import datetime
import time
from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
import os
from ieee_grss_dse.learning.loss import crossentropy_loss
import seaborn as sns
from matplotlib import colors

# Set up plotting parameters
colors_xkcd = ['cobalt', 'very dark purple', "amber",
               "faded green", 'terracotta', "pale purple", 'grape',
               'salmon pink', 'greyish', 'dark turquoise', 'pastel blue'
               ]

colors_ieee = ['#ff0000', '#0000ff', '#ffff00', '#b266ff']

# sns.set_palette(sns.color_palette(palette=colors_ieee))
# cmap = colors.ListedColormap(colors_ieee)

cmap = colors.ListedColormap(sns.xkcd_palette(colors_xkcd)[0:4])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_images(args, model, s2_input, target, epoch, ix):
    '''
    Generate predicted image for an input stack. Create plots for comparing input stack,
    ground truth NDVI layer, and predicted NDVI layer. Print a scale bar on the ground
    truth image
    '''
    print('Generating images for visualization')
    
    prediction = model(s2_input, training=False)

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 8))

    if tf.rank(s2_input) == 4:
        test_input_for_plotting = s2_input[-1, ..., 4:1:-1]
    elif tf.rank(s2_input) == 5:
        test_input_for_plotting = s2_input[0, -1, ..., 4:1:-1]

    

    test_input_for_plotting = (test_input_for_plotting * args.INPUT_BANDS_STD[4:1:-1] + \
                              args.INPUT_BANDS_MEAN[4:1:-1])/10000

    print(np.min(test_input_for_plotting, axis = (0,1)))
    print(np.mean(test_input_for_plotting, axis = (0,1)))
    print(np.max(test_input_for_plotting, axis = (0,1)))
    #test_input_for_plotting = np.transpose(test_input_for_plotting, (1,2,0))


    prediction_sparse =  np.argmax(prediction[0], axis=-1)
    target_sparse =  np.argmax(target[0], axis=-1)


    display_list = [test_input_for_plotting, prediction_sparse, target_sparse]
    images_list = []


    title = ["RGB Image 20200831", "Prediction", "Ground Truth"]


    # Plot 3 images on a single matplotlib figure
    for i in range(3):
        ax[i].set_title(title[i])
        images_list.append(ax[i].imshow(display_list[i], cmap=cmap, vmin=0, vmax=3))
        ax[i].axis("off")

        ## Add scale bar
        if i == 0:
            fontprops = fm.FontProperties(size=12)
            bar_width = 50

            scalebar = AnchoredSizeBar(ax[i].transData,
                                       bar_width, '500m', 'lower right',
                                       pad=0.3,
                                       color='Black',
                                       frameon=True,
                                       size_vertical=2,
                                       fontproperties=fontprops)

            ax[i].add_artist(scalebar)



    bounds_all_preds = np.arange(start=-0.5, stop=4.5, step=1)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cbar = fig.colorbar(images_list[-1], cax=cax, orientation='vertical',
                        boundaries=bounds_all_preds, ticks=range(0, 4),
                        values=range(0,4))
    cbar.ax.set_yticklabels(['Yes Sett.\nNo Elec.\n(ROI)', 'No Sett.\nNo Elec.',
                             'Yes Sett.\nYes Elec.', 'No Sett.\nYes Elec.'])

    # Save figure for visual inspection
    image_dir = f'{args.IMAGE_DIR}/{dir_time}/epoch_{epoch}'
    os.makedirs(image_dir, exist_ok=True)

    plt.tight_layout()

    plt.savefig(f'{image_dir}/epoch_{epoch}_img_{ix}_input_prediction_target.png', bbox_inches='tight')
    plt.close()

def calculate_val_metrics(args, model, val_ds, epoch):

    # Generate images for visualization
    # for ix, (s2_input, target) in enumerate(val_ds.take(1)):
    #     generate_images(args, model, s2_input, target, epoch, ix)

    conf_matrix_tensor = tf.zeros(shape=[4,4], dtype=tf.dtypes.int32)

    for ix, (input_image, target_image) in val_ds.enumerate():
        prediction = model(input_image, training=False)

        prediction_sparse = np.argmax(prediction[0], axis=-1)
        target_sparse = np.argmax(target_image[0], axis=-1)

        conf_matrix = tf.math.confusion_matrix(tf.reshape(target_sparse, [-1]),
                                               tf.reshape(prediction_sparse, [-1]),
                                               num_classes=4)

        conf_matrix_tensor += conf_matrix

        if ix < 16:
            generate_images(args, model, input_image, target_image, epoch, ix)

    print('Validation dataset confusion matrix')
    print(conf_matrix_tensor)


    precision = np.zeros(4)
    recall = np.zeros(4)
    f1 = np.zeros(4)

    for ix in range(4):
        TP = conf_matrix_tensor[ix, ix]
        FP = tf.math.reduce_sum(conf_matrix_tensor[:, ix]) - TP
        FN = tf.math.reduce_sum(conf_matrix_tensor[ix]) - TP

        precision[ix] = TP / (TP + FP)
        recall[ix]    = TP / (TP + FN)

        if np.sum([precision[ix], recall[ix]]) > 0:
            f1[ix]        = 2 * precision[ix] * recall[ix] / (precision[ix] + recall[ix])

    total_acc = np.sum(np.diag(conf_matrix_tensor)) / np.sum(conf_matrix_tensor)

    print(f'Epoch {epoch}, validation set total accuracy: {total_acc}')
    
    if args.LOG:
        
        with summary_writer.as_default():
            tf.summary.scalar("val_total_acc", total_acc, step=epoch)
            for ix in range(4):
                tf.summary.scalar(f"val_precision_class_{ix+1}", precision[ix], step=epoch)
                tf.summary.scalar(f"val_recall_class_{ix+1}", recall[ix], step=epoch)
                tf.summary.scalar(f"val_f1_class_{ix+1}", f1[ix], step=epoch)


@tf.function
def train_step(args, input_image, target, model, epoch):
    '''
    Function that applies a training step for both generator and discriminator
    '''

    with tf.GradientTape() as model_tape:
        # Generate target iamge
        model_output = model(input_image, training=True)

        # Create model loss terms
        model_loss = crossentropy_loss(model_output, target)

        # Find gradients
        model_gradients = model_tape.gradient(model_loss, model.trainable_variables)

        # Apply gradients
        optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))

        # Write out loss and correlation terms
        if args.LOG:
            
            with summary_writer.as_default():
                tf.summary.scalar("train_total_loss", model_loss, step=epoch)

    
        return model_loss

def fit(args, train_ds, val_ds):
    '''
    Fit function. This function generates a set of images for visualization every epoch,
    and applies the training function

    Calculating the total length of the training set is optional: It takes time to calculate
    once, but after calculating once, I recommend assigning the length to pbar.

    Model checkpointing also occurs once every n epochs.
    '''

    total_training_imgs = int(args.TRAIN_PATH.strip('.tfrecords').split('_')[-1])

    print('Training')
    for epoch in range(args.EPOCHS):
        start = time.time()
        display.clear_output(wait=True)

        print("Epoch: ", epoch)

        # Train and track progress with pbar
        total_steps = np.ceil(total_training_imgs/args.BATCH_SIZE)
        pbar = tqdm(total=total_steps, ncols=100)

        #print('Calculating validation ds metrics')
        #calculate_val_metrics(args, model, val_ds, epoch)

        for n, (input_batch, target_batch) in train_ds.enumerate():
            pbar.update(1)
            step = epoch + n/total_steps
            model_loss = train_step(args, input_batch, target_batch, model, epoch)

            pbar.set_description(f'Training loss: {model_loss}')

        print('Calculating validation ds metrics')
        calculate_val_metrics(args, model, val_ds, epoch)

        print("Time taken for epoch {} is {} sec\n".format(epoch,
                                                           time.time() - start))


def prepare_datasets(args):
    train_funcs = [
        type_transform,
        lambda inp, tar: apply_band_normalization(args, inp, tar),
        one_hot_encoding_target,
        list_imagery_by_ts,

    ]

    # Similarly, define the functions that should be mapped onto the validation tf.data.Dataset
    val_funcs = [
        type_transform,
        lambda inp, tar: apply_band_normalization(args, inp, tar),
        one_hot_encoding_target,
        list_imagery_by_ts,
    ]

    ## NEED TO DEFINE NORMALIZATION FUNCTIONS HERE TOO

    # Create tf.data.Dataset for training data -- Not sure about optimal # calls
    # or prefetch parameters
    train_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Map functions onto training dataset, then shuffle and batch.
    for func in train_funcs:
        train_dataset = train_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    train_dataset = train_dataset.shuffle(args.BUFFER_SIZE).batch(args.BATCH_SIZE).prefetch(1)

    # Create tf.data.Dataset for validation data -- Not sure about optimal # calls
    # or prefetch parameters
    val_dataset = tf.data.TFRecordDataset(args.VAL_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Map functions onto testing dataset, then shuffle and batch.
    for func in val_funcs:
        val_dataset = val_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    val_dataset = val_dataset.batch(1).prefetch(1)

    return train_dataset, val_dataset



if __name__ == '__main__':

    '''
    Runs when main.py is called.
    '''

    # Get arguments from params.yaml
    args = get_args()
    args = dotdict(vars(args))

    # Get discriminator model, define loss + optimizers
    model = xception_model(args)
    optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{dir_time}/'

    if not args.LOAD_EXISTING:
        # Define checkpoint object + save path
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
        print('Training from scratch')
    else:
        # Load existing checkpoint
        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.CHECKPOINT_FOLDER}/'
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(prev_checkpoint_prefix)}')
        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix))

    # Save new normalization file if necessary
    if args.CALCULATE_NORM:
        norm_dir = dir_time
        calculate_normalization(args, dir_time)
    else:
        print('Loading existing normalization')
        norm_dir = '20210122-140846'

    # Load normalization
    load_normalization_arrays(args, norm_dir)

    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    tf.Graph().finalize()




    # Create a directory to store training results based on new model start time
    if args.LOG:
        summary_writer = tf.summary.create_file_writer(f'{args.LOG_DIR}/{dir_time}')

    # Define the functions that get mapped onto the training tf.data.Dataset
    # These are imported from learning/datagenerator.py

    train_dataset, val_dataset = prepare_datasets(args)

    if args.TRAIN:
        fit(args, train_dataset, val_dataset)
