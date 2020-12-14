import tensorflow as tf
import rasterio
import numpy as np
from data_processing.data_processing_utils import get_args
from learning.model import Discriminator
from learning.datagenerator import type_transform, use_select_input_bands, parse_example
import datetime
import time
from IPython import display
from tqdm import tqdm

def prepare_datasets():
    train_funcs = [
        type_transform,
        use_select_input_bands,
    ]

    # Similarly, define the functions that should be mapped onto the validation tf.data.Dataset
    val_funcs = [
        type_transform,
        use_select_input_bands,
    ]

    ## NEED TO DEFINE NORMALIZATION FUNCTIONS HERE TOO

    # Create tf.data.Dataset for training data -- Not sure about optimal # calls
    # or prefetch parameters
    train_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    print(train_dataset)
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

    # Get discriminator model, define loss + optimizers
    discriminator = Discriminator(args)
    disc_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    disc_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator.compile(optimizer=discriminator_optimizer,
                          loss=disc_loss_object,
                          metrics=disc_metrics)


    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{dir_time}/'

    if not args.LOAD_EXISTING:
        # Define checkpoint object + save path
        checkpoint = tf.train.Checkpoint(
            discriminator_optimizer=discriminator_optimizer,
            discriminator=discriminator,
        )
        print('Training from scratch')
    else:
        # Load existing checkpoint
        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.CHECKPOINT_FOLDER}/'
        checkpoint = tf.train.Checkpoint(
            discriminator_optimizer=discriminator_optimizer,
            discriminator=discriminator,
        )
        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(prev_checkpoint_prefix)}')
        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix))

    # Create a directory to store training results based on new model start time
    if args.LOG:
        summary_writer = tf.summary.create_file_writer(f'{args.LOG_DIR}/fit/{dir_time}')

    # Define the functions that get mapped onto the training tf.data.Dataset
    # These are imported from learning/datagenerator.py

    train_dataset, val_dataset = prepare_datasets()

    if args.TRAIN:
        discriminator.fit(train_dataset, epochs=args.EPOCHS, validation_data=val_dataset)