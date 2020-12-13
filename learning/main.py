import tensorflow as tf
import rasterio
import numpy as np
from data_processing.data_processing_utils import get_args
from learning.model import Discriminator
from learning.datagenerator import type_transform, use_select_input_bands, parse_example
from learning.loss import discriminator_loss
import datetime
import time
from IPython import display
from tqdm import tqdm

@tf.function
def train_step(input_image, target, epoch, disc_loss_object):
    '''
    Function that applies a training step for both generator and discriminator
    '''

    with tf.GradientTape() as disc_tape:
        # Generate target image

        # Generate discriminator outputs for real and generated data
        pred_output = discriminator(input_image, training=True)

        # Create discriminator loss terms
        disc_loss = discriminator_loss(target, pred_output, disc_loss_object)

        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        ### ADD OTHER METRICS CALCULATIONS HERE

        # Write out loss and correlation terms
        with summary_writer.as_default():
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)


def fit(args, train_ds, val_ds, disc_loss_object):
    '''
    Fit function. This function generates a set of images for visualization every epoch,
    and applies the training function.

    Calculating the total length of the training set is optional: It takes time to calculate
    once, but after calculating once, I recommend assigning the length to pbar.

    Model checkpointing also occurs once every n epochs.
    '''

    min_val_loss = np.inf
    total_training_imgs = int(args.TRAIN_PATH.strip('.tfrecords').split('_')[-1])

    print('Training')
    for epoch in range(args.EPOCHS):
        start = time.time()
        display.clear_output(wait=True)

        print(f'Epoch: {epoch}')

        # Train and track progress with pbar
        pbar = tqdm(total=np.ceil(total_training_imgs / args.BATCH_SIZE), ncols=60)

        for n, (input_batch, target_batch) in train_ds.enumerate():
            pbar.update(1)
            train_step(input_batch, target_batch, epoch, disc_loss_object)

        ## NEED TO IMPLEMENT TEST SET ACCURACY CALCULATIONS

        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix,
                                            max_to_keep=1,
                                            checkpoint_name=f'epoch_{epoch}_cMSE_{min_val_loss}')
        manager.save()


        print("Time taken for epoch {} is {} sec\n".format(epoch,
                                                           time.time() - start))


if __name__ == '__main__':

    '''
    Runs when main.py is called.
    '''

    # Get arguments from params.yaml
    args = get_args()

    # Get discriminator model, define loss + optimizers
    discriminator = Discriminator(args)
    disc_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          label_smoothing=0.2)

    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


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
    summary_writer = tf.summary.create_file_writer(f'{args.LOG_DIR}/fit/{dir_time}')

    # Define the functions that get mapped onto the training tf.data.Dataset
    # These are imported from learning/datagenerator.py
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

    if args.TRAIN:
        fit(args, train_dataset, val_dataset, disc_loss_object)
