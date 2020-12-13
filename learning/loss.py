import tensorflow as tf


def discriminator_loss(target, pred, disc_loss_object):
    '''
    Discriminator loss function.
    Binary cross entropy loss is applied.
    '''
    pred_loss = disc_loss_object(target, pred)

    return pred_loss