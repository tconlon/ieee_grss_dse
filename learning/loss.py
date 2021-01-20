import tensorflow as tf

def crossentropy_loss(model_output, target):
    '''
    model_output and target are both tensors of shape: (batch, 4, 4, 4)
    '''

    loss = tf.keras.losses.categorical_crossentropy(target, model_output)

    mean_loss = tf.math.reduce_sum(loss)

    return mean_loss
