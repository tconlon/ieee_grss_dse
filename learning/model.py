import tensorflow as tf

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.ELU())

    return result


def Discriminator(args):
    ### NEEDS TO BE FIXED TO ACCOUNT FOR PROPER SPATIAL DIMENSIONS

    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(
        shape=[args.INPUT_HEIGHT, args.INPUT_WIDTH, args.INPUT_CHANNELS],
        name="input_image"
    )

    down1 = downsample(64, 4, False)(inp)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    down4 = downsample(256, 4)(down3)  # (bs, 32, 32, 256)
    down5 = downsample(256, 4)(down4)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5)  # (bs, 34, 34, 256)

    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(4, 5, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
