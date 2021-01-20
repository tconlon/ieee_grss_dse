import tensorflow as tf
from ieee_grss_dse.learning.xception import xception_top


def last_layer(filters, size, strides, padding):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False,
            name='last_layer_with_softmax'
        )
    )

    result.add(tf.keras.layers.Softmax())

    return result


def conv2d_layer(filters, size, strides, padding, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.ELU())

    return result

def conv2dtranspose_layer(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=size,
                strides=strides,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def ts_conv_layer(imagery_list_by_ts, filters, size, strides, padding='same'):

    output_list = []
    conv_layer = conv2d_layer(filters=filters, size=size, strides=strides, padding=padding)

    for image in imagery_list_by_ts:
        output_list.append(conv_layer(image))

    return output_list

def ts_max_pool(imagery_list_by_ts, pool_size):

    output_list = []
    max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='valid')

    for image in imagery_list_by_ts:
        output_list.append(max_pool_layer(image))

    return output_list


def average_across_ts(imagery_list_by_ts):

    imagery_list = tf.keras.layers.Average()(imagery_list_by_ts)

    return imagery_list

def padding_2d_across_ts(imagery_list_by_ts, padding):
    output_list = []
    padding_2d_layer = tf.keras.layers.ZeroPadding2D(padding=padding)

    for image in imagery_list_by_ts:
        output_list.append(padding_2d_layer(image))

    return output_list


def conv2dlstm_layer_avg_and_reduce(concat_layer, filters, size):


    conv2dlstm = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=size,
                                            padding='same',
                                            return_sequences=True)

    out = conv2dlstm(concat_layer)

    # Average across the temporal dimension
    out = tf.math.reduce_mean(out, axis=1)

    ## Reduce channels
    out = conv2d_layer(filters=256, size=3, strides=1, padding='same')(out)
    out = conv2d_layer(filters=128, size=3, strides=1, padding='same')(out)
    out = conv2d_layer(filters=64, size=3, strides=1, padding='same')(out)
    out = conv2d_layer(filters=32, size=3, strides=1, padding='same')(out)

    out = last_layer(filters=4, size=3, strides=1, padding='same')(out)

    return out

def s2_convlstm_network(args):


    inp = tf.keras.layers.Input(
            shape=[args.n_s2_timesteps, args.INPUT_HEIGHT, args.INPUT_WIDTH, args.INPUT_S2_CHANNELS],
            name=f"input_image"
    )
    inp_list = []

    for i in range(args.n_s2_timesteps):
        inp_list.append(inp[:, i, ...])


    # Top level of UNET
    l1_imagery_list = ts_conv_layer(inp_list, filters=16, size=3, strides=1) # [list of (bn, 200, 200, 16)]
    l1_imagery_list = ts_conv_layer(l1_imagery_list, filters=16, size=3, strides=1) # [list of (bn, 200, 200, 16)]

    # Drop an imagery list to level 2
    l2_imagery_list = ts_max_pool(l1_imagery_list, pool_size=2) # [list of (bn, 100, 100, 16)]
    l2_imagery_list = ts_conv_layer(l2_imagery_list, filters=32, size=3, strides=1) # [list of (bn, 100, 100, 32)]
    l2_imagery_list = ts_conv_layer(l2_imagery_list, filters=32, size=3, strides=1) # [list of (bn, 100, 100, 32)]

    # Drop an imagery_list to level 3
    l3_imagery_list = ts_max_pool(l2_imagery_list, pool_size=2) # [list of (bn, 50, 50, 32)]
    l3_imagery_list = ts_conv_layer(l3_imagery_list, filters=64, size=3, strides=1) # [list of (bn, 50, 50, 64)]
    l3_imagery_list = ts_conv_layer(l3_imagery_list, filters=64, size=3, strides=1) # [list of (bn, 50, 50, 64)]

    # Drop an imagery_list to level 4
    l4_imagery_list = ts_max_pool(l3_imagery_list, pool_size=2)  # [list of (bn, 25, 25, 64)]
    l4_imagery_list = ts_conv_layer(l4_imagery_list, filters=128, size=3, strides=1)  # [list of (bn, 25, 25, 128)]
    l4_imagery_list = ts_conv_layer(l4_imagery_list, filters=128, size=3, strides=1)  # [list of (bn, 25, 25, 128)]

    # Drop an imagery_list to level 5
    l5_imagery_list = ts_max_pool(l4_imagery_list, pool_size=2)  # [list of (bn, 12, 12, 128)]
    l5_imagery_list = ts_conv_layer(l5_imagery_list, filters=256, size=3, strides=1)  # [list of (bn, 12, 12, 256)]
    l5_imagery_list = ts_conv_layer(l5_imagery_list, filters=256, size=3, strides=1)  # [list of (bn, 12, 12, 256)]

    # Drop imagery to level 6 and pad
    l6_imagery_list = ts_max_pool(l5_imagery_list, pool_size=2)  # [list of (bn, 6, 6, 256)]
    l6_imagery_list = padding_2d_across_ts(l6_imagery_list, padding=1) # [list of (bn, 8, 8, 256)]

    # One final max pooling to get spatial dimensions to align
    l7_imagery_list =  ts_max_pool(l6_imagery_list, pool_size=2)

    # Concatenate
    concat_layer = tf.keras.layers.Concatenate(axis=0)([tf.expand_dims(i, axis=0) for i in l7_imagery_list])
    # Reorder dims so that time dimension is second and overall dims are: (samples, time, rows, cols, channels)
    concat_layer = tf.transpose(concat_layer, perm=[1,0,2,3,4])

    # Input concatenated layer to LSTM + channel-reducing layers
    model_output = conv2dlstm_layer_avg_and_reduce(concat_layer, filters=256, size=3)

    return tf.keras.Model(inputs=inp, outputs=model_output)



def xception_model(args,):

    # Set up input layers
    inp = tf.keras.layers.Input(
            shape=[args.n_s2_timesteps, args.INPUT_HEIGHT, args.INPUT_WIDTH, args.INPUT_S2_CHANNELS],
            name=f"input_image"
    )

    inp_list = []
    out_list = []

    for i in range(args.n_s2_timesteps):
        inp_list.append(inp[:, i, ...])

    model = xception_top(args, pooling=None)

    for inp_img in inp_list:
        out_list.append(model(inp_img))

    concat_layer = tf.keras.layers.Concatenate(axis=0)([tf.expand_dims(i, axis=0) for i in out_list])
    concat_layer = tf.transpose(concat_layer, perm=[1,0,2,3,4])
    model_output = conv2dlstm_layer_avg_and_reduce(concat_layer, filters=256, size=3)

    model = tf.keras.Model(inputs=inp, outputs=model_output)

    return model


