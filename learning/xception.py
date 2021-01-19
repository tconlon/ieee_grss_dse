# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""Xception V1 model for Keras.
On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.
Reference:
  - [Xception: Deep Learning with Depthwise Separable Convolutions](
      https://arxiv.org/abs/1610.02357) (CVPR 2017)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.util.tf_export import keras_export



layers = VersionAwareLayers()

def xception_top(args, pooling=None, ):
    """Instantiates the Xception architecture.
    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](
      https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Note that the default input image size for this model is 299x299.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For Xception, call `tf.keras.applications.xception.preprocess_input` on your
    inputs before passing them to the model.
    Arguments:

    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True,
      and if no `weights` argument is specified.

    Returns:
    A `keras.Model` instance.
    Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """

    img_input = tf.keras.layers.Input(
            shape=[args.INPUT_HEIGHT, args.INPUT_WIDTH, args.INPUT_S2_CHANNELS],
            name=f"input_image"
    )


    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = layers.Conv2D(
      32, (3, 3),
      strides=(2, 2),
      use_bias=False,
      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(
      128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(
      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(
      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
      256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
      728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
      1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = layers.BatchNormalization(
      axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(
      1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = layers.BatchNormalization(
      axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block13_pool')(x)
    x = layers.add([x, residual])


    x = layers.SeparableConv2D(
      1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = layers.BatchNormalization(
      axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    ### TC HAS CHANGED THIS LAYER TO A CONV KERNEL WITH 4x4 dimensions
    x = layers.SeparableConv2D(
      2048, (4, 4), padding='valid', use_bias=False, name='block14_sepconv2')(x)
    x = layers.BatchNormalization(
      axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)


    # if include_top:
    #     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #     # imagenet_utils.validate_activation(classifier_activation, weights)
    #     x = layers.Dense(classes, activation=classifier_activation,
    #                      name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #       x = layers.GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #       x = layers.GlobalMaxPooling2D()(x)


    inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name='xception')

    return model
