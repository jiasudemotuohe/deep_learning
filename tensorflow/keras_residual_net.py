# -*- coding: utf-8 -*-
# @Time    : 2020-04-16 11:40
# @Author  : speeding_moto

from tensorflow import keras
import data_utils
import tensorflow_docs as tf_docs
from matplotlib import pyplot
import tensorflow_docs.plots
import tensorflow as tf

"""
use keras funciton api to imlements the cnn redisual network

"""


def identity_block(x, filters, kernel_sizes, strides, paddings, block):
    """
    identity block,
    """

    short_cut_x = x
    x = keras.layers.Conv2D(filters=filters[0], kernel_size=kernel_sizes[0], strides=strides[0], padding=paddings[0],
                            data_format='channels_last', activation=keras.activations.relu, name='main_route'+block +
                                                                                                 'a1')(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    conv2d = keras.layers.Conv2D(filters=filters[1], kernel_size=kernel_sizes[1], strides=strides[1], padding=paddings[1]
                                 , data_format='channels_last', use_bias=True, name='main_route'+block+'2a')

    """ 
    here we need to use the short_cut_x as the short_cut,
    if the shape of the short_cut_x not equal to the shape of the x, we need to use w*short_cut_x to reshape the 
    short_cut_x, thus we can add the sort_cut_x and x
    """

    x = keras.layers.BatchNormalization(axis=3)(x)
    x = conv2d(x) + short_cut_x

    return keras.layers.Activation(keras.activations.relu)(x)


def create_residual_model():
    img_inputs = keras.Input(shape=(64, 64, 3))

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(img_inputs)
    x = identity_block(x, filters=[64, 64], kernel_sizes=[(3, 3), [3, 3]], strides=[(1, 1), (1, 1)],
                       paddings=['same', 'same'], block='block_1')

    x = keras.layers.MaxPool2D()(x)
    x = identity_block(x, filters=[64, 64], kernel_sizes=[(3, 3), (3, 3)], strides=[(1, 1), (1, 1)],
                       paddings=['same', 'same'], block='block_2')

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=img_inputs, outputs=x)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model


def train(model):
    train_x, train_y, test_x, test_y = data_utils.load_gesture_symbol_data()

    call_backs = [keras.callbacks.TensorBoard(log_dir='logs')]
    history = model.fit(train_x, train_y, epochs=5, callbacks=call_backs, batch_size=128,
                        validation_data=(test_x, test_y))

    plooter = tf_docs.plots.HistoryPlotter()
    plooter.plot({"residual": history}, metric='accuracy')
    plooter.plot({"residual": history}, metric='loss')
    pyplot.show()


if __name__ == '__main__':
    model = create_residual_model()
    train(model)
