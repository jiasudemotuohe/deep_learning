# -*- coding: utf-8 -*-
# @Time    : 2020-04-16 00:09
# @Author  : speeding_moto

from tensorflow import keras
import tensorflow as tf
import data_utils
from matplotlib import pyplot


def model(input_shape):

    img_inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="valid",
                            activation='relu', name='first_convolution')(img_inputs)

    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, )(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="valid",
                            activation='relu', name='second_convolution')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10)(x)

    return keras.Model(inputs=img_inputs, outputs=x, name='keras_fa_model')


def train(model):
    train_x, train_labels, test_x, test_labels = data_utils.load_gesture_symbol_data()

    call_backs = [keras.callbacks.TensorBoard('logs')]

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(train_x, train_labels, callbacks=call_backs, epochs=10, batch_size=128)
    draw_history(history)


def draw_history(history):
    pyplot.plot(history.history['loss'], color='black')
    pyplot.plot(history.history['accuracy'], color='black',)
    pyplot.show()



if __name__ == '__main__':
    model = model((64, 64, 3))
    model.summary()
    # keras.utils.plot_model(model, 'first_model_graph.png')
    train(model)