# -*- coding: utf-8 -*-
# @Time    : 2020-04-12 18:32
# @Author  : speeding_moto

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot


def train():
    (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.fashion_mnist.load_data()
    # (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.mnist.load_data()

    trains_image = trains_image / 255.0
    tests_image = tests_image / 255.0

    sequential = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(10)])

    sequential.compile(optimizer=keras.optimizers.Adam(0.01),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['accuracy'])

    call_backs = [tf.keras.callbacks.TensorBoard(log_dir='./fashion_logs')]

    sequential.fit(trains_image, trains_labels, epochs=100, batch_size=1000, callbacks=call_backs)


def explore_image():
    (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.mnist.load_data()

    pyplot.figure(figsize=(10, 10))

    for i in range(30):
        pyplot.subplot(5, 6, i+1)
        pyplot.grid(False)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(trains_image[i], cmap=pyplot.cm.binary)
        pyplot.xlabel(trains_labels[i])



    pyplot.show()


if __name__ == '__main__':
    train()
    # explore_image()