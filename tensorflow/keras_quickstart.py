# -*- coding: utf-8 -*-
# @Time    : 2020-04-12 18:32
# @Author  : speeding_moto

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import tensorflow_docs
import tensorflow_docs.modeling
import pandas
import tensorflow_docs.plots
import numpy as np

# /Users/anyongyi/PycharmProjects/deep_learning/tensorflow/datasets/cifar-10-python.tar

def train():
    # (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.fashion_mnist.load_data()
    (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.mnist.load_data()

    trains_image = trains_image / 255.0
    tests_image = tests_image / 255.0

    sequential = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(10)])

    sequential.compile(optimizer=keras.optimizers.Adam(0.01),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

    call_backs = [tf.keras.callbacks.TensorBoard(log_dir='./fashion_logs'),
                  tensorflow_docs.modeling.EpochDots()]

    histroy = sequential.fit(trains_image, trains_labels, epochs=200, callbacks=call_backs,
                             validation_data=[tests_image, test_labels])

    hist = pandas.DataFrame(histroy.history)
    print(hist.tail())

    plooter = tensorflow_docs.plots.HistoryPlotter(smoothing_std=2)
    plooter.plot({"basic": histroy}, metric='acc')
    pyplot.ylim([0, 1])
    pyplot.ylabel("ACC")
    pyplot.show()

    probability_model = keras.Sequential([sequential, keras.layers.Softmax()])
    y_predict_labels = probability_model.predict(tests_image)

    print(y_predict_labels[0:5])
    for i in range(10):
        print((np.argmax(y_predict_labels[i])), test_labels[i])


# def explore_image():
#     (trains_image, trains_labels), (tests_image, test_labels) = keras.datasets.mnist.load_data()
#
#     pyplot.figure(figsize=(10, 10))
#
#     for i in range(30):
#         pyplot.subplot(5, 6, i+1)
#         pyplot.grid(False)
#         pyplot.xticks([])
#         pyplot.yticks([])
#         pyplot.imshow(trains_image[i], cmap=pyplot.cm.binary)
#         pyplot.xlabel(trains_labels[i])
#
#
#
#     pyplot.show()


if __name__ == '__main__':
    # explore_image()
    train()
