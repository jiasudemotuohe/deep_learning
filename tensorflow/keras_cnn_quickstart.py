# -*- coding: utf-8 -*-
# @Time    : 2020-04-13 16:56
# @Author  : speeding_moto

import data_utils
from matplotlib import pyplot
from tensorflow import keras


def train():
    train_images, train_labels, test_images, test_labels = data_utils.load_gesture_symbol_data()

    sequential = keras.Sequential()
    sequential.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                       input_shape=(64, 64, 3), activation='relu'))
    sequential.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    sequential.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    sequential.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    sequential.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    # here is the fully connection layer

    sequential.add(keras.layers.Flatten())
    sequential.add(keras.layers.Dense(64, activation='relu'))
    sequential.add(keras.layers.Dense(6))

    sequential.summary()

    sequential.compile(optimizer='adam',
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    call_backs = [keras.callbacks.TensorBoard(log_dir='logs')]

    history = sequential.fit(train_images, train_labels, epochs=100, callbacks=call_backs,
                             validation_data=[test_images, test_labels])

    pyplot.plot(history.history['accuracy'], label='accuracy')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('accuracy')
    pyplot.legend(loc='lower right')
    pyplot.show()

    sequential.save('./weights/sequential')
# def load_cifar_data():
#     (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
#
#     # with withopen("./datasets/cifar-10-python.tar", mode='rb') as fo:
#     #     dict = pickle.load(fo, encoding='bytes')
#     #
#     #     print(dict)
#     return train_images, train_labels, test_images, test_labels
#     # return None,None,None,None


def explore_data():
    train_images, train_labels, test_images, test_labels = data_utils.load_data()
    """

    :param train_images:  (64, 64, 3)
    :param train_labels: 5
    """
    print(train_images.shape, train_images[0].shape, train_labels[0])

    pyplot.figure(figsize=(10, 10))
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.xlabel([train_labels[i]])
        pyplot.yticks([])
        pyplot.ylim((0, 64))
        pyplot.imshow(train_images[i])
    pyplot.show()


if __name__ == '__main__':

    # explore_data()
    train()
    train_images, train_labels, test_images, test_labels = data_utils.load_gesture_symbol_data()

    sequential = keras.models.load_model('./weights/sequential')
    loss, acc = sequential.evaluate(test_images, test_labels)
    print("loss=%s   acc=%s" % (loss, acc))

