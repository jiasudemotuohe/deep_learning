# -*- coding: utf-8 -*-
# @Time    : 2020-04-11 22:00
# @Author  : speeding_moto

from tensorflow import keras
import numpy as np


def main():

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    input_datas = keras.Input(shape=(32,))
    dense1 = keras.layers.Dense(64, activation='relu')
    x = dense1(input_datas)

    dense2 = keras.layers.Dense(128, activation='relu')
    x = dense2(x)

    out_puts = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=input_datas, outputs=out_puts, name="ming_model")
    model.summary()

    keras.utils.plot_model(model, 'first_model.png', show_shapes=True)


if __name__ == '__main__':
    data = np.reshape(range(-10, 10), newshape=(4, 5))
    print(data)

    # normalization
    # normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # print(normalized_data)
    #
    # normalization_data = (data - np.mean(data)) / np.max(data)
    # print(normalization_data)

    # normalized = np.normalize_axis_index(data, axis=0)
    normalized = np.normalize_axis_tuple(data, axis=0)
    print(normalized)

