# -*- coding: utf-8 -*-
# @Time    : 2020-04-11 19:17
# @Author  : speeding_moto

import tensorflow as tf
from tensorflow import keras
import numpy as np
import eunite_data


def store_model(model):
    model.save_weights('/weight/model')


def load_model(model):
    return model.load_weight('./weight/model')


def train(train_x, train_y, test_x, test_y):

    model = keras.Sequential()
    model.add(keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                 bias_regularizer=tf.keras.regularizers.l1(0.01)))

    # model.add(keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    #                              bias_regularizer=tf.keras.regularizers.l1(0.01)))
    #
    model.add(keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                 bias_regularizer=tf.keras.regularizers.l1(0.01)))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.MSE,
                  metrics=['mae', 'mse'])

    call_backs = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
                  tf.keras.callbacks.TensorBoard(log_dir='./logs')]

    model.fit(train_x, train_y, epochs=5000, batch_size=500, callbacks=call_backs,
              validation_data=(test_x, test_y))

    return model


def predict(model, test_x):
    y_predict = model.predict(test_x)
    return y_predict


def main():
    train_x, train_y, test_x, test_y = eunite_data.load_eunite_train_data()
    model = train(train_x, train_y, test_x, test_y)

    # y_predict = predict(model, train_x)
    #
    # loss = model.evaluate(test_x, test_y)
    # print("evaluate=%s" % loss)

    model.save('./model/load_model')
    model.summary()


if __name__ == '__main__':
    main()

