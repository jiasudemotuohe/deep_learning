# -*- coding: utf-8 -*-
# @Time    : 2020-04-12 00:50
# @Author  : speeding_moto

from matplotlib import pyplot
from tensorflow import keras
import eunite_data


def draw_image(y_predit, y_true):

    pyplot.plot(y_predit)
    pyplot.plot(y_true)
    pyplot.title("Prediction Fit Graph")
    pyplot.show()


if __name__ == '__main__':

    x, y = eunite_data.load_eunite_data()

    model = keras.models.load_model("./model/load_model")
    y_predict = model.predict(x)

    draw_image(y_predict, y)
