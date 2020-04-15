# -*- coding: utf-8 -*-
# @Time    : 2020-04-12 00:50
# @Author  : speeding_moto

from matplotlib import pyplot
from tensorflow import keras
import eunite_data
import numpy as np


def draw_image(y_predit, y_true):

    pyplot.plot(y_predit, color='blue')
    pyplot.plot(y_true, color='red')
    pyplot.title("Prediction Fit Graph")
    pyplot.show()


def analysis_result():
    """
    use the model predict the test data , then  analysis the y_predict and y_ture
    """
    train_x, train_y, test_x, test_y = eunite_data.load_eunite_train_data()

    model = keras.models.load_model("./model/load_model")
    # test_x.shape = 229 * 13
    y_predict = model.predict(train_x)

    draw_image(y_predict, train_y)

    mae = np.mean(np.abs(y_predict - train_y))
    mae_percent = np.mean(mae / train_y)

    print("example_number =%s, mean mae= %s" % (test_x.shape[0], mae))
    print("example_number =%s, mean mae percent= %s" % (test_x.shape[0], mae_percent))


if __name__ == '__main__':
    analysis_result()

