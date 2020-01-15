# -*- coding: utf-8 -*-
# @Time    : 2020-01-13 21:44
# @Author  : AnYongYi
'''linear regression demo'''

import random
import numpy as np


def main():

    x_list, y_list = generate_data()
    train(x_list, y_list)


def generate_data():
    theta_1 = 10
    bias_1 = 5

    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(1, 10)
        y = x * theta_1 + bias_1 + random.random() * random.randint(-1, 1)

        x_list.append(x)
        y_list.append(y)
    return np.array(x_list).reshape((-1, 1)), np.array(y_list)


def train(x, y):
    max_step = 1000
    learning_rate = 0.01
    w = [random.random()]
    b = [random.random()]

    for i in range(max_step):
        y_pred = inference(x, w, b)

        dw, db = gradient(y, y_pred, x, b)

        w -= learning_rate * dw
        b -= learning_rate * db

        loss = (y - y_pred) ** 2
        loss = loss.reshape((-1, 1))

        print("i=%s,  loss=%s" % (i, np.sum(loss)))
    print("the final theta and bais is %s   %s  " % (w, b))


def inference(x, w, bias):
    return np.dot(x, w) + bias


def gradient(y_true, y_pred, x, b):
    loss = (y_pred - y_true)
    loss = np.reshape(np.array(loss), (-1, 1))

    dw = loss * x
    db = loss * 1

    dw_avg = np.nanmean(dw, axis=0)
    db_avg = np.nanmean(db, axis=0)

    return dw_avg, db_avg


def test():
    x = [[1, 2], [3, 4]]
    w = [1, 2]

    x = np.array(x)
    w = np.array(w)

    mean = np.nanmean(x, axis=1)
    print(mean)


if __name__ == "__main__":
    # test()
    main()