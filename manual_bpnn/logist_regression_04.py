# -*- coding: utf-8 -*-
# @Time    : 2020-01-15 22:06
# @Author  : mo_tuo_che

import numpy as np
import random
import math

extreme_min_number = 0


def main():
    x_list, y_list = generate_data()    # generate the data
    train(x_list, y_list)


def train(x, y):
    learning_rate = 0.1
    max_step = 10000

    w = np.array([random.random(), random.random()])
    bias = np.array([random.random()])

    for i in range(max_step):
        y_pred = inference(x, w, bias)

        dw, db = gradient(x, y, y_pred)
        w += -learning_rate * dw
        bias += -learning_rate * db

        loss = eval_loss(y, y_pred)
        if i % 100 == 0:
            print("i = %s dw=%s  db=%s  loss = %s" % (i, dw, db, loss))

    print("the final w=%s  bias=%s" % (w, bias))


def gradient(x, y, y_pred):
    y_diff = (y - y_pred).reshape((-1, 1))
    dw = x * y_diff
    db = 1 * y_diff

    dw_avg = dw.mean(axis=0)
    db_avg = db.mean(axis=0)

    return dw_avg, db_avg


def eval_loss(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):

        if y_pred[i] == 0:

            loss += y_true[i] * math.log(y_pred[i] + extreme_min_number) + (1.0 - y_true[i]) * math.log(1.0 - y_pred[i])
        elif y_pred[i] == 1:

            loss += y_true[i] * math.log(y_pred[i]) + (1.0 - y_true[i]) * math.log(1.0 - y_pred[i] + extreme_min_number)
        else:

            loss += y_true[i] * math.log(y_pred[i]) + (1.0 - y_true[i]) * math.log(1.0 - y_pred[i])

    return loss


def inference(x, w, bias):
    y_value = np.dot(x, w.T) + bias
    return sigmoid_function(y_value, label=0)


def generate_data():
    theta1 = 2
    theta2 = 3
    bias_1 = 1

    x_list = []
    y_list = []

    n_samples = 300
    for i in range(n_samples):
        x_1 = random.randint(-1, 1)
        x_2 = random.randint(-1, 1)
        y_value = theta1 * x_1 + theta2 * x_2 + bias_1

        y_label = sigmoid_function(y_value, label=1)

        x_list.append([x_1, x_2])
        y_list.append(y_label)

    return np.array(x_list), np.array(y_list)


def sigmoid_function(y_value, label):
    if label == 1:

        return 1 if 1 / (1 + math.exp(-y_value)) >= 0.5 else 0
    else:
        y_proba = []
        for i in range(len(y_value)):
            try:

                temp = 1 / (1 + math.exp(-y_value[i] + extreme_min_number))
                y_proba.append(temp)
            except Exception as ex:

                print(-y_value[i] + extreme_min_number, ex)
        return y_proba


if __name__ == "__main__":
    # test()
    main()
