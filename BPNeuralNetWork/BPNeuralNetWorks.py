# -*- coding: utf-8 -*-
# @Time    : 2020-01-17 01:10
# @Author  : jia_su_de_mo_tuo_che

import numpy as np
import random
import BreastLoader
import math


'''
    30:20:1
'''

MINIMUN_NUMBER = 0.0
LOG = True


class BPNNeuralClassification:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.bias = [np.random.randn(n, 1) for n in sizes[1:]]  # bias
        self.weights = [np.random.randn(c, r) for c, r in zip(sizes[1:], sizes[:-1])]  # weight

    def train(self, x_batch, y_batch, learning_rate=0.01, max_step=100):
        self.n_samples = len(x_batch)
        self.learning_rate = learning_rate

        for i in range(max_step):
            delta_w_batch = [np.zeros(w.shape) for w in self.weights]
            delta_b_batch = [np.zeros(b.shape) for b in self.bias]

            loss_sum = 0
            for x, y in zip(x_batch, y_batch):

                delta_w, delta_b, loss = self.back_propagation(x, y)

                delta_b_batch = [bb + dbb for bb, dbb in zip(delta_b_batch, delta_b)]
                delta_w_batch = [bw + dbw for bw, dbw in zip(delta_w_batch, delta_w)]
                loss_sum += loss

            self.weights = [w - dw/self.n_samples * learning_rate for w, dw in zip(self.weights, delta_w_batch)]
            self.bias = [b - db/self.n_samples * learning_rate for b, db in zip(self.bias, delta_b_batch)]

            if LOG:
                print("loss=%s" % loss_sum)

    def back_propagation(self, a, y):
        a = a.reshape((-1, 1))

        activations = [a]
        zs = []

        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)

            zs.append(z)
            activations.append(a)

        # back propagation, to calculate the loss function, and use the loss to calculate the delta_w, delta_b

        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.bias]

        loss = self.loss(y, activations[-1])

        for i in range(1, self.num_layers):  # -1, -2

            if i == 1:  # back_calculate the delta_w, delta_b, i==1 calculate the last layer's delta

                delta_b[-i] = loss * self.sigmoid_derivative(zs[-i])
                delta_w[-i] = np.dot(delta_b[-i], activations[-i-1].T)

            else:
                delta_b[-i] = np.dot(self.weights[-i+1].T, delta_b[-i+1]) * self.sigmoid_derivative(zs[-i])
                delta_w[-i] = np.dot(delta_b[-i], activations[-i-1].T)

        return delta_w, delta_b, loss

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss_dirivative(self, y, y_pred):
        return -(y * (1 / y_pred) + (1 - y) * (1 / (1-y_pred)))

    def loss(self, y, y_pred):
        loss = y * np.log(y_pred + MINIMUN_NUMBER) + (1-y) * np.log(1-y_pred + MINIMUN_NUMBER)
        return -loss

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def predict(self, x, y):
        z1 = np.dot(self.weights[0], x.T) + self.bias[0]
        a1 = self.sigmoid(z1)

        z2 = np.dot(self.weights[1], a1) + self.bias[1]
        a2 = self.sigmoid(z2)

        print(a2)
        print(y)


def run():
    x, target, feature_names, target_names = BreastLoader.load_cancer_data()

    size = [30, 1]
    model = BPNNeuralClassification(size)
    model.train(x, target)

    # model.predict(x, target)


if __name__ == "__main__":
    # test()
    run()
