# -*- coding: utf-8 -*-
# @Time    : 2020-04-03 19:42
# @Author  : jia_su_de_mo_tuo_che

import numpy as np
import sys
import Utils
import h5py
import time

MINIMUN_NUMBER = 0.0000001


def zero_pad(x, pad):
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), "constant", constant_values=0)


def single_convolution_step(a_slice, w, b):
    s = np.multiply(a_slice, w) + b
    return np.sum(s, axis=None)


def conv_forward(a_prev, weights, biases, hyper_parameter):
    """
    convolution layer

    """

    m, h_prev, w_prev, c_prev = a_prev.shape
    m_filter, h_filter, w_filter, c_filter = weights.shape

    stride = hyper_parameter["stride"]
    pad = hyper_parameter["pad"]

    n_h = int((h_prev - h_filter + 2 * pad) / stride + 1)
    n_w = int((w_prev - w_filter + 2 * pad) / stride + 1)

    a_prev = zero_pad(a_prev, pad)
    z = np.zeros(shape=(m, n_h, n_w, m_filter))

    for i in range(m):

        vertical_start, vertical_end = 0, h_filter
        for j in range(n_h):

            horizontal_start, horizontal_end = 0, w_filter
            for k in range(n_w):
                a_slice = a_prev[i][vertical_start: vertical_end, horizontal_start: horizontal_end, :]

                z[i][j][k] = [single_convolution_step(a_slice, w, b) for w, b in zip(weights, biases)]

                horizontal_start += stride
                horizontal_end += stride

            vertical_start += stride
            vertical_end += stride

    assert (z.shape == (m, n_h, n_w, m_filter))

    return z


def pool_forward(z, hyper_parameter, mode="max"):

    """
    use the max_pooling to get the new pooled data, mean pooling is not accomplish yet

    """
    filter_size = hyper_parameter['filter_size']
    stride = hyper_parameter["stride"]

    m, n_h, n_w, n_channel = z.shape

    h = int((n_h - filter_size) / stride + 1)
    w = int((n_w - filter_size) / stride + 1)

    a = np.zeros((m, h, w, n_channel))

    if mode != "max":  # only finished the max pooling code
        raise Exception("invalid  mode type")

    if stride != filter_size:
        raise Exception("invalid stride, stride need to equals the filter_size")

    for i in range(m):
        vertical_start, vertical_end = 0, filter_size

        for j in range(h):

            horizontal_start, horizontal_end = 0, filter_size
            for k in range(w):
                a_slice = z[i][vertical_start: vertical_end, horizontal_start:horizontal_end, :]

                m1 = np.max(a_slice, axis=0)
                m2 = np.max(m1, axis=0)

                horizontal_start += stride
                horizontal_end += stride
                a[i][j][k] = m2

            vertical_start += stride
            vertical_end += stride

    return a


def init_variable():
    w1 = np.random.randn(10, 3, 3, 3)
    b1 = np.random.randn(10, 1, 1, 1)

    w2 = np.random.randn(6, 10240)
    b2 = np.random.randn(6, 1)

    return [w1, w2], [b1, b2]


def conv_backforward(cache):
    (z, a_prev_paded, W, B, hyper_parameter) = cache

    stride = hyper_parameter["stride"]
    pad = hyper_parameter['pad']

    n_z, h_z, w_z, c_z = z.shape
    n_a, h_a, w_a, c_a = a_prev_paded.shape
    n_f, h_f, w_f, c_f = W.shape

    pad_back = int(((h_a - 1) * stride + h_f - h_z) / 2)
    z_paded = zero_pad(z, pad_back)

    W_fliped = Utils.flip_180(W)
    delta_a = np.zeros(a_prev_paded.shape)

    for i in range(n_z):

        vertical_start, vertical_end = 0, h_f
        for j in range(h_a):

            horizontal_start, horizontal_end = 0, w_f
            for k in range(w_a):

                for c in range(n_f):

                    z_slice = z_paded[i][vertical_start: vertical_end, horizontal_start: horizontal_end, c:c+1]
                    delta_a[i][j][k] += np.sum(z_slice * W_fliped[c], axis=(0, 1))

                horizontal_start += 1
                horizontal_end += 1

            vertical_start += 1
            vertical_end += 1

    # we calculate the delta a finally, but it's the padded data, so we need to cut out the original data,
    return delta_a[:, pad: -pad, pad: -pad, :]


def pool_backforward(delta, cache):
    """
    Notes
    -----
    here we just accomplish the max pool_backforward, and the max_pool stride need to equals the filter_hight & width

    """

    a_prev, hyper_parameter = cache

    m, n_h, n_w, n_c = delta.shape

    pool_f = hyper_parameter['f']
    stride = hyper_parameter["stride"]

    assert (pool_f == stride)


def fully_connection_nn(a, w, b):
    """
    First:
    we need to transform the original data from 3 Dimension to 2 Dimension

    Second:
    we have to init the original W
    --------
    """
    n, n_h, h_w, n_c = a.shape

    a = np.reshape(a, newshape=(n, -1))

    z = np.dot(w, a.T) + b
    a = relu_activation_function(z)

    return a


def relu_activation_function(z):
    z[z < 0] = 0
    return z


def cross_entropy_loss(y_pred, y):
    loss = [-np.sum(y_true * np.log(y_p + MINIMUN_NUMBER))for y_p, y_true in zip(y_pred, y)]
    return loss


def soft_max(z):
    return [item / np.sum(item) for item in z]


def soft_max_loss_derivative(y_pred, y):
    # when
    loss_derivative = y_pred - y

    return loss_derivative


def back_propagation(y_pred, train_y, z2):

    delta_z2 = soft_max_loss_derivative(y_pred, train_y)
    dw3 = [np.dot(delta_item.T, z2.item) for delta_item, z2_item in zip(delta_z2, z2.T)]
    print(dw3.shape)


def main():
    train_x, train_y, test_x, test_y = Utils.load_data_set(one_hot=True)
    ws, bs = init_variable()

    z1 = conv_forward(train_x, ws[0], bs[0], hyper_parameter={"stride": 1, "pad": 1})
    a1 = pool_forward(z1, hyper_parameter={"filter_size": 2, "stride": 2}, mode="max")
    #
    z2 = fully_connection_nn(a1, ws[1], bs[1])
    y_pred = soft_max(z2.T)

    back_propagation(y_pred, train_y, z2)
    loss = cross_entropy_loss(y_pred, train_y)

    # pool_backforward(delta_a, cache_pool)
    # delta_a, dw, db = conv_backforward(dz, cache)
    # print(z2.shape, z2.T)



if __name__ == "__main__":
    main()
    print("cost time is %s" % time.clock())

