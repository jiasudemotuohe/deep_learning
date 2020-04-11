# -*- coding: utf-8 -*-
# @Time    : 2020-04-03 19:42
# @Author  : jia_su_de_mo_tuo_che

import numpy as np
import utils
import time

MINIMUN_NUMBER = 0.000001


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

    hyper_parameter1 = {"stride": 1, "pad": 1}
    hyper_parameter2 = {"stride": 2, "pad": 0, "filter_size": 2}

    return [w1, w2], [b1, b2], [hyper_parameter1, hyper_parameter2]


# def conv_backforward(cache, delta_z, W, hyper_parameter):
#     (z, a_prev_paded, W, B, hyper_parameter) = cache
#
#     stride = hyper_parameter["stride"]
#     pad = hyper_parameter['pad']
#
#     n_z, h_z, w_z, c_z = z.shape
#     n_a, h_a, w_a, c_a = a_prev_paded.shape
#     n_f, h_f, w_f, c_f = W.shape
#
#     pad_back = int(((h_a - 1) * stride + h_f - h_z) / 2)
#     z_paded = zero_pad(z, pad_back)
#
#     W_fliped = Utils.flip_180(W)
#     delta_a = np.zeros(a_prev_paded.shape)
#
#     for i in range(n_z):
#
#         vertical_start, vertical_end = 0, h_f
#         for j in range(h_a):
#
#             horizontal_start, horizontal_end = 0, w_f
#             for k in range(w_a):
#
#                 for c in range(n_f):
#
#                     z_slice = z_paded[i][vertical_start: vertical_end, horizontal_start: horizontal_end, c:c+1]
#                     delta_a[i][j][k] += np.sum(z_slice * W_fliped[c], axis=(0, 1))
#
#                 horizontal_start += 1
#                 horizontal_end += 1
#
#             vertical_start += 1
#             vertical_end += 1
#
#     # we calculate the delta a finally, but it's the padded data, so we need to cut out the original data,
#     return delta_a[:, pad: -pad, pad: -pad, :]


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
    # a = relu_activation_function(z)

    return z


def relu_activation_function(z):
    z[z < 0] = 0
    return z


def cross_entropy_loss(y_pred, y):
    loss = [-np.sum(y_true * np.log(y_p + MINIMUN_NUMBER))for y_p, y_true in zip(y_pred, y)]
    return np.sum(loss)


def soft_max(z):
    z = z - np.max(z)
    return np.array([np.exp(item) / np.sum(np.exp(item)) for item in z])


def soft_max_loss_derivative(y_pred, y):
    return y_pred - y


def back_propagation(y_pred, train_y, ws, bs, activations, parameters):
    """
    delta_b1_batch: here delta_b1_batch is the derivative after the max_pooling, we need use this number to compute the
                    previous derivative

    delta_w2_batch: this is the derivative of the full_connection layer


    """
    m, classes = y_pred.shape

    a1 = np.reshape(activations[-1], newshape=(m, -1))  # 10240 * 6
    z1_repeats = np.repeat(activations[-1], repeats=2, axis=2).repeat(repeats=2, axis=1)
    z1_mask = np.equal(activations[-2], z1_repeats)

    delta_b2_batch = soft_max_loss_derivative(y_pred, train_y)
    delta_w2_batch = []
    delta_b1_batch = []
    delta_w1_batch = []

    for i in range(m):
        dz2 = np.array(delta_b2_batch[i]).reshape(6, 1)
        a1_item = np.array(a1[i]).reshape(1, -1)

        delta_b1 = np.dot(ws[1].T, dz2).reshape(32, 32, 10)
        delta_b1_reverse_max_pool = np.repeat(delta_b1, repeats=2, axis=0).repeat(repeats=2, axis=1)

        delta_b1 = delta_b1_reverse_max_pool * z1_mask[i]
        delta_w1 = signal_conv_back_forward(activations[-3][i], delta_b1, ws[0], parameters[0])

        delta_w2_batch.append(np.dot(dz2, a1_item))

        delta_b1 = np.sum(delta_b1, axis=(0, 1), keepdims=True)
        delta_b1_batch.append(delta_b1)
        delta_w1_batch.append(delta_w1)

    dw2 = np.mean(delta_w2_batch, axis=0)
    dw1 = np.mean(delta_w1_batch, axis=0)

    db2 = np.mean(delta_b2_batch, axis=0)
    db1 = np.mean(delta_b1_batch, axis=0)
    return [dw1, dw2], [db1.reshape(bs[0].shape), db2.reshape(bs[1].shape)]

    # w2_shape = np.array(delta_w2_batch).shape
    # b1_shape = np.array(delta_b1_batch).shape
    # w1_shape = np.array(delta_w1_batch).shape
    # print(w2_shape, b1_shape, w1_shape)


def signal_conv_back_forward(a_prev, delta_b, w, params):
    stride = params['stride']

    h_delta_b, w_delta_b, c_delta_b = delta_b.shape
    n_kernel, h_kernel, w_kernel, c_kernel = w.shape

    delta_w = np.zeros(w.shape)

    for i in range(h_delta_b):
        vertical_start, vertical_end = 0, h_kernel

        for j in range(w_delta_b):
            horizontal_start, horizontal_end = 0, w_kernel

            a_slice = a_prev[vertical_start: vertical_end, horizontal_start: horizontal_end, :]

            for k in range(n_kernel):
                b = delta_b[i:i+1, j:j+1, k:k+1]

                delta_w[k] += b * a_slice

            horizontal_start += stride
            horizontal_end += stride

        vertical_start += stride
        vertical_end += stride

    return delta_w


def train(learning_rate=0.01, max_step=5):
    train_x, train_y, test_x, test_y = utils.load_data_set(one_hot=True)

    ws, bs, hyper_parameters = init_variable()

    for step in range(max_step):

        z1 = conv_forward(train_x, ws[0], bs[0], hyper_parameters[0])
        a1 = pool_forward(z1, hyper_parameters[1], mode="max")

        a2 = fully_connection_nn(a1, ws[1], bs[1])
        y_pred = soft_max(a2.T)

        activations = [zero_pad(train_x, hyper_parameters[0]["pad"]), z1, a1]
        dws, dbs = back_propagation(y_pred, train_y, ws, bs, activations, hyper_parameters)

        ws = [w - learning_rate * dw for w, dw in zip(ws, dws)]
        bs = [b - learning_rate * db for b, db in zip(bs, dbs)]

        print("step=%s loss is %s " % (step, cross_entropy_loss(y_pred, train_y)))


if __name__ == "__main__":
    train()
    print("cost time is %s" % time.clock())
