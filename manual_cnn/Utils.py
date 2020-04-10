# -*- coding: utf-8 -*-
# @Time    : 2020-04-07 18:19
# @Author  : jia_su_de_mo_tuo_che

import numpy as np
import h5py
from matplotlib import pyplot

DEBUGE = False


def flip_180(W):
    m, n_h, n_w, n_c = W.shape

    assert (n_h == n_w)
    e_1 = np.identity(n_h, dtype=int)[:, ::-1]

    return [np.dot(e_1, w).dot(e_1) for w in W]


def load_data_set(one_hot=True):

    train = h5py.File("datasets/train_signs.h5")
    test = h5py.File("datasets/test_signs.h5")

    train_x = train['train_set_x']
    train_y = train['train_set_y']
    test_x = test['test_set_x']
    test_y = test['test_set_y']

    if DEBUGE:
        pyplot.imshow(train_x[0])
        pyplot.show()

    train_x = train_x[:10]
    train_y = train_y[:10]

    if one_hot:
        classes = train["list_classes"].shape[0]
        train_y = np.identity(classes, dtype=int)[train_y]
        test_y = np.identity(classes, dtype=int)[test_y]

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    DEBUGE = True
    load_data_set()
