# -*- coding: utf-8 -*-
# @Time    : 2020-04-14 00:02
# @Author  : speeding_moto

import h5py


def load_gesture_symbol_data():
    train = h5py.File('datasets/train_signs.h5', 'r')
    test = h5py.File('datasets/test_signs.h5', 'r')

    train_x = train['train_set_x'][:]
    train_y = train['train_set_y'][:]
    test_x = test['test_set_x'][:]
    test_y = test['test_set_y'][:]

    # (1080, 64, 64, 3)(64, 64, 3)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    load_gesture_symbol_data()