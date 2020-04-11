# -*- coding: utf-8 -*-
# @Time    : 2020-01-17 17:42
# @Author  : jia_su_de_mo_tuo_che

from sklearn.datasets import load_breast_cancer
import numpy as np


def load_cancer_data():
    data = load_breast_cancer(return_X_y=False)

    feature_names = data['feature_names']
    target_names = data['target_names']
    target = data['target']
    x = data['data']

    n_max = np.max(x)
    n_min = np.min(x)

    x_normalize = (x - n_min) / (n_max - n_min)

    return x_normalize, target.reshape(-1, 1), feature_names, target_names


if __name__ == "__main__":
    load_cancer_data()