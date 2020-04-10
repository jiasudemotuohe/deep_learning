# -*- coding: utf-8 -*-
# @Time    : 2020-01-12 19:55
# @Author  : AnYongYi

import cv2
import numpy as np

img_path_zhangyi = "./image/zhangyi.jpg"
img_path_dark = "./image/dark.jpg"


def gauss():
    img = cv2.imread(img_path_zhangyi)
    img_gauss = cv2.GaussianBlur(img, (7, 7), 1)
    cv2.imshow('img_gauss', img_gauss)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


def gauss_contrast():
    kernal = cv2.getGaussianKernel(17, 5)

    img = cv2.imread(img_path_zhangyi)
    img_gauss1 = cv2.GaussianBlur(img, (17, 17), 5)
    img_gauss2 = cv2.sepFilter2D(img, -1, kernal, kernal)
    cv2.imshow('img_g1', img_gauss1)
    cv2.imshow('img_g2', img_gauss2)

    key = cv2.waitKey()
    if key == 17:
        cv2.destroyAllWindows()


def kernal_derivative():
    kernal_lap = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
    img = cv2.imread(img_path_zhangyi)
    img_lap = cv2.filter2D(img, -1, kernal_lap)
    cv2.imshow('img_lap', img_lap)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


def kernal_sharpen():
    # 图像+edge=更锐利地图像，因为突出边缘
    img = cv2.imread(img_path_zhangyi)

    # kernel_sharp = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)
    # lap_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
    # cv2.imshow('sharp_lenna', lap_img)
    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()
    # 这样不对，因为，周围有4个1，中间是-3，虽然有边缘效果，但是周围得1会使得原kernel有滤波效果，使图像模糊；
    # 解决：所以取kernel_lap得相反数，再加上原图像，这样突出了中心像素，效果类似于小方差的高斯，所以
    #      可以既有边缘效果，又保留图像清晰度

    # kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # lap_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
    # cv2.imshow('sharp_lenna', lap_img)
    # key = cv2.waitKey()
    # if key == 27:
    #     cv2.destroyAllWindows()

    # 更“凶猛”的边缘效果
    # 不仅考虑x-y方向上的梯度，同时考虑了对角线方向上的梯度
    kernel_sharp = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)

    ######## Edge #########
    # x轴
    edgex = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    sharp_img = cv2.filter2D(img, -1, kernel=edgex)
    cv2.imshow('edgex_lenna', sharp_img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # gauss()
    # gauss_contrast()
    # kernal_derivative()
    kernal_sharpen()