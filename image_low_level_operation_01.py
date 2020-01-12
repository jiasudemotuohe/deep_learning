# -*- coding: utf-8 -*-
# @Time    : 2020-01-11 13:23
# @Author  : AnYongYi

import cv2
import numpy as np
import random
import pandas

image_path = './image/dark.jpg'
image_dark_path = './image/zhangyi.jpg'


def image_crop():
    img = cv2.imread(image_path)

    img_crop = img[30:280, 200:493]
    print('img.shape=%s' % str(img.shape))
    print('img_crop.shape=%s' % str(img_crop.shape))

    cv2.imshow('img', img)
    cv2.imshow('img_crop', img_crop)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


def crop_test():
    data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                     [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])
    print(data.shape)
    data_crop = data[0:3, 0:2]
    print(data_crop)


def image_split():
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)

    cv2.imshow('b_img', b)
    cv2.imshow('g_img', g)
    cv2.imshow('r_img', r)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


def be_lighter(img):
    # print('img.shape:  %s' % str(img.shape))
    random_int = random.randint(0, 100)

    if random_int == 0:
        pass
    elif random_int > 0:
        limit = 255 - random_int
        #不可直接+255，否则可能会出现黑点等乱象
        img[img > limit] = 255
        img[img <= limit] = random_int + img[img <= limit]

    return img


def image_split_merge():
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)

    b_transformed = be_lighter(b)
    g_transformed = be_lighter(g)
    r_transformed = be_lighter(r)

    img_merge = cv2.merge((b_transformed, g_transformed, r_transformed))
    cv2.imshow('img_merge', img_merge)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


# gamma adjust
def img_dark_lighter():
    img_dark = cv2.imread(image_path)

    table = []
    gamma = 1.0/1.5

    for i in range(256):
        # table.append(((i / 255.0) ** gamma) * 255)
        table.append(((i / 255.0) ** gamma) * 255)

    table = np.array(table).astype('uint8')
    img_lighter = cv2.LUT(img_dark, table)

    cv2.imshow('dark', img_dark)
    cv2.imshow('lighter', img_lighter)
    key = cv2.waitKey()

    if key == 27:
        cv2.destroyAllWindows()


def rotation():
    img = cv2.imread(image_dark_path)

    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 20, 0.8)
    image_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow('image_rotated', image_rotated)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # crop_test()
    # image_crop()
    # image_split()
    # image_split_merge()

    # img_dark_lighter()
    rotation()