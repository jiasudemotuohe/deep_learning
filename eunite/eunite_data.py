# -*- coding: utf-8 -*-
# @Time    : 2020-04-11 12:34
# @Author  : speeding_moto

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


EUNITE_PATH = "dataset/eunite.xlsx"
PARSE_TABLE_NAME = "mainData"


def load_eunite_data():
    """
    return the generated load data, include all the features wo handle

    """
    data = open_file()
    X, Y = generate_features(data)

    return X.values, Y.values


def load_eunite_train_data():
    X, Y = load_eunite_data()

    trains_test_rate = int(len(X) * 0.7)

    train_x = X[0: trains_test_rate]
    train_y = Y[0: trains_test_rate]

    test_x = X[trains_test_rate:]
    test_y = Y[trains_test_rate:]

    return train_x, train_y, test_x, test_y


def generate_features(df):
    """
    parse the data, wo need to transfer the class number to ont_hot for our calculate later

    """
    months = df["Month"]
    days = df["Day"]

    one_hot_months = cast_to_one_hot(months, n_classes=12)
    days = cast_to_one_hot(days, n_classes=31)
    one_hot_months = pd.DataFrame(one_hot_months)
    days = pd.DataFrame(days)

    df = pd.merge(left=df, right=one_hot_months, left_index=True, right_index=True)
    df = pd.merge(left=df, right=days, left_index=True, right_index=True)

    y = df['Max Load']

    # think, maybe wo need to normalization the temperature data,
    temperature = normalization(df['Temp'].values)
    temperature = pd.DataFrame(temperature)

    df = pd.merge(left=df, right=temperature, left_index=True, right_index=True)

    drop_columns = ["ID", "Month", "Day", "Year", "Max Load", "Temp"]

    df.drop(drop_columns, axis=1, inplace=True)

    print(df[0:10], "\n", y[0])
    return df, y


def normalization(data):
    return (data - np.mean(data)) / np.max(np.abs(data))


def cast_to_one_hot(data, n_classes):
    """
    cast the classifier data to one hot

    """
    one_hot_months = np.eye(N=n_classes)[[data - 1]]
    return one_hot_months


def show_month_temperature_load_image(df):
    plt.title("relation of temperature and load")

    max_load = df["Max Load"]
    temp = df['Temp'] * 15

    plt.plot(max_load)
    plt.plot(temp)
    plt.xlabel('time')
    plt.annotate('temperature', xy=[200, 200], xytext=(300, 200))
    plt.annotate('load', xy=[200, 600], xytext=(200, 800))

    plt.show()


def open_file():
    """
    open the eunite load excel file to return
    """
    xlsx_file = pd.ExcelFile(EUNITE_PATH)
    return xlsx_file.parse(PARSE_TABLE_NAME)


if __name__ == '__main__':
    df = open_file()

    show_month_temperature_load_image(df)

    x, y  = load_eunite_data()

    print(x.shape)
