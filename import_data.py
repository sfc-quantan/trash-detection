# coding: utf-8
from sklearn.cross_validation import train_test_split
import numpy as np
import keras
import gc


def make_data():
    gominum = 7

    x = np.load("x.npy")
    y = np.load("y.npy")
    z = np.load("z.npy")
    z = keras.utils.to_categorical(z, 8)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, z, train_size=0.8)

    y_xywh_train = np.zeros((len(y_train), gominum, 4))
    y_c_train = np.zeros((len(y_train), gominum))
    for i in range(len(y_train)):
        for j in range(len(y_train[0])):
            y_xywh_train[i][j] = y_train[i][j][:4]
            if(y_train[i][j][4] == 1):
                y_c_train[i][j] = 1

    y_xywh_test = np.zeros((len(y_test), gominum, 4))
    y_c_test = np.zeros((len(y_test), gominum))
    for i in range(len(y_test)):
        for j in range(len(y_test[0])):
            y_xywh_test[i][j] = y_test[i][j][:4]
            if(y_test[i][j][4] == 1):
                y_c_test[i][j] = 1

    for i in range(len(y_xywh_train)):
        for j in range(len(y_xywh_train[0])):
            y_xywh_train[i][j][0] = y_xywh_train[i][j][0]/640
            y_xywh_train[i][j][1] = y_xywh_train[i][j][1]/480
            y_xywh_train[i][j][2] = y_xywh_train[i][j][2]/640
            y_xywh_train[i][j][3] = y_xywh_train[i][j][3]/480

    for i in range(len(y_xywh_test)):
        for j in range(len(y_xywh_test[0])):
            y_xywh_test[i][j][0] = y_xywh_test[i][j][0]/640
            y_xywh_test[i][j][1] = y_xywh_test[i][j][1]/480
            y_xywh_test[i][j][2] = y_xywh_test[i][j][2]/640
            y_xywh_test[i][j][3] = y_xywh_test[i][j][3]/480

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_xywh_train = y_xywh_train.reshape(-1, 28).astype('float32')
    y_xywh_test = y_xywh_test.reshape(-1, 28).astype('float32')

    y_c_train = y_c_train.reshape(-1, 7)
    y_c_test = y_c_test.reshape(-1, 7)

    del x, y
    gc.collect()

    return x_train, x_test, y_xywh_train, y_xywh_test, y_c_train, y_c_test, z_train, z_test
