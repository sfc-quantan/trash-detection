# coding: utf-8
import gc
import keras
import numpy as np
from sklearn.cross_validation import train_test_split


NUMBER_OF_TRASH = 7


def y_split(y_array):
    y_xywh_t = np.zeros((len(y_array), NUMBER_OF_TRASH, 4))
    y_c_t = np.zeros((len(y_array), NUMBER_OF_TRASH))
    print(y_xywh_t.shape, y_c_t.shape)
    for i in range(len(y_array)):
        for j in range(len(y_array[0])):
            y_xywh_t[i][j] = y_array[i][j][:4]
            y_xywh_t[i][j][0:4:2]=y_xywh_t[i][j][0:4:2] / 640
            y_xywh_t[i][j][1:4:2]=y_xywh_t[i][j][1:4:2] / 480
            if(y_array[i][j][4] == 1):
                y_c_t[i][j] = 1

    return y_xywh_t, y_c_t

'''
def y_normalization(y_array):
    for i in range(len(y_array)):
        for j in range(len(y_array[0])):
            y_array[i][j][0] = y_array[i][j][0] / 640
            y_array[i][j][1] = y_array[i][j][1] / 480
            y_array[i][j][2] = y_array[i][j][2] / 640
            y_array[i][j][3] = y_array[i][j][3] / 480

    return y_array
'''

def make_data():
    data=np.load('data.npz')

    x = data['images']
    y = data['boxes']
    z = data['number']
    z = keras.utils.to_categorical(z, 8)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, z, train_size=0.8)

    (y_xywh_train, y_c_train) = y_split(y_train)
    (y_xywh_test, y_c_test) = y_split(y_test)

   # y_xywh_train = y_normalization(y_xywh_train)
   # y_xywh_test = y_normalization(y_xywh_test)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_xywh_train = y_xywh_train.reshape(-1, 28).astype('float32')
    y_xywh_test = y_xywh_test.reshape(-1, 28).astype('float32')

    del x, y
    gc.collect()

    return x_train, x_test, y_xywh_train, y_xywh_test, y_c_train, y_c_test, z_train, z_test
