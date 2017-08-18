# coding: utf-8
from keras import backend as K


def org_mse(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 7, 4))
    xy_true = y_true[:, :, 0:2]
    wh_true = y_true[:, :, 2:4]

    y_pred = K.reshape(y_pred, (-1, 7, 4))
    xy_pred = y_pred[:, :, 0:2]
    wh_pred = y_pred[:, :, 2:4]

    temp = K.square(xy_pred - xy_true)
    temp = temp + K.square(K.sqrt(wh_pred) - K.sqrt(wh_true))
    return temp
