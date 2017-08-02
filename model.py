# coding: utf-8
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, LeakyReLU
from keras.layers import BatchNormalization, GlobalAveragePooling2D


def make_model():
    width = 480
    height = 640

    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_shape=(width, height, 3))
    inputs = vgg16.output
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(4096)(x)

    main_output = Dense(28, activation='sigmoid', name='main_output')(x)
    side_output = Dense(7, activation='sigmoid', name='side_output')(x)
    numbers_output = Dense(8, activation='softmax', name='numbers_output')(x)
    inputs = vgg16.input

    return inputs, main_output, side_output, numbers_output
