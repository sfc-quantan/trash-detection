# coding: utf-8
import gc
import gpu_config
import tensorflow
from keras.preprocessing.image import load_img, img_to_array,array_to_img,ImageDataGenerator
import matplotlib.pyplot as Plt
import keras
from keras import metrics
from keras.models import Sequential,Model,load_model,model_from_json
from keras.layers import Dense,Activation,Input,LeakyReLU
from keras.layers import Conv2D,Flatten,BatchNormalization
from keras.layers import MaxPooling2D,Dropout,GlobalAveragePooling2D
import numpy as np
import json
from keras.optimizers import Adam
import pprint
from sklearn.cross_validation import train_test_split
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras import backend as K

def org_mse(y_true, y_pred):
   y_true=K.reshape(y_true,(-1,7,4))
   xy=y_true[:,:,0:2]
   wh=y_true[:,:,2:4]
   
   y_pred=K.reshape(y_pred, (-1,7,4))
   xy2=y_pred[:,:,0:2]
   wh2=y_true[:,:,2:4]

   temp=K.square(xy2-xy)
   temp=temp+K.square(K.sqrt(wh2)-K.sqrt(wh))
   return temp

