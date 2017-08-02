# coding: utf-8
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

