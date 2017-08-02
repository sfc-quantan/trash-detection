# coding: utf-8

import gpu_config
import tensorflow
import numpy as np
import keras
from keras import metrics
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
from loss import org_mse
from import_data import make_data
from model import make_model
from evaluate_model import evaluate1


batch_size=8
epoch=1
gpu_config.set_tensorflow([1])

(x_train,x_test,y_xywh_train,y_xywh_test,y_c_train,y_c_test,z_train,z_test)=make_data()

(inputs,main_output,side_output,numbers_output)=make_model()

model = Model(inputs=inputs, outputs=[main_output, side_output,numbers_output])
    
adam=Adam(lr=0.0003)
checkpointer=ModelCheckpoint("model1.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=190,verbose=0, mode='auto')

model.compile(optimizer=adam,loss={'main_output':org_mse, 'side_output': 'binary_crossentropy','numbers_output':'categorical_crossentropy'}, 
	metrics={'main_output': 'mae', 'side_output': 'accuracy','numbers_output':'accuracy'},
	loss_weights={'main_output': 1., 'side_output': 0.2,'numbers_output':1})

model.fit(x=x_train,y=[y_xywh_train,y_c_train,z_train],batch_size=batch_size,epochs=epoch,verbose=1,
        validation_data=(x_test,[y_xywh_test,y_c_test,z_test]),callbacks=[checkpointer,early_stopping])

score=model.evaluate(x_test,[y_xywh_test,y_c_test,z_test],verbose=1,batch_size=8)
print(score[0],score[1])

json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('param.hdf5')
model.save('model.h5') 

evaluate1()

