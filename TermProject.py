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
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint,EarlyStopping
from PIL import Image
from keras import backend as K
from loss import org_mse
import random
k=1367
gpu_config.set_tensorflow([1])
width=480
height=640
gominum=7
xywhc=5

x=np.load("x.npy")
y=np.load("y.npy")
z=np.load("z.npy")
z=keras.utils.to_categorical(z,8)
x_train, x_test, y_train, y_test,z_train,z_test = train_test_split(x, y,z, train_size=0.8)


y_xywh_train=np.zeros((len(y_train),gominum,4))
y_c_train=np.zeros((len(y_train),gominum))
for i in range(len(y_train)):
    for j in range(len(y_train[0])):
        y_xywh_train[i][j]=y_train[i][j][:4]
        if(y_train[i][j][4]==1):
            y_c_train[i][j]=1
            

y_xywh_test=np.zeros((len(y_test),gominum,4))
y_c_test=np.zeros((len(y_test),gominum))                      
for i in range(len(y_test)):
    for j in range(len(y_test[0])):
        y_xywh_test[i][j]=y_test[i][j][:4]
        if(y_test[i][j][4]==1):
            y_c_test[i][j]=1

for i in range(len(y_xywh_train)):
    for j in range(len(y_xywh_train[0])):
        y_xywh_train[i][j][0]= y_xywh_train[i][j][0]/640
        y_xywh_train[i][j][1]= y_xywh_train[i][j][1]/480
        y_xywh_train[i][j][2]= y_xywh_train[i][j][2]/640
        y_xywh_train[i][j][3]= y_xywh_train[i][j][3]/480

for i in range(len(y_xywh_test)):
    for j in range(len(y_xywh_test[0])):
        y_xywh_test[i][j][0]= y_xywh_test[i][j][0]/640
        y_xywh_test[i][j][1]= y_xywh_test[i][j][1]/480
        y_xywh_test[i][j][2]= y_xywh_test[i][j][2]/640
        y_xywh_test[i][j][3]= y_xywh_test[i][j][3]/480
        

print(y_xywh_train[77])  
print(x_train.shape)
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
y_xywh_train=y_xywh_train.reshape(-1,28).astype('float32')
y_xywh_test=y_xywh_test.reshape(-1,28).astype('float32')

y_c_train=y_c_train.reshape(-1,7)
y_c_test=y_c_test.reshape(-1,7)

del x,y 
gc.collect()

batch_size=8
epoch=5
inceptionv3 = InceptionV3(weights='imagenet', include_top=False,input_shape=(width,height,3))
vgg16 = VGG16(include_top=False, weights='imagenet' ,input_shape=(width,height,3))
inputs=vgg16.output

x=GlobalAveragePooling2D()(inputs)
x=Dense(4096)(x)
x=LeakyReLU(alpha=0.1)(x)
x=BatchNormalization()(x)
x=Dense(4096)(x)
x=LeakyReLU(alpha=0.1)(x)
x=BatchNormalization()(x)
x=Dense(4096)(x)

#inputs =Input(shape=(480,640,3))
#x=Conv2D(32,kernel_size=(3,3))(inputs)
#x=LeakyReLU(alpha=0.1)(x)
#x=Conv2D(32,(3,3))(x)
#x=LeakyReLU(alpha=0.1)(x)
#x=MaxPooling2D((2,2))(x)
#x=Dropout(0.25)(x)
#x=Flatten()(x)
#x=Dense(32)(x)

main_output=Dense(28,activation='sigmoid',name='main_output')(x)

side_output=Dense(7, activation='sigmoid',name='side_output')(x)
numbers_output=Dense(8,activation='softmax',name='numbers_output')(x)
model = Model(inputs=vgg16.input, outputs=[main_output, side_output,numbers_output])
#model = Model(inputs=inceptionv3.input,output=[main_output, side_output,numbers_output])
#model=Model(inputs=inceptionv3.input,output=[numbers_output])

#for layer in model.layers[:15]:
#        layer.trainable = False


#model = Model(inputs=inceptionv3.input, outputs=[main_output, side_output,numbers_output])
adam=Adam(lr=0.0003)
#model = Model(inputs=inputs, outputs=[main_output, side_output])

checkpointer=ModelCheckpoint("model1.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(optimizer=adam,loss={'main_output':org_mse, 'side_output': 'binary_crossentropy','numbers_output':'categorical_crossentropy'}, metrics={'main_output': 'mae', 'side_output': 'accuracy','numbers_output':'accuracy'},loss_weights={'main_output': 1., 'side_output': 0.2,'numbers_output':1})


print(model.summary())
early_stopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=190,verbose=0, mode='auto')
model.fit(x=x_train,y=[y_xywh_train,y_c_train,z_train],batch_size=batch_size,epochs=epoch,verbose=1,
        validation_data=(x_test,[y_xywh_test,y_c_test,z_test]),callbacks=[checkpointer,early_stopping])



score=model.evaluate(x_test,[y_xywh_test,y_c_test,z_test],verbose=1,batch_size=8)
print(score[0],score[1])
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save('model.h5') 
model.save_weights('param.hdf5')

model = load_model("model.h5",{"org_mse":org_mse})

k=random.randint(1,200)
image = Image.open("1/"+str(k)+".jpg")
f=open("1/annotations/"+str(k)+".json","r")
json_dict=json.load(f)
#image=Plt.imread("1/122.jpg")
data = np.asarray(image, dtype=float)/255
print(data.shape)
data = data.reshape(1,480,640,3)

testdata=model.predict(data)
print(testdata)
test1=testdata[0]
test2=testdata[1]
test1=np.array(test1)
test2=np.array(test2)
print(test1.shape)

test1=test1.reshape(7,4)

for i in range(7):
    test1[i][0]=test1[i][0]*640
    test1[i][1]=test1[i][1]*480
    test1[i][2]=test1[i][2]*640
    test1[i][3]=test1[i][3]*480


print(test1.astype('int32'))
print(test2)
for i in range(len(json_dict["objects"])):
    print(json_dict["objects"][i]["x_y_w_h"])



