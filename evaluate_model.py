from keras.models import Model,load_model
import json
import random
import numpy as np
from PIL import Image
from loss import org_mse
def evaluate1():


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


