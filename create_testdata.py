# coding: utf-8
import numpy as np
import json
import matplotlib.pyplot as Plt

k = 1367
width = 480
height = 640
gominum = 7
xywhc = 5


y = np.zeros((k, gominum, xywhc))
x = np.arange(k*width*height*3).reshape(k, width, height, 3)

for i in range(1, k+1):
    img = Plt.imread("./1/"+str(i)+".jpg")
    x[i-1] = img

for i in range(1, k+1):
    f = open("./1/annotations/"+str(i)+".json", "r")
    json_dict = json.load(f)
    tmp = len(json_dict["objects"])
    for j in range(tmp):
        y[i-1][j] = json_dict["objects"][j]["x_y_w_h"]+[1]

z = np.zeros(k)
for i in range(1, k+1):
    f = open("./1/annotations/"+str(i)+".json", "r")
    json_dict = json.load(f)
    z[i-1] = int(len(json_dict["objects"]))

np.save('x.npy', x)
np.save('y.npy', y)
np.save('z.npy', z)
