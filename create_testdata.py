# coding: utf-8
import json
import numpy as np
import matplotlib.pyplot as Plt

K = 10
WIDTH = 480
HEIGHT = 640
GOMINUM = 7
XYWHC = 5

y = np.zeros((K, GOMINUM, XYWHC))
x = np.arange(K * WIDTH * HEIGHT * 3).reshape(K, WIDTH, HEIGHT, 3)

for i in range(1, K + 1):
    img = Plt.imread("./1/{}.jpg".format(i))
    x[i - 1] = img

for i in range(1, K + 1):
    f = open("./1/annotations/{}.json".format(i), "r")
    json_dict = json.load(f)
    tmp = len(json_dict["objects"])
    for j in range(tmp):
        y[i - 1][j] = json_dict["objects"][j]["x_y_w_h"] + [1]

z = np.zeros(K)
for i in range(1, K + 1):
    f = open("./1/annotations/{}.json".format(i), "r")
    json_dict = json.load(f)
    z[i - 1] = int(len(json_dict["objects"]))

np.save('x.npy', x)
np.save('y.npy', y)
np.save('z.npy', z)
