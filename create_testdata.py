# coding: utf-8
import os.path
import argparse
import json
import numpy as np
import matplotlib.pyplot as Plt

parser = argparse.ArgumentParser(description="import data")
parser.add_argument('--path', '-p', default='data', help="path to dataset")
parser.add_argument('--number_of_images', '-n', default=10,
                    type=int, help="number of images")

args = parser.parse_args()
img_path = args.path
number_of_images = args.number_of_images

file_name = os.listdir(img_path)[5]
file = os.path.join(img_path, file_name)
annotations = os.listdir(img_path + "/annotations")

img = Plt.imread(file)

width = img.shape[0]
height = img.shape[1]
rgb = img.shape[2]
NUMBER_OF_TRASH = 7
XYWHC = 5
# 画像フォルダと画像の枚数を指定可能：画像の大きさと拡張子を取得する。


def main():
    y = np.zeros((number_of_images, NUMBER_OF_TRASH, XYWHC))
    x = np.zeros((number_of_images, width, height, rgb))
    z = np.zeros(number_of_images)
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if img_name.endswith(('.jpg', '.png')) and number_of_images > idx - 1:
            file = os.path.join(img_path, img_name)
            img = Plt.imread(file)
            print(idx)
            x[idx - 1] = img
            annotations_path = os.path.join(
                img_path + "/annotations", annotations[idx - 1])
            with open(annotations_path, "r") as f:
                json_dict = json.load(f)
                number_of_objects = int(len(json_dict["objects"]))
                z[idx - 1] = number_of_objects
                tmp = number_of_objects
                for j in range(tmp):
                    y[idx - 1][j] = json_dict["objects"][j]["x_y_w_h"] + [1]

    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('z.npy', z)
    print("save complete")

if __name__ == "__main__":
    main()
