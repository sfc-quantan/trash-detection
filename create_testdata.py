# coding: utf-8
import os.path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


NUMBER_OF_TRASH = 7
XYWHC = 5


def main():
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

    img = plt.imread(file)

    width = img.shape[0]
    height = img.shape[1]
    rgb = img.shape[2]
    y = np.zeros((number_of_images, NUMBER_OF_TRASH, XYWHC), np.uint8)
    x = np.zeros((number_of_images, width, height, rgb), dtype=np.uint8)
    z = np.zeros(number_of_images)
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if img_name.endswith(('.jpg', '.png')) and number_of_images > idx:
            file = os.path.join(img_path, img_name)
            img = plt.imread(file)
            print(idx)
            x[idx] = img
            annotations_path = os.path.join(
                img_path + "/annotations", annotations[idx])
            with open(annotations_path, "r") as f:
                json_dict = json.load(f)
                number_of_objects = int(len(json_dict["objects"]))
                z[idx] = number_of_objects
                tmp = number_of_objects
                for j in range(tmp):
                    y[idx][j] = json_dict["objects"][j]["x_y_w_h"] + [1]
                    # y[idx - 1][j] = json_dict["objects"][j]["x_y_w_h"]
                    # y = y.astype(''dtype=np.uint8)
    print(x[-1])
    np.savez_compressed('data', images=x, boxes=y, number=z)
    print("save complete")
    nap = np.load('data.npz')
    print(nap['images'], nap['boxes'],nap['number'])


if __name__ == "__main__":
    main()
