# Writen by Zhenyao 09/03/2020
from skimage import io
from collections import namedtuple
import os
import numpy as np
from PIL import Image


Label = namedtuple('Label', ['name', 'id', 'color'])

atlantis = [
    # name id  color
    Label('background', 0, (0, 0, 0)),
    Label('river', 1, (160, 96, 224))
]


def csv_writer(path, input_list, fields=["label", "pixels", "counter"]):
    import csv
    csv_path = os.path.join(path, "labels_stat.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fields)

        csv_writer.writeheader()
        for item in input_list:
            csv_writer.writerow(item)


def fun_color2id(img, labels=atlantis):

    color2id = {label.color: label.id for label in labels}
    num_of_classes = len(labels)

    w, h = img.size
    id_map = np.zeros((h, w), dtype=np.uint8)
    pixels_list = np.zeros(num_of_classes, dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            color = img.getpixel((x, y))
            id_map[y, x] = color2id[color]
            pixels_list[color2id[color]] += 1
    return id_map, pixels_list


def main(function, path, labels=atlantis):

    masks_path = os.path.join(path, "test")
    SegID_path = os.path.join(path, "test_")

    try:
        os.makedirs(SegID_path)
    except FileExistsError as err:
        print(f"FileExistsError: {err}")

    for root, dirs, imgs in os.walk(masks_path):

        total_pixels = np.zeros(len(labels))
        total_labels = np.zeros(len(labels))
        for img in imgs:
            if img.endswith(".png"):
                mask = Image.open(os.path.join(root, img))
                id_map, pixels_list = fun_color2id(mask)
                total_pixels += pixels_list
                io.imsave(os.path.join(SegID_path, img), id_map)
                for unq_id in np.unique(id_map):
                    total_labels[unq_id] += 1

    dir_pixels = []
    for label in labels:
        img_dict = {}
        img_dict["label"] = label.name
        img_dict["pixels"] = total_pixels[label.id]
        img_dict["counter"] = total_labels[label.id]
        dir_pixels.append(img_dict)

    csv_writer(path, dir_pixels)
    print(f"{path} id done!")


if __name__ == "__main__":
    import sys
    # main(fun_color2id, sys.argv[1])
    main(fun_color2id, "dataset/masks/")
