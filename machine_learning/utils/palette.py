import json
import numpy as np
from PIL import Image

with open("./utils/labels_info.json", 'r') as jf:
    dataset = json.load(jf)

palette = []
for label in dataset:
    palette.append(label["color"][0])
    palette.append(label["color"][1])
    palette.append(label["color"][2])

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask, padding_size=False):

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')

    if padding_size:
        h, w =  padding_size
        new_mask = new_mask.resize((w, h), Image.NEAREST)

    new_mask.putpalette(palette)
    return new_mask
