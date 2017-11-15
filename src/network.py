from numpy import matlib, fill_diagonal, outer, zeros, ndarray
import numpy as np
import os
import sys
from PIL import Image

patterns=[]
size = 0
weight_matrix = None



def load_patterns():
    img = None
    for f in os.listdir('../converted_road_signs'):
        f = '../converted_road_signs/' + f
        if os.path.isfile(f):
            img = Image.open(f)
            img = img.convert('1')
            i =[]
            if not img.width == img.height == 120:
                continue

            for x in range(0, img.width):
                for y in range(0, img.height):
                    i.append(float(img.getpixel((x,y)) & 0x1)) #converted image has 255 if black, 0 if white)
            patterns.append(i)

        print ("len of patterns",len(patterns))
        print("len of pattern", len(patterns[0]))
        print("image size", img.width, img.height, img.width * img.height)


def prepare_weight_matrix():
    global weight_matrix

    # weight_matrix = matlib.rand(14400,14400)
    size = len(patterns[0])
    weight_matrix = matlib.zeros(shape=(120*120, 120*120), dtype=float)
    # fill_diagonal(weight_matrix, 0)


def train():
    global patterns
    global weight_matrix
    weight_matrix = zeros(shape=(120*120, 120*120), dtype=float)
    i = 0
    for pattern in patterns:
        i+= 1
        pattern = np.asarray(pattern, dtype=float)
        out = outer(pattern, pattern)

        weight_matrix += out
        print ("trained %d of %d" % (i, len(patterns)))
    weight_matrix /= 120*120

print ("loading patterns")
load_patterns()
print ("patterns loaded")
# prepare_weight_matrix()
print ("matrix prepared")
print("train")
train()
print ("trained")
print (weight_matrix)
print(weight_matrix.max())


