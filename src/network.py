from numpy import matlib, fill_diagonal, outer, zeros, ndarray
import numpy as np
import os
import sys
from PIL import Image

import memory_profiler

patterns=[]
size = 0
weight_matrix = None
list_of_files = None


# @profile
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
            del i

        print("len of patterns",len(patterns))
        print("len of pattern", len(patterns[0]))
        print("image size", img.width, img.height, img.width * img.height)


def prepare_weight_matrix():
    global weight_matrix

    # weight_matrix = matlib.rand(14400,14400)
    size = len(patterns[0])
    weight_matrix = matlib.zeros(shape=(120*120, 120*120), dtype=float)
    # fill_diagonal(weight_matrix, 0)

@profile
def train():
    global patterns
    global weight_matrix
    global list_of_files
    weight_matrix = zeros(shape=(120*120, 120*120), dtype=float)
    i = 0
    for p in list_of_files:
        i+= 1
        pattern = convert_file_to_pattern(p)
        o = outer(pattern,pattern)
        weight_matrix += o
        # pattern = None
        print ("trained %d of %d" % (i, len(list_of_files)))
        del pattern
        del o
    weight_matrix /= len(list_of_files)


def get_list_of_files():
    l = []
    for f in os.listdir('../converted_road_signs'):
        f = '../converted_road_signs/' + f
        if os.path.isfile(f):
            img = Image.open(f)
            if img.height == img.width == 120:
                l.append(f)
    return l

# @profile
def convert_file_to_pattern(f):
    img = Image.open(f)
    img = img.convert('1')
    i = []

    for x in range(0, img.width):
        for y in range(0, img.height):
            i.append(float(img.getpixel((x, y)) & 0x1))  # converted image has 255 if black, 0 if white)
    return np.asarray(i, dtype=float)

# @profile
def test_on_oryginals():
    for i, pattern in enumerate(list_of_files):
        p = convert_file_to_pattern(pattern)
        if p != outer(p, weight_matrix):
            print("Fail in pattern no", i)
        del p


@profile
def main():
    global weight_matrix
    global list_of_files

    print ("loading patterns")
    # load_patterns()
    list_of_files = get_list_of_files()
    print ("patterns loaded")
    # prepare_weight_matrix()
    print ("matrix prepared")
    print("train")
    train()
    print ("trained")
    print (weight_matrix)
    print(weight_matrix.max())
    print(sys.getsizeof(patterns), sys.getsizeof(weight_matrix))
    test_on_oryginals()
    print ("END")

if __name__ == '__main__':
    main()

