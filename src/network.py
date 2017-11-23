from numpy import outer, zeros
import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)
DIR_NAME = '9x9'
N = 9
DIR_NAME = 'converted_road_signs'
N = 120

patterns=[]
size = 0
weight_matrix = None
list_of_files = None

def train_hebb():
    global patterns
    global weight_matrix
    global list_of_files
    weight_matrix = zeros(shape=(N*N, N*N), dtype=float)
    i = 0
    for p in list_of_files:
        i+= 1
        pattern = convert_file_to_pattern(p)
        o = outer(pattern,pattern)
        weight_matrix += o
        print ("trained %d of %d" % (i, len(list_of_files)))
        del pattern
        del o
    weight_matrix /= len(list_of_files)
    weight_matrix = np.asmatrix(weight_matrix)


def get_list_of_files():
    l = []
    for f in os.listdir('../'+DIR_NAME):
        f = '../'+DIR_NAME+'/'+f
        if os.path.isfile(f):
            img = Image.open(f)
            if img.height * img.width <= N*N:
                l.append(f)
                print(f)
    return l


def convert_file_to_pattern(f):
    img = Image.open(f)
    img = img.convert('1')
    i = []

    for x in range(0, img.height):
        for y in range(0, img.width):
            i.append(1.0 if img.getpixel((y,x)) & 0x1 else -1.0)  # converted image has 255 if black, 0 if white)
    if len(i) < N*N:
        i += [-1] * (N*N - len(i))
    return np.asarray(i, dtype=float)


def test_images(output_dir='../recognized/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for im in list_of_files:

        p = convert_file_to_pattern(im)
        o = recognize(p)
        print(im)

        i = []
        for x in range(0, N*N):
            i.append((0, 0, 0) if o[0][x] < 0 else (255, 255, 255))

        im1 = Image.new('RGB', (N,N))
        im1.putdata(i)
        im1.save(output_dir + os.path.basename(im))


def recognize(pattern, max_steps=5):
    global weight_matrix
    for i in range(0, max_steps):
        changes = False
        pattern = np.asarray(np.dot(pattern, weight_matrix))

        for j in range(0, N*N):
            if 1 > pattern[0][j] > 0:
                changes = True
                pattern[0][j] = 1
            elif -1 < pattern[0][j] < 0:
                changes = True
                pattern[0][j] = -1

        print ("changes", changes)
        if not changes:
            print("stabilized after:", i)
            return pattern

    return pattern

def pseudo_inversion():
    global list_of_files
    global weight_matrix
    x = []
    for i, f in enumerate(list_of_files):
        x.append(convert_file_to_pattern(f))

    x = np.asmatrix(x, dtype=float).transpose()
    weight_matrix = np.dot(x, np.linalg.pinv(x))


def main():
    global weight_matrix
    global list_of_files

    list_of_files = get_list_of_files()

    print("train hebb")
    train_hebb()
    print("trained, testing")

    test_images(output_dir='../recognized/hebb/')
    print("END hebb")

    print("train pseudo inversion")
    pseudo_inversion()
    test_images(output_dir='../recognized/pseudo_inv/')



if __name__ == '__main__':
    main()

