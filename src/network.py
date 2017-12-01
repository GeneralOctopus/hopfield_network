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


class NeuronNetwork:
    def __init__(self, path_to_patterns, size_x, size_y):
        self.path_to_patterns, self.size_x, self.size_y = (os.path.abspath(path_to_patterns) + '/'), size_x, size_y
        self.weight_matrix = zeros(shape=(size_x*size_x, size_y*size_y), dtype=float)
        self.list_of_patterns = []
    
    def _convert_image_size(self, image):
        if image.width != self.size_x or image.height != self.size_y:
            print "Converting file size to pattern:", str(self.size_x) + "x" + str(self.size_y)
            oryginal_size = image.size
            changed_size = (self.size_x, (oryginal_size[1]*self.size_y)/oryginal_size[0])
            image.thumbnail(changed_size, Image.ANTIALIAS)
        return image

    def _convert_image_to_binary(self, image, depth):
        print "Converting file to binary"
        gray_image = image.convert('L')
        binary_file = gray_image.point(lambda x: 0 if x<depth else 1, '1')
        return binary_file

    def convert_file_to_pattern(self, image):
        image = self._convert_image_size(image)
        image = self._convert_image_to_binary(image, 128)
        i = []
    
        for x in range(0, image.width):
            for y in range(0, image.height):
                i.append(float(image.getpixel((x,y)) & 0x1)) #converted image has 255 if black, 0 if white)
        return np.asarray(i, dtype=float)

    def load_patterns(self):
        print "Loading patterns, please wait..."
        for image in os.listdir(self.path_to_patterns):
            image_path = self.path_to_patterns + image
            if os.path.isfile(image_path):
                print "Loading pattern:", os.path.abspath(image)
                image = Image.open(image_path)
                converted_image = self.convert_file_to_pattern(image)
                self.list_of_patterns.append(converted_image)
            print "Pattern added to list to patterns"
        print "Patterns loaded!"

    def _prepare_weight_matrix(self):
        global weight_matrix

        # weight_matrix = matlib.rand(14400,14400)
        size = len(patterns[0])
        weight_matrix = matlib.zeros(shape=(self.size_x*self.size_x, self.size_y*self.size_y), dtype=float)
        # fill_diagonal(weight_matrix, 0)            



def test_images(output_dir='/../recognized/'):
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
    hebb_network = NeuronNetwork("converted_road_signs", 60, 60)
    hebb_network.load_patterns()
    print hebb_network.path_to_patterns
#    print hebb_network.weight_matrix
#    list_of_files = get_list_of_files()

#    print("train hebb")
#    train_hebb()
#    print("trained, testing")

#    test_images(output_dir='../recognized/hebb/')
#    print("END hebb")

#    print("train pseudo inversion")
#    pseudo_inversion()
#    test_images(output_dir='../recognized/pseudo_inv/')



if __name__ == '__main__':
    main()

