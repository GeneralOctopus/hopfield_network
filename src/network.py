from numpy import outer, zeros
import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)


class NeuronNetwork:
    def __init__(self, path_to_patterns, size_x, size_y):
        self.path_to_patterns, self.size_x, self.size_y = (os.path.abspath(path_to_patterns) + '/'), size_x, size_y
        self.weight_matrix = zeros(shape=(self.size_x*self.size_x, self.size_y*self.size_y), dtype=float)
        self.list_of_patterns = []
        self.load_patterns()
    
    def _convert_image_size(self, image):
        if image.width != self.size_x or image.height != self.size_y:
            print "Converting file size to pattern:", str(self.size_x) + "x" + str(self.size_y)
            image = image.resize((self.size_x, self.size_y), Image.ANTIALIAS)

        return image

    def _convert_image_to_binary(self, image, depth):
        print "Converting file to binary"
        image = image.convert('L')
        new_image = []

        for x in range(0, image.height):
            for y in range(0, image.width):
                new_image.append(1.0 if image.getpixel((y,x)) & 0x1 else -1.0)  # converted image has 255 if black, 0 if white)
        if len(new_image) < self.size_x*self.size_y:
            new_image += [-1.0] * (self.size_x*self.size_y - len(new_image))

        return np.asarray(new_image, dtype=float)

    def convert_file_to_pattern(self, image_path):
        image = Image.open(image_path)
        image = self._convert_image_size(image)
        image = self._convert_image_to_binary(image, 128)
        return image

    def load_patterns(self):
        print "Loading patterns, please wait..."
        for image in os.listdir(self.path_to_patterns):
            image_path = self.path_to_patterns + image
            print "Loading pattern:", image_path
            if os.path.isfile(image_path):
                self.list_of_patterns.append(image_path)
                print "Pattern added to list to patterns"
            else:
                print "Wrong pattern"
        print "Patterns loaded!"

    def train_network_by_hebb(self):
        print "Clear weight matrix before learning..."
        self.zeros_weight_matrix()
        print "Start learning by hebb method..."
        for index, pattern in enumerate(self.list_of_patterns):
            i = index + 1
            o = outer(pattern, pattern)
            self.weight_matrix += o
            print ("trained %d of %d" % (i, len(self.list_of_patterns)))
        self.weight_matrix /= len(self.list_of_patterns)
        self.weight_matrix = np.asmatrix(self.weight_matrix)
        print "End learning by hebb method..."

    def train_network_by_pseudo_inversion(self):
        print "Clear weight matrix before learning..."
        self.zeros_weight_matrix()
        print "Start learning by pseudo inversion..."
        self.x = []
        for index, pattern_path in enumerate(self.list_of_patterns):
            print pattern_path
            self.x.append(self.convert_file_to_pattern(pattern_path))

        self.x = np.asmatrix(self.x, dtype=float).transpose()
        self.weight_matrix = np.dot(self.x, np.linalg.pinv(self.x))
        print "End learning by pseudo inversion..."

    def recognize_image(self, image_path, max_steps=7):
        pattern = self.convert_file_to_pattern(image_path)
        for i in range(0, max_steps):
            changes = False
            response = np.asarray(np.dot(pattern, self.weight_matrix))

            for j in range(0, self.size_x*self.size_y):
                if 1 > response[0][j] > 0:
                    changes = True
                    response[0][j] = 1.0
                elif -1 < response[0][j] < 0:
                    changes = True
                    response[0][j] = -1.0

            print ("changes", changes)
            if not changes:
                print("stabilized after:", i)
                return response 

        return response
        
    def test_image(self, image_path):
        print "test image"
        if os.path.isfile(image_path):
            response = self.recognize_image(image_path)
            i = []

            for x in range(0, self.size_x*self.size_y):
                i.append((0, 0, 0) if response[0][x] < 0 else (255, 255, 255))
            
            im1 = Image.new('RGB', (self.size_x, self.size_y))
            im1.putdata(i)
            im1.save(os.path.abspath(image_path) + os.path.basename(image_path)[::-7] + "_recognize.png" )

    def test_cut_images(self, input_directory):
        print "Testing cut images..."
        number_of_recognized = 0
        path = input_directory + "cut/"

        for directory in os.listdir(path):
            for image_name in os.listdir(path + directory):
                self.test_image(path + directory + "/" + image_name)

    def test_noised_images(self, input_directory):
        print "Testing noised images..."
        number_of_recognized = 0
        path = input_directory + "noised/"
 
        for directory in os.listdir(path):
            for image_name in os.listdir(path + directory):
                self.test_image(path + directory + "/" + image_name)

    def test_network(self, input_directory):
        print "Testing network size", str(self.size_x) + "x" + str(self.size_y) + "..."
        self.test_cut_images(input_directory)
        self.test_noised_images(input_directory)

    def zeros_weight_matrix(self):
        self.weight_matrix = zeros(shape=(self.size_x*self.size_x, self.size_y*self.size_y), dtype=float)


def main():
    size = 60
    tests_directories = "tests/" + str(size) + 'x' + str(size) + '/'
    patterns_directory = tests_directories + "patterns/"
# 
#    hebb_network = NeuronNetwork(patterns_directory, size, size)
#    hebb_network.train_network_by_hebb()

#    hebb_network.test_network(tests_directories)
    ps_inv_network = NeuronNetwork(patterns_directory, size, size)
    ps_inv_network.train_network_by_pseudo_inversion()

    ps_inv_network.test_network(tests_directories)
#    test_images(output_dir='../recognized/hebb/')
#    print("END hebb")

#    print("train pseudo inversion")
#    pseudo_inversion()
#    test_images(output_dir='../recognized/pseudo_inv/')


if __name__ == '__main__':
    main()

