import os
import sys
import random
from PIL import Image


class Converter:
    def __init__(self, images_list, size):
        self.size = size
        self.images_list = images_list
        self.path_to_tests = 'tests/'+str(self.size)+'x'+str(self.size)
        self.path_to_patterns = self.path_to_tests+'/patterns/'
        self.patterns_list = []

    def _resize_file(self, file_name):
        oryginal_image = Image.open(file_name)
        oryginal_size = oryginal_image.size
        changed_size = (self.size, (oryginal_size[1]*self.size)/oryginal_size[0])
        
        oryginal_image.thumbnail(changed_size, Image.ANTIALIAS)
        return oryginal_image

    def _convert_to_binary(self, color_image):
        gray_image = color_image.convert('L')
        binary_file = gray_image.point(lambda x: 0 if x<128 else 255, '1')
        return binary_file

    def _prepare_patterns(self):
        for image in self.images_list:
            resized_image = self._resize_file(image)
            binary_image = self._convert_to_binary(resized_image)
            binary_image.save(self.path_to_patterns+os.path.basename(image))

    def _get_patterns_list(self):
        absolute_path_to_patterns = os.path.abspath(self.path_to_patterns)

        return [absolute_path_to_patterns+'/'+image for image in os.listdir(absolute_path_to_patterns) if os.path.isfile(absolute_path_to_patterns+'/'+image)]

    def _add_noise(self, file_name, percent):
        image = Image.open(file_name)
        pixels = image.load()
        image_size = image.size
        amount_of_changed_pixels = (image_size[0]*image_size[1]*percent)/100
    
        for pixel in range(0, amount_of_changed_pixels):
            random_x = int(random.random() * image_size[0])
            random_y = int(random.random() * image_size[1])
            pixel_value = pixels[random_x, random_y] 
            image.putpixel((random_x, random_y), 0 if pixel_value>0 else 255)

        image.save(self.path_to_tests+'/noised/'+str(percent)+'_percent_of_noise/'+os.path.basename(file_name))

    def _remove_image_block(self, file_name, percent):
        image = Image.open(file_name)
        image_size = image.size
        removed_lines = (image_size[1]*percent)/100

        for x in range(0, image_size[0]):
            for y in range(0, removed_lines):
                image.putpixel((x, y), 0)
        image.save(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_begin/'+os.path.basename(file_name))

        image = Image.open(file_name)
        for x in range(0, image_size[0]):
            for y in range(image_size[1]-removed_lines, image_size[1]):
                image.putpixel((x, y), 0)
        image.save(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_end/'+os.path.basename(file_name))

    def _generate_noised_files(self):
        for percent in range(10, 91, 20):
            if not os.path.exists(self.path_to_tests+'/noised/'+str(percent)+'_percent_of_noise/'):
                os.makedirs(self.path_to_tests+'/noised/'+str(percent)+'_percent_of_noise/')
            for image in self.patterns_list:
                self._add_noise(image, percent)

    def _generate_cut_files(self):
        for percent in range(20, 81, 15):
            if not os.path.exists(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_end/'):
                os.makedirs(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_end/')
            if not os.path.exists(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_begin/'):
                os.makedirs(self.path_to_tests+'/cut/'+str(percent)+'_percent_at_begin/')
            for image in self.patterns_list:
                self._remove_image_block(image, percent)

    def prepare_test_framework(self):
        if not os.path.exists(self.path_to_patterns):
            os.makedirs(self.path_to_patterns)
 
        print "Generating binary patterns {0} x {1} ...".format(self.size, self.size)
        self._prepare_patterns()
        self.patterns_list = self._get_patterns_list()
        print "Generating noise..."
        self._generate_noised_files()
        print "Cut patterns..."
        self._generate_cut_files()
        print "Tests generated!"


if __name__=='__main__':
    if len(sys.argv) < 2:
        print "Usage: python converter.py file1.py file2.py ... size"

    print "Amount of files ", len(sys.argv[1:-1:])
    print "Size {0} x {1} ...".format(sys.argv[-1], sys.argv[-1])
    converter = Converter(sys.argv[1:-1:], int(sys.argv[-1]))
    converter.prepare_test_framework()

