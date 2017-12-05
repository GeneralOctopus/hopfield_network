import sys
import random
from PIL import Image


def add_noise(file_name, percentage):
    image = Image.open(file_name)
    pixels = image.load()
    image_size = image.size
    amount_of_changed_pixels = (image_size[0]*image_size[1]*percentage)/100

    for pixel in range(0, amount_of_changed_pixels):
        random_x = int(random.random() * image_size[0])
        random_y = int(random.random() * image_size[1])
        pixel_value = pixels[random_x, random_y] 
        image.putpixel((random_x, random_y), 0 if pixel_value>0 else 255)
 
    image.save('test_images/noised_images/' + file_name[:-4:] + '_noise_percentage_' + percentage + '.png')


def remove_image_block(file_name, percentage):
    pass


def convert_to_binary(file_name):
    color_image = Image.open(file_name)
    gray_image = color_image.convert('L')
    binary_file = gray_image.point(lambda x: 0 if x<128 else 255, '1')
    image_size = color_image.size
    binary_file.save('binary_files/' + image_size[0] + 'x' + image_size[0] + '/' + file_name)


def resize_file(file_name, output_size):
    oryginal_image = Image.open(file_name)
    oryginal_size = oryginal_image.size
    changed_size = (output_size, (oryginal_size[1]*output_size)/oryginal_size[0])
    
    oryginal_image.thumbnail(changed_size, Image.ANTIALIAS)
    oryginal_image.save('resized_files/' + output_size + 'x' + output_size + '/' + file_name)


if __name__=='__main__':
    if len(sys.argv) < 2:
        print "Usage: python converter.py file1.py file2.py ... percentage_of_noise"

    for arg in sys.argv[1:-1:]:
        print arg
        add_noise(arg, int(sys.argv[-1]))

#if __name__=='__main__':
#    if len(sys.argv) < 2:
#        print "Usage: python converter.py file1.py file2.py ... size"
#
#    
#    for arg in sys.argv[1:-1:]:
#        print arg
#        resize_file(arg, int(sys.argv[-1]))
#        convert_to_binary('resized_files/' + sys.argv[-1] + 'x' + sys.argv[-1] + '/')
#   

