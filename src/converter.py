import sys
from PIL import Image


def convert_to_binary(file_name):
    color_image = Image.open(file_name)
    gray_image = color_image.convert('L')
    binary_file = gray_image.point(lambda x: 0 if x<128 else 255, '1')
    binary_file.save(file_name[0:-8:] + '_mono.png')


def resize_file(file_name, output_size):
    oryginal_image = Image.open(file_name)
    oryginal_size = oryginal_image.size
    changed_size = (output_size, (oryginal_size[1]*output_size)/oryginal_size[0])
    
    oryginal_image.thumbnail(changed_size, Image.ANTIALIAS)
    oryginal_image.save(file_name[0:-8:] + '_resized.png')


if __name__=='__main__':
    if len(sys.argv) < 2:
        print "Usage: python converter.py file1.py file2.py ... size"

    for arg in sys.argv[1::]:
        convert_to_binary(arg)

