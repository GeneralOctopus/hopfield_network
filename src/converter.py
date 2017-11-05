import sys
from PIL import Image


def convert_to_binary(file_name):
    color_image = Image.open(file_name)
    gray_image = color_image.convert('L')
    binary_file = gray_image.point(lambda x: 0 if x<128 else 255, '1')
    binary_file.save(file_name[0:-8:] + '_mono.png')


if __name__=='__main__':
    if len(sys.argv) < 2:
        print "Usage: python converter.py file1.py file2.py ..."

    for arg in sys.argv[1::]:
        convert_to_binary(arg)
