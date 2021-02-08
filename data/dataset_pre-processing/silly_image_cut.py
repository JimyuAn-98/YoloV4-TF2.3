"""
Include function:
    1. Cut the images into blocks of fixed resolution
"""

import cv2 as cv
import math
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('input', './test/input/tiny_test.jpg', 'path to save processed images')
flags.DEFINE_integer('size', 416, 'the input size of your model')
flags.DEFINE_string('output', './test/input', 'path to save processed images')


def pure_image_cut():
    img_path = FLAGS.input
    input_size = FLAGS.size
    output_path = FLAGS.output

    img_name = img_path.split("/")[-1].split(".")[0]
    img = cv.imread(img_path)   # read the image
    img_height = img.shape[0]
    img_width = img.shape[1]
    blocks_x = math.ceil(img_width / input_size)  # number of blocks in x axis
    blocks_y = math.ceil(img_height / input_size)  # number of blocks in y axis
    count = 1
    for i in range(blocks_x):
        for j in range(blocks_y):
            x_min = 0 + i * input_size
            x_max = input_size + x_min
            y_min = 0 + j * input_size
            y_max = input_size + y_min
            if x_max > img_width:
                x_min = img_width - input_size
                x_max = img_width
            if y_max > img_height:
                y_min = img_height - input_size
                y_max = img_height
            img_cut = img[y_min:y_max, x_min:x_max]
            new_img_name = img_name + '_%d' % i + '_%d.jpg' % j
            new_img_path = '/'.join([output_path, new_img_name])
            cv.imwrite(new_img_path, img_cut)
            print('Done No. %d'%count)
            count += 1

def main(_argv):
    pure_image_cut()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass