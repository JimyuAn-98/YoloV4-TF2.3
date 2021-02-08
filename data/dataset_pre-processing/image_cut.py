from absl import app, flags
import cv2 as cv
import numpy as np
import math
import os
from absl.flags import FLAGS

flags.DEFINE_string('fap', './data/dataset/tiny_train.txt', 'path to the former annotation txt file')
flags.DEFINE_integer('size', 416, 'the input size of your model')
flags.DEFINE_string('oip', './data/tiny_set/train/cut_images', 'path to save processed images')
flags.DEFINE_string('oivp', './data/tiny_set/train/cut_images_visualized', 'path to save visualized images')
flags.DEFINE_string('oap', './data/dataset/tiny_cut_train.txt', 'path to save processed annotation file')


def load_file():
    anno_path = FLAGS.fap   # get the path of the former annotation txt file
    with open(anno_path, "r") as f:     # open file
        txt = f.readlines()     # read by line
        annotations = [line.strip() for line in txt]
        return annotations

def cut_images():
    output_anno_path = FLAGS.oap    # get the path to save annotation file
    output_img_path = FLAGS.oip     # get the path to save images
    output_img_visualized_path = FLAGS.oivp     # get the path to save bboxes visualized imaged
    input_size = FLAGS.size         # get the input size
    anno = load_file()      # get the former annotation file

    if not os.path.exists(output_img_path):     # if the path of saving images is not exist, than create one
        os.mkdir(output_img_path)

    if os.path.exists(output_anno_path):    # if there is a annotation file which has the same name, than erase it
        os.remove(output_anno_path)

    with open(output_anno_path, 'a') as txt:    # create and open the output annotation file
        count = 1
        for line in anno:   # read in line
            lines = line.split(":")     # split the sources by ":", you can change it into yours acquirement
            img_path = lines[0]
            # By getting the name of the original picture to make it easier for naming the processed picture
            img_name = img_path.split("/")[-1].split(".")[0]
            bboxes = np.array(
                [list(map(int, box.split(","))) for box in lines[1:]]   # get the bboxes coordinate
            )
            img = cv.imread(img_path)   # read the image
            img_height = img.shape[0]
            img_width = img.shape[1]
            blocks_x = math.ceil(img_width / input_size)  # number of blocks in x axis
            blocks_y = math.ceil(img_height / input_size)  # number of blocks in y axis
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
                    new_img_path = '/'.join([output_img_path, new_img_name])
                    new_img_visualized_name = 'v_' + new_img_name
                    new_img_visualized_path = '/'.join([output_img_visualized_path, new_img_visualized_name])

                    x_cen = (x_min + x_max) / 2
                    y_cen = (y_min + y_max) / 2
                    annotation, img_cut_visualized = gen_anno(new_img_path, input_size, x_cen, y_cen, bboxes, i)    # get the annotation
                    if annotation is None:
                        pass
                    else:
                        cv.imwrite(new_img_visualized_path, img_cut_visualized)
                        cv.imwrite(new_img_path, img_cut)
                        txt.write(annotation + "\n")
            print("finish process No.%d origin image" % count)
            count += 1

def gen_anno(img, img_path, input_size, x_cen, y_cen, bboxes, i):
    annotation = img_path
    for bbox in bboxes:
        category = bbox[4]
        bx_cen = (bbox[0] + bbox[2]) / 2
        by_cen = (bbox[1] + bbox[3]) / 2
        b_l = bbox[2] - bbox[0]
        b_h = bbox[3] - bbox[1]
        dx = abs(x_cen - bx_cen)
        dy = abs(y_cen - by_cen)

        if dx > (b_l / 2 + input_size / 2) or dy > (b_h / 2 + input_size / 2):
            pass
        else:
            if dx <= (input_size / 2 - b_l / 2):  # x axis full in
                L = b_l
            else:
                L = b_l / 2 - (dx - input_size / 2)

            if dy <= (input_size / 2 - b_h / 2):  # y axis full in
                H = b_h
            else:
                H = b_h / 2 - (dy - input_size / 2)

            if (L * H) / (b_l * b_h) <= 0.2:  # less than 20% of the origin area
                pass
            else:
                if x_cen - bx_cen > 0:  # left side
                    if dx < (input_size / 2 - b_l / 2):  # full in
                        new_bx_cen = input_size / 2 - dx
                    else:
                        new_bx_cen = L / 2
                else:  # right side
                    if dx < (input_size / 2 - b_l / 2):
                        new_bx_cen = input_size / 2 + dx
                    else:
                        new_bx_cen = input_size - L / 2

                if y_cen - by_cen < 0:  # down side
                    if dy < (input_size / 2 - b_h / 2):
                        new_by_cen = input_size / 2 + dy
                    else:
                        new_by_cen = input_size - H / 2
                else:  # up side
                    if dy < (input_size / 2 - b_h / 2):
                        new_by_cen = input_size / 2 - dy
                    else:
                        new_by_cen = H / 2

                bx_min = int(new_bx_cen - L / 2)
                bx_max = int(new_bx_cen + L / 2)
                by_min = int(new_by_cen - H / 2)
                by_max = int(new_by_cen + H / 2)
                annotation += ':' + ','.join([str(bx_min), str(by_min), str(bx_max), str(by_max), str(category)])
                cv.rectangle(img, (bx_min, by_min), (bx_max, by_max), (0, 255, 0), 1)
                font = cv.FONT_ITALIC
                if int(category) == 0:
                    text = 'sea_person'
                    cv.putText(img, text, (bx_min, by_min - 5), font, 0.5, (0, 255, 0), 1)
                elif int(category) == 1:
                    text = 'earth_person'
                    cv.putText(img, text, (bx_min, by_min - 5), font, 0.5, (0, 0, 255), 1)
                else:
                    text = 'None'
                    cv.putText(img, text, (bx_min, by_min - 5), font, 0.5, (255, 0, 0), 1)

    if annotation == img_path:
        return None, None
    else:
        return annotation, img

def main(_argv):
    cut_images()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
